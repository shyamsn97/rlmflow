import asyncio

from helpers import StubLLM
from pydantic import BaseModel

from rlmflow import (
    SYSTEM_PROMPT,
    ActionNode,
    AgentConfig,
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    FinalQuery,
    Flow,
    LLMOutput,
    Node,
    OutputNode,
    PlanQuery,
    PromptBuilder,
    Runtime,
    UserQuery,
    start,
)
from rlmflow.graph.nodes import (
    EXECUTION_POLICY,
    FIRST_TURN_SAFEGUARD,
    LAST_TURN_ADDENDUM,
    PLANNING_ACTION,
    TURN_ONE_FINISH_SENTENCE,
    USER_PROMPT,
    WORKER_ADDENDUM,
)
from rlmflow.prompts.prompts import MAX_STATIC_PROMPT_CHARS


def take(flow, node):
    return asyncio.run(flow.step(node))


def plan_instruction(agent):
    turns = agent.llm_turns()
    max_iters = agent.config.max_iters
    if max_iters is None:
        body = f"Turn {turns + 1}:"
    else:
        body = USER_PROMPT.format(iter_1=turns + 1, max_iter=max_iters)
    if turns == 0:
        body = FIRST_TURN_SAFEGUARD + body
    return f"{body}\n\n{PLANNING_ACTION}"


def test_the_default_prompt_stays_under_its_size_ceiling():
    assert len(SYSTEM_PROMPT) <= MAX_STATIC_PROMPT_CHARS, (
        f"static prompt is {len(SYSTEM_PROMPT)} chars against a {MAX_STATIC_PROMPT_CHARS} ceiling"
    )
    assert "Recursive Language Model" in SYSTEM_PROMPT
    assert EXECUTION_POLICY in SYSTEM_PROMPT


def test_messages_are_a_pure_projection_of_the_transcript():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query")
    reply = LLMOutput(content="assistant", code="print('x')")
    root.append(reply)
    reply.append(ExecOutput(content="observation"))
    before = [node.id for node in root.transcript()]

    messages = flow.build_messages(root.frontier)

    assert [node.id for node in root.transcript()] == before
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-1]["content"].startswith(
        "observation\n\nEnd the REPL block normally to continue."
    )
    assert messages[-1]["content"].endswith("- finish(answer) — Submit your final answer.")


def test_a_user_query_subclass_projects_without_a_builder_edit():
    class Note(UserQuery):
        pass

    root = start("query")
    note = Note(content="from a subclass")
    root.append(note)

    assert note.render() == [{"role": "user", "content": "from a subclass"}]
    assert note.project()[-1] == {"role": "user", "content": "from a subclass"}


def test_project_flattens_multi_message_nodes_and_skips_invisible_nodes():
    class Exchange(Node):
        def render(self) -> list[dict[str, str]]:
            return [
                {"role": "assistant", "content": "ack"},
                {"role": "user", "content": self.content},
            ]

    root = start("original")
    hidden = ExecAction(code="pass")
    root.append(hidden)
    exchange = Exchange(content="review this")
    hidden.append(exchange)

    assert exchange.project(keep=1) == [
        {"role": "user", "content": "review this"},
    ]
    assert exchange.project(keep=2) == [
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "review this"},
    ]
    assert exchange.project(keep=3) == [
        {"role": "user", "content": "original"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "review this"},
    ]


def test_custom_renderer_applies_only_to_the_current_frontier():
    root = start("original")
    assistant = LLMOutput(content="canonical assistant")
    root.append(assistant)
    frontier = ExecOutput(content="canonical observation")
    assistant.append(frontier)

    def render_current(_runtime: Runtime, _node: Node) -> list[dict[str, str]]:
        return [{"role": "user", "content": "custom frontier"}]

    messages = Flow(
        StubLLM(lambda _messages: "unused"),
        render_fn=render_current,
    ).build_messages(frontier)

    assert [message["content"] for message in messages[1:3]] == [
        "original",
        "canonical assistant",
    ]
    assert messages[-1]["content"].startswith(
        "custom frontier\n\nEnd the REPL block normally to continue."
    )


def test_default_renderer_keeps_current_output_with_background_status():
    root = start("query", keep_n_messages=2)
    reply = LLMOutput(content="working")
    root.append(reply)
    action = ExecAction(code="pass")
    reply.append(action)
    action.append(
        AgentStart(
            content="child",
            config=root.config.child("child"),
        )
    )
    frontier = ExecOutput(content="observation")
    action.append(frontier)

    messages = Flow(StubLLM(lambda _messages: "unused")).build_messages(frontier)

    assert [message["role"] for message in messages[-2:]] == ["user", "user"]
    assert "Background subagents:" in messages[-2]["content"]
    assert messages[-1]["content"].startswith(
        "observation\n\nEnd the REPL block normally to continue."
    )


def test_errors_are_projected_as_user_observations():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query")
    root.append(ErrorOutput(content="NameError: missing"))

    assert flow.build_messages(root.frontier)[-1]["role"] == "user"
    assert flow.build_messages(root.frontier)[-1]["content"].startswith(
        "NameError: missing\n\nEnd the REPL block normally to continue."
    )


def test_explicit_output_schema_is_in_system_prompt():
    class Answer(BaseModel):
        value: str

    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query", output_schema=Answer.model_json_schema())

    assert '"value"' in flow.build_messages(root.frontier)[0]["content"]


def test_child_output_schema_is_in_the_child_prompt_not_the_root():
    from rlmflow import CODING_HANDOFF_SCHEMA

    root = start("query", max_depth=1, child_output_schema=CODING_HANDOFF_SCHEMA)
    leaf = start("leaf", config=root.config.child("leaf"))
    flow = Flow(StubLLM(lambda _messages: "unused"))

    root_prompt = flow.build_messages(root.frontier)[0]["content"]
    leaf_prompt = flow.build_messages(leaf.frontier)[0]["content"]
    assert "diff_summary" not in root_prompt
    assert "diff_summary" in leaf_prompt
    assert "followups" in leaf_prompt


def test_structured_child_output_guidance_is_on_by_default():
    root = start("query", max_depth=1)
    default_prompt = Flow(StubLLM(lambda _messages: "unused")).build_messages(root.frontier)[0][
        "content"
    ]
    disabled_prompt = Flow(
        StubLLM(lambda _messages: "unused"),
        enable_structured_output=False,
    ).build_messages(root.frontier)[0]["content"]

    assert "Child results are JSON-compatible Python values by default" in default_prompt
    assert "Child results are JSON-compatible Python values by default" not in disabled_prompt
    assert "object is listed as fields" in default_prompt
    assert "object is listed as fields" not in disabled_prompt
    assert default_prompt.count("Example —") == 3
    assert disabled_prompt.count("Example —") == 3
    assert "## Examples" not in default_prompt


def test_in_category_snippets_follow_the_matching_capability():
    root = start("query", max_depth=1)
    leaf = start("leaf", config=root.config.child("leaf"))
    baseline = Flow(StubLLM(lambda _messages: "unused")).build_messages(
        start("baseline", max_depth=0).frontier
    )[0]["content"]
    plain = Flow(StubLLM(lambda _messages: "unused")).build_messages(root.frontier)[0]["content"]
    deeper = Flow(StubLLM(lambda _messages: "unused")).build_messages(
        start("deeper", max_depth=2).frontier
    )[0]["content"]
    queried = Flow(
        StubLLM(lambda _messages: "unused"),
        use_llm_query=True,
    ).build_messages(root.frontier)[0]["content"]
    tree = Flow(
        StubLLM(lambda _messages: "unused"),
        use_agent_tree=True,
    ).build_messages(root.frontier)[0]["content"]
    leaf_prompt = Flow(StubLLM(lambda _messages: "unused")).build_messages(leaf.frontier)[0][
        "content"
    ]

    inspect = 'print({name: value[:500] for name, value in INPUTS.items()} or "(empty INPUTS)")'
    query = "Use `llm_query` for a one-shot transform"
    launch_wait = "print(await asyncio.gather(first, second))"
    agents = "`AGENTS` is a read-only snapshot"

    assert inspect in plain
    assert inspect in leaf_prompt
    assert baseline.count("Example —") == leaf_prompt.count("Example —") == 1
    assert plain.count("Example —") == deeper.count("Example —") == 3
    assert all(
        len(prompt) <= MAX_STATIC_PROMPT_CHARS for prompt in (baseline, plain, deeper, leaf_prompt)
    )
    assert query not in plain
    assert query in queried
    assert launch_wait in plain
    assert launch_wait not in baseline
    assert launch_wait not in leaf_prompt
    assert "Produce deliverable A." in plain
    assert 'model="default"' in plain
    assert "Use `llm_query_batched` for many independent one-shot transforms." in queried
    assert agents not in plain
    assert agents in tree
    assert "## Examples" not in queried
    assert "## Examples" not in tree


def test_this_agent_schema_section_includes_a_finish_example():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    plain = start("query", max_depth=1)
    structured = start(
        "query",
        max_depth=1,
        output_schema={
            "type": "object",
            "properties": {"n": {"type": "number"}},
            "required": ["n"],
        },
    )
    plain_prompt = flow.build_messages(plain.frontier)[0]["content"]
    structured_prompt = flow.build_messages(structured.frontier)[0]["content"]

    assert 'finish({"n": 41})' not in plain_prompt
    assert "This run requires structured output" in structured_prompt
    assert 'finish({"n": 41})' in structured_prompt


def test_prompt_documents_the_complete_builtin_repl_api():
    flow = Flow(StubLLM(lambda _messages: "unused"), use_llm_query=True)
    root = flow.start("query", max_depth=1)
    plan = take(flow, root.frontier)

    prompt = flow.build_messages(plan)[0]["content"]
    compact = " ".join(prompt.split())

    assert "You are a Recursive Language Model" in prompt
    for field in (
        "goal: str",
        "name: str | None",
        "inputs: dict[str, str | AgentHandle] | None",
        "model: str",
        "output_schema: object | None",
        "prompt_profile: str | None",
        "reuse_repl: bool",
    ):
        assert field in prompt
    assert "await launch_subagent(" in prompt
    assert "await llm_query(prompt: str" in prompt
    assert "await llm_query_batched(prompts: list[str]" in prompt
    assert "`model=` on `launch_subagent` / `llm_query` is required:" in prompt
    assert "- `default` — current model" in prompt
    assert "`finish(answer: object) -> None`" in compact
    assert "`await launch_subagent(goal: str, *, model: str," in compact
    assert "`INPUTS: dict[str, str]`" in prompt
    assert "the REPL persists across turns" in compact
    assert "Only printed stdout is returned" in compact
    assert "4,000 characters" not in compact
    assert "Outputs over ~20K characters are truncated" in compact
    assert "Launch ready independent work together" in compact
    assert "known work product depends on another" in compact
    assert "Consume and validate child results" in compact
    assert "print(await asyncio.gather(first, second))" in prompt
    assert "declare a known dependency without waiting first" in prompt
    assert 'inputs={"artifact": source}' in prompt
    assert "print(await review)" in prompt
    assert "after:" not in prompt
    assert "wait: bool" not in prompt
    assert "done(" not in prompt
    assert "`PLAN`" not in prompt
    assert "## Examples" not in prompt


def test_prompt_exposes_each_model_choice_and_marks_the_current_one():
    default = StubLLM(lambda _messages: "unused")
    fast = StubLLM(lambda _messages: "unused")
    flow = Flow(default, llm_clients={"fast": fast})
    root = start("query", model="fast", max_depth=1)

    prompt = flow.build_messages(root.frontier)[0]["content"]

    assert "- `default`" in prompt
    assert "- `fast` — current model" in prompt
    assert "using model key **`fast`**" in prompt


def test_delegation_model_is_marked_in_the_model_manifest():
    default = StubLLM(lambda _messages: "unused")
    worker = StubLLM(lambda _messages: "unused")
    flow = Flow(
        default,
        llm_clients={"worker": worker},
        delegation_model="worker",
        use_llm_query=True,
    )
    root = flow.start("query", max_depth=1)

    prompt = flow.build_messages(root.frontier)[0]["content"]

    assert "- `default` — current model" in prompt
    assert "- `worker` — default for delegated work" in prompt
    assert 'model="worker"' in prompt
    spawn_example = prompt[prompt.index("Example — launch ready") : prompt.index("Use `llm_query`")]
    assert 'model="worker"' in spawn_example
    assert 'model="default"' not in spawn_example


def test_default_prompt_guides_iterative_investigation_and_grounded_delegation():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", inputs={"context": "two substantial scopes"}, max_depth=1)
    plan = take(flow, root.frontier)

    prompt = flow.build_messages(plan)[0]["content"]
    compact = " ".join(prompt.split())

    assert "Python REPL" in compact
    assert "must launch" not in compact
    assert "Work allocation and graph policy" in compact
    assert "Launch ready independent work together" in compact
    assert "prerequisite `AgentHandle` as a named `inputs` value" in compact
    assert EXECUTION_POLICY in prompt
    assert "## Examples" not in prompt
    assert "print(await asyncio.gather(first, second))" in prompt
    assert "await asyncio.gather(" not in EXECUTION_POLICY


def test_every_agent_starts_with_a_plan_and_gets_the_shared_execution_policy():
    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        root_config=AgentConfig(max_depth=1),
    )
    eligible = flow.start("query", inputs={"context": "material"})
    without_inputs = flow.start("query")
    leaf = start(
        "leaf",
        config=eligible.config.child("leaf"),
        inputs={"scope": "focused material"},
    )

    eligible_plan = take(flow, eligible.frontier)
    without_inputs_plan = take(flow, without_inputs.frontier)
    leaf_plan = take(flow, leaf.frontier)

    assert isinstance(eligible_plan, PlanQuery)
    assert eligible_plan.instruction() == plan_instruction(eligible)
    assert isinstance(without_inputs_plan, PlanQuery)
    assert without_inputs_plan.instruction() == plan_instruction(without_inputs)
    assert isinstance(leaf_plan, PlanQuery)
    assert leaf_plan.instruction() == plan_instruction(leaf)
    assert EXECUTION_POLICY in flow.build_messages(eligible_plan)[0]["content"]
    assert WORKER_ADDENDUM in flow.build_messages(leaf_plan)[0]["content"]
    assert EXECUTION_POLICY not in flow.build_messages(leaf_plan)[0]["content"]
    assert "You are a Recursive Language Model" in flow.build_messages(leaf_plan)[0]["content"]
    assert "`await launch_subagent(goal: str" not in flow.build_messages(leaf_plan)[0]["content"]


def test_plan_query_user_turn_is_turn_n_of_max_with_a_first_turn_safeguard():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", max_iters=8)
    first = take(flow, root.frontier)

    assert isinstance(first, PlanQuery)
    assert first.instruction().startswith(FIRST_TURN_SAFEGUARD)
    assert "Turn 1/8:" in first.instruction()
    assert first.instruction().endswith(PLANNING_ACTION)
    assert len(FIRST_TURN_SAFEGUARD) < 300


def test_shared_execution_policy_plans_the_known_subgraph_in_one_prompt():
    policy = " ".join(EXECUTION_POLICY.split())
    assert "Keep one coherent, tightly coupled deliverable local" in policy
    assert "distinct multi-step investigation" in policy
    assert "bounded, non-overlapping deliverable" in policy
    assert "Launch ready independent work together" in policy
    assert "prerequisite `AgentHandle` as a named `inputs` value" in policy
    assert "Do not duplicate delegated work locally" in policy
    assert "Consume and validate child results" in policy
    assert "Stop launching when the answer is supported" in policy
    assert len(EXECUTION_POLICY) < 1_200


def test_plan_query_makes_dependency_graph_construction_the_default():
    compact = " ".join(PLANNING_ACTION.split())

    assert "the next objective" in compact
    assert "what stays local" in compact
    assert "what, if anything, gets its own owner" in compact
    assert "which known work products depend on which others" in compact
    assert "one REPL block" in compact
    assert len(PLANNING_ACTION) < 400


def test_leaf_plan_query_gets_the_shared_policy_without_the_launch_api():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", max_depth=0)
    plan = take(flow, root.frontier)
    prompt = flow.build_messages(plan)[0]["content"]

    assert WORKER_ADDENDUM in prompt
    assert EXECUTION_POLICY not in prompt
    assert "`await launch_subagent(goal: str" not in prompt
    assert "Baseline mode: no sub-agents available" in prompt
    assert "Do not leave TODOs" in " ".join(prompt.split())


def test_final_budget_guard_precedes_agent_start_working_query():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", max_iters=1)

    assert isinstance(take(flow, root), FinalQuery)


def test_child_plan_query_is_depth_aware():
    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        root_config=AgentConfig(max_depth=2),
    )
    root = flow.start("query", inputs={"context": "material"})
    child = start(
        "focused scope",
        config=root.config.child("worker"),
        inputs={"scope": "focused material"},
    )

    child_plan = take(flow, child.frontier)
    child_system = flow.build_messages(child_plan)[0]["content"]

    assert isinstance(child_plan, PlanQuery)
    assert child_plan.instruction() == plan_instruction(child)
    assert EXECUTION_POLICY in child_system
    assert "You are a Recursive Language Model" in child_system
    assert "`await launch_subagent(goal: str" in child_system


def test_every_repl_observation_automatically_enters_a_new_plan_action():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", inputs={"context": "material"}, max_depth=1)
    initial_system = flow.build_messages(root.frontier)[0]["content"]

    plan = take(flow, root.frontier)
    plan_system = flow.build_messages(plan)[0]["content"]
    output = LLMOutput(content="inspect", code="print('observed')")
    plan.append(output)
    action = ExecAction(code="print('observed')")
    output.append(action)
    observation = ExecOutput(content="observed")
    action.append(observation)
    next_plan = take(flow, observation)
    messages = flow.build_messages(next_plan)

    assert isinstance(next_plan, PlanQuery)
    assert EXECUTION_POLICY in initial_system
    assert EXECUTION_POLICY in plan_system
    assert messages[0]["content"] == plan_system
    assert messages[-1]["content"].startswith(plan_instruction(root))
    assert FIRST_TURN_SAFEGUARD not in messages[-1]["content"]
    assert "Turn 2:" in messages[-1]["content"]
    assert messages[-1]["content"].count(PLANNING_ACTION) == 1

    output = LLMOutput(content="work", code="print('done')")
    next_plan.append(output)
    action = ExecAction(code="print('done')")
    output.append(action)
    observation = ExecOutput(content="done")
    action.append(observation)
    third_plan = take(flow, observation)

    assert isinstance(third_plan, PlanQuery)
    assert sum(isinstance(node, PlanQuery) for node in root.transcript()) == 3


def test_local_path_repeats_the_plan_action_boundary():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", inputs={"context": "material"}, max_depth=1)
    plan = take(flow, root.frontier)
    inspect = LLMOutput(content="inspect")
    plan.append(inspect)
    inspect_action = ExecAction(code="print('shape')")
    inspect.append(inspect_action)
    observation = ExecOutput(content="shape")
    inspect_action.append(observation)
    next_plan = take(flow, observation)
    output = LLMOutput(content="choose local", code="print('working locally')")
    next_plan.append(output)
    action = ExecAction(code="print('working locally')")
    output.append(action)
    local_observation = ExecOutput(content="working locally")
    action.append(local_observation)
    turn = take(flow, local_observation)

    assert isinstance(turn, PlanQuery)
    assert sum(isinstance(node, PlanQuery) for node in root.transcript()) == 3


def test_background_status_has_no_repeated_action_guidance():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", inputs={"context": "material"}, max_depth=1)
    reply = LLMOutput(content="launch", code="unused")
    root.append(reply)
    action = ExecAction(code="unused")
    reply.append(action)
    action.append(start("child", config=root.config.child("research")))
    action.append(ExecOutput(content="launched"))

    turn = take(flow, root.frontier)
    content = "\n".join(message["content"] for message in flow.build_messages(turn))

    assert "Background subagents:" in content
    assert "`research` (root.research): running" in content
    assert "Gather the child results" not in content
    assert "Retrieve needed results" not in content


def test_final_budget_action_replaces_ordinary_free_form_continuation():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start(
        "query",
        inputs={"context": "material"},
        max_depth=1,
        max_iters=2,
    )
    inspect = LLMOutput(content="inspect", code="print('observed')")
    root.append(inspect)
    inspect.append(ExecOutput(content="observed"))

    turn = take(flow, root.frontier)
    user = (
        turn.content if isinstance(turn, FinalQuery) else flow.build_messages(turn)[-1]["content"]
    )
    system = flow.build_messages(turn)[0]["content"]

    assert isinstance(turn, FinalQuery)
    assert "This is your last turn" in user
    assert LAST_TURN_ADDENDUM in system
    assert TURN_ONE_FINISH_SENTENCE not in system
    assert EXECUTION_POLICY in system
    assert "`await launch_subagent(goal: str" not in system
    assert "`await llm_query(prompt: str" not in system
    assert "Inspect this turn" not in system
    assert "`finish(answer: object) -> None`" in " ".join(system.split())
    assert "submit your best inference rather than ending the run with nothing" in " ".join(
        user.split()
    )


def test_one_shot_query_guidance_matches_enabled_builtins():
    plain = Flow(StubLLM(lambda _messages: "unused"))
    single = Flow(
        StubLLM(lambda _messages: "unused"),
        use_llm_query=True,
        use_llm_query_batched=False,
    )
    equipped = Flow(StubLLM(lambda _messages: "unused"), use_llm_query=True)
    root = start("solve", inputs={"task": "{}"}, max_depth=2)

    without = plain.build_messages(root.frontier)[0]["content"]
    with_single = single.build_messages(root.frontier)[0]["content"]
    within = equipped.build_messages(root.frontier)[0]["content"]

    assert "await llm_query(" not in without
    assert "await llm_query_batched(" not in without
    assert "await llm_query(" in with_single
    assert "await llm_query_batched(" not in with_single
    assert "await llm_query(" in within
    assert "await llm_query_batched(" in within

    isolation = "Keep bulky values in the REPL or pass them through `inputs=`"
    oneshot = "Use `llm_query` for a one-shot transform"
    plain_plan = take(plain, start("solve", max_depth=2))
    single_plan = take(single, start("solve", max_depth=2))
    equipped_plan = take(equipped, start("solve", max_depth=2))
    plain_prompt = plain.build_messages(plain_plan)[0]["content"]
    single_prompt = single.build_messages(single_plan)[0]["content"]
    equipped_prompt = equipped.build_messages(equipped_plan)[0]["content"]
    assert isolation in plain_prompt
    assert isolation in single_prompt
    assert isolation in equipped_prompt
    assert oneshot not in plain_prompt
    assert oneshot in single_prompt
    assert oneshot in equipped_prompt
    assert "Slice long text through `llm_query`" not in equipped_prompt
    batched = "Use `llm_query_batched` for many independent one-shot transforms."
    assert batched in equipped_prompt
    assert batched not in single_prompt
    assert equipped_prompt.index("Example — launch ready") < equipped_prompt.index(oneshot)


def test_leaf_prompt_keeps_shared_policy_but_omits_the_launch_api():
    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        root_config=AgentConfig(max_depth=1),
    )
    root = flow.start("root")
    leaf = start("leaf", config=root.config.child("leaf"))
    leaf_plan = take(flow, leaf.frontier)

    prompt = flow.build_messages(leaf_plan)[0]["content"]
    compact = " ".join(prompt.split())

    assert "`await launch_subagent(goal: str" not in prompt
    assert "AgentHandle" not in prompt
    assert "Think in dataflow" not in prompt
    assert "You are a Recursive Language Model" in compact
    assert WORKER_ADDENDUM in prompt
    assert EXECUTION_POLICY not in prompt
    assert "`finish(answer: object) -> None`" in prompt


def test_repl_contract_is_available_to_root_and_leaf():
    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        root_config=AgentConfig(max_depth=1),
    )
    root = flow.start("root")
    leaf = start("leaf", config=root.config.child("leaf"), inputs={"scope": "focused"})

    for agent in (root, leaf):
        prompt = flow.build_messages(agent.frontier)[0]["content"]
        compact = " ".join(prompt.split())
        assert "Recursive Language Model" in compact
        assert "the REPL persists across turns" in compact
        assert "Only printed stdout is returned" in compact


def test_inputs_manifest_never_inlines_values():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("build the thing", inputs={"context": "Requirement: use a dark background."})

    prompt = flow.build_messages(root.frontier)[0]["content"]

    assert "- context: str, 35 chars" in prompt
    assert "Requirement: use a dark background." not in prompt


def test_inputs_manifest_resolves_agent_markers_without_leaking_them():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("root")
    action = ExecAction(code="launch")
    root.append(action)
    source = AgentStart(content="source", config=root.config.child("source"))
    action.append(source)
    source.append(DoneOutput(result={"fact": 1}))
    consumer = AgentStart(
        content="consumer",
        config=root.config.child(
            "consumer",
            inputs={"evidence": {"$rlmflow_agent": source.id}},
        ),
    )
    action.append(consumer)

    prompt = flow.build_messages(consumer.frontier)[0]["content"]

    assert "- evidence: str, 11 chars" in prompt
    assert "$rlmflow_agent" not in prompt


def test_root_routes_while_children_execute_until_max_depth():
    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        root_config=AgentConfig(max_depth=3),
    )
    root = flow.start("root")
    child = start("child", config=root.config.child("child"))
    grandchild = start("grandchild", config=child.config.child("grandchild"))
    leaf = start("leaf", config=grandchild.config.child("leaf"))

    root_plan = take(flow, root.frontier)
    root_prompt = flow.build_messages(root_plan)[0]["content"]
    assert EXECUTION_POLICY in root_prompt
    assert "await launch_subagent(" in root_prompt

    for agent in (child, grandchild):
        plan = take(flow, agent.frontier)
        prompt = flow.build_messages(plan)[0]["content"]
        compact = " ".join(prompt.split())
        assert "You are a Recursive Language Model" in compact
        assert "launch_subagent" in prompt
        assert EXECUTION_POLICY in prompt

    leaf_plan = take(flow, leaf.frontier)
    leaf_prompt = flow.build_messages(leaf_plan)[0]["content"]
    assert "`await launch_subagent(goal: str" not in leaf_prompt
    assert "You are a Recursive Language Model" in leaf_prompt
    assert WORKER_ADDENDUM in leaf_prompt
    assert EXECUTION_POLICY not in leaf_prompt


def test_prompt_uses_the_active_agents_max_depth_override():
    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        root_config=AgentConfig(max_depth=3),
    )
    root = flow.start("root", max_depth=1)
    leaf = start("leaf", config=root.config.child("leaf"))

    root_prompt = flow.build_messages(root.frontier)[0]["content"]
    leaf_prompt = flow.build_messages(leaf.frontier)[0]["content"]

    assert "depth **0** of max **1**" in root_prompt
    assert "`await launch_subagent(goal: str" in root_prompt
    assert "depth **1** of max **1**" in leaf_prompt
    assert "You cannot spawn sub-agents" in leaf_prompt
    assert "`await launch_subagent(goal: str" not in leaf_prompt


def test_default_prompt_leaves_file_guidance_to_tool_descriptions():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", max_depth=1)

    prompt = flow.build_messages(root.frontier)[0]["content"]

    assert "workspace" not in prompt
    assert "`ls`" not in prompt


def test_file_tool_descriptions_carry_shared_write_rules():
    from rlmflow import FILE_TOOLS

    flow = Flow(StubLLM(lambda _messages: "unused"), tools=[FILE_TOOLS])
    root = flow.start("query", max_depth=1)
    leaf = start("leaf", config=root.config.child("worker"))

    planner = flow.build_messages(root.frontier)[0]["content"]
    worker = flow.build_messages(leaf.frontier)[0]["content"]

    assert "Agents in one flow may share this working directory" in planner
    assert "write only paths assigned to your scope" in planner
    assert "If another agent wrote a file, inspect it in place" in planner
    assert "bind only on agents that cannot spawn" in planner
    assert "`write_file(" not in planner
    assert "Replaces the whole file with no warning" in worker
    assert "`write_file(" in worker


def test_prompt_builder_subclass_appears_on_plan_query_turns():
    class MarkerPrompt(PromptBuilder):
        def __call__(self, flow=None, node=None) -> str:
            return super().__call__(flow, node) + "\n\nMARKER_SKILL_TEXT"

    flow = Flow(StubLLM(lambda _messages: "unused"), system_prompt=MarkerPrompt())
    root = flow.start("query", max_depth=1)
    plan = take(flow, root.frontier)
    prompt = flow.build_messages(plan)[0]["content"]

    assert "MARKER_SKILL_TEXT" in prompt
    assert EXECUTION_POLICY in prompt
    assert prompt.index(EXECUTION_POLICY) < prompt.index("MARKER_SKILL_TEXT")


def test_custom_node_can_replace_or_wrap_the_system_prompt():
    class ReviewQuery(UserQuery):
        def build_system_prompt(self, flow) -> str:
            return "You are a reviewer."

    class WrappedQuery(UserQuery):
        def build_system_prompt(self, flow) -> str:
            return flow.build_system_prompt(self) + "\n\nReview rules."

    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query")
    review = ReviewQuery(content="review")
    root.append(review)
    wrapped = WrappedQuery(content="wrap")
    start("query").append(wrapped)

    assert review.build_system_prompt(flow) == "You are a reviewer."
    assert "You are a Recursive Language Model" in wrapped.build_system_prompt(flow)
    assert wrapped.build_system_prompt(flow).endswith("Review rules.")


def test_query_action_output_bases_keep_prompt_hook_on_queries():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query")
    query = PlanQuery()
    root.append(query)
    action = ExecAction(code="x = 1")
    start("query").append(action)
    output = ExecOutput(content="1")
    start("query").append(output)
    done = DoneOutput(result="ok")
    start("query").append(done)
    reply = LLMOutput(content="hi")
    start("query").append(reply)

    assert isinstance(query, UserQuery)
    assert isinstance(action, ActionNode)
    assert isinstance(output, OutputNode)
    assert isinstance(done, OutputNode)
    assert isinstance(reply, OutputNode)
    assert not isinstance(root, UserQuery)
    assert hasattr(query, "build_system_prompt")
    assert not hasattr(action, "build_system_prompt")
    assert not hasattr(output, "build_system_prompt")
    assert not hasattr(reply, "build_system_prompt")
    assert "You are a Recursive Language Model" in flow.build_messages(root)[0]["content"]
    assert EXECUTION_POLICY in flow.build_messages(query)[0]["content"]
    assert EXECUTION_POLICY in flow.build_messages(output)[0]["content"]


def test_subclass_on_userquery_covers_plan():
    seen = []

    class Tracking(Flow):
        transitions = Flow.transitions.derive()

    from rlmflow.engine.steps import complete

    @Tracking.transitions.on(UserQuery)
    async def track(flow, node):
        seen.append(type(node).__name__)
        return await complete(flow, node)

    flow = Tracking(StubLLM(lambda _messages: "unused"))
    root = start("query")
    root.append(PlanQuery())

    take(flow, root.frontier)
    assert seen == ["PlanQuery"]


def test_subclass_can_skip_plan_on_start():
    from rlmflow.engine.steps import complete

    class SkipPlan(Flow):
        transitions = Flow.transitions.derive()

    @SkipPlan.transitions.on(AgentStart)
    async def start_chats(flow, node):
        return await complete(flow, node)

    flow = SkipPlan(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", inputs={"context": "material"}, max_depth=1)

    assert not isinstance(take(flow, root), PlanQuery)


def test_follow_up_user_query_after_done_inserts_plan_once():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", inputs={"context": "material"}, max_depth=1)
    done = DoneOutput(result="first")
    root.append(done)
    done.append(UserQuery(content="again"))

    landed = take(flow, root.frontier)
    assert isinstance(landed, PlanQuery)
    assert sum(isinstance(node, PlanQuery) for node in root.transcript()) == 1
