import ast
import asyncio
import re

from helpers import StubLLM
from pydantic import BaseModel

from rlmflow import (
    SYSTEM_PROMPT,
    AgentConfig,
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    FinalQuery,
    Flow,
    LLMOutput,
    LLMRequestStep,
    Node,
    PlanQuery,
    Runtime,
    StepFunction,
    UserQuery,
    start,
)
from rlmflow.graph.nodes import ORCHESTRATOR_ADDENDUM, WORKING_ACTION
from rlmflow.prompts.prompts import MAX_STATIC_PROMPT_CHARS


def take(flow, node):
    return asyncio.run(flow.step(node)).created


def test_the_default_prompt_stays_under_its_size_ceiling():
    assert len(SYSTEM_PROMPT) <= MAX_STATIC_PROMPT_CHARS, (
        f"static prompt is {len(SYSTEM_PROMPT)} chars against a {MAX_STATIC_PROMPT_CHARS} ceiling"
    )


def test_messages_are_a_pure_projection_of_the_transcript():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query")
    root.append(LLMOutput(content="assistant", code="print('x')")).append(
        ExecOutput(content="observation")
    )
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
        "observation\n\nEnd the REPL block with one:"
    )
    assert messages[-1]["content"].endswith(
        "- finish(answer) — Submit your final answer."
    )


def test_a_user_query_subclass_projects_without_a_builder_edit():
    class Note(UserQuery):
        pass

    root = start("query")
    note = root.append(Note(content="from a subclass"))

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
    hidden = root.append(ExecAction(code="pass"))
    exchange = hidden.append(Exchange(content="review this"))

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
    assistant = root.append(LLMOutput(content="canonical assistant"))
    frontier = assistant.append(ExecOutput(content="canonical observation"))

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
        "custom frontier\n\nEnd the REPL block with one:"
    )


def test_default_renderer_keeps_current_output_with_background_status():
    root = start("query", keep_n_messages=2)
    action = root.append(LLMOutput(content="working")).append(ExecAction(code="pass"))
    action.append(
        AgentStart(
            content="child",
            config=root.config.child("child"),
        )
    )
    frontier = action.append(ExecOutput(content="observation"))

    messages = Flow(StubLLM(lambda _messages: "unused")).build_messages(frontier)

    assert [message["role"] for message in messages[-2:]] == ["user", "user"]
    assert "Background subagents:" in messages[-2]["content"]
    assert messages[-1]["content"].startswith(
        "observation\n\nEnd the REPL block with one:"
    )


def test_errors_are_projected_as_user_observations():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query")
    root.append(ErrorOutput(content="NameError: missing"))

    assert flow.build_messages(root.frontier)[-1]["role"] == "user"
    assert flow.build_messages(root.frontier)[-1]["content"].startswith(
        "NameError: missing\n\nEnd the REPL block with one:"
    )


def test_explicit_output_schema_is_in_system_prompt():
    class Answer(BaseModel):
        value: str

    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query", output_schema=Answer.model_json_schema())

    assert '"value"' in flow.build_messages(root.frontier)[0]["content"]


def test_structured_child_output_guidance_is_opt_in():
    root = start("query", max_depth=1)
    default_prompt = Flow(StubLLM(lambda _messages: "unused")).build_messages(root.frontier)[0][
        "content"
    ]
    enabled_prompt = Flow(
        StubLLM(lambda _messages: "unused"),
        enable_structured_output=True,
    ).build_messages(root.frontier)[0]["content"]

    assert "Child results are plain text by default" not in default_prompt
    assert "Child results are plain text by default" in enabled_prompt


def test_prompt_documents_the_complete_builtin_repl_api():
    flow = Flow(StubLLM(lambda _messages: "unused"), use_llm_query=True)
    root = start("query", max_depth=1)

    prompt = flow.build_messages(root.frontier)[0]["content"]
    compact = " ".join(prompt.split())

    assert "# REPL and Delegation" in prompt
    for field in (
        "goal: str",
        "name: str | None",
        "inputs: dict[str, str] | None",
        "model: str",
        "output_schema: object | None",
        "prompt_profile: str | None",
        "reuse_repl: bool",
    ):
        assert field in prompt
    assert "await launch_subagent(" in prompt
    assert "await llm_query(prompt: str" in prompt
    assert "await llm_query_batched(prompts: list[str]" in prompt
    assert "Registered models (`model=` is required)" in prompt
    assert "- `default` — current model" in prompt
    assert 'inputs={"paths": "\\n".join(batch)}' in prompt
    assert 'assert records and all("value" in record for record in records)' in prompt
    assert "assert reports and all(report.strip() for report in reports)" in prompt
    assert "candidate = \"\\n\".join(reports)" in prompt
    assert "finish(candidate)" in prompt
    assert "output_schema=" not in prompt
    assert "`finish(value: object) -> None`" in compact
    assert "first compute and bind the candidate" in compact
    assert "Never submit a value you have not inspected" in compact
    assert "print it with its checks" in compact
    assert "After reading that output on the next turn, call `finish(candidate)`" in compact
    assert (
        "Both `transition(...)` and `finish(...)` terminate the block immediately"
        in compact
    )
    assert "Verification must test the candidate against the original source" in compact
    assert "`INPUTS: dict[str, str]`" in prompt

    # The REPL's two non-obvious mechanics. Both are stated once now: the prompt was
    # rebuilt from official-rlm's structure after reaching 6,972 chars against their
    # 2,094 for the same score, and restating a rule was most of the difference.
    assert "the REPL persists across turns" in compact
    assert "only `print(...)` output (stdout) is shown back to you" in compact
    assert "a bare expression on the last line is silently discarded" in compact

    # Naming the 4,000-char cap backfired: official states ~20K, so the same sentence
    # told our model to look at a fifth as much. Cap-hit rate fell 37% -> 12% and the
    # long-context tasks regressed. State that output truncates, not the number.
    assert "4,000 characters" not in compact
    assert "REPL outputs over ~20K characters are truncated" in compact
    working = " ".join(WORKING_ACTION.split())
    assert "start by probing the available context" in working
    assert "plan briefly in prose and execute one ```repl``` block" in working

    assert "wait: bool" not in prompt
    assert "done(" not in prompt
    assert "`PLAN`" not in prompt


def test_prompt_exposes_each_model_choice_and_marks_the_current_one():
    default = StubLLM(lambda _messages: "unused")
    fast = StubLLM(lambda _messages: "unused")
    flow = Flow(default, llm_clients={"fast": fast})
    root = start("query", model="fast", max_depth=1)

    prompt = flow.build_messages(root.frontier)[0]["content"]

    assert "- `default`" in prompt
    assert "- `fast` — current model" in prompt
    assert "using model key **`fast`**" in prompt


def test_default_prompt_guides_iterative_investigation_and_grounded_delegation():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", inputs={"context": "two substantial scopes"}, max_depth=1)

    prompt = flow.build_messages(root.frontier)[0]["content"]
    compact = " ".join(prompt.split())
    working = " ".join(WORKING_ACTION.split())

    assert "Python REPL" in compact
    assert "must launch" not in compact
    assert "Act as an orchestrator" not in compact
    assert "Use its output as feedback" in working
    assert "independence and parallelism are not reasons" not in compact
    assert "Read anything smaller yourself" not in compact
    assert "you cannot see a child's reasoning" not in compact

    assert "## Examples" in prompt
    assert "**Local work — probe, print the candidate, then submit it.**" in prompt
    assert "**Delegation — a source too large to read here.**" in prompt
    assert "await handle.wait_for_result()" in prompt


def test_agent_start_selects_one_depth_aware_working_query():
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
    leaf_system = flow.build_messages(leaf.frontier)[0]["content"]

    eligible_plan = take(flow, eligible.frontier)
    without_inputs_plan = take(flow, without_inputs.frontier)
    leaf_plan = take(flow, leaf.frontier)

    assert isinstance(eligible_plan, PlanQuery)
    assert "Structural INPUTS profile:" in eligible_plan.instruction()
    assert ORCHESTRATOR_ADDENDUM in eligible_plan.instruction()
    assert isinstance(without_inputs_plan, PlanQuery)
    assert "Structural INPUTS profile:" not in without_inputs_plan.instruction()
    assert isinstance(leaf_plan, PlanQuery)
    assert leaf_plan.instruction().startswith(WORKING_ACTION)
    assert ORCHESTRATOR_ADDENDUM not in leaf_plan.instruction()
    assert "Delegation:" not in leaf_plan.instruction()
    assert "recursive agent" not in leaf_plan.instruction().casefold()
    assert "You are a Recursive Language Model" in leaf_system
    assert "This task was assigned by a parent agent" in leaf_system
    assert "launch_subagent" not in leaf_system


def test_working_query_combines_probe_plan_and_action_guidance():
    compact = " ".join(WORKING_ACTION.split())
    assert "start by probing the available context" in compact
    assert "print a few lines, count them" in compact
    assert "plan briefly in prose" in compact
    assert "execute one ```repl``` block" in compact
    assert "Use its output as feedback" in compact
    assert len(WORKING_ACTION) < 1_000

    orchestrator = " ".join(ORCHESTRATOR_ADDENDUM.split())
    assert "act as an orchestrator" in orchestrator
    assert "Launch independent subagents together" in orchestrator
    assert "do not spend one model turn per item or batch" in orchestrator


def test_final_budget_guard_precedes_agent_start_working_query():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", max_iters=1)

    assert isinstance(take(flow, root), FinalQuery)


def test_child_planning_query_is_depth_aware():
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
    assert child_plan.instruction().startswith(WORKING_ACTION)
    assert ORCHESTRATOR_ADDENDUM in child_plan.instruction()
    assert "You are a Recursive Language Model" in child_system
    assert "This task was assigned by a parent agent" in child_system
    assert "**Optional delegation" not in child_system


def test_plan_query_enters_the_ordinary_repl_loop():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", inputs={"context": "material"}, max_depth=1)
    initial_system = flow.build_messages(root.frontier)[0]["content"]

    plan = take(flow, root.frontier)
    messages = flow.build_messages(plan)

    assert isinstance(plan, PlanQuery)
    assert messages[0]["content"] == initial_system
    assert messages[-1]["content"].startswith(WORKING_ACTION)

    output = plan.append(LLMOutput(content="inspect", code="print('observed')"))
    action = output.append(ExecAction(code="print('observed')"))
    action.requested_transition = "act"
    observation = action.append(ExecOutput(content="observed"))
    next_turn = take(flow, observation)

    assert isinstance(next_turn, LLMOutput)
    assert sum(isinstance(node, PlanQuery) for node in root.transcript()) == 1


def test_local_path_does_not_repeat_plan_query():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", inputs={"context": "material"}, max_depth=1)
    plan = take(flow, root.frontier)
    output = plan.append(LLMOutput(content="choose local", code="print('working locally')"))
    action = output.append(ExecAction(code="print('working locally')"))
    action.requested_transition = "act"
    local_observation = action.append(ExecOutput(content="working locally"))
    before_plans = [node.id for node in root.transcript() if isinstance(node, PlanQuery)]

    turn = take(flow, local_observation)

    assert [node.id for node in root.transcript() if isinstance(node, PlanQuery)] == (before_plans)
    assert not isinstance(turn, PlanQuery)


def test_background_status_has_no_repeated_action_guidance():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", inputs={"context": "material"}, max_depth=1)
    action = root.append(LLMOutput(content="launch", code="unused")).append(
        ExecAction(code="unused")
    )
    action.append(start("child", config=root.config.child("research")))
    action.append(ExecOutput(content="launched"))

    turn = take(flow, root.frontier)
    content = "\n".join(message["content"] for message in flow.build_messages(turn))

    assert "Background subagents:" in content
    assert "`research` (root.research): running" in content
    assert "Gather the child results" not in content
    assert "Retrieve needed results" not in content
    assert "before finishing" not in content


def test_final_budget_action_replaces_ordinary_free_form_continuation():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start(
        "query",
        inputs={"context": "material"},
        max_depth=1,
        max_iters=2,
    )
    root.append(LLMOutput(content="inspect", code="print('observed')")).append(
        ExecOutput(content="observed")
    )

    turn = take(flow, root.frontier)
    content = (
        turn.content if isinstance(turn, FinalQuery) else flow.build_messages(turn)[-1]["content"]
    )

    assert isinstance(turn, FinalQuery)
    assert "Using the REPL observation immediately above" not in content
    assert "launch a useful set of coherent workstreams" not in content
    assert "This is your last turn" in content

    # The standing contract is print-then-submit, which needs a turn that reads the
    # print. There is none here, so the last turn must override it and guess rather
    # than return nothing.
    assert "submit your best inference rather than ending the run with nothing" in " ".join(
        content.split()
    )


def test_one_shot_query_guidance_matches_enabled_builtins():
    plain = Flow(StubLLM(lambda _messages: "unused"))
    equipped = Flow(StubLLM(lambda _messages: "unused"), use_llm_query=True)
    root = start("solve", inputs={"task": "{}"}, max_depth=2)

    without = plain.build_messages(root.frontier)[0]["content"]
    within = equipped.build_messages(root.frontier)[0]["content"]

    assert "cheap way to read at volume" not in without
    assert "cheap way to read at volume" not in within
    assert "await llm_query(" not in without
    assert "await llm_query_batched(" not in without
    assert "await llm_query(" in within
    assert "await llm_query_batched(" in within


def test_leaf_prompt_omits_delegation_guidance():
    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        root_config=AgentConfig(max_depth=1),
    )
    root = flow.start("root")
    leaf = start("leaf", config=root.config.child("leaf"))

    prompt = flow.build_messages(leaf.frontier)[0]["content"]
    compact = " ".join(prompt.split())

    assert "launch_subagent" not in prompt
    assert "AgentHandle" not in prompt
    assert "You are a Recursive Language Model" in compact
    assert "This task was assigned by a parent agent" in compact
    assert "**Local work — probe, print the candidate, then submit it.**" in prompt
    assert "`finish(value: object) -> None`" in prompt


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
        assert prompt.count("## REPL and Delegation") == 1
        assert "the REPL persists across turns" in compact
        assert "only `print(...)` output (stdout) is shown back to you" in compact


def test_inputs_manifest_never_inlines_values():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("build the thing", inputs={"context": "Requirement: use a dark background."})

    prompt = flow.build_messages(root.frontier)[0]["content"]

    assert "- context: str, 35 chars" in prompt
    assert "Requirement: use a dark background." not in prompt


def test_root_routes_while_children_execute_until_max_depth():
    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        root_config=AgentConfig(max_depth=3),
    )
    root = flow.start("root")
    child = start("child", config=root.config.child("child"))
    grandchild = start("grandchild", config=child.config.child("grandchild"))
    leaf = start("leaf", config=grandchild.config.child("leaf"))

    root_prompt = flow.build_messages(root.frontier)[0]["content"]
    assert "Act as an orchestrator" not in root_prompt
    assert "plan briefly in prose" in " ".join(WORKING_ACTION.lower().split())
    assert "**Local work — probe, print the candidate, then submit it.**" in root_prompt
    assert "**Delegation — a source too large to read here.**" in root_prompt

    for agent in (child, grandchild):
        prompt = flow.build_messages(agent.frontier)[0]["content"]
        compact = " ".join(prompt.split())
        assert "You are a Recursive Language Model" in compact
        assert "This task was assigned by a parent agent" in compact
        assert "launch_subagent" in prompt
        assert "Act as an orchestrator" not in compact
        assert "**Delegation — a source too large" not in prompt

    leaf_prompt = flow.build_messages(leaf.frontier)[0]["content"]
    assert "launch_subagent" not in leaf_prompt
    assert "You are a Recursive Language Model" in leaf_prompt
    assert "This task was assigned by a parent agent" in leaf_prompt
    assert "**Local work — probe, print the candidate, then submit it.**" in leaf_prompt


def test_examples_render_for_each_available_capability():
    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        root_config=AgentConfig(max_depth=1),
    )
    with_inputs = flow.start("root", inputs={"context": "material"})
    without_inputs = flow.start("root")
    leaf = start(
        "leaf",
        config=with_inputs.config.child("leaf"),
        inputs={"scope": "focused material"},
    )

    input_prompt = flow.build_messages(with_inputs.frontier)[0]["content"]
    no_input_prompt = flow.build_messages(without_inputs.frontier)[0]["content"]
    leaf_prompt = flow.build_messages(leaf.frontier)[0]["content"]

    local_heading = "**Local work — probe, print the candidate, then submit it.**"
    delegation_heading = "**Delegation — a source too large to read here.**"
    assert local_heading in input_prompt
    assert delegation_heading in input_prompt
    assert local_heading in no_input_prompt
    assert delegation_heading in no_input_prompt
    assert local_heading in leaf_prompt
    assert delegation_heading not in leaf_prompt

    # Local leads. Delegation led while it was the differentiating feature, but it
    # fired on 3 of 34 benchmark rows — all on one task, which scored 0.000 nine
    # times out of nine — while print-then-submit is what every row depends on.
    for prompt in (input_prompt, no_input_prompt):
        assert prompt.index(local_heading) < prompt.index(delegation_heading)


def test_example_blocks_import_every_module_they_use():
    # The REPL namespace holds only builtins, INPUTS, ENV, and injected tools, so an
    # example that reaches for an unimported module teaches code that raises NameError
    # on the turn it is copied — and a failed first delegation turn sends the model
    # back to local execution.
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("solve", inputs={"task": "{}"}, max_depth=2)
    prompt = flow.build_messages(root.frontier)[0]["content"]

    stdlib_roots = {"asyncio", "csv", "json", "math", "os", "re", "sys"}
    blocks = re.findall(r"```repl\s*\n(.*?)\n```", prompt, re.DOTALL)
    assert blocks

    for code in blocks:
        tree = ast.parse(code)
        imported = {
            alias.asname or alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        used = {
            node.value.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        } & stdlib_roots
        assert used <= imported, f"{sorted(used - imported)} used unimported in:\n{code}"


def test_no_example_finishes_from_the_block_that_computes_the_answer():
    # finish() ends the run, so a block that computes and finishes never shows the
    # model its own answer. gpt-5-mini submitted a regex-truncated `109` for a
    # `10984.869565217392` total this way. Prose alone does not fix it: when the
    # examples disagreed with the prose, the examples won.
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("solve", inputs={"task": "{}"}, max_depth=2)
    prompt = flow.build_messages(root.frontier)[0]["content"]

    blocks = re.findall(r"```repl\s*\n(.*?)\n```", prompt, re.DOTALL)
    assert blocks

    for code in blocks:
        body = [
            node
            for node in ast.parse(code).body
            if not isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        finishes = [
            node
            for node in ast.walk(ast.parse(code))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "finish"
        ]
        if not finishes:
            continue
        assert len(body) == 1, f"finish shares a block with other work:\n{code}"
        assert isinstance(body[0], ast.Expr), f"finish is not the whole block:\n{code}"


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
    assert "launch_subagent" in root_prompt
    assert "depth **1** of max **1**" in leaf_prompt
    assert "You cannot spawn sub-agents" in leaf_prompt
    assert "launch_subagent" not in leaf_prompt


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

    prompt = flow.build_messages(root.frontier)[0]["content"]

    assert "Replaces the whole file with no warning" in prompt
    assert "Agents in one flow may share this working directory" in prompt
    assert "write only paths assigned to your scope" in prompt
    assert "If another agent wrote a file, inspect it in place" in prompt


def test_update_step_fn_on_userquery_covers_plan_via_mro():
    seen = []

    class TrackStep(StepFunction):
        async def __call__(self, node):
            seen.append(type(node).__name__)
            return node.append(LLMOutput(content="ok"))

    flow = Flow(StubLLM(lambda _messages: "unused"))
    flow.update_step_fn(UserQuery, TrackStep)
    root = start("query")
    root.append(PlanQuery())

    take(flow, root.frontier)
    assert seen == ["PlanQuery"]


def test_update_step_fn_can_skip_plan_on_start():
    class SkipPlanStep(LLMRequestStep):
        async def __call__(self, node):
            return await self.chat(node)

    flow = Flow(StubLLM(lambda _messages: "unused"))
    flow.update_step_fn(AgentStart, SkipPlanStep)
    root = flow.start("query", inputs={"context": "material"}, max_depth=1)

    assert not isinstance(take(flow, root), PlanQuery)


def test_follow_up_user_query_after_done_inserts_plan_once():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", inputs={"context": "material"}, max_depth=1)
    root.append(DoneOutput(result="first")).append(UserQuery(content="again"))

    landed = take(flow, root.frontier)
    assert isinstance(landed, PlanQuery)
    assert sum(isinstance(node, PlanQuery) for node in root.transcript()) == 1
