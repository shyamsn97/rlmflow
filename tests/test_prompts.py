import asyncio

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
    InspectQuery,
    LLMOutput,
    LLMRequestStep,
    Node,
    PlanQuery,
    Runtime,
    StepFunction,
    UserQuery,
    start,
)
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
    assert messages[-1]["content"] == "observation"


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

    assert [message["content"] for message in messages[1:]] == [
        "original",
        "canonical assistant",
        "custom frontier",
    ]


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
    assert messages[-1]["content"] == "observation"


def test_errors_are_projected_as_user_observations():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query")
    root.append(ErrorOutput(content="NameError: missing"))

    assert flow.build_messages(root.frontier)[-1]["role"] == "user"
    assert flow.build_messages(root.frontier)[-1]["content"].endswith("NameError: missing")


def test_explicit_output_schema_is_in_system_prompt():
    class Answer(BaseModel):
        value: str

    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query", output_schema=Answer.model_json_schema())

    assert '"value"' in flow.build_messages(root.frontier)[0]["content"]


def test_prompt_documents_the_complete_builtin_repl_api():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query", max_depth=1)

    prompt = flow.build_messages(root.frontier)[0]["content"]

    assert "# Built-in REPL API" in prompt
    for field in ("goal", "name", "inputs", "model", "output_schema", "prompt_profile"):
        assert f"`{field}`" in prompt
    assert "`await launch_subagent(...) -> AgentHandle`" in prompt
    assert "model: str," in prompt
    assert "`model` (required)" in prompt
    assert "Select it explicitly from **Available models**" in " ".join(prompt.split())
    assert "Available models (`model=` is required" in prompt
    assert "- `default` — current model" in prompt
    assert 'model="default"' in prompt
    assert "`wait_for_result()`" in prompt
    assert "Every launch starts one in the background" in prompt
    assert "`finish(answer: object) -> NoReturn`" in prompt
    assert "`INPUTS: dict[str, str]`" in prompt
    assert "`ENV: dict[str, object]`" in prompt
    assert "`print(...)` is the observation channel" in prompt
    assert "Only the names documented here" in prompt
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


def test_default_prompt_keeps_free_form_strategy_without_turn_gates():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", inputs={"context": "two substantial scopes"}, max_depth=1)

    prompt = flow.build_messages(root.frontier)[0]["content"]
    compact = " ".join(prompt.split())

    assert "The REPL persists across turns" in compact
    assert "Before producing deliverables, inspect the complete relevant content" in compact
    assert "Size up work" in compact
    assert "Keep small or tightly coupled work local" in compact
    assert "act as an orchestrator" in compact
    assert "pass exact relevant context through `inputs`" in compact
    assert "launch independent children before waiting" in compact
    assert "components with explicit interfaces are independent workstreams" in compact
    assert "even within one final product" in compact
    assert "choose each child's registered model explicitly" in compact
    assert "Keep synthesis with the parent" in compact
    assert "without redundant delegation or review loops" in compact
    assert "## Examples" in prompt
    assert "**Inspection example — task text.**" in prompt
    assert "**Local trajectory — one deterministic result.**" in prompt
    assert "**Fan-out trajectory — several independent substantial outputs.**" in prompt
    assert "## Turn Guidance" not in prompt
    assert "Inspection turn only" not in prompt
    assert "Post-inspection orchestration turn" not in prompt
    assert "two or more" not in compact
    assert prompt.index("## Status") < prompt.index("Size up work")
    assert compact.endswith("finish without redundant delegation or review loops.")


def test_inspection_only_guidance_is_scoped_to_agents_with_inputs_and_children():
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

    eligible_turn = take(flow, eligible.frontier)
    without_inputs_turn = take(flow, without_inputs.frontier)
    leaf_turn = take(flow, leaf.frontier)

    assert isinstance(eligible_turn, InspectQuery)
    eligible_content = flow.build_messages(eligible_turn)[-1]["content"]
    assert "Inspect the complete relevant content of `INPUTS` now" in eligible_content
    assert "Preserve explicit requirements in REPL state" in eligible_content
    assert "Do not produce deliverables or finish on this turn" in eligible_content
    assert "Using the REPL observation immediately above" not in eligible_content
    assert not isinstance(without_inputs_turn, InspectQuery)
    assert (
        "Inspect the complete relevant content of `INPUTS` now"
        not in flow.build_messages(without_inputs_turn)[-1]["content"]
    )
    assert not isinstance(leaf_turn, InspectQuery)
    assert (
        "Inspect the complete relevant content of `INPUTS` now"
        not in flow.build_messages(leaf_turn)[-1]["content"]
    )


def test_first_repl_observation_receives_one_grounded_orchestration_prompt():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", inputs={"context": "material"}, max_depth=1)
    initial_system = flow.build_messages(root.frontier)[0]["content"]
    root.append(InspectQuery()).append(
        LLMOutput(content="inspect", code="print('observed')")
    ).append(ExecOutput(content="observed"))

    turn = take(flow, root.frontier)
    messages = flow.build_messages(turn)

    assert isinstance(turn, PlanQuery)
    assert messages[0]["content"] == initial_system
    assert messages[-2]["content"] == "observed"
    assert "Now size up the observed work" in messages[-1]["content"]
    assert "choose local execution or delegation" in messages[-1]["content"]
    assert "substantial components with explicit interfaces" in messages[-1]["content"]
    assert (
        "launch multiple coherent workstreams before local implementation"
        in messages[-1]["content"]
    )
    assert "components in one final product still qualify" in messages[-1]["content"]
    assert "or each scope is trivial" in messages[-1]["content"]
    assert "Choose `model=` explicitly for each child" in messages[-1]["content"]
    assert "keep final synthesis with the parent" in messages[-1]["content"]


def test_local_path_does_not_repeat_the_orchestration_prompt():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", inputs={"context": "material"}, max_depth=1)
    first_observation = (
        root.append(InspectQuery())
        .append(LLMOutput(content="inspect", code="print('observed')"))
        .append(ExecOutput(content="observed"))
    )
    plan = take(flow, first_observation)
    local_observation = plan.append(
        LLMOutput(content="choose local", code="print('working locally')")
    ).append(ExecOutput(content="working locally"))
    before_plans = [node.id for node in root.transcript() if isinstance(node, PlanQuery)]

    turn = take(flow, local_observation)

    assert [node.id for node in root.transcript() if isinstance(node, PlanQuery)] == (before_plans)
    assert not isinstance(turn, PlanQuery)
    assert (
        "Now size up the observed work" not in flow.build_messages(local_observation)[-1]["content"]
    )


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
    assert "You have used the full iteration budget" in content
    assert content.rstrip().endswith("Do not investigate further.")


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
    assert "Delegate substantive" not in prompt
    assert "Stay centered on the assigned goal" in compact
    assert "return material the parent can directly use" in compact
    assert "Do not expand into the entire parent task" in compact
    assert "`finish(answer: object) -> NoReturn`" in prompt


def test_context_section_is_available_to_root_and_leaf():
    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        root_config=AgentConfig(max_depth=1),
    )
    root = flow.start("root")
    leaf = start("leaf", config=root.config.child("leaf"), inputs={"scope": "focused"})

    for agent in (root, leaf):
        prompt = flow.build_messages(agent.frontier)[0]["content"]
        compact = " ".join(prompt.split())
        assert prompt.count("## Context and Working Memory") == 1
        assert "The REPL persists across turns" in compact
        assert "retain useful state in variables" in compact
        assert "Print bounded observations" in compact


def test_inputs_manifest_never_inlines_values():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("build the thing", inputs={"context": "Requirement: use a dark background."})

    prompt = flow.build_messages(root.frontier)[0]["content"]

    assert "- context: str, 35 chars" in prompt
    assert "Requirement: use a dark background." not in prompt


def test_same_free_form_strategy_is_used_until_max_depth():
    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        root_config=AgentConfig(max_depth=3),
    )
    root = flow.start("root")
    child = start("child", config=root.config.child("child"))
    grandchild = start("grandchild", config=child.config.child("grandchild"))
    leaf = start("leaf", config=grandchild.config.child("leaf"))

    for agent in (root, child, grandchild):
        prompt = flow.build_messages(agent.frontier)[0]["content"]
        compact = " ".join(prompt.split())
        assert "act as an orchestrator" in compact
        assert "launch independent children before waiting" in compact
        assert "launch_subagent" in prompt
        assert "**Fan-out trajectory — several independent substantial outputs.**" in prompt

    leaf_prompt = flow.build_messages(leaf.frontier)[0]["content"]
    assert "Delegate substantive" not in leaf_prompt
    assert "launch_subagent" not in leaf_prompt
    assert "Stay centered on the assigned goal" in leaf_prompt
    assert "## Examples" not in leaf_prompt


def test_examples_render_only_for_available_inputs_and_capabilities():
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

    assert "**Inspection example — task text.**" in input_prompt
    assert "**Fan-out trajectory — several independent substantial outputs.**" in input_prompt
    assert "**Inspection example — task text.**" not in no_input_prompt
    assert "**Fan-out trajectory — several independent substantial outputs.**" in no_input_prompt
    assert "**Inspection example — task text.**" in leaf_prompt
    assert "**Fan-out trajectory — several independent substantial outputs.**" not in leaf_prompt


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

    flow = Flow(StubLLM(lambda _messages: "unused"), tools=FILE_TOOLS)
    root = flow.start("query", max_depth=1)

    prompt = flow.build_messages(root.frontier)[0]["content"]

    assert "Replaces the whole file with no warning" in prompt
    assert "Agents in one flow may share this working directory" in prompt
    assert "write only paths assigned to your scope" in prompt
    assert "If another agent wrote a file, inspect it in place" in prompt


def test_update_step_fn_on_userquery_covers_inspect_via_mro():
    seen = []

    class TrackStep(StepFunction):
        async def __call__(self, node):
            seen.append(type(node).__name__)
            return node.append(LLMOutput(content="ok"))

    flow = Flow(StubLLM(lambda _messages: "unused"))
    flow.update_step_fn(UserQuery, TrackStep)
    root = start("query")
    root.append(InspectQuery())

    take(flow, root.frontier)
    assert seen == ["InspectQuery"]


def test_update_step_fn_can_skip_inspect_on_start():
    class SkipInspectStep(LLMRequestStep):
        async def __call__(self, node):
            return await self.chat(node)

    flow = Flow(StubLLM(lambda _messages: "unused"))
    flow.update_step_fn(AgentStart, SkipInspectStep)
    root = flow.start("query", inputs={"context": "material"}, max_depth=1)

    assert not isinstance(take(flow, root), InspectQuery)


def test_follow_up_user_query_after_done_inserts_inspect_once():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("query", inputs={"context": "material"}, max_depth=1)
    root.append(DoneOutput(result="first")).append(UserQuery(content="again"))

    landed = take(flow, root.frontier)
    assert isinstance(landed, InspectQuery)
    assert sum(isinstance(node, InspectQuery) for node in root.transcript()) == 1
