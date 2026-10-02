from helpers import StubLLM, TestRuntime

from rlmflow import (
    AgentConfig,
    AgentStart,
    Flow,
    StructuredResult,
    TransitionDecision,
    UserQuery,
    VerificationFailure,
)

SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
    "additionalProperties": False,
}
OK_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
SUM_SCHEMA = {
    "type": "object",
    "properties": {"sum": {"type": "integer"}},
    "required": ["sum"],
    "additionalProperties": False,
}


class HandoffSemanticEngine:
    def __init__(self):
        self.calls = 0

    def supports(self, request):
        return ()

    async def generate(self, request, codec):
        self.calls += 1
        read_set = request.state["read_set"]
        matched = any("[handoff " in item["content"] for item in read_set)
        return StructuredResult(
            request.state["candidates"][0]["transition"] if matched else "continue",
            {"model": "semantic-test"},
        )


def test_minimal_interface_end_to_end_replays_without_work(tmp_path):
    calls: list[str] = []

    def reply(messages):
        text = "\n".join(message["content"] for message in messages)
        calls.append(text)
        if "semantic route selected" in text:
            return "```repl\nfinish('done')\n```"
        if "join typed children" in text:
            return "```repl\nfinish({'sum': 2})\n```"
        if "right typed child" in text:
            if "set ok to true" in text:
                return "```repl\nfinish({'ok': True})\n```"
            return "```repl\nfinish({'ok': False})\n```"
        if "left typed child" in text:
            return "```repl\nfinish({'value': 1})\n```"
        return (
            "```repl\n"
            "def verify(value):\n"
            "    return None if value['ok'] else 'set ok to true'\n"
            f"left = launch_subagent(task='left typed child', schema={SCHEMA!r})\n"
            f"right = launch_subagent(task='right typed child', schema={OK_SCHEMA!r}, verify=verify)\n"
            "primary = left if True else right\n"
            "joined = launch_subagent(\n"
            "    task='join typed children',\n"
            f"    schema={SUM_SCHEMA!r},\n"
            "    primary=primary,\n"
            "    audit=right,\n"
            ")\n"
            "print(await joined)\n"
            "```"
        )

    engine = HandoffSemanticEngine()
    flow = Flow(
        StubLLM(reply),
        runtime=TestRuntime(),
        structured_engine=engine,
    )

    @flow.transition(
        source=UserQuery,
        when="The joined child result is available to its parent.",
    )
    def select_handoff(node):
        return UserQuery(content="semantic route selected")

    root = flow.start(
        "conformance root",
        config=AgentConfig(max_depth=1, max_iters=8),
    )
    assert flow.run(root) == "done"

    children = list(root.iter_agents())[1:]
    assert len(children) == 3
    assert {child.config.output_schema["required"][0] for child in children} == {
        "value",
        "ok",
        "sum",
    }
    assert len(next(child for child in children if "right typed" in child.content).transcript()) > 3
    assert sum(isinstance(node, VerificationFailure) for node in root.walk()) == 1
    matched = next(
        node
        for node in root.walk()
        if isinstance(node, TransitionDecision) and node.selected_rule_id is not None
    )
    assert any(child.argument_references for child in children)
    assert matched.next is not None

    saved = root.save(tmp_path / "conformance")
    loaded = AgentStart.load(saved)
    call_count = len(calls)
    semantic_count = engine.calls

    class NoWorkEngine:
        def supports(self, request):
            raise AssertionError("semantic engine repeated during terminal replay")

    def no_model_call(_messages):
        raise AssertionError("model repeated during terminal replay")

    assert Flow(
        StubLLM(no_model_call),
        runtime=TestRuntime(),
        structured_engine=NoWorkEngine(),
    ).run(loaded) == "done"
    assert len(calls) == call_count
    assert engine.calls == semantic_count

    forked = matched.fork()
    assert isinstance(forked.frontier, TransitionDecision)
    assert forked.frontier.next is None
