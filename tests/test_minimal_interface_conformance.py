from helpers import StubLLM, TestRuntime, first_user, run_result

from rlmflow import DEPENDS_ON, AgentConfig, Flow, Graph

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


def test_minimal_interface_end_to_end_reruns_without_work(tmp_path):
    calls: list[str] = []

    def reply(messages):
        task = first_user(messages)
        last = messages[-1]["content"]
        calls.append(task)
        if task == "join typed children":
            return "```repl\nsubmit({'sum': primary['value'] + int(audit['ok'])})\n```"
        if task == "right typed child":
            if "ok" in last and messages[-1] is not messages[1]:
                return "```repl\nsubmit({'ok': True})\n```"
            return "```repl\nsubmit({'ok': 'yes'})\n```"
        if task == "left typed child":
            return "```repl\nsubmit({'value': 1})\n```"
        if "'sum': 2" in last:
            return "```repl\nsubmit('done')\n```"
        return (
            "```repl\n"
            "left = launch_agent('left typed child', config='value')\n"
            "right = launch_agent('right typed child', config='ok')\n"
            "joined = launch_agent('join typed children', config='sum', primary=left, audit=right)\n"
            "print(await joined)\n"
            "```"
        )

    g = Graph()
    root = g.create_agent("conformance root")
    configs = [
        AgentConfig(name=name, output_schema=schema, max_iters=4)
        for name, schema in (("value", SCHEMA), ("ok", OK_SCHEMA), ("sum", SUM_SCHEMA))
    ]
    flow = Flow(StubLLM(reply), configs=configs, runtime=TestRuntime())

    assert run_result(flow, g, root, root_config=AgentConfig(max_iters=8)) == "done"
    children = g.tree(root)[1:]
    assert [child.config.name for child in children] == ["value", "ok", "sum"]
    left, right, joined = children
    assert g.result(left) == {"value": 1}
    assert g.result(right) == {"ok": True}
    assert g.result(joined) == {"sum": 2}
    rejected = [n for n in g.history(right) if n.type == "code" and n.data.get("error")]
    assert len(rejected) == 1
    assert {e.target for e in g.edges(kind=DEPENDS_ON, source=joined.id)} == {left.id, right.id}

    g.save(tmp_path / "conformance")
    loaded = Graph.load(tmp_path / "conformance")
    call_count = len(calls)

    def no_model_call(_messages):
        raise AssertionError("model repeated during terminal rerun")

    assert (
        Flow(StubLLM(no_model_call), configs=configs, runtime=TestRuntime()).run(loaded) is loaded
    )
    assert loaded.result(loaded.node(root.id)) == "done"
    assert len(calls) == call_count

    cell = next(n for n in g.history(root) if n.type == "code")
    forked = g.fork(cell)
    assert g.frontier(forked) is cell
    assert not g.terminal(forked)
