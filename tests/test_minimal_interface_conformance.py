from helpers import ScriptedFlow, StubLLM, TestRuntime, code_nodes, first_user, run_result

from rlmflow import RETURN, AgentConfig, Graph
from rlmflow.graph.calls import launched

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
            "primary, audit = await asyncio.gather(left, right)\n"
            "joined = launch_agent('join typed children', config='sum', "
            "primary=primary, audit=audit)\n"
            "print(await joined)\n"
            "```"
        )

    g = Graph()
    root = g.start("conformance root")
    configs = [
        AgentConfig(name=name, output_schema=schema, max_iters=4)
        for name, schema in (("value", SCHEMA), ("ok", OK_SCHEMA), ("sum", SUM_SCHEMA))
    ]
    flow = ScriptedFlow(StubLLM(reply), configs=configs, runtime=TestRuntime())

    assert run_result(flow, g, root, root_config=AgentConfig(max_iters=8)) == "done"
    children = root.subtree()[1:]
    assert [child.config.name for child in children] == ["value", "ok", "sum"]
    left, right, joined = children
    assert left.final_result() == {"value": 1}
    assert right.final_result() == {"ok": True}
    assert joined.final_result() == {"sum": 2}
    assert joined.input == {
        "task": "join typed children",
        "primary": {"value": 1},
        "audit": {"ok": True},
    }
    rejected = [n for n in code_nodes(g, right) if n.error]
    assert len(rejected) == 1
    code_node = code_nodes(g, root)[0]
    assert {e.source for e in g.edges(kind=RETURN, target=code_node.id)} == {
        child.nodes()[-1].id for child in children
    }
    assert launched(code_node) == [
        left.id,
        right.id,
        joined.id,
    ]

    g.save(tmp_path / "conformance")
    loaded = Graph.load(tmp_path / "conformance")
    call_count = len(calls)

    def no_model_call(_messages):
        raise AssertionError("model repeated during terminal rerun")

    assert (
        ScriptedFlow(StubLLM(no_model_call), configs=configs, runtime=TestRuntime()).run(loaded)
        is loaded
    )
    assert loaded.agent(root.id).final_result() == "done"
    assert len(calls) == call_count

    branch = g.fork(code_node)
    again = branch.node(code_node.id)
    assert branch.agent(root.id).nodes()[-1] is again and again.state == "ready"
    assert branch.agent(root.id).children() == [] and branch.agent(root.id).status() != "done"
    assert root.status() == "done" and root.subtree()[1:] == children
