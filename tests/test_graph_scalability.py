import json
from dataclasses import dataclass
from typing import ClassVar

import pytest
from helpers import agent_graph

from rlmflow import (
    DEPENDS_ON,
    AgentConfig,
    AgentStart,
    Edge,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    InvariantError,
    LLMOutput,
    LLMUsage,
    Node,
    TurnMode,
    UserQuery,
    ops,
    persistence,
    reads,
)


def chain(length: int) -> tuple[Graph, AgentStart, list[Node]]:
    g, root = agent_graph("deep")
    nodes: list[Node] = [root]
    for index in range(length - 1):
        nodes.append(ops.append(g, root, UserQuery(content=str(index))))
    return g, root, nodes


def test_deep_graph_operations_are_iterative(tmp_path):
    g, root, nodes = chain(10_000)

    assert reads.history(g, root) == nodes
    assert reads.history(g, nodes[-1]) == nodes

    g.save(tmp_path / "deep")
    loaded = Graph.load(tmp_path / "deep")
    assert len(reads.history(loaded, loaded.node(root.id))) == 10_000
    assert len(loaded) == len(g)

    fork = ops.fork(g, nodes[4_999])
    assert reads.history(g, fork) == [*nodes[:5_000], fork]
    assert reads.frontier(g, fork) is nodes[4_999]


def test_fork_and_persistence_preserve_full_config_without_sharing_values(tmp_path):
    g, root = agent_graph(
        "configured",
        args={"doc": "value"},
        output_schema={"type": "string"},
        child_config={
            "output_schema": {
                "type": "object",
                "properties": {"done": {"type": "boolean"}},
                "required": ["done"],
            },
            "model": "fast",
        },
        max_depth=4,
        max_iters=7,
        max_budget=123,
        keep_n_messages=5,
        max_output_length=999,
        max_query_chars=888,
        resume_on_child_done=True,
    )

    fork = ops.fork(g, root, name="root")
    g.save(tmp_path / "configured")
    loaded = Graph.load(tmp_path / "configured").node(root.id)

    assert fork.config == root.config
    assert loaded.config == root.config
    assert fork.config is not root.config
    assert fork.args is not root.args
    assert fork.config.output_schema is not root.config.output_schema
    assert fork.config.child_config is not root.config.child_config


def test_reads_count_usage_agents_and_errors():
    g, root = agent_graph("indexed")
    ops.append(
        g, root, LLMOutput(content="answer", usage=LLMUsage(input_tokens=7, output_tokens=3))
    )
    action = ops.append(g, root, ExecAction(code="delegate"))
    child = ops.spawn(g, "child", by=action, name="child")
    error = ops.append(g, child, ErrorOutput(content="failed"))

    assert reads.usage(g) == LLMUsage(input_tokens=7, output_tokens=3)
    assert reads.usage(g, [child]) == LLMUsage()
    assert reads.tree(g, root) == [root, child]
    assert g.node(child.id) is child
    assert g.nodes(ErrorOutput) == [error]
    assert len(g.nodes()) == 5


def test_saved_log_is_flat_ordered_and_strict(tmp_path):
    g, _root, _nodes = chain(4)
    run = g.save(tmp_path / "run")
    document = json.loads((run / "graph.json").read_text())

    assert set(document) == {"id", "items"}
    assert [item["seq"] for item in document["items"]] == list(range(len(document["items"])))
    assert all("children" not in item and "parent_id" not in item for item in document["items"])

    def broken(edit) -> Graph:
        data = json.loads(json.dumps(document))
        edit(data["items"])
        (run / "graph.json").write_text(json.dumps(data))
        return Graph.load(run)

    with pytest.raises(ValueError, match="no message snapshot field"):
        broken(lambda items: items[0]["payload"].pop("messages"))
    with pytest.raises(ValueError, match="out of log order"):
        broken(lambda items: items.reverse())
    with pytest.raises(ValueError, match="unknown node type 'unknown_query'"):
        broken(lambda items: items[1].update(type="unknown_query"))


def test_multi_message_snapshots_survive_persistence_and_fork(tmp_path):
    @dataclass
    class BoardQuery(UserQuery):
        type: ClassVar[str] = "test_board_query_snapshot"

    g, root = agent_graph("root")
    query = ops.append(g, root, BoardQuery(content="move"))
    query.snapshot_messages(
        [
            {"role": "user", "content": "board"},
            {"role": "user", "content": "move"},
        ]
    )

    g.save(tmp_path / "run")
    loaded = Graph.load(tmp_path / "run")
    forked = ops.fork(g, query)

    assert reads.frontier(loaded, loaded.node(root.id)).message_dicts() == query.message_dicts()
    assert reads.frontier(g, forked).message_dicts() == query.message_dicts()


def test_graph_rejects_dependency_cycles_and_missing_producers():
    g, root = agent_graph("root")
    action = ops.append(g, root, ExecAction(code="launch"))
    left = ops.spawn(g, "left", by=action, name="left")
    right = ops.spawn(g, "right", by=action, name="right", depends_on=[left])

    with pytest.raises(InvariantError):
        g.add_edge(Edge(kind=DEPENDS_ON, source=left.id, target=right.id))
    with pytest.raises(InvariantError):
        ops.spawn(g, "orphan", depends_on=[AgentStart(content="missing")])


def test_persistence_validates_and_round_trips_saved_tool_calls(tmp_path):
    g, root = agent_graph("root")
    action = ops.append(g, root, ExecAction(code="query", turn_mode=TurnMode.FINAL))
    action.record_tool_call("llm_query", 0, {"answer": 42})
    ops.append(g, root, ExecOutput(content="done"))

    g.save(tmp_path / "run")
    [loaded_action] = Graph.load(tmp_path / "run").nodes(ExecAction)
    assert loaded_action.turn_mode is TurnMode.FINAL
    assert loaded_action.saved_tool_calls == {"llm_query": {0: {"answer": 42}}}

    action.saved_tool_calls["llm_query"][1] = object()
    with pytest.raises(TypeError, match="must be strict JSON"):
        g.save(tmp_path / "bad")


def test_graph_file_is_the_atomic_save_commit_point(tmp_path, monkeypatch):
    g, root, _nodes = chain(2)
    run = g.save(tmp_path / "run")
    committed = [item.id for item in g.items()]
    ops.append(g, root, UserQuery(content="new work"))
    write_json = persistence._write_json

    def interrupt_graph_commit(path, data):
        if path.name == "graph.json":
            raise OSError("simulated crash before graph commit")
        write_json(path, data)

    monkeypatch.setattr(persistence, "_write_json", interrupt_graph_commit)
    with pytest.raises(OSError, match="simulated crash"):
        g.save(run)

    assert [item.id for item in Graph.load(run).items()] == committed


def test_large_args_are_content_addressed_once(tmp_path):
    large = "payment,row\n" + ("merchant,100.00\n" * 8_000)
    g, root = agent_graph("root", args={"data": large})
    action = ops.append(g, root, ExecAction(code="delegate"))
    child = ops.spawn(g, "child", by=action, name="child", data=large)

    run = g.save(tmp_path / "run")
    graph_text = (run / "graph.json").read_text(encoding="utf-8")
    refs = [
        item["payload"]["args"]["data"]
        for item in json.loads(graph_text)["items"]
        if item.get("type") == "agent_start"
    ]

    assert len(list((run / "blobs").iterdir())) == 1
    assert len({json.dumps(ref, sort_keys=True) for ref in refs}) == 1
    assert large not in graph_text

    loaded = Graph.load(run)
    assert loaded.node(root.id).args["data"] == large
    assert loaded.node(child.id).args["data"] == large


def test_removed_tree_apis_have_no_compatibility_aliases():
    for name in ("to_dict", "tokens", "walk", "fork", "append", "parent_agent", "next", "prev"):
        assert not hasattr(Node, name), name
    for name in ("frontier", "sub_agents", "transcript", "result", "save", "load"):
        assert not hasattr(AgentStart, name), name
    assert not hasattr(persistence, "to_document")


def test_registered_custom_node_round_trips(tmp_path):
    @dataclass
    class ReviewOutput(Node):
        type: ClassVar[str] = "test_review_output"
        verdict: str = ""

    g, root = agent_graph("review")
    ops.append(g, root, ReviewOutput(verdict="ship"))

    g.save(tmp_path / "run")
    loaded = Graph.load(tmp_path / "run")
    frontier = reads.frontier(loaded, loaded.node(root.id))

    assert isinstance(frontier, ReviewOutput)
    assert frontier.verdict == "ship"


def test_graph_saved_with_child_output_schema_loads_as_child_config(tmp_path):
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    g, root = agent_graph("legacy")
    g.save(tmp_path / "run")
    path = tmp_path / "run" / "graph.json"
    data = json.loads(path.read_text())
    config = data["items"][0]["payload"]["config"]
    del config["child_config"]
    config["child_output_schema"] = schema
    path.write_text(json.dumps(data))

    loaded = Graph.load(tmp_path / "run").node(root.id)

    assert loaded.config.child_config == {"output_schema": schema}


def test_child_config_applies_to_every_descendant():
    root = AgentConfig(max_depth=3, child_config={"model": "fast", "prompt_profile": "worker"})

    child = root.child("child")
    grandchild = child.child("grandchild", output_schema={"type": "string"})

    assert (child.model, child.prompt_profile) == ("fast", "worker")
    assert (grandchild.model, grandchild.prompt_profile) == ("fast", "worker")
    assert grandchild.output_schema == {"type": "string"}
    assert grandchild.max_depth == 3
    assert root.model == "default"
