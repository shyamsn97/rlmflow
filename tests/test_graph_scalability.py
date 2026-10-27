import json

import pytest
from helpers import agent_graph

from rlmflow import (
    AGENT_START,
    CODE,
    DEPENDS_ON,
    LLM_OUTPUT,
    Edge,
    Graph,
    InvariantError,
    LLMUsage,
    Node,
    persistence,
)


def chain(length: int) -> tuple[Graph, Node, list[Node]]:
    g, root = agent_graph("deep")
    nodes: list[Node] = [root]
    for index in range(length - 1):
        nodes.append(g.append(root, Node(CODE, code=str(index), output="")))
    return g, root, nodes


def test_deep_graph_operations_are_iterative(tmp_path):
    g, root, nodes = chain(10_000)

    assert g.history(root) == nodes
    assert g.history(nodes[-1]) == nodes

    g.save(tmp_path / "deep")
    loaded = Graph.load(tmp_path / "deep")
    assert len(loaded.history(loaded.node(root.id))) == 10_000
    assert len(loaded) == len(g)

    fork = g.fork(nodes[4_999])
    assert g.history(fork) == [*nodes[:5_000], fork]
    assert g.frontier(fork) is nodes[4_999]


def test_fork_and_persistence_preserve_full_config_without_sharing_values(tmp_path):
    g, root = agent_graph(
        "configured",
        args={"doc": "value"},
        output_schema={"type": "string"},
        model="fast",
        max_depth=4,
        max_iters=7,
        max_budget=123,
        keep_n_messages=5,
        max_output_length=999,
        max_query_chars=888,
    )

    fork = g.fork(root, name="root")
    g.save(tmp_path / "configured")
    loaded = Graph.load(tmp_path / "configured").node(root.id)

    assert fork.config == root.config
    assert loaded.config == root.config
    assert fork.args is not root.args
    assert fork.config.output_schema is not root.config.output_schema


def test_reads_count_usage_agents_and_errors():
    g, root = agent_graph("indexed")
    g.append(root, Node(LLM_OUTPUT, text="answer", usage={"input_tokens": 7, "output_tokens": 3}))
    cell = g.append(root, Node(CODE, code="delegate", output=""))
    child = g.spawn("child", by=cell, name="child")
    error = g.append(child, Node(CODE, code="1/0", output="ZeroDivisionError", error=True))

    assert g.usage() == LLMUsage(input_tokens=7, output_tokens=3)
    assert g.usage([child]) == LLMUsage()
    assert g.tree(root) == [root, child]
    assert g.node(child.id) is child
    assert [n for n in g.nodes(CODE) if n.data.get("error")] == [error]
    assert len(g.nodes()) == 5


def test_saved_log_is_flat_ordered_and_strict(tmp_path):
    g, _root, _nodes = chain(4)
    run = g.save(tmp_path / "run")
    document = json.loads((run / "graph.json").read_text())

    assert set(document) == {"format", "id", "configs", "items"}
    assert [item["seq"] for item in document["items"]] == list(range(len(document["items"])))
    assert all("children" not in item and "parent_id" not in item for item in document["items"])

    def broken(edit) -> Graph:
        data = json.loads(json.dumps(document))
        edit(data["items"])
        (run / "graph.json").write_text(json.dumps(data))
        return Graph.load(run)

    with pytest.raises(ValueError, match="out of log order"):
        broken(lambda items: items.reverse())
    with pytest.raises(ValueError, match="unknown node type 'unknown_query'"):
        broken(lambda items: items[1].update(type="unknown_query"))
    with pytest.raises(TypeError, match="data must be an object"):
        broken(lambda items: items[1].update(data=[]))


def test_graph_rejects_dependency_cycles_and_missing_producers():
    g, root = agent_graph("root")
    cell = g.append(root, Node(CODE, code="launch", output=""))
    left = g.spawn("left", by=cell, name="left")
    right = g.spawn("right", by=cell, name="right", depends_on=[left])

    with pytest.raises(InvariantError):
        g.add_edge(Edge(kind=DEPENDS_ON, source=left.id, target=right.id))
    with pytest.raises(InvariantError):
        g.spawn("orphan", depends_on=[Node(AGENT_START, task="missing")])


def test_persistence_validates_and_round_trips_recorded_calls(tmp_path):
    g, root = agent_graph("root")
    cell = Node(CODE, code="query", output="done")
    cell.record_call("llm_query", 0, {"answer": 42})
    g.append(root, cell)

    g.save(tmp_path / "run")
    [loaded_cell] = Graph.load(tmp_path / "run").nodes(CODE)
    assert loaded_cell.calls("llm_query") == {0: {"answer": 42}}

    cell.data["calls"]["llm_query"]["1"] = object()
    with pytest.raises(TypeError, match="must be strict JSON"):
        g.save(tmp_path / "bad")


def test_graph_file_is_the_atomic_save_commit_point(tmp_path, monkeypatch):
    g, root, _nodes = chain(2)
    run = g.save(tmp_path / "run")
    committed = [item.id for item in g.items()]
    g.append(root, Node(CODE, code="new work", output=""))
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
    cell = g.append(root, Node(CODE, code="delegate", output=""))
    child = g.spawn("child", by=cell, name="child", data=large)

    run = g.save(tmp_path / "run")
    graph_text = (run / "graph.json").read_text(encoding="utf-8")
    refs = [
        item["data"]["args"]["data"]
        for item in json.loads(graph_text)["items"]
        if item.get("type") == AGENT_START
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
        assert not hasattr(Node, name), name
    assert not hasattr(persistence, "to_document")
