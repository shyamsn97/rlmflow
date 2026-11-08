import json

import pytest
from helpers import agent_graph

from rlmflow import (
    CODE,
    LLM,
    Agent,
    AgentConfig,
    Graph,
    LLMUsage,
    Node,
    persistence,
)
from rlmflow.graph import calls


def chain(length: int) -> tuple[Graph, Agent, list[Node]]:
    g, root = agent_graph("deep")
    nodes = [g.append(root, Node(CODE, code=str(index), output="")) for index in range(length)]
    return g, root, nodes


def test_deep_graph_operations_are_iterative(tmp_path):
    g, root, nodes = chain(10_000)

    assert g.history(root) == nodes
    assert g.history(nodes[-1]) == nodes

    g.save(tmp_path / "deep")
    loaded = Graph.load(tmp_path / "deep")
    assert len(loaded.history(loaded.agent(root.id))) == 10_000
    assert len(loaded) == len(g)

    fork = g.fork(nodes[4_999])
    assert g.history(fork) == nodes[:5_000]
    assert g.history(fork)[-1] is nodes[4_999]


def test_deep_agent_chains_are_iterative():
    g, root = agent_graph("deep")
    agent = root
    for _ in range(2_000):
        agent = g.add_agent("deeper", parent=agent)

    assert (len(g.lineage(agent)) - 1) == 2_000
    assert len(g.subtree(root)) == 2_001
    assert g.lineage(agent)[0] is root


def test_fork_and_persistence_preserve_full_config_without_sharing_values(tmp_path):
    g, root = agent_graph("configured", args={"doc": "value"})
    root.bind_config(
        AgentConfig(
            output_schema={"type": "string"},
            model="fast",
            max_depth=4,
            max_iters=7,
            max_budget=123,
            keep_n_messages=5,
            max_output_length=999,
            max_query_chars=888,
        )
    )

    fork = g.fork(root, name="forked")
    g.save(tmp_path / "configured")
    loaded = Graph.load(tmp_path / "configured").agent(root.id)

    assert fork.config_name == root.config_name is None
    assert loaded.config == root.config
    assert fork.args == root.args and fork.args is not root.args
    assert loaded.config.output_schema is not root.config.output_schema


def test_reads_count_usage_agents_and_errors():
    g, root = agent_graph("indexed")
    g.append(root, Node(LLM, text="answer", usage={"input_tokens": 7, "output_tokens": 3}))
    cell = g.append(root, Node(CODE, code="delegate", output=""))
    child = g.add_agent("child", parent=cell, name="child")
    error = g.append(child, Node(CODE, code="1/0", output="ZeroDivisionError", error=True))

    assert g.usage() == LLMUsage(input_tokens=7, output_tokens=3)
    assert g.usage([child]) == LLMUsage()
    assert g.subtree(root) == [root, child]
    assert g.agent(child.id) is child
    assert [n for n in g.nodes(CODE) if n.data.get("error")] == [error]
    assert len(g.nodes()) == 3 and len(g.agents()) == 2


def test_saved_graph_is_flat_ordered_and_strict(tmp_path):
    g, _root, _nodes = chain(4)
    run = g.save(tmp_path / "run")
    document = json.loads((run / "graph.json").read_text())

    assert set(document) == {"format", "id", "agents", "nodes", "edges", "edits"}
    seqs = [item["seq"] for item in document["nodes"]]
    assert seqs == sorted(seqs)
    assert all("children" not in item and "parent_id" not in item for item in document["agents"])

    def broken(edit) -> Graph:
        data = json.loads(json.dumps(document))
        edit(data["nodes"])
        (run / "graph.json").write_text(json.dumps(data))
        return Graph.load(run)

    with pytest.raises(ValueError, match="unknown node type 'unknown_query'"):
        broken(lambda items: items[1].update(type="unknown_query"))
    with pytest.raises(ValueError, match="unknown node type 'llm_output'"):
        broken(lambda items: items[1].update(type="llm_output"))
    with pytest.raises(TypeError, match="data must be an object"):
        broken(lambda items: items[1].update(data=[]))


def test_persistence_validates_and_round_trips_recorded_calls(tmp_path):
    g, root = agent_graph("root")
    cell = Node(CODE, code="query", output="done")
    calls.record(cell, "llm_query", "q", value={"answer": 42})
    g.append(root, cell)

    g.save(tmp_path / "run")
    [loaded_cell] = Graph.load(tmp_path / "run").nodes(CODE)
    assert calls.claim(loaded_cell, "llm_query", "q")["value"] == {"answer": 42}

    cell.data["calls"].append({"tool": "llm_query", "value": object()})
    with pytest.raises(TypeError, match="must be strict JSON"):
        g.save(tmp_path / "bad")


def test_graph_file_is_the_atomic_save_commit_point(tmp_path, monkeypatch):
    g, root, _nodes = chain(2)
    run = g.save(tmp_path / "run")
    committed = [node.id for node in g.nodes()]
    g.append(root, Node(CODE, code="new work", output=""))
    write_text = persistence._write_text

    def interrupt_graph_commit(path, text):
        if path.name == "graph.json":
            raise OSError("simulated crash before graph commit")
        write_text(path, text)

    monkeypatch.setattr(persistence, "_write_text", interrupt_graph_commit)
    with pytest.raises(OSError, match="simulated crash"):
        g.save(run)

    assert [node.id for node in Graph.load(run).nodes()] == committed


def test_large_args_are_content_addressed_once(tmp_path):
    large = "payment,row\n" + ("merchant,100.00\n" * 8_000)
    g, root = agent_graph("root", args={"data": large})
    cell = g.append(root, Node(CODE, code="delegate", output=""))
    child = g.add_agent("child", parent=cell, name="child", data=large)

    run = g.save(tmp_path / "run")
    graph_text = (run / "graph.json").read_text(encoding="utf-8")
    refs = [agent["args"]["data"] for agent in json.loads(graph_text)["agents"]]

    assert len(list((run / "blobs").iterdir())) == 1
    assert len(refs) == 2 and len({json.dumps(ref, sort_keys=True) for ref in refs}) == 1
    assert large not in graph_text

    loaded = Graph.load(run)
    assert loaded.agent(root.id).args["data"] == large
    assert loaded.agent(child.id).args["data"] == large


def test_removed_tree_apis_have_no_compatibility_aliases():
    for name in ("to_dict", "tokens", "walk", "fork", "append", "parent_agent", "next", "prev"):
        assert not hasattr(Node, name), name
    for name in ("frontier", "sub_agents", "transcript", "result", "save", "load"):
        assert not hasattr(Node, name), name
    for name in ("attach", "merge", "as_of", "items", "depends_on", "inputs", "tree", "terminal"):
        assert not hasattr(Graph, name), name
    assert not hasattr(persistence, "to_document")
