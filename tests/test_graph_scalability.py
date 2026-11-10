import json

import pytest
from helpers import agent_graph, ran

import rlmflow
from rlmflow import AgentConfig, Graph, LLMUsage, Node, persistence
from rlmflow.graph import calls


def chain(length: int) -> tuple[Graph, Node, list[Node]]:
    g, root = agent_graph("deep")
    nodes = [g.append(root, ran(str(index))) for index in range(length)]
    return g, root, nodes


def test_deep_graph_operations_are_iterative(tmp_path):
    g, root, nodes = chain(10_000)

    assert root.nodes() == [root, *nodes]
    assert nodes[-1].transcript() == [root, *nodes]

    g.save(tmp_path / "deep")
    loaded = Graph.load(tmp_path / "deep")
    assert len(loaded.agent(root.id).nodes()) == 10_001
    assert len(loaded) == len(g)

    branch = g.fork(nodes[4_999])
    assert [n.id for n in branch.agent(root.id).nodes()] == [n.id for n in [root, *nodes[:5_000]]]
    assert len(root.nodes()) == 10_001


def test_deep_agent_chains_are_iterative():
    g, root = agent_graph("deep")
    agent = root
    for _ in range(2_000):
        agent = g.start("deeper", parent=agent)

    assert (len(agent.lineage()) - 1) == 2_000
    assert len(root.subtree()) == 2_001
    assert agent.lineage()[0] is root


def test_persistence_preserves_full_config_without_sharing_values(tmp_path):
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

    branch = g.fork(root)
    g.save(tmp_path / "configured")
    loaded = Graph.load(tmp_path / "configured").agent(root.id)

    assert branch.agent(root.id).config == root.config
    assert loaded.config == root.config
    assert loaded.input == root.input and loaded.input is not root.input
    assert loaded.config.output_schema is not root.config.output_schema


def test_reads_count_usage_agents_and_errors():
    g, root = agent_graph("indexed")
    g.append(
        root, Node("assistant", "answer", data={"usage": {"input_tokens": 7, "output_tokens": 3}})
    )
    code_node = g.append(root, ran("delegate"))
    child = g.start("child", parent=code_node, name="child")
    error = g.append(child, ran("1/0", "ZeroDivisionError", error=True))

    assert g.usage() == LLMUsage(input_tokens=7, output_tokens=3)
    assert g.usage([child]) == LLMUsage()
    assert root.subtree() == [root, child]
    assert g.agent(child.id) is child
    assert [n for n in g.nodes() if n.error] == [error]
    assert len(g.nodes()) == 5 and len(g.agents()) == 2


def test_saved_graph_is_flat_ordered_and_strict(tmp_path):
    g, _root, _nodes = chain(4)
    run = g.save(tmp_path / "run")
    document = json.loads((run / "graph.json").read_text())

    assert set(document) == {"format", "id", "metadata", "nodes", "edges"}
    seqs = [item["seq"] for item in document["nodes"]]
    assert seqs == sorted(seqs)
    assert all("children" not in item and "parent_id" not in item for item in document["nodes"])

    def broken(edit) -> Graph:
        data = json.loads(json.dumps(document))
        edit(data["nodes"])
        (run / "graph.json").write_text(json.dumps(data))
        return Graph.load(run)

    with pytest.raises(ValueError, match="unknown state 'pending'"):
        broken(lambda items: items[1].update(state="pending"))
    with pytest.raises(ValueError, match="role must be 'user' or 'assistant'"):
        broken(lambda items: items[1].update(role="tool"))
    with pytest.raises(TypeError, match="data must be an object"):
        broken(lambda items: items[1].update(data=[]))


def test_persistence_validates_and_round_trips_recorded_calls(tmp_path):
    g, root = agent_graph("root")
    code_node = ran("query", "done")
    calls.record(code_node, "llm_query", "q", value={"answer": 42})
    g.append(root, code_node)

    g.save(tmp_path / "run")
    loaded_code = Graph.load(tmp_path / "run").node(code_node.id)
    assert calls.CallLog(loaded_code).claim("llm_query", "q")["value"] == {"answer": 42}

    code_node.calls.append({"tool": "llm_query", "value": object()})
    with pytest.raises(TypeError, match="must be strict JSON"):
        g.save(tmp_path / "bad")


def test_graph_file_is_the_atomic_save_commit_point(tmp_path, monkeypatch):
    g, root, _nodes = chain(2)
    run = g.save(tmp_path / "run")
    committed = [node.id for node in g.nodes()]
    g.append(root, ran("new work"))
    write_text = persistence._write_text

    def interrupt_graph_commit(path, text):
        if path.name == "graph.json":
            raise OSError("simulated crash before graph commit")
        write_text(path, text)

    monkeypatch.setattr(persistence, "_write_text", interrupt_graph_commit)
    with pytest.raises(OSError, match="simulated crash"):
        g.save(run)

    assert [node.id for node in Graph.load(run).nodes()] == committed


def test_large_inputs_are_content_addressed_once(tmp_path):
    large = "payment,row\n" + ("merchant,100.00\n" * 8_000)
    g, root = agent_graph("root", args={"data": large})
    code_node = g.append(root, ran("delegate"))
    child = g.start("child", parent=code_node, name="child", data=large)

    run = g.save(tmp_path / "run")
    graph_text = (run / "graph.json").read_text(encoding="utf-8")
    nodes = json.loads(graph_text)["nodes"]
    refs = [node["input"]["data"] for node in nodes if "data" in node["input"]]

    assert len(list((run / "blobs").iterdir())) == 1
    assert len(refs) == 2 and len({json.dumps(ref, sort_keys=True) for ref in refs}) == 1
    assert large not in graph_text

    loaded = Graph.load(run)
    assert loaded.agent(root.id).input["data"] == large
    assert loaded.agent(child.id).input["data"] == large


def test_removed_apis_have_no_compatibility_aliases():
    for name in ("to_dict", "tokens", "walk", "fork", "append", "parent_agent", "next", "prev"):
        assert not hasattr(Node, name), name
    for name in ("frontier", "sub_agents", "save", "load", "type", "agent_id", "root_data"):
        assert not hasattr(Node, name), name
    for name in ("attach", "merge", "as_of", "items", "depends_on", "inputs", "tree", "terminal"):
        assert not hasattr(Graph, name), name
    for name in (
        "add_agent",
        "add_edge",
        "remove_edge",
        "link_return",
        "deliver",
        "busy",
        "subscribe",
        "history",
        "tail",
        "status",
        "result",
        "children",
        "parent",
        "lineage",
        "subtree",
        "path",
        "last_turn",
        "owner",
    ):
        assert not hasattr(Graph, name), name
    for name in ("RootData", "CODE", "LLM", "MESSAGE", "IDLE", "PooledLLMClient"):
        assert not hasattr(rlmflow, name), name
    assert not hasattr(persistence, "to_document")
