import json
from collections import Counter
from dataclasses import dataclass

import pytest

from rlmflow import (
    AgentStart,
    ErrorOutput,
    ExecAction,
    LLMOutput,
    LLMUsage,
    Node,
    UserQuery,
    persistence,
    start,
)


def chain(length: int) -> tuple[AgentStart, list[Node]]:
    root = start("deep")
    nodes: list[Node] = [root]
    frontier: Node = root
    for index in range(length - 1):
        frontier = frontier.append(UserQuery(content=str(index)))
        nodes.append(frontier)
    return root, nodes


def assert_index_matches_tree(root: AgentStart) -> None:
    nodes = list(root.walk())
    agents = [node for node in nodes if isinstance(node, AgentStart)]
    errors = [node for node in nodes if isinstance(node, ErrorOutput)]
    usage = sum(
        (node.usage for node in nodes if isinstance(node, LLMOutput)),
        LLMUsage(),
    )

    assert root.stats.node_count == len(nodes)
    assert root.stats.node_counts == Counter(node.type for node in nodes)
    assert root.stats.agent_count == len(agents)
    assert list(root.iter_agents()) == agents
    assert root.errors() == tuple(errors)
    assert root.usage == usage


def test_deep_graph_operations_are_iterative(tmp_path):
    root, nodes = chain(10_000)

    assert list(root.walk()) == nodes
    assert list(nodes[-1].iter_backwards()) == list(reversed(nodes))

    loaded = AgentStart.load(root.save(tmp_path / "deep"))
    assert len(list(loaded.walk())) == 10_000
    assert_index_matches_tree(loaded)

    fork = nodes[4_999].fork()
    assert len(list(fork.walk())) == 5_000
    assert {node.id for node in fork.walk()}.isdisjoint(node.id for node in root.walk())
    assert_index_matches_tree(fork)


def test_fork_and_persistence_preserve_full_config_without_sharing_values(tmp_path):
    root = start(
        "configured",
        inputs={"doc": "value"},
        output_schema={"type": "string"},
        max_depth=4,
        max_iters=7,
        child_max_iters=3,
        max_budget=123,
        keep_n_messages=5,
        max_output_length=999,
        max_query_chars=888,
    )

    fork = root.fork()
    loaded = AgentStart.load(root.save(tmp_path / "configured"))

    assert fork.config == root.config
    assert loaded.config == root.config
    assert fork.config is not root.config
    assert fork.config.inputs is not root.config.inputs
    assert fork.config.output_schema is not root.config.output_schema


def test_run_index_tracks_counts_usage_agents_and_errors():
    root = start("indexed")
    output = root.append(
        LLMOutput(
            content="answer",
            usage=LLMUsage(input_tokens=7, output_tokens=3),
        )
    )
    action = output.append(ExecAction(code="delegate"))
    child = action.append(AgentStart(content="child", config=root.config.child("child")))
    child.append(ErrorOutput(content="failed"))

    assert root.usage == LLMUsage(input_tokens=7, output_tokens=3)
    assert root.stats.node_count == 5
    assert root.stats.node_counts == {
        "agent_start": 2,
        "llm_output": 1,
        "exec_action": 1,
        "error_output": 1,
    }
    assert list(root.iter_agents()) == [root, child]
    assert root.find_agent(root.id) is root
    assert root.find_agent(child.id) is child
    assert root.errors() == (child.frontier,)
    assert_index_matches_tree(root)
    with pytest.raises(TypeError):
        root.stats.node_counts["node"] = 1  # type: ignore[index]


def test_v3_is_flat_strict_and_rejects_v2():
    root, _ = chain(4)
    document = persistence.to_document(root)

    assert document["version"] == 3
    assert "root" not in document
    assert all("children" not in record for record in document["nodes"])
    assert persistence.to_document(persistence.from_document(document)) == document

    with pytest.raises(persistence.UnsupportedGraphVersion, match="requires v3"):
        persistence.from_document({"version": 2, "root": {}})

    broken = {**document, "nodes": [dict(record) for record in document["nodes"]]}
    broken["nodes"][1]["parent_id"] = "later"
    with pytest.raises(ValueError, match="missing or later parent"):
        persistence.from_document(broken)

    unknown = {**document, "nodes": [dict(record) for record in document["nodes"]]}
    unknown["nodes"][1]["type"] = "unknown_query"
    with pytest.raises(ValueError, match="unknown node type 'unknown_query'"):
        persistence.from_document(unknown)


def test_graph_file_is_the_atomic_save_commit_point(tmp_path, monkeypatch):
    root, nodes = chain(2)
    run = persistence.save(root, tmp_path / "run")
    committed_ids = [node.id for node in root.walk()]
    nodes[-1].append(UserQuery(content="new work"))
    write_json = persistence._write_json

    def interrupt_graph_commit(path, data):
        if path.name == "graph.json":
            raise OSError("simulated crash before graph commit")
        write_json(path, data)

    monkeypatch.setattr(persistence, "_write_json", interrupt_graph_commit)
    with pytest.raises(OSError, match="simulated crash"):
        persistence.save(root, run)

    assert [node.id for node in persistence.load(run).walk()] == committed_ids


def test_large_inputs_are_content_addressed_once_across_graph_views(tmp_path):
    large = "payment,row\n" + ("merchant,100.00\n" * 8_000)
    root = start("root", inputs={"data": large})
    action = root.append(ExecAction(code="delegate"))
    child = action.append(
        AgentStart(
            content="child",
            config=root.config.child("child", inputs={"data": large}),
        )
    )

    run = persistence.save(root, tmp_path / "run")
    graph_text = (run / "graph.json").read_text(encoding="utf-8")
    graph = json.loads(graph_text)
    refs = [
        record["payload"]["inputs"]["data"]
        for record in graph["nodes"]
        if record["type"] == "agent_start"
    ]
    blobs = list((run / "input_blobs").iterdir())

    assert len(blobs) == 1
    assert len({ref[persistence.INPUT_BLOB_REF_KEY] for ref in refs}) == 1
    assert large not in graph_text
    assert large not in (run / "agents" / "root" / "agent.json").read_text()
    assert large not in (run / "agents" / "root" / "session.jsonl").read_text()
    assert large not in (run / "agents" / "root" / "child" / "agent.json").read_text()

    loaded = persistence.load(run)

    assert loaded.config.inputs["data"] == large
    assert loaded.find_agent(child.id).config.inputs["data"] == large


def test_removed_graph_apis_have_no_compatibility_aliases():
    root = start("clean break")

    assert not hasattr(Node, "to_dict")
    assert not hasattr(Node, "tokens")
    assert not hasattr(persistence, "to_dict")
    assert not hasattr(persistence, "from_dict")
    with pytest.raises(TypeError):
        list(root.walk(reverse=True))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        root.fork(new_ids=False)  # type: ignore[call-arg]


def test_registered_custom_node_round_trips():
    @persistence.register_node_type
    @dataclass
    class ReviewOutput(Node):
        type = "test_review_output"
        verdict: str = ""

        def to_record(self):
            record = super().to_record()
            record["payload"]["verdict"] = self.verdict
            return record

    root = start("review")
    root.append(ReviewOutput(verdict="ship"))

    loaded = persistence.from_document(persistence.to_document(root))

    assert isinstance(loaded.frontier, ReviewOutput)
    assert loaded.frontier.verdict == "ship"
