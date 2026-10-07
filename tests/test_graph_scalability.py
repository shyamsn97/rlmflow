import json
from collections import Counter
from dataclasses import dataclass
from typing import ClassVar

import pytest

from rlmflow import (
    AgentStart,
    ErrorOutput,
    ExecAction,
    ExecOutput,
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
        nxt = UserQuery(content=str(index))
        frontier.append(nxt)
        frontier = nxt
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
        args={"doc": "value"},
        output_schema={"type": "string"},
        child_output_schema={
            "type": "object",
            "properties": {"done": {"type": "boolean"}},
            "required": ["done"],
        },
        max_depth=4,
        max_iters=7,
        max_budget=123,
        keep_n_messages=5,
        max_output_length=999,
        max_query_chars=888,
        resume_on_child_done=True,
    )

    fork = root.fork()
    loaded = AgentStart.load(root.save(tmp_path / "configured"))

    assert fork.config == root.config
    assert loaded.config == root.config
    assert fork.config is not root.config
    assert fork.args is not root.args
    assert fork.config.output_schema is not root.config.output_schema
    assert fork.config.child_output_schema is not root.config.child_output_schema


def test_run_index_tracks_counts_usage_agents_and_errors():
    root = start("indexed")
    output = LLMOutput(
        content="answer",
        usage=LLMUsage(input_tokens=7, output_tokens=3),
    )
    root.append(output)
    action = ExecAction(code="delegate")
    output.append(action)
    child = AgentStart(content="child", config=root.config.child("child"))
    action.append(child)
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


def test_graph_document_is_flat_strict_and_unversioned():
    root, _ = chain(4)
    document = persistence.to_document(root)

    assert "version" not in document
    assert "root" not in document
    assert all("children" not in record for record in document["nodes"])
    assert persistence.to_document(persistence.from_document(document)) == document

    missing_snapshot = json.loads(json.dumps(document))
    del missing_snapshot["nodes"][0]["payload"]["messages"]
    with pytest.raises(ValueError, match="no message snapshot field"):
        persistence.from_document(missing_snapshot)

    broken = {**document, "nodes": [dict(record) for record in document["nodes"]]}
    broken["nodes"][1]["parent_id"] = "later"
    with pytest.raises(ValueError, match="missing or later parent"):
        persistence.from_document(broken)

    unknown = {**document, "nodes": [dict(record) for record in document["nodes"]]}
    unknown["nodes"][1]["type"] = "unknown_query"
    with pytest.raises(ValueError, match="unknown node type 'unknown_query'"):
        persistence.from_document(unknown)


def test_multi_message_snapshots_survive_persistence_and_fork():
    @dataclass
    class BoardQuery(UserQuery):
        type: ClassVar[str] = "test_board_query_snapshot"

    root = start("root")
    query = BoardQuery(content="move")
    root.append(query)
    query.snapshot_messages(
        [
            {"role": "user", "content": "board"},
            {"role": "user", "content": "move"},
        ]
    )

    loaded = persistence.from_document(persistence.to_document(root))
    forked = query.fork()

    assert loaded.frontier.message_dicts() == query.message_dicts()
    assert forked.frontier.message_dicts() == query.message_dicts()


def test_persistence_rejects_invalid_argument_references_and_cycles():
    root = start("root")
    action = ExecAction(code="launch")
    root.append(action)
    left = AgentStart(content="left", config=root.config.child("left"))
    right = AgentStart(content="right", config=root.config.child("right"))
    action.append(left)
    action.record_tool_call("launch_subagent", 0, {"$rlmflow_agent": left.id})
    action.append(right)
    action.record_tool_call("launch_subagent", 1, {"$rlmflow_agent": right.id})

    left.args["files"] = ["a.txt", "b.txt"]
    loaded = persistence.from_document(persistence.to_document(root))
    assert loaded.find_agent(left.id).args["files"] == ["a.txt", "b.txt"]
    left.args.clear()

    left.args["right"] = {"$rlmflow_agent": right.id}
    right.args["left"] = {"$rlmflow_agent": left.id}
    with pytest.raises(ValueError, match="dependency cycle"):
        persistence.to_document(root)

    right.args.clear()
    left.args["right"] = {"$rlmflow_agent": "missing"}
    with pytest.raises(ValueError, match="references missing agent"):
        persistence.to_document(root)

    left.args["right"] = {"$rlmflow_agent": left.id}
    with pytest.raises(ValueError, match="cannot depend on itself"):
        persistence.to_document(root)

    left.args["right"] = {"$rlmflow_agent": right.id, "extra": "bad"}
    with pytest.raises(TypeError, match="exact agent reference"):
        persistence.to_document(root)


def test_persistence_validates_and_round_trips_saved_tool_calls():
    root = start("root")
    action = ExecAction(code="query")
    root.append(action)
    action.record_tool_call("llm_query", 0, {"answer": 42})
    action.append(ExecOutput(content="done"))

    loaded = persistence.from_document(persistence.to_document(root))
    loaded_action = next(node for node in loaded.walk() if isinstance(node, ExecAction))
    assert loaded_action.saved_tool_calls == {"llm_query": {0: {"answer": 42}}}

    action.saved_tool_calls["llm_query"][1] = object()
    with pytest.raises(TypeError, match="must be strict JSON"):
        persistence.to_document(root)


def test_persistence_rejects_out_of_scope_input_reference():
    root = start("root", max_depth=2)
    root_action = ExecAction(code="launch siblings")
    root.append(root_action)
    owner = AgentStart(content="owner", config=root.config.child("owner"))
    source = AgentStart(content="source", config=root.config.child("source"))
    root_action.append(owner)
    root_action.record_tool_call("launch_subagent", 0, {"$rlmflow_agent": owner.id})
    root_action.append(source)
    root_action.record_tool_call("launch_subagent", 1, {"$rlmflow_agent": source.id})

    owner_action = ExecAction(code="launch consumer")
    owner.append(owner_action)
    consumer = AgentStart(
        content="consumer",
        args={"source": {"$rlmflow_agent": source.id}},
        config=owner.config.child("consumer"),
    )
    owner_action.append(consumer)
    owner_action.record_tool_call("launch_subagent", 0, {"$rlmflow_agent": consumer.id})

    with pytest.raises(ValueError, match="outside its launch scope"):
        persistence.to_document(root)


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


def test_large_args_are_content_addressed_once_across_graph_views(tmp_path):
    large = "payment,row\n" + ("merchant,100.00\n" * 8_000)
    root = start("root", args={"data": large})
    action = ExecAction(code="delegate")
    root.append(action)
    child = AgentStart(
        content="child",
        args={"data": large},
        config=root.config.child("child"),
    )
    action.append(child)

    run = persistence.save(root, tmp_path / "run")
    graph_text = (run / "graph.json").read_text(encoding="utf-8")
    graph = json.loads(graph_text)
    refs = [
        record["payload"]["args"]["data"]
        for record in graph["nodes"]
        if record["type"] == "agent_start"
    ]
    blobs = list((run / "arg_blobs").iterdir())

    assert len(blobs) == 1
    assert len({ref[persistence.ARG_BLOB_REF_KEY] for ref in refs}) == 1
    assert large not in graph_text
    assert large not in (run / "agents" / "root" / "agent.json").read_text()
    assert large not in (run / "agents" / "root" / "session.jsonl").read_text()
    assert large not in (run / "agents" / "root" / "child" / "agent.json").read_text()

    loaded = persistence.load(run)

    assert loaded.args["data"] == large
    assert loaded.find_agent(child.id).args["data"] == large


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
