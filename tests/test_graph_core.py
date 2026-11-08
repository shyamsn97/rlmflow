"""The graph store, its writes, and its agent queries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import finish, own, ran

from rlmflow.graph import calls, persistence, queries
from rlmflow.graph.config import AgentConfig
from rlmflow.graph.graph import Graph, GraphError
from rlmflow.graph.nodes import (
    FORK,
    READY,
    RETURN,
    RUNNING,
    SPAWN,
    WRITE,
    Node,
    record_metadata,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_writes_assign_consecutive_seq():
    g = Graph()
    a = g.add_agent("task")
    reply = g.append(a, Node("assistant", "hi"))
    child = g.add_agent("child", parent=reply)
    [edge] = g.edges(kind=SPAWN)

    assert [a.seq, reply.seq, child.seq, edge.seq] == [0, 1, 2, 3]


def test_a_failed_write_leaves_the_graph_unchanged():
    g = Graph()
    a = g.add_agent("task")
    stranger = Node("user", "elsewhere")
    with pytest.raises(KeyError, match="no agent"):
        g.append(stranger, ran())
    with pytest.raises(TypeError, match="text must be a string"):
        g.send(a, 123)
    with pytest.raises(TypeError, match="task must be a string"):
        g.add_agent(123, parent=a)
    with pytest.raises(TypeError, match="strict JSON"):
        g.link("peer", a, a, bad=object())
    assert g.agents() == [a] and g.nodes() == [a] and g.edges() == []


def test_ids_are_unique():
    g = Graph()
    a = g.add_agent("task")
    node = g.append(a, Node("assistant"))
    with pytest.raises(GraphError, match="already in the graph"):
        g.append(a, node)
    assert own(a) == [node]


def test_containment_owner_and_node_queries():
    g = Graph()
    a = g.add_agent("task")
    reply = g.append(a, Node("assistant", "hi"))
    message = g.send(a, "later")

    assert a.id in g and reply in g and message.id in g and "node_missing" not in g
    assert g.agent(a.id) is a and g.agent(a) is a and g.node(reply.id) is reply
    assert g.node(a.id) is a and g.owner(message) is a and len(g) == 3
    assert g.nodes(agent=a) == [a, reply, message] and g.nodes() == [a, reply, message]
    assert list(g) == [a] and g.roots() == [a]
    with pytest.raises(KeyError, match="no node"):
        g.node("node_missing")
    with pytest.raises(KeyError, match="no agent"):
        g.agent(reply.id)


def test_application_edges_need_known_endpoints():
    g = Graph()
    a = g.add_agent("task")
    with pytest.raises(GraphError, match="not in this graph"):
        g.link("peer", a, "node_missing")
    assert g.edges() == []

    edge = g.link("peer", a, a, weight=2)
    assert (edge.kind, edge.source, edge.target, edge.data) == ("peer", a.id, a.id, {"weight": 2})
    assert not edge.structural
    assert g.edges(kind="peer", touching=a.id) == [edge]


def test_status_is_done_idle_or_running():
    g = Graph()
    parent = g.add_agent("parent")
    child = g.add_agent("child", parent=g.append(parent, ran()))
    assert g.status(parent) == "running" and g.status(child) == "running"
    assert g.history(child) == [child] and g.result(child) is None

    g.append(parent, Node("assistant", seen=len(g.history(parent)), idle=True))
    assert g.status(parent) == "idle"
    g.send(parent, "nudge")
    # Any message wakes an idle session.
    assert g.status(parent) == "running"

    finish(g, child, 1)
    assert g.status(child) == "done" and g.result(child) == 1


def test_code_that_has_not_run_keeps_its_agent_running():
    g = Graph()
    a = g.add_agent("task")
    finish(g, a, 1)
    code_node = g.append(a, Node("assistant", code="x = 1"))
    assert code_node.state == READY and g.status(a) == "running" and g.result(a) is None


def test_spawn_by_a_node_or_a_root_links_the_parent():
    g = Graph()
    parent = g.add_agent("parent")
    code_node = g.append(parent, ran("launch"))
    child = g.add_agent("child", parent=code_node)
    hosted = g.add_agent("hosted", parent=parent)
    grandchild = g.add_agent("grandchild", parent=child)

    assert g.parent(child) is parent and g.parent(hosted) is parent
    assert g.children(parent) == [child, hosted]
    assert g.lineage(grandchild) == [parent, child, grandchild]
    assert g.subtree(parent) == [parent, child, grandchild, hosted]
    assert [(e.source, e.target) for e in g.edges(kind=SPAWN)] == [
        (code_node.id, child.id),
        (parent.id, hosted.id),
        (child.id, grandchild.id),
    ]
    assert g.roots() == [parent]

    with pytest.raises(GraphError, match="not in this graph"):
        g.add_agent("lost", parent=ran())


def test_unlink_removes_an_edge_once():
    g = Graph()
    a = g.add_agent("a")
    b = g.add_agent("b", parent=a)
    peer = g.link("peer", a, b)
    g.unlink(peer)
    assert g.edges(kind="peer") == [] and len(g.edges(kind=SPAWN)) == 1
    with pytest.raises(KeyError, match="no edge"):
        g.unlink(peer.id)


def test_fork_history_shares_the_prefix_and_an_empty_fork_stands_at_its_base():
    g = Graph()
    base = g.add_agent("task")
    reply = g.append(base, Node("assistant", "plan"))
    g.append(base, ran("x", "out"))

    fork = g.fork(reply)
    assert g.base(fork) is reply and own(fork) == []
    assert g.history(fork) == [base, reply, fork]
    assert g.status(fork) == "running"

    code_node = g.append(fork, ran("y"))
    assert g.history(fork) == [base, reply, fork, code_node] == g.history(code_node)
    assert own(fork) == [code_node]
    assert len(g.history(base)) == 3

    nested = g.fork(code_node)
    assert (
        g.history(nested) == [base, reply, fork, code_node, nested] and g.base(nested) is code_node
    )
    assert [(e.kind, e.source, e.target) for e in g.edges()] == [
        (FORK, reply.id, fork.id),
        (FORK, code_node.id, nested.id),
    ]


def test_fork_at_a_root_starts_over_from_its_inputs():
    g = Graph()
    base = g.add_agent("task", limit=3)
    g.append(base, Node("assistant", "plan"))

    fork = g.fork(base)
    assert g.history(fork) == [base, fork]
    assert fork.root_data.task == base.root_data.task and fork.input == {}
    assert g.history(fork)[0].input == {"task": "task", "limit": 3}


def test_fork_needs_a_node_in_the_graph():
    g = Graph()
    base = g.add_agent("task")
    with pytest.raises(GraphError, match="not in this graph"):
        g.fork(ran())
    assert g.agents() == [base]


def test_names_and_paths_derive_from_configs_and_launchers():
    g = Graph()
    root = g.add_agent("root")
    code_node = g.append(root, ran())
    first = g.add_agent("a", parent=code_node, config="worker")
    second = g.add_agent("b", parent=code_node, config="worker")
    named = g.add_agent("c", parent=code_node, name="checker")

    assert [a.root_data.name for a in (root, first, second, named)] == [
        "root",
        "worker-0",
        "worker-1",
        "checker",
    ]
    assert g.path(second) == "root.worker-1"
    assert g.lineage(second) == [root, second]
    assert [a.root_data.config_name for a in (root, second, named)] == [None, "worker", "default"]
    assert g.add_agent("again").root_data.name == "root-0"
    top = g.add_agent("solo", config="worker")
    assert (top.root_data.name, top.root_data.config_name, g.parent(top)) == (
        "worker",
        "worker",
        None,
    )
    assert g.add_agent("more", config="worker").root_data.name == "worker-0"
    with pytest.raises(ValueError, match="invalid agent name"):
        g.add_agent("bad", name="no spaces")


def test_add_agent_makes_a_root_whose_keywords_are_repl_inputs():
    g = Graph()
    lead = g.add_agent("x", name="lead", limit=3)
    data = lead.root_data
    assert (data.name, data.config_name, lead.input) == ("lead", None, {"task": "x", "limit": 3})
    assert (lead.role, lead.text, lead.owner, data.history) == ("user", "x", lead.id, [lead])
    with pytest.raises(AttributeError, match="until a Flow runs it"):
        data.config
    data.bind_config(AgentConfig(model="big"))
    assert data.config.model == "big" and data.metadata["config"]["model"] == "big"
    assert g.add_agent("y").root_data.name == "root"


def test_send_appends_a_message_that_binds_its_inputs():
    g = Graph()
    a = g.add_agent()
    reply = g.append(a, Node("assistant", "code"))
    first = g.send(a, "one", by=reply, n=2)
    second = g.send(a, "two")
    [edge] = g.edges(kind=WRITE)

    assert (edge.source, edge.target, edge.data) == (reply.id, first.id, {})
    assert (first.text, first.input, first.producer) == ("one\nSet: `n`", {"n": 2}, a.id)
    assert second.producer == "user" and second.role == "user"
    assert g.send(a, "three", sender="boss").producer == "boss"
    assert own(a)[1:3] == [first, second]
    with pytest.raises(TypeError, match="strict JSON"):
        g.send(a, "bad", value=object())


def test_fork_writes_an_agent_continuing_from_the_node():
    g = Graph()
    base = g.add_agent("task", config="m")
    reply = g.append(base, Node("assistant", "plan"))
    code_node = g.append(base, ran("x = 1", "ran"))
    edges = len(g.edges())

    fork = g.fork(reply, name="retry", config="other")
    assert [e.kind for e in g.edges()[edges:]] == [FORK] and g.base(fork) is reply
    assert (fork.root_data.name, fork.root_data.config_name) == ("retry", "other")
    assert base.root_data.config_name == "m"
    assert g.tail(base) is code_node and g.parent(fork) is None
    second = g.fork(code_node)
    assert (second.root_data.name, second.root_data.config_name) == ("m-fork0", "m")

    edges = len(g.edges())
    judged = g.fork(reply, parent=code_node)
    assert [e.kind for e in g.edges()[edges:]] == [FORK, SPAWN]
    assert g.parent(judged) is base and judged in g.children(base)

    fresh = g.append(fork, ran("x = 2"))
    assert g.history(fork) == [base, reply, fork, fresh]
    assert g.history(base) == [base, reply, code_node]


def test_fork_at_a_finished_turn_is_done_until_a_message_reopens_it():
    g = Graph()
    base = g.add_agent()
    done = finish(g, base, 1)
    fork = g.fork(done)
    assert g.status(fork) == "done" and g.result(fork) == 1

    message = g.send(fork, "Try again.")
    assert g.status(fork) == "running"
    assert g.history(fork) == [base, done, fork, message]
    assert g.status(base) == "done"


def test_usage_and_cost_sum_every_node_that_reports_them():
    g = Graph()
    a = g.add_agent("task")
    g.append(a, Node("assistant", usage={"input_tokens": 3, "output_tokens": 2}))
    child = g.add_agent("child", parent=a)
    finish(g, child, 1, usage={"input_tokens": 10, "output_tokens": 1})
    g.append(child, Node("assistant", cost_usd=0.25))
    assert g.usage().total == 16
    assert g.usage([a]).total == 5
    assert g.usage().cost_usd == 0.25 and g.usage([a]).cost_usd == 0


def test_saved_calls_are_claimed_by_key_in_order():
    g = Graph()
    root = g.add_agent("root")
    code_node = g.append(root, ran("await launch_agent('x')"))
    child = g.add_agent("child", parent=code_node)
    same = calls.key("llm_query", "q")
    assert same == calls.key("llm_query", "q") != calls.key("llm_query", "other")
    calls.record(code_node, "llm_query", same, value="first")
    calls.record(code_node, calls.LAUNCH, calls.key(calls.LAUNCH, "x"), agent=child.id)
    calls.record(code_node, "llm_query", same, value="second")
    calls.record(code_node, "big", "b", value="x" * (calls.MAX_SAVED_BYTES + 1))
    calls.record(code_node, "live", "l", value=object())

    log = calls.CallLog(code_node)
    assert log.claim("llm_query", same)["value"] == "first"
    assert log.claim("llm_query", same)["value"] == "second"
    assert log.claim("llm_query", same) is None
    assert log.claim("big", "b") is None and log.claim("live", "l") is None
    assert [entry["tool"] for entry in log.unclaimed()] == [calls.LAUNCH]
    assert [entry["value"] for entry in code_node.calls] == ["first", "second"]

    calls.record(code_node, calls.LAUNCH, calls.key(calls.LAUNCH, "x"), agent=child.id)
    assert calls.launched(code_node) == [child.id]
    with pytest.raises(TypeError, match="must be JSON"):
        calls.key("tool", object())
    calls.rebind(code_node, child.id, "agent_other")
    assert calls.launched(code_node) == ["agent_other"]


def test_a_return_edge_marks_a_childs_result_as_seen():
    g = Graph()
    root = g.add_agent("root")
    code_node = g.append(root, ran())
    child = g.add_agent("child", parent=code_node)
    done = finish(g, child, 1)
    assert queries.unseen_results(g, root) == [(child, done)]

    g.link(RETURN, done, code_node)
    assert [(e.source, e.target) for e in g.edges(kind=RETURN)] == [(done.id, code_node.id)]
    assert queries.unseen_results(g, root) == []


# -- persistence ----------------------------------------------------------------


def test_save_and_load_round_trip_nodes_edges_histories_and_blobs(tmp_path):
    g = Graph()
    root = g.add_agent("task", doc="x" * 70_000)
    root.root_data.bind_config(AgentConfig(max_budget=500))
    record_metadata("experiment", "a", node=root)
    peer = g.add_agent("peer", name="peer", config="worker")
    gone = g.link("peer", root, peer, w=1)
    note = g.link("note", root, peer, k=2)
    reply = g.append(root, Node("assistant", "hi", usage={"input_tokens": 1}))
    record_metadata("score", 0.5, node=reply)
    code_node = g.append(root, ran("x = 1", "ok", result=[1, 2]))
    code_node.started_at = code_node.finished_at - 1
    fork = g.fork(reply, name="alt")
    g.unlink(gone)
    message = g.send(peer, "queued", by=code_node, n=3)

    run = g.save(tmp_path / "run")
    assert sorted(path.name for path in run.iterdir()) == ["blobs", "graph.json"]
    assert len(list((run / "blobs").iterdir())) == 1
    graph_text = (run / "graph.json").read_text()
    assert "x" * 70_000 not in graph_text
    document = json.loads(graph_text)
    assert document["format"] == persistence.FORMAT_VERSION == 6
    assert set(document) == {"format", "id", "nodes", "edges"}
    loaded = Graph.load(run)

    assert loaded.id == g.id
    for get in (Graph.agents, Graph.nodes):
        assert [(n.id, n.role, n.owner, n.seq) for n in get(loaded)] == [
            (n.id, n.role, n.owner, n.seq) for n in get(g)
        ]
    assert [(e.id, e.kind, e.source, e.target, e.data) for e in loaded.edges()] == [
        (e.id, e.kind, e.source, e.target, e.data) for e in g.edges()
    ]
    for agent in g.agents():
        assert [n.id for n in loaded.history(loaded.agent(agent.id))] == [
            n.id for n in g.history(agent)
        ]

    loaded_root = loaded.agent(root.id)
    assert loaded_root.input == {"task": "task", "doc": "x" * 70_000}
    assert loaded_root.root_data.config_name is None
    assert loaded_root.root_data.config.max_budget == 500
    assert loaded_root.metadata == {"experiment": "a"}
    assert loaded.agent(peer.id).root_data.config_name == "worker"
    with pytest.raises(AttributeError, match="until a Flow runs it"):
        loaded.agent(peer.id).root_data.config
    assert [e.id for e in loaded.edges(kind="note")] == [note.id]
    assert loaded.edges(kind="peer") == []

    loaded_reply = loaded.node(reply.id)
    assert loaded_reply.data == {"usage": {"input_tokens": 1}, "producer": "self"}
    assert loaded_reply.metadata == {"score": 0.5}
    loaded_code = loaded.node(code_node.id)
    assert (loaded_code.code, loaded_code.output, loaded_code.result) == ("x = 1", "ok", [1, 2])
    assert loaded_code.state == "done"
    assert loaded_code.finished_at - loaded_code.started_at == pytest.approx(1)
    loaded_message = loaded.node(message.id)
    assert (loaded_message.input, loaded_message.producer) == ({"n": 3}, root.id)
    assert loaded.base(loaded.agent(fork.id)) is loaded_reply

    later = finish(loaded, loaded_root, 1)
    assert later.seq > max(item.seq for item in [*g.nodes(), *g.edges()])


def test_a_node_that_was_running_loads_ready_to_run_again(tmp_path):
    g = Graph()
    root = g.add_agent("task")
    code_node = g.append(root, Node("assistant", code="x = 1"))
    code_node.state = RUNNING
    g.save(tmp_path / "run")

    loaded = Graph.load(tmp_path / "run")
    assert loaded.node(code_node.id).state == READY
    assert loaded.status(loaded.agent(root.id)) == "running"


def test_load_validates_that_every_node_is_placed_once(tmp_path):
    g = Graph()
    root = g.add_agent("task")
    reply = g.append(root, Node("assistant", "hi"))
    fork = g.fork(reply)
    run = g.save(tmp_path / "run")
    original = json.loads((run / "graph.json").read_text())

    def load(edit) -> Graph:
        document = json.loads(json.dumps(original))
        edit(document)
        (run / "graph.json").write_text(json.dumps(document))
        return Graph.load(run)

    def record(document, node):
        return next(item for item in document["nodes"] if item["id"] == node.id)

    def unplace(document):
        record(document, root)["root"]["history"].remove(reply.id)

    def misplace(document):
        record(document, root)["root"]["history"].append("node_missing")

    def steal(document):
        record(document, fork)["root"]["history"].append(reply.id)

    def headless(document):
        record(document, root)["root"]["history"].pop(0)

    def dangle(document):
        document["edges"][0]["target"] = "node_missing"

    with pytest.raises(GraphError, match="exactly one agent's history"):
        load(unplace)
    with pytest.raises(GraphError, match="lists node 'node_missing'"):
        load(misplace)
    with pytest.raises(GraphError, match="doesn't own"):
        load(steal)
    with pytest.raises(GraphError, match="must start with its root"):
        load(headless)
    with pytest.raises(GraphError, match="points at missing"):
        load(dangle)


def test_load_refuses_older_formats(tmp_path):
    g = Graph()
    g.add_agent("task")
    run = g.save(tmp_path / "run")
    document = json.loads((run / "graph.json").read_text())
    document["format"] = 3
    (run / "graph.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="run format 3"):
        Graph.load(run)


def test_load_refuses_a_format_5_run_with_agents_and_typed_nodes(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    agent = {"id": "agent_a", "name": "root", "task": "t", "args": {}, "nodes": ["node_a"]}
    node = {"id": "node_a", "type": "llm", "seq": 1, "data": {"text": "hi"}}
    document = {"format": 5, "id": "graph_a", "agents": [agent], "nodes": [node], "edges": []}
    (run / "graph.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="run format 5, and this rlmflow reads only format 6"):
        Graph.load(run)


def test_load_refuses_runs_saved_before_the_harness_refactor():
    with pytest.raises(ValueError, match="run format 1"):
        Graph.load(FIXTURES / "v1_tree_run")


def test_copy_is_independent_and_keeps_ids():
    g = Graph()
    root = g.add_agent("task")
    reply = g.append(root, Node("assistant", "hi"))
    child = g.add_agent("child", parent=reply)

    clone = g.copy()
    assert clone.id == g.id and clone.agent(child.id) is not child
    assert clone.parent(clone.agent(child.id)).id == root.id
    finish(clone, clone.agent(root.id), 1)
    clone.node(reply.id).text = "changed"
    assert own(root) == [reply] and reply.text == "hi"
