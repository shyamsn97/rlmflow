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
    a = g.start("task")
    reply = g.append(a, Node("assistant", "hi"))
    child = g.start("child", parent=reply)
    [edge] = g.edges(kind=SPAWN)

    assert [a.seq, reply.seq, child.seq, edge.seq] == [0, 1, 2, 3]


def test_a_failed_write_leaves_the_graph_unchanged():
    g = Graph()
    a = g.start("task")
    stranger = Node("user", "elsewhere")
    with pytest.raises(KeyError, match="no agent"):
        g.append(stranger, ran())
    with pytest.raises(TypeError, match="text must be a string"):
        g.send(a, 123)
    with pytest.raises(TypeError, match="task must be a string"):
        g.start(123, parent=a)
    with pytest.raises(TypeError, match="strict JSON"):
        g.link("peer", a, a, bad=object())
    assert g.agents() == [a] and g.nodes() == [a] and g.edges() == []


def test_ids_are_unique():
    g = Graph()
    a = g.start("task")
    node = g.append(a, Node("assistant"))
    with pytest.raises(GraphError, match="already in the graph"):
        g.append(a, node)
    assert own(a) == [node]


def test_containment_owner_and_node_queries():
    g = Graph()
    a = g.start("task")
    reply = g.append(a, Node("assistant", "hi"))
    message = g.send(a, "later")

    assert a.id in g and reply in g and message.id in g and "node_missing" not in g
    assert g.agent(a.id) is a and g.agent(a) is a and g.node(reply.id) is reply
    assert g.node(a.id) is a and message.agent is a and len(g) == 3
    assert a.nodes() == [a, reply, message] and g.nodes() == [a, reply, message]
    assert list(g) == [a] and g.roots() == [a]
    with pytest.raises(KeyError, match="no node"):
        g.node("node_missing")
    with pytest.raises(KeyError, match="no agent"):
        g.agent(reply.id)


def test_application_edges_need_known_endpoints():
    g = Graph()
    a = g.start("task")
    with pytest.raises(GraphError, match="not in this graph"):
        g.link("peer", a, "node_missing")
    assert g.edges() == []

    edge = g.link("peer", a, a, weight=2)
    assert (edge.kind, edge.source, edge.target, edge.data) == ("peer", a.id, a.id, {"weight": 2})
    assert not edge.structural
    assert g.edges(kind="peer", touching=a.id) == [edge]


def test_status_is_done_idle_or_running():
    g = Graph()
    parent = g.start("parent")
    child = g.start("child", parent=g.append(parent, ran()))
    assert parent.status() == "running" and child.status() == "running"
    assert child.nodes() == [child] and child.final_result() is None

    g.append(parent, Node("assistant", data={"seen": len(parent.nodes()), "idle": True}))
    assert parent.status() == "idle"
    g.send(parent, "nudge")
    # Any message wakes an idle session.
    assert parent.status() == "running"

    finish(g, child, 1)
    assert child.status() == "done" and child.final_result() == 1


def test_code_that_has_not_run_keeps_its_agent_running():
    g = Graph()
    a = g.start("task")
    finish(g, a, 1)
    code_node = g.append(a, Node("assistant", code="x = 1"))
    assert code_node.state == READY and a.status() == "running" and a.final_result() is None


def test_spawn_by_a_node_or_a_root_links_the_parent():
    g = Graph()
    parent = g.start("parent")
    code_node = g.append(parent, ran("launch"))
    child = g.start("child", parent=code_node)
    hosted = g.start("hosted", parent=parent)
    grandchild = g.start("grandchild", parent=child)

    assert child.parent() is parent and hosted.parent() is parent
    assert parent.children() == [child, hosted]
    assert grandchild.lineage() == [parent, child, grandchild]
    assert parent.subtree() == [parent, child, grandchild, hosted]
    assert [(e.source, e.target) for e in g.edges(kind=SPAWN)] == [
        (code_node.id, child.id),
        (parent.id, hosted.id),
        (child.id, grandchild.id),
    ]
    assert g.roots() == [parent]

    with pytest.raises(GraphError, match="not in this graph"):
        g.start("lost", parent=ran())


def test_unlink_removes_an_edge_once():
    g = Graph()
    a = g.start("a")
    b = g.start("b", parent=a)
    peer = g.link("peer", a, b)
    g.unlink(peer)
    assert g.edges(kind="peer") == [] and len(g.edges(kind=SPAWN)) == 1
    with pytest.raises(KeyError, match="no edge"):
        g.unlink(peer.id)


def test_fork_is_a_new_graph_with_the_same_ids_cut_after_the_node():
    g = Graph()
    base = g.start("task")
    reply = g.append(base, Node("assistant", "plan"))
    code_node = g.append(base, ran("x", "out"))

    branch = g.fork(reply)
    copy = branch.agent(base.id)
    assert branch.id != g.id and copy is not base
    assert branch.metadata == {"forked_from": {"graph": g.id, "node": reply.id}}
    assert [n.id for n in copy.nodes()] == [base.id, reply.id]
    assert copy.status() == "running"

    fresh = branch.append(copy, ran("y"))
    assert fresh.id not in g and base.nodes() == [base, reply, code_node]
    assert g.metadata == {}


def test_fork_at_a_root_starts_over_from_its_inputs():
    g = Graph()
    base = g.start("task", limit=3)
    g.append(base, Node("assistant", "plan"))

    branch = g.fork(base)
    copy = branch.agent(base.id)
    assert copy.nodes() == [copy] and copy.input == {"task": "task", "limit": 3}


def test_fork_copies_other_top_level_agents_and_their_messages():
    g = Graph()
    root = g.start("task")
    child = g.start("child", parent=root)
    other = g.start("other", name="other")
    note = g.send(root, "from the other tree", by=other)

    branch = g.fork(child)
    assert [a.id for a in branch.agents()] == [root.id, child.id, other.id]
    assert [n.id for n in own(branch.agent(root.id))] == [note.id]


def test_fork_takes_back_what_a_peer_sent_after_the_node():
    g = Graph()
    root = g.start("task")
    peer = g.start("peer", name="peer")
    first = g.append(peer, Node("assistant", "thinking"))
    later = g.append(peer, Node("assistant", "send"))
    note = g.send(root, "from the peer", by=later)

    branch = g.fork(first)
    assert [a.id for a in branch.agents()] == [root.id, peer.id]
    assert own(branch.agent(root.id)) == [] and note.id not in branch
    assert note.id in g


def test_fork_needs_a_node_in_the_graph_and_a_known_repl_choice():
    g = Graph()
    base = g.start("task")
    with pytest.raises(GraphError, match="not in this graph"):
        g.fork(ran())
    with pytest.raises(ValueError, match="repl must be"):
        g.fork(base, repl="shared")
    assert g.agents() == [base]


def test_names_and_paths_derive_from_configs_and_launchers():
    g = Graph()
    root = g.start("root")
    code_node = g.append(root, ran())
    first = g.start("a", parent=code_node, config="worker")
    second = g.start("b", parent=code_node, config="worker")
    named = g.start("c", parent=code_node, name="checker")

    assert [a.name for a in (root, first, second, named)] == [
        "root",
        "worker-0",
        "worker-1",
        "checker",
    ]
    assert second.path() == "root.worker-1"
    assert second.lineage() == [root, second]
    assert [a.config_name for a in (root, second, named)] == [None, "worker", "default"]
    assert g.start("again").name == "root-0"
    top = g.start("solo", config="worker")
    assert (top.name, top.config_name, top.parent()) == (
        "worker",
        "worker",
        None,
    )
    assert g.start("more", config="worker").name == "worker-0"
    with pytest.raises(ValueError, match="invalid agent name"):
        g.start("bad", name="no spaces")


def test_add_agent_makes_a_root_whose_keywords_are_repl_inputs():
    g = Graph()
    lead = g.start("x", name="lead", limit=3)
    assert (lead.name, lead.config_name, lead.input) == ("lead", None, {"task": "x", "limit": 3})
    assert (lead.role, lead.text, lead.owner, lead.nodes()) == ("user", "x", lead.id, [lead])
    with pytest.raises(AttributeError, match="until a Flow runs it"):
        lead.config
    lead.bind_config(AgentConfig(model="big"))
    assert lead.config.model == "big" and lead.config_record["model"] == "big"
    assert g.start("y").name == "root"


def test_send_appends_a_message_that_binds_its_inputs():
    g = Graph()
    a = g.start()
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


def test_fork_gives_agents_new_repls_or_keeps_the_ones_still_in_step():
    g = Graph()
    base = g.start("task", config="m")
    peer = g.start("peer", name="peer", parent=base)
    reply = g.append(base, Node("assistant", "plan"))
    tail = g.append(base, Node("assistant", "more"))
    assert (base.repl_id, peer.repl_id) == (base.id, peer.id)

    copied = g.fork(reply)
    assert {a.repl_id for a in copied.agents()}.isdisjoint({base.id, peer.id})

    rewound = g.fork(reply, repl="same")
    assert rewound.agent(peer.id).repl_id == peer.id
    assert rewound.agent(base.id).repl_id != base.id
    assert rewound.agent(base.id).config_name == "m"

    at_tail = g.fork(tail, repl="same")
    assert [a.repl_id for a in at_tail.agents()] == [base.id, peer.id]


def test_fork_at_a_finished_turn_is_done_until_a_message_reopens_it():
    g = Graph()
    base = g.start()
    done = finish(g, base, 1)
    branch = g.fork(done)
    copy = branch.agent(base.id)
    assert copy.status() == "done" and copy.final_result() == 1

    branch.send(copy, "Try again.")
    assert copy.status() == "running"
    assert base.status() == "done"


def test_usage_and_cost_sum_every_node_that_reports_them():
    g = Graph()
    a = g.start("task")
    g.append(a, Node("assistant", data={"usage": {"input_tokens": 3, "output_tokens": 2}}))
    child = g.start("child", parent=a)
    finish(g, child, 1, usage={"input_tokens": 10, "output_tokens": 1})
    g.append(child, Node("assistant", data={"cost_usd": 0.25}))
    assert g.usage().total == 16
    assert g.usage([a]).total == 5
    assert g.usage().cost_usd == 0.25 and g.usage([a]).cost_usd == 0


def test_saved_calls_are_claimed_by_key_in_order():
    g = Graph()
    root = g.start("root")
    code_node = g.append(root, ran("await launch_agent('x')"))
    child = g.start("child", parent=code_node)
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
    root = g.start("root")
    code_node = g.append(root, ran())
    child = g.start("child", parent=code_node)
    done = finish(g, child, 1)
    assert queries.unseen_results(g, root) == [(child, done)]

    g.link(RETURN, done, code_node)
    assert [(e.source, e.target) for e in g.edges(kind=RETURN)] == [(done.id, code_node.id)]
    assert queries.unseen_results(g, root) == []


# -- persistence ----------------------------------------------------------------


def test_save_and_load_round_trip_nodes_edges_histories_and_blobs(tmp_path):
    g = Graph()
    root = g.start("task", doc="x" * 70_000)
    root.bind_config(AgentConfig(max_budget=500))
    record_metadata("experiment", "a", node=root)
    peer = g.start("peer", name="peer", config="worker")
    gone = g.link("peer", root, peer, w=1)
    note = g.link("note", root, peer, k=2)
    reply = g.append(root, Node("assistant", "hi", data={"usage": {"input_tokens": 1}}))
    record_metadata("score", 0.5, node=reply)
    code_node = g.append(root, ran("x = 1", "ok", result=[1, 2]))
    code_node.started_at = code_node.finished_at - 1
    g.unlink(gone)
    message = g.send(peer, "queued", by=code_node, n=3)

    run = g.save(tmp_path / "run")
    assert sorted(path.name for path in run.iterdir()) == ["blobs", "graph.json"]
    assert len(list((run / "blobs").iterdir())) == 1
    graph_text = (run / "graph.json").read_text()
    assert "x" * 70_000 not in graph_text
    document = json.loads(graph_text)
    assert document["format"] == persistence.FORMAT_VERSION == 7
    assert set(document) == {"format", "id", "metadata", "nodes", "edges"}
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
        assert [n.id for n in loaded.agent(agent.id).nodes()] == [n.id for n in agent.nodes()]

    loaded_root = loaded.agent(root.id)
    assert loaded_root.input == {"task": "task", "doc": "x" * 70_000}
    assert loaded_root.config_name is None
    assert loaded_root.config.max_budget == 500
    assert loaded_root.metadata == {"experiment": "a"}
    assert loaded.agent(peer.id).config_name == "worker"
    with pytest.raises(AttributeError, match="until a Flow runs it"):
        loaded.agent(peer.id).config
    assert [e.id for e in loaded.edges(kind="note")] == [note.id]
    assert loaded.edges(kind="peer") == []

    loaded_reply = loaded.node(reply.id)
    assert loaded_reply.data == {"usage": {"input_tokens": 1}} and loaded_reply.producer == "self"
    assert loaded_reply.metadata == {"score": 0.5}
    loaded_code = loaded.node(code_node.id)
    assert (loaded_code.code, loaded_code.output, loaded_code.result) == ("x = 1", "ok", [1, 2])
    assert loaded_code.state == "done"
    assert loaded_code.finished_at - loaded_code.started_at == pytest.approx(1)
    loaded_message = loaded.node(message.id)
    assert (loaded_message.input, loaded_message.producer) == ({"n": 3}, root.id)

    later = finish(loaded, loaded_root, 1)
    assert later.seq > max(item.seq for item in [*g.nodes(), *g.edges()])


def test_a_node_that_was_running_loads_ready_to_run_again(tmp_path):
    g = Graph()
    root = g.start("task")
    code_node = g.append(root, Node("assistant", code="x = 1"))
    code_node.state = RUNNING
    g.save(tmp_path / "run")

    loaded = Graph.load(tmp_path / "run")
    assert loaded.node(code_node.id).state == READY
    assert loaded.agent(root.id).status() == "running"


def test_load_validates_that_every_node_belongs_to_an_agent(tmp_path):
    g = Graph()
    root = g.start("task")
    reply = g.append(root, Node("assistant", "hi"))
    other = g.start("other", name="other")
    g.link("note", reply, other)
    run = g.save(tmp_path / "run")
    original = json.loads((run / "graph.json").read_text())

    def load(edit) -> Graph:
        document = json.loads(json.dumps(original))
        edit(document)
        (run / "graph.json").write_text(json.dumps(document))
        return Graph.load(run)

    def record(document, node):
        return next(item for item in document["nodes"] if item["id"] == node.id)

    def orphan(document):
        record(document, reply)["owner"] = "node_missing"

    def early(document):
        record(document, reply)["seq"] = -5

    def stray(document):
        record(document, other)["owner"] = root.id

    def dangle(document):
        document["edges"][0]["target"] = "node_missing"

    with pytest.raises(GraphError, match="which is no agent here"):
        load(orphan)
    with pytest.raises(GraphError, match="comes before its agent"):
        load(early)
    with pytest.raises(GraphError, match="must own itself"):
        load(stray)
    with pytest.raises(GraphError, match="points at missing"):
        load(dangle)


def test_load_reads_format_6_with_its_history_lists(tmp_path):
    g = Graph()
    root = g.start("task")
    reply = g.append(root, Node("assistant", "hi"))
    root.metadata["experiment"] = "a"
    run = g.save(tmp_path / "run")
    document = json.loads((run / "graph.json").read_text())
    document["format"] = 6
    for record in document["nodes"]:
        record["data"]["producer"] = record.pop("producer")
        if "agent" in record:
            record["root"] = record.pop("agent")
            record["root"]["history"] = [root.id, reply.id]
            record["root"]["metadata"] = {"config": {"model": "old"}, "tag": "t"}
    (run / "graph.json").write_text(json.dumps(document))

    loaded = Graph.load(run)
    agent = loaded.agent(root.id)
    assert [n.id for n in agent.nodes()] == [root.id, reply.id]
    assert agent.metadata == {"tag": "t", "experiment": "a"}
    assert agent.config_record == {"model": "old"}
    assert loaded.node(reply.id).producer == "self" and "producer" not in loaded.node(reply.id).data


def test_a_saved_fork_keeps_where_it_came_from_and_its_repls(tmp_path):
    g = Graph()
    root = g.start("task")
    reply = g.append(root, Node("assistant", "hi"))
    branch = g.fork(reply)

    loaded = Graph.load(branch.save(tmp_path / "branch"))
    assert (loaded.id, loaded.metadata) == (branch.id, branch.metadata)
    repl = loaded.agent(root.id).repl_id
    assert repl == branch.agent(root.id).repl_id != root.id


def test_load_refuses_a_run_with_fork_edges(tmp_path):
    g = Graph()
    root = g.start("task")
    peer = g.start("peer", name="peer")
    g.link("note", root, peer)
    run = g.save(tmp_path / "run")
    document = json.loads((run / "graph.json").read_text())
    document["edges"][0]["kind"] = "fork"
    (run / "graph.json").write_text(json.dumps(document))
    with pytest.raises(GraphError, match="fork edge"):
        Graph.load(run)


def test_load_refuses_older_formats(tmp_path):
    g = Graph()
    g.start("task")
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
    with pytest.raises(ValueError, match="run format 5, and this rlmflow reads only formats 6, 7"):
        Graph.load(run)


def test_load_refuses_runs_saved_before_the_harness_refactor():
    with pytest.raises(ValueError, match="run format 1"):
        Graph.load(FIXTURES / "v1_tree_run")


def test_copy_is_independent_and_keeps_ids():
    g = Graph()
    root = g.start("task")
    reply = g.append(root, Node("assistant", "hi"))
    child = g.start("child", parent=reply)

    clone = g.copy()
    assert clone.id == g.id and clone.agent(child.id) is not child
    assert clone.agent(child.id).parent().id == root.id
    finish(clone, clone.agent(root.id), 1)
    clone.node(reply.id).text = "changed"
    assert own(root) == [reply] and reply.text == "hi"
