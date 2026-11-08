"""The graph store, its writes, and its agent queries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlmflow.graph import calls, persistence
from rlmflow.graph.config import AgentConfig
from rlmflow.graph.graph import Graph, GraphError
from rlmflow.graph.nodes import (
    CODE,
    DONE,
    IDLE,
    LLM,
    MESSAGE,
    RETURN,
    SPAWN,
    Agent,
    Edge,
    Node,
    record_metadata,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_writes_assign_consecutive_seq_and_notify_subscribers():
    g = Graph()
    seen: list[str] = []
    unsubscribe = g.subscribe(lambda kind, _item: seen.append(kind))
    a = g.add_agent("task")
    llm = g.append(a, Node(LLM, text="hi"))
    child = g.add_agent("child", parent=llm)
    [edge] = g.edges(kind=SPAWN)

    assert [a.seq, llm.seq, child.seq, edge.seq] == [0, 1, 2, 3]
    assert seen == ["agent", "node", "agent", "edge"]
    assert g.revision == 4
    unsubscribe()
    g.append(a, Node(DONE, result=1))
    assert len(seen) == 4 and g.revision == 5


def test_a_failed_write_leaves_the_graph_unchanged():
    g = Graph()
    a = g.add_agent("task")
    stranger = Agent("elsewhere", name="stranger")
    with pytest.raises(KeyError, match="no agent"):
        g.append(stranger, Node(CODE, code=""))
    with pytest.raises(TypeError, match="text must be a string"):
        g.send(a, 123)
    with pytest.raises(TypeError, match="task must be a string"):
        g.add_agent(123, parent=a)
    assert g.agents() == [a] and g.nodes() == [] and g.edges() == [] and a.inbox == []


def test_ids_are_unique():
    g = Graph()
    a = g.add_agent("task")
    node = g.append(a, Node(LLM))
    with pytest.raises(GraphError, match="already in the graph"):
        g.append(a, node)
    assert a.nodes == [node]


def test_containment_owner_and_node_queries():
    g = Graph()
    a = g.add_agent("task")
    llm = g.append(a, Node(LLM, text="hi"))
    queued = g.send(a, "later")

    assert a.id in g and llm in g and queued.id in g and "node_missing" not in g
    assert g.agent(a.id) is a and g.agent(a) is a and g.node(llm.id) is llm
    assert g.owner(queued) is a and len(g) == 2
    assert g.nodes(agent=a) == [llm, queued] and g.nodes(MESSAGE) == [queued]
    assert list(g) == [a] and g.roots() == [a]
    with pytest.raises(KeyError, match="no node"):
        g.node(a.id)
    with pytest.raises(KeyError, match="no agent"):
        g.agent(llm.id)


def test_application_edges_need_known_endpoints_and_a_non_structural_kind():
    g = Graph()
    a = g.add_agent("task")
    with pytest.raises(GraphError, match="not in this graph"):
        g.add_edge(Edge(kind="peer", source=a.id, target="node_missing"))
    for kind in (SPAWN, RETURN, MESSAGE):
        with pytest.raises(GraphError, match="written by the engine"):
            g.add_edge(Edge(kind=kind, source=a.id, target=a.id))
    with pytest.raises(TypeError, match="strict JSON"):
        g.add_edge("peer", a, a, bad=object())
    assert g.edges() == []

    edge = g.add_edge("peer", a, a, weight=2)
    assert (edge.kind, edge.source, edge.target, edge.data) == ("peer", a.id, a.id, {"weight": 2})
    assert g.edges(kind="peer", touching=a.id) == [edge]
    with pytest.raises(GraphError, match="already"):
        g.add_edge(edge)
    assert g.edges() == [edge]


def test_undelivered_message_waits_in_the_inbox():
    g = Graph()
    a = g.add_agent("task")
    done = g.append(a, Node(DONE, result=1))
    message = g.send(a, "again")

    assert a.inbox == [message]
    assert g.history(a)[-1] is done
    assert g.status(a) == "running"

    assert g.deliver(a) == [message]
    assert a.inbox == []
    assert g.history(a)[-1] is message
    assert g.history(a) == [done, message]


def test_status_is_done_idle_or_running():
    g = Graph()
    parent = g.add_agent("parent")
    child = g.add_agent("child", parent=g.append(parent, Node(CODE, code="")))
    assert g.status(parent) == "running" and g.status(child) == "running"
    assert g.history(child) == [] and g.result(child) is None

    g.append(parent, Node(IDLE))
    assert g.status(parent) == "idle"
    g.send(parent, "nudge")
    # Any message wakes an idle session.
    assert g.status(parent) == "running"

    g.append(child, Node(DONE, result=1))
    assert g.status(child) == "done" and g.result(child) == 1


def test_a_pending_cell_keeps_its_agent_running():
    g = Graph()
    a = g.add_agent("task")
    g.append(a, Node(DONE, result=1))
    cell = g.append(a, Node(CODE, code="x = 1", pending=True))
    assert cell.pending and g.status(a) == "running" and g.result(a) is None
    g.send(a, "wait")
    assert g.deliver(a) == []  # a pending cell must run before messages arrive
    g.update(cell, output="", pending=False)
    assert not cell.pending and g.deliver(a)


def test_spawn_by_a_cell_or_an_agent_links_the_parent():
    g = Graph()
    parent = g.add_agent("parent")
    cell = g.append(parent, Node(CODE, code="launch"))
    child = g.add_agent("child", parent=cell)
    hosted = g.add_agent("hosted", parent=parent)
    grandchild = g.add_agent("grandchild", parent=child)

    assert g.parent(child) is parent and g.parent(hosted) is parent
    assert g.children(parent) == [child, hosted]
    assert g.lineage(grandchild) == [parent, child, grandchild]
    assert (len(g.lineage(grandchild)) - 1) == 2 and (len(g.lineage(parent)) - 1) == 0
    assert g.subtree(parent) == [parent, child, grandchild, hosted]
    assert [(e.source, e.target) for e in g.edges(kind=SPAWN)] == [
        (cell.id, child.id),
        (parent.id, hosted.id),
        (child.id, grandchild.id),
    ]
    assert g.roots() == [parent]

    stray = Node(CODE, code="")
    with pytest.raises(GraphError, match="not in this graph"):
        g.add_agent("lost", parent=stray)


def test_structural_edges_are_never_removed_and_application_ones_are():
    g = Graph()
    a = g.add_agent("a")
    b = g.add_agent("b", parent=a)
    with pytest.raises(GraphError, match="written by the engine"):
        g.remove_edge(g.edges(kind=SPAWN)[0])

    peer = g.add_edge(Edge(kind="peer", source=a.id, target=b.id))
    g.remove_edge(peer)
    assert g.edges(kind="peer") == []
    assert len(g.edges(kind=SPAWN)) == 1
    with pytest.raises(KeyError, match="no edge"):
        g.remove_edge(peer.id)


def test_fork_history_shares_the_prefix_and_an_empty_fork_stands_at_its_base():
    g = Graph()
    base = g.add_agent("task")
    llm = g.append(base, Node(LLM, text="plan"))
    g.append(base, Node(CODE, code="x", output="out"))

    fork = g.fork(llm)
    assert fork.base is llm and fork.nodes == []
    assert g.history(fork)[-1] is llm
    assert g.history(fork) == [llm]
    assert not g.status(fork) == "done" and g.status(fork) == "running"

    cell = g.append(fork, Node(CODE, code="y"))
    assert g.history(fork)[-1] is cell
    assert g.history(fork) == [llm, cell]
    assert fork.nodes == [cell]
    assert [node.type for node in g.history(base)] == [LLM, CODE]
    assert g.history(cell) == [llm, cell]

    nested = g.fork(cell)
    assert g.history(nested) == [llm, cell] and nested.base is cell
    assert g.edges() == []


def test_fork_at_an_agent_starts_from_nothing():
    g = Graph()
    base = g.add_agent("task", limit=3)
    g.append(base, Node(LLM, text="plan"))

    fork = g.fork(base)
    assert fork.base is None and g.history(fork) == []
    assert fork.task == base.task and fork.args == base.args and fork.args is not base.args


def test_fork_needs_a_landed_node():
    g = Graph()
    base = g.add_agent("task")
    queued = g.send(base, "later")
    with pytest.raises(GraphError, match="not in a history"):
        g.fork(queued)
    assert g.agents() == [base]


def test_names_and_paths_derive_from_configs_and_launchers():
    g = Graph()
    root = g.add_agent("root")
    cell = g.append(root, Node(CODE, code=""))
    first = g.add_agent("a", parent=cell, config="worker")
    second = g.add_agent("b", parent=cell, config="worker")
    named = g.add_agent("c", parent=cell, name="checker")

    assert [a.name for a in (root, first, second, named)] == [
        "root",
        "worker-0",
        "worker-1",
        "checker",
    ]
    assert g.path(second) == "root.worker-1"
    assert g.lineage(second) == [root, second]
    assert (root.config_name, second.config_name, named.config_name) == (None, "worker", "default")
    assert g.add_agent("again").name == "root-0"
    top = g.add_agent("solo", config="worker")
    assert (top.name, top.config_name, g.parent(top)) == ("worker", "worker", None)
    assert g.add_agent("more", config="worker").name == "worker-0"
    with pytest.raises(ValueError, match="invalid agent name"):
        g.add_agent("bad", name="no spaces")


def test_create_agent_makes_a_root_whose_keywords_are_repl_args():
    g = Graph()
    lead = g.add_agent("x", name="lead", limit=3)
    assert (lead.name, lead.config_name, lead.args) == ("lead", None, {"limit": 3})
    assert lead.type == "agent" and lead.agent_id == lead.id
    with pytest.raises(AttributeError, match="until a Flow steps it"):
        lead.config
    lead.bind_config(AgentConfig(model="big"))
    assert lead.config.model == "big" and lead.metadata["config"]["model"] == "big"
    assert g.add_agent("y").name == "root"


def test_send_queues_and_deliver_moves_messages_at_a_boundary():
    g = Graph()
    a = g.add_agent()
    llm = g.append(a, Node(LLM, text="code"))
    first = g.send(a, "one", by=llm, kind="note")
    second = g.send(a, "two")
    [edge] = g.edges(kind=MESSAGE)
    assert (edge.source, edge.target, edge.data) == (llm.id, first.id, {})
    assert first.data["kind"] == "note" and first.data["sender"] == a.id
    assert second.data["sender"] == "user"
    assert g.send(a, "three", sender="boss").data["sender"] == "boss"
    assert a.inbox[:2] == [first, second]

    assert g.deliver(a) == []  # mid-turn: the llm's code must run first
    out = g.append(a, Node(CODE, code="", output="ran"))
    delivered = g.deliver(a)
    assert delivered[:2] == [first, second]
    assert g.history(a)[-3:] == delivered
    assert g.history(a)[:2] == [llm, out]
    assert a.inbox == []


def test_fork_writes_an_agent_based_at_the_node():
    g = Graph()
    base = g.add_agent("task", config="m")
    llm = g.append(base, Node(LLM, text="plan"))
    cell = g.append(base, Node(CODE, code="x = 1", output="ran"))
    edges = len(g.edges())

    fork = g.fork(llm, name="retry", config="other")
    assert g.edges()[edges:] == [] and fork.base is llm
    assert (fork.name, fork.config_name, base.config_name) == ("retry", "other", "m")
    assert g.history(fork)[-1] is llm
    assert g.history(base)[-1] is cell
    assert g.parent(fork) is None
    second = g.fork(cell)
    assert (second.name, second.config_name) == ("m-fork0", "m")

    edges = len(g.edges())
    judged = g.fork(llm, parent=cell)
    assert [e.kind for e in g.edges()[edges:]] == [SPAWN]
    assert g.parent(judged) is base and judged in g.children(base)

    fresh = g.append(fork, Node(CODE, code="x = 2"))
    assert g.history(fork) == [llm, fresh]
    assert g.history(base) == [llm, cell]


def test_fork_at_done_is_finished_until_a_message_reopens_it():
    g = Graph()
    base = g.add_agent()
    done = g.append(base, Node(DONE, result=1))
    fork = g.fork(done)
    assert g.status(fork) == "done" and g.result(fork) == 1

    message = g.send(fork, "Try again.")
    assert g.status(fork) == "running"
    assert g.deliver(fork) == [message]
    assert g.history(fork) == [done, message]
    assert g.status(base) == "done"


def test_usage_and_cost_sum_every_node_that_reports_them():
    g = Graph()
    a = g.add_agent("task")
    g.append(a, Node(LLM, text="", usage={"input_tokens": 3, "output_tokens": 2}))
    child = g.add_agent("child", parent=a)
    g.append(child, Node(DONE, result=1, usage={"input_tokens": 10, "output_tokens": 1}))
    g.append(child, Node(IDLE, cost_usd=0.25))
    assert g.usage().total == 16
    assert g.usage([a]).total == 5
    assert g.usage().cost_usd == 0.25 and g.usage([a]).cost_usd == 0


def test_saved_calls_are_claimed_by_key_in_order():
    g = Graph()
    root = g.add_agent("root")
    cell = g.append(root, Node(CODE, code="await launch_agent('x')"))
    child = g.add_agent("child", parent=cell)
    same = calls.key("llm_query", "q")
    assert same == calls.key("llm_query", "q") != calls.key("llm_query", "other")
    calls.record(cell, "llm_query", same, value="first")
    calls.record(cell, calls.LAUNCH, calls.key(calls.LAUNCH, "x"), agent=child.id)
    calls.record(cell, "llm_query", same, value="second")
    calls.record(cell, "big", "b", value="x" * (calls.MAX_SAVED_BYTES + 1))
    calls.record(cell, "live", "l", value=object())

    calls.begin(cell)
    assert calls.claim(cell, "llm_query", same)["value"] == "first"
    assert calls.claim(cell, "llm_query", same)["value"] == "second"
    assert calls.claim(cell, "llm_query", same) is None
    assert calls.claim(cell, "big", "b") is None and calls.claim(cell, "live", "l") is None
    assert [entry["tool"] for entry in calls.finish(cell)] == [calls.LAUNCH]
    assert [entry["value"] for entry in cell.data["calls"]] == ["first", "second"]

    calls.record(cell, calls.LAUNCH, calls.key(calls.LAUNCH, "x"), agent=child.id)
    assert calls.launched(cell) == [child.id]
    with pytest.raises(TypeError, match="must be JSON"):
        calls.key("tool", object())
    calls.rebind(cell, child.id, "agent_other")
    assert calls.launched(cell) == ["agent_other"]


def test_link_return_draws_one_edge():
    g = Graph()
    root = g.add_agent("root")
    cell = g.append(root, Node(CODE, code=""))
    child = g.add_agent("child", parent=cell)
    done = g.append(child, Node(DONE, result=1))
    first = g.link_return(done, cell)
    assert g.link_return(done, cell) is first
    assert [(e.source, e.target) for e in g.edges(kind=RETURN)] == [(done.id, cell.id)]


# -- persistence ----------------------------------------------------------------


def test_save_and_load_round_trip_agents_nodes_edges_and_blobs(tmp_path):
    g = Graph()
    root = g.add_agent("task", doc="x" * 70_000)
    root.bind_config(AgentConfig(max_budget=500))
    peer = g.add_agent("peer", name="peer", config="worker")
    edge = g.add_edge(Edge(kind="peer", source=root.id, target=peer.id, data={"w": 1}))
    note = g.add_edge("note", root, peer, k=2)
    llm = g.append(root, Node(LLM, text="hi", usage={"input_tokens": 1}))
    record_metadata("score", 0.5, node=llm)
    fork = g.fork(llm, name="alt")
    g.remove_edge(edge)
    g.send(peer, "queued")

    run = g.save(tmp_path / "run")
    assert sorted(path.name for path in run.iterdir()) == ["blobs", "graph.json"]
    document = json.loads((run / "graph.json").read_text())
    assert document["format"] == persistence.FORMAT_VERSION == 5
    assert set(document) == {"format", "id", "agents", "nodes", "edges", "edits"}
    loaded = Graph.load(run)

    assert loaded.id == g.id
    assert [(a.id, a.name, a.seq) for a in loaded.agents()] == [
        (a.id, a.name, a.seq) for a in g.agents()
    ]
    assert [(n.id, n.type, n.seq) for n in loaded.nodes()] == [
        (n.id, n.type, n.seq) for n in g.nodes()
    ]
    assert [(e.id, e.kind, e.seq) for e in loaded.edges()] == [
        (e.id, e.kind, e.seq) for e in g.edges()
    ]
    loaded_root = loaded.agent(root.id)
    assert loaded_root.args == {"doc": "x" * 70_000}
    assert loaded_root.config_name is None and loaded_root.config.max_budget == 500
    assert loaded.agent(peer.id).config_name == "worker"
    with pytest.raises(AttributeError, match="until a Flow steps it"):
        loaded.agent(peer.id).config
    assert [e.id for e in loaded.edges(kind="note")] == [note.id]
    assert loaded.edges(kind="peer") == []
    assert [m.data["text"] for m in loaded.agent(peer.id).inbox] == ["queued"]
    assert loaded.history(loaded_root)[-1].data == {"text": "hi", "usage": {"input_tokens": 1}}
    assert loaded.history(loaded_root)[-1].metadata == {"score": 0.5}
    assert loaded.agent(fork.id).base is loaded.node(llm.id)

    later = loaded.append(loaded_root, Node(DONE, result=1))
    assert later.seq > max(item.seq for item in [*g.agents(), *g.nodes(), *g.edges()])


def test_load_validates_that_every_node_is_placed_once(tmp_path):
    g = Graph()
    root = g.add_agent("task")
    llm = g.append(root, Node(LLM, text="hi"))
    fork = g.fork(llm)
    g.add_edge("note", root, llm)
    run = g.save(tmp_path / "run")
    original = json.loads((run / "graph.json").read_text())

    def load(edit) -> Graph:
        document = json.loads(json.dumps(original))
        edit(document)
        (run / "graph.json").write_text(json.dumps(document))
        return Graph.load(run)

    def unplace(document):
        document["agents"][0]["nodes"] = []

    def misplace(document):
        document["agents"][0]["nodes"].append("node_missing")

    def rebase(document):
        next(a for a in document["agents"] if a["id"] == fork.id)["base"] = "node_missing"

    def dangle(document):
        document["edges"][0]["target"] = "node_missing"

    with pytest.raises(GraphError, match="exactly one agent's history or inbox"):
        load(unplace)
    with pytest.raises(GraphError, match="unknown node"):
        load(misplace)
    with pytest.raises(GraphError, match="unknown base"):
        load(rebase)
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


def test_load_refuses_runs_saved_before_the_harness_refactor():
    with pytest.raises(ValueError, match="run format 1"):
        Graph.load(FIXTURES / "v1_tree_run")


def test_copy_is_independent_and_keeps_ids():
    g = Graph()
    root = g.add_agent("task")
    llm = g.append(root, Node(LLM, text="hi"))
    child = g.add_agent("child", parent=llm)

    clone = g.copy()
    assert clone.id == g.id and clone.agent(child.id) is not child
    assert clone.parent(clone.agent(child.id)).id == root.id
    clone.append(clone.agent(root.id), Node(DONE, result=1))
    clone.node(llm.id).data["text"] = "changed"
    assert root.nodes == [llm] and llm.data["text"] == "hi"
