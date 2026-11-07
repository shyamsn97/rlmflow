"""Editing a graph: edit, delete, remove and replace, and what each invalidates."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rlmflow import AgentConfig, Flow
from rlmflow.graph import calls
from rlmflow.graph.graph import Graph, GraphError, Invalidation
from rlmflow.graph.nodes import (
    CODE,
    DONE,
    LLM,
    MESSAGE,
    RETURN,
    SPAWN,
    Edge,
    Node,
)
from rlmflow.graph.persistence import to_records
from rlmflow.graph.queries import unseen_results
from rlmflow.view.text import render_tree
from rlmflow.view.timeline import replay, snapshot, steps, timeline


def launch(g: Graph, cell: Node, task: str, index: int = 0, **kwargs):
    """``launch_agent(task)`` inside ``cell``: the child and the cell's saved launch."""
    child = g.add_agent(task, parent=cell, **kwargs)
    calls.record(cell, calls.LAUNCH, index, agent=child.id)
    return child


def awaited(g: Graph, cell: Node, *children) -> None:
    """``await child`` inside ``cell``: the saved awaits, and their return edges."""
    for index, child in enumerate(children):
        done = g.history(child)[-1]
        calls.record(cell, calls.AWAIT, index, agent=child.id, done=done.id)
        g.link_return(done, cell)


def delegation() -> SimpleNamespace:
    """root: plan -> cell (launches and awaits worker) -> reply -> final; worker: think -> done."""
    g = Graph()
    root = g.add_agent("root")
    plan = g.append(root, Node(LLM, text="plan"))
    cell = g.append(root, Node(CODE, code="print(await launch_agent('work'))", output="1"))
    worker = launch(g, cell, "work", name="worker")
    think = g.append(worker, Node(LLM, text="think"))
    done = g.append(worker, Node(DONE, result=1))
    awaited(g, cell, worker)
    reply = g.append(root, Node(LLM, text="answer"))
    final = g.append(root, Node(DONE, result=1))
    return SimpleNamespace(**locals())


def records(g: Graph) -> str:
    return json.dumps(to_records(g), sort_keys=True)


def call(cell: Node, tool: str, index: int = 0) -> int:
    """Where the ``index``-th call to ``tool`` sits in the cell's saved calls."""
    positions = [i for i, entry in enumerate(cell.data["calls"]) if entry["tool"] == tool]
    return positions[index]


# -- edit -----------------------------------------------------------------------


def test_edit_changes_data_and_rewinds_the_owner_to_the_node():
    d = delegation()
    g = d.g
    inv = g.edit(d.plan, text="new plan")

    assert d.plan.data["text"] == "new plan"
    assert d.root.nodes == [d.plan]
    assert inv.rewound == [d.root.id]
    assert inv.removed == [d.worker.id]
    assert set(inv.dropped) == {d.cell.id, d.reply.id, d.final.id, d.think.id, d.done.id}
    assert inv.rerun == []
    assert d.worker.id not in g and g.edges() == []
    assert g.edits == [{"op": "edit", "target": d.plan.id, "text": "new plan", **inv.as_dict()}]
    assert g.status(d.root) == "running"


def test_edit_a_childs_done_reruns_the_cell_that_awaited_it():
    d = delegation()
    g = d.g
    inv = g.edit(d.done, result=2)

    assert d.done.data["result"] == 2 and g.result(d.worker) == 2
    assert inv.rerun == [d.cell.id]
    assert inv.rewound == [d.root.id]
    assert inv.dropped == [d.final.id, d.reply.id]
    assert inv.removed == []
    assert d.root.nodes == [d.plan, d.cell] and d.cell.pending
    assert d.worker.nodes == [d.think, d.done] and g.status(d.worker) == "done"
    assert g.edges(kind=RETURN) == []
    assert g.history(d.root)[-1] is d.cell
    assert g.status(d.root) == "running"


def test_edit_a_result_still_in_the_inbox_takes_the_message_back():
    g = Graph()
    root = g.add_agent("root")
    root.bind_config(AgentConfig())
    cell = g.append(root, Node(CODE, code="launch_agent('work')"))
    worker = launch(g, cell, "work")
    done = g.append(worker, Node(DONE, result=1))
    flow = Flow()
    message = flow.deliver_results(g, root)
    assert root.inbox == [message]
    assert [(e.source, e.target) for e in g.edges(kind=RETURN)] == [(done.id, message.id)]

    inv = g.edit(done, result=2)

    assert root.inbox == [] and message.id not in g
    assert inv.dropped == [message.id] and inv.rewound == [] and inv.rerun == []
    assert unseen_results(g, root) == [(worker, done)]
    again = flow.deliver_results(g, root)
    assert g.edges(kind=RETURN) == g.edges(kind=RETURN, target=again.id)


def test_edit_a_delivered_result_rewinds_the_receiver_to_before_it():
    g = Graph()
    root = g.add_agent("root")
    root.bind_config(AgentConfig())
    cell = g.append(root, Node(CODE, code="launch_agent('work')"))
    worker = launch(g, cell, "work")
    done = g.append(worker, Node(DONE, result=1))
    message = Flow().deliver_results(g, root)
    g.deliver(root)
    later = g.append(root, Node(LLM, text="got it"))

    inv = g.edit(done, result=2)

    assert root.nodes == [cell]
    assert inv.rewound == [root.id]
    assert set(inv.dropped) == {message.id, later.id}
    assert g.edges(kind=RETURN) == []


def test_editing_an_inbox_message_changes_only_its_data():
    g = Graph()
    root = g.add_agent("root")
    llm = g.append(root, Node(LLM, text="hi"))
    message = g.send(root, "a")

    inv = g.edit(message, text="b")

    assert not inv and inv == Invalidation()
    assert message.data["text"] == "b" and root.inbox == [message] and root.nodes == [llm]
    assert g.edits[-1]["op"] == "edit"


def test_edit_rewinds_a_fork_based_at_the_node_and_removes_forks_based_later():
    g = Graph()
    root = g.add_agent("root")
    plan = g.append(root, Node(LLM, text="plan"))
    cell = g.append(root, Node(CODE, code="x = 1"))
    at_plan = g.fork(plan, name="at_plan")
    own = g.append(at_plan, Node(CODE, code="x = 2"))
    at_cell = g.fork(cell, name="at_cell")

    inv = g.edit(plan, text="other plan")

    assert at_plan.id in g and at_plan.nodes == [] and at_plan.base is plan
    assert g.history(at_plan) == [plan]
    assert at_cell.id not in g
    assert set(inv.rewound) == {root.id, at_plan.id}
    assert inv.removed == [at_cell.id]
    assert set(inv.dropped) == {cell.id, own.id}
    assert [a for a in g.agents() if a.base is not None] == [at_plan]


def test_rerun_removes_forks_taken_at_the_cell():
    d = delegation()
    g = d.g
    fork = g.fork(d.cell, name="retry")

    inv = g.edit(d.done, result=2)

    assert fork.id not in g and inv.removed == [fork.id]


def test_edit_rejects_non_json_data_and_unknown_nodes():
    d = delegation()
    g = d.g
    before = records(g)
    with pytest.raises(TypeError, match="strict JSON"):
        g.edit(d.plan, text=object())
    with pytest.raises(KeyError, match="not in this graph"):
        g.edit(Node(LLM, text="stray"), text="x")
    assert records(g) == before and g.edits == []


# -- delete and remove ------------------------------------------------------------


def test_delete_drops_the_node_and_everything_after_it():
    d = delegation()
    g = d.g
    inv = g.delete(d.reply)

    assert d.root.nodes == [d.plan, d.cell]
    assert inv.dropped == [d.final.id, d.reply.id]
    assert inv.rewound == [d.root.id] and inv.removed == [] and inv.rerun == []
    assert d.worker.id in g and len(g.edges(kind=RETURN)) == 1
    assert g.edits[-1] == {"op": "delete", "target": d.reply.id, **inv.as_dict()}


def test_deleting_an_inbox_message_takes_it_back():
    g = Graph()
    root = g.add_agent("root")
    sender = g.add_agent("sender")
    cell = g.append(sender, Node(CODE, code="send"))
    message = g.send(root, "hi", by=cell)

    inv = g.delete(message)

    assert root.inbox == [] and message.id not in g
    assert inv.dropped == [message.id] and inv.rewound == []
    assert g.edges(kind=MESSAGE) == [] and sender.nodes == [cell]


def test_deleting_a_cell_removes_its_children_and_takes_back_its_messages():
    g = Graph()
    root = g.add_agent("root")
    plan = g.append(root, Node(LLM, text="plan"))
    cell = g.append(root, Node(CODE, code="launch and send"))
    worker = launch(g, cell, "work", name="worker")
    nested = g.add_agent("nested", parent=worker, name="nested")
    queued_peer = g.add_agent("queued", name="queued")
    queued = g.send(queued_peer, "for later", by=cell)
    read_peer = g.add_agent("read", name="read")
    before = g.append(read_peer, Node(CODE, code="b0"))
    read = g.send(read_peer, "read me", by=cell)
    g.deliver(read_peer)
    after = g.append(read_peer, Node(LLM, text="replying"))

    inv = g.delete(cell)

    assert root.nodes == [plan]
    assert set(inv.removed) == {worker.id, nested.id}
    assert queued_peer.inbox == [] and queued.id not in g
    assert read_peer.nodes == [before]
    assert set(inv.rewound) == {root.id, read_peer.id}
    assert set(inv.dropped) == {cell.id, queued.id, read.id, after.id}
    assert g.edges(kind=SPAWN) == [] and g.edges(kind=MESSAGE) == []
    assert g.agents() == [root, queued_peer, read_peer]


def test_remove_takes_out_the_subtree_and_what_it_caused():
    d = delegation()
    g = d.g
    nested = g.add_agent("nested", parent=d.worker, name="nested")
    peer = g.add_agent("peer", name="peer")
    note = g.send(peer, "from the worker", by=d.think)

    inv = g.remove(d.worker)

    assert set(inv.removed) == {d.worker.id, nested.id}
    assert d.worker.id not in g and nested.id not in g
    assert inv.rerun == [d.cell.id] and d.cell.pending
    assert d.root.nodes == [d.plan, d.cell]
    assert peer.inbox == [] and note.id not in g
    assert g.children(d.root) == [] and g.edges(kind=SPAWN) == []
    assert inv.rewound == [d.root.id]
    assert g.edits[-1] == {"op": "remove", "target": d.worker.id, **inv.as_dict()}


def test_remove_an_agent_removes_forks_based_on_its_nodes():
    g = Graph()
    root = g.add_agent("root")
    plan = g.append(root, Node(LLM, text="plan"))
    fork = g.fork(plan, name="branch")
    g.append(fork, Node(LLM, text="own"))
    copy = g.fork(root, name="copy")

    inv = g.remove(root)

    assert g.agents() == [copy] and set(inv.removed) == {root.id, fork.id}


# -- replace ------------------------------------------------------------------------


def test_replace_a_tool_call_with_a_value():
    g = Graph()
    root = g.add_agent("root")
    cell = g.append(root, Node(CODE, code="a = await llm_query('x'); b = await llm_query('y')"))
    calls.record(cell, "llm_query", 0, value="old")
    calls.record(cell, "llm_query", 1, value="keep")
    reply = g.append(root, Node(LLM, text="next"))
    final = g.append(root, Node(DONE, result="old"))

    inv = g.replace(cell, call(cell, "llm_query"), "new")

    assert cell.data["calls"] == [
        {"tool": "llm_query", "value": "new"},
        {"tool": "llm_query", "value": "keep"},
    ]
    assert cell.pending
    assert inv.rerun == [cell.id] and inv.rewound == [root.id]
    assert inv.dropped == [final.id, reply.id] and inv.removed == []
    assert root.nodes == [cell]
    assert g.edits[-1] == {
        "op": "replace",
        "target": cell.id,
        "tool": "llm_query",
        "index": 0,
        "value": "new",
        **inv.as_dict(),
    }


def test_replace_an_await_with_a_value_keeps_the_old_result_from_being_delivered():
    d = delegation()
    g = d.g
    inv = g.replace(d.cell, call(d.cell, calls.AWAIT), 5)

    assert d.cell.data["calls"][call(d.cell, calls.AWAIT)] == {"tool": "await", "value": 5}
    assert inv.rerun == [d.cell.id] and inv.removed == []
    assert d.worker.id in g and g.status(d.worker) == "done"
    [edge] = g.edges(kind=RETURN)
    assert (edge.source, edge.target, edge.data) == (d.done.id, d.cell.id, {"replaced": True})
    assert unseen_results(g, d.root) == []


def test_replace_an_await_with_another_agent():
    d = delegation()
    g = d.g
    other = g.add_agent("other", name="other")

    inv = g.replace(d.cell, call(d.cell, calls.AWAIT), agent=other)

    assert d.cell.data["calls"][call(d.cell, calls.AWAIT)] == {"tool": "await", "agent": other.id}
    assert inv.rerun == [d.cell.id]
    assert g.edits[-1]["agent"] == other.id


def test_replace_a_launch_with_another_agent_removes_the_old_child():
    d = delegation()
    g = d.g
    other = g.add_agent("prepared", name="prepared")
    g.append(other, Node(DONE, result=9))

    inv = g.replace(d.cell, call(d.cell, calls.LAUNCH), agent=other)

    assert calls.launched(d.cell) == [other.id]
    assert inv.removed == [d.worker.id] and d.worker.id not in g
    assert inv.rerun == [d.cell.id]
    assert g.parent(other) is d.root and g.children(d.root) == [other]
    assert g.roots() == [d.root]
    [edge] = g.edges(kind=SPAWN)
    assert (edge.source, edge.target, edge.data) == (d.cell.id, other.id, {})


def test_replace_leaves_no_receipt_pointing_at_a_removed_agent():
    d = delegation()
    g = d.g
    other = g.add_agent("prepared", name="prepared")
    g.append(other, Node(DONE, result=9))

    g.replace(d.cell, call(d.cell, calls.LAUNCH), agent=other)

    assert all(entry["agent"] in g for entry in d.cell.data["calls"] if "agent" in entry)


def test_replace_refuses_mismatched_replacements():
    d = delegation()
    g = d.g
    before = records(g)
    with pytest.raises(GraphError, match="only be replaced by another agent"):
        g.replace(d.cell, call(d.cell, calls.LAUNCH), "value")
    calls.record(d.cell, "llm_query", 0, value="answer")
    before = records(g)
    with pytest.raises(GraphError, match="only be replaced by a value"):
        g.replace(d.cell, call(d.cell, "llm_query"), agent=d.root)
    with pytest.raises(GraphError, match="saved no call 5"):
        g.replace(d.cell, 5, "x")
    with pytest.raises(TypeError, match="exactly one"):
        g.replace(d.cell, call(d.cell, "llm_query"))
    with pytest.raises(TypeError, match="exactly one"):
        g.replace(d.cell, call(d.cell, "llm_query"), "x", agent=d.root)
    with pytest.raises(TypeError, match="strict JSON"):
        g.replace(d.cell, call(d.cell, "llm_query"), object())
    assert records(g) == before and g.edits == []


# -- every edit -----------------------------------------------------------------------

#: One of each edit on ``delegation()``; ``d.other`` is a finished top-level agent.
EDITS = {
    "edit_done": lambda d, **kw: d.g.edit(d.done, result=2, **kw),
    "edit_plan": lambda d, **kw: d.g.edit(d.plan, text="x", **kw),
    "delete": lambda d, **kw: d.g.delete(d.cell, **kw),
    "remove": lambda d, **kw: d.g.remove(d.worker, **kw),
    "replace_value": lambda d, **kw: d.g.replace(d.cell, call(d.cell, calls.AWAIT), 5, **kw),
    "replace_agent": lambda d, **kw: d.g.replace(
        d.cell, call(d.cell, calls.LAUNCH), agent=d.other, **kw
    ),
}


def delegation_with_other() -> SimpleNamespace:
    d = delegation()
    d.other = d.g.add_agent("other", name="other")
    d.g.append(d.other, Node(DONE, result=9))
    return d


@pytest.mark.parametrize("name", sorted(EDITS))
def test_dry_run_computes_the_invalidation_and_leaves_the_graph_untouched(name):
    d = delegation_with_other()
    g = d.g
    before, revision = records(g), g.revision
    events: list[str] = []
    g.subscribe(lambda kind, _item: events.append(kind))

    planned = EDITS[name](d, dry_run=True)

    assert planned
    assert records(g) == before and g.revision == revision and events == []
    assert g.edits == []
    assert not d.cell.pending

    applied = EDITS[name](d)
    assert applied == planned
    assert records(g) != before and events[-1] == "edit"
    assert g.edits[-1] == {**g.edits[-1], **applied.as_dict()}


def test_edits_are_logged_and_survive_save_and_load(tmp_path):
    d = delegation()
    g = d.g
    peer = g.add_agent("peer", name="peer")
    g.append(peer, Node(LLM, text="hi"))
    g.edit(d.done, result=2)
    g.replace(d.cell, call(d.cell, calls.AWAIT), 7)
    g.delete(peer.nodes[0])
    g.remove(peer)
    assert [entry["op"] for entry in g.edits] == ["edit", "replace", "delete", "remove"]

    loaded = Graph.load(g.save(tmp_path / "run"))

    assert loaded.edits == g.edits
    cell = loaded.node(d.cell.id)
    assert cell.pending and cell.data["calls"][call(cell, calls.AWAIT)] == {
        "tool": "await",
        "value": 7,
    }
    assert loaded.status(loaded.agent(d.root.id)) == "running"


def test_subscribers_see_each_edit_and_its_invalidation():
    d = delegation()
    g = d.g
    seen: list[tuple[str, object]] = []
    g.subscribe(lambda kind, item: seen.append((kind, item)))
    revision = g.revision

    inv = g.edit(d.done, result=2)

    assert seen[-1] == ("edit", inv) and g.revision > revision


@pytest.mark.parametrize("name", sorted(EDITS))
def test_edits_refuse_while_an_affected_agent_is_stepping(name):
    d = delegation_with_other()
    g = d.g
    g.busy.add(d.root.id)
    before = records(g)

    with pytest.raises(GraphError, match="'root' is stepping"):
        EDITS[name](d, dry_run=True)
    with pytest.raises(GraphError, match="'root' is stepping"):
        EDITS[name](d)

    assert records(g) == before
    assert g.edits == []
    assert not d.cell.pending and d.root.nodes[-1] is d.final


def test_an_edit_that_would_touch_a_stepping_descendant_changes_nothing():
    d = delegation()
    g = d.g
    g.busy.add(d.worker.id)
    before = records(g)

    with pytest.raises(GraphError, match="'worker' is stepping"):
        g.edit(d.plan, text="new plan")
    with pytest.raises(GraphError, match="'worker' is stepping"):
        g.delete(d.cell, dry_run=True)

    assert records(g) == before and g.edits == []
    assert d.root.nodes == [d.plan, d.cell, d.reply, d.final]
    # Edits that don't reach the worker still go through.
    assert g.delete(d.final).dropped == [d.final.id]


def test_structural_edges_cannot_be_added_or_removed_by_hand():
    d = delegation()
    g = d.g
    fork = g.fork(d.plan)
    g.send(d.root, "hi", by=d.think)
    for kind in (SPAWN, RETURN, MESSAGE):
        [existing] = g.edges(kind=kind)
        with pytest.raises(GraphError, match="written by the engine"):
            g.remove_edge(existing)
        with pytest.raises(GraphError, match="written by the engine"):
            g.add_edge(Edge(kind, d.root.id, fork.id))
        with pytest.raises(GraphError, match="written by the engine"):
            g.add_edge(kind, d.root, fork)
    assert len(g.edges()) == 3


def test_an_edited_await_marks_the_cell_pending_and_drops_its_done():
    g = Graph()
    root = g.add_agent("root")
    g.append(root, Node(LLM, text="plan"))
    cell = g.append(root, Node(CODE, code="launch"))
    worker = launch(g, cell, "work")
    done = g.append(worker, Node(DONE, result=1))
    submitting = g.append(root, Node(CODE, code="submit(await w)", output=""))
    awaited(g, submitting, worker)
    final = g.append(root, Node(DONE, result=1))
    assert g.status(root) == "done" and g.result(root) == 1

    inv = g.edit(done, result=2)

    assert inv.rerun == [submitting.id] and submitting.pending
    assert inv.dropped == [final.id] and "output" not in submitting.data
    assert g.history(root)[-1] is submitting
    assert g.status(root) == "running" and g.result(root) is None
    message = g.send(root, "while it reruns")
    assert g.deliver(root) == [] and root.inbox == [message]


def test_a_cell_cut_off_mid_run_keeps_what_it_launched_through_save_and_load(tmp_path):
    g = Graph()
    root = g.add_agent("root")
    g.append(root, Node(LLM, text="plan"))
    cell = g.append(root, Node(CODE, code="launch and send", pending=True))
    child = launch(g, cell, "child", name="child")
    peer = g.add_agent("peer", name="peer")
    g.send(peer, "for later", by=cell)

    loaded = Graph.load(g.save(tmp_path / "run"))

    again = loaded.node(cell.id)
    assert again.pending and loaded.status(loaded.agent(root.id)) == "running"
    assert loaded.parent(loaded.agent(child.id)).id == root.id
    assert calls.launched(again) == [child.id]
    assert [m.data["text"] for m in loaded.agent(peer.id).inbox] == ["for later"]


def test_load_refuses_a_dangling_edge(tmp_path):
    d = delegation()
    run = d.g.save(tmp_path / "run")
    document = json.loads((run / "graph.json").read_text())
    [edge] = [e for e in document["edges"] if e["kind"] == RETURN]
    edge["source"] = "node_missing"
    (run / "graph.json").write_text(json.dumps(document))

    with pytest.raises(GraphError, match="points at missing"):
        Graph.load(run)


# -- timeline -------------------------------------------------------------------------


def small_run() -> Graph:
    g = Graph()
    root = g.add_agent("root")
    g.append(root, Node(LLM, text="plan"))
    cell = g.append(root, Node(CODE, code="launch"))
    child = launch(g, cell, "work", name="worker")
    g.append(child, Node(DONE, result=1))
    awaited(g, cell, child)
    g.fork(cell, name="retry")
    g.append(root, Node(DONE, result=1))
    for agent in g.agents():
        agent.bind_config(AgentConfig())
    return g


def test_timeline_lists_agents_as_their_own_steps():
    g = small_run()

    ordered = steps(g)

    assert [step.kind for step in ordered] == [
        "agent",
        "llm",
        "code",
        "agent",
        "done",
        "agent",
        "done",
    ]
    assert [step.agent for step in ordered if step.kind == "agent"] == [
        "root",
        "root.worker",
        "retry",
    ]
    assert ordered[0].summary.startswith("start root")
    assert [step.index for step in ordered] == list(range(len(ordered)))
    assert [item.id for item in timeline(g)] == [step.node.id for step in ordered]


def test_replay_snapshots_grow_monotonically_to_the_whole_graph():
    g = small_run()
    total = len(timeline(g))

    snaps = list(replay(g))

    assert len(snaps) == total
    sizes = [len(s.agents()) + len(s.nodes()) for s in snaps]
    assert sizes == list(range(1, total + 1))
    agents = [len(s.agents()) for s in snaps]
    edges = [len(s.edges()) for s in snaps]
    assert agents == sorted(agents) and edges == sorted(edges)
    assert records(snaps[-1]) == records(g)
    for snap in snaps:
        assert all(e.source in snap and e.target in snap for e in snap.edges())
        assert render_tree(snap)


def test_snapshot_cuts_the_graph_at_a_step():
    g = small_run()
    root = g.agents()[0]
    worker = g.children(root)[0]

    cut = snapshot(g, 4)

    assert [a.id for a in cut.agents()] == [root.id, worker.id]
    assert cut.agent(worker.id).nodes == [] and cut.parent(cut.agent(worker.id)).id == root.id
    assert cut.agent(root.id).config == root.config
    assert cut is not g and len(g.agents()) == 3


def test_replay_works_on_agents_no_flow_has_stepped():
    g = Graph()
    root = g.add_agent("root")
    g.append(root, Node(LLM, text="plan"))

    assert [step.kind for step in steps(g)] == ["agent", "llm"]
    assert [len(s.nodes()) for s in replay(g)] == [0, 1]
