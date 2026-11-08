"""Editing a graph: edit, delete, remove and replace, and what each invalidates."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from helpers import TestRuntime, agent_graph, finish, own, ran

from rlmflow import Flow, Node
from rlmflow.graph import calls
from rlmflow.graph.edits import RULES, GraphError, Invalidation
from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import DONE, FORK, READY, RETURN, RUNNING, SPAWN, WRITE
from rlmflow.graph.persistence import to_records
from rlmflow.graph.queries import unseen_results
from rlmflow.view.text import render_tree
from rlmflow.view.timeline import replay, snapshot, steps, timeline


def launch(g: Graph, code_node: Node, task: str, index: int = 0, **kwargs) -> Node:
    """``launch_agent(task)`` inside ``code_node``: the child and the code_node's saved launch."""
    child = g.add_agent(task, parent=code_node, **kwargs)
    calls.record(code_node, calls.LAUNCH, f"launch-{index}", agent=child.id)
    return child


def awaited(g: Graph, code_node: Node, *children: Node) -> None:
    """``await child`` inside ``code_node``: the saved awaits, and their return edges."""
    for index, child in enumerate(children):
        done = g.tail(child)
        calls.record(code_node, calls.AWAIT, f"await-{index}", agent=child.id, done=done.id)
        g.link(RETURN, done, code_node)


def delegation() -> SimpleNamespace:
    """root: plan -> code_node (launches and awaits worker) -> reply -> final; worker: think -> done."""
    g, root = agent_graph("root")
    plan = g.append(root, Node("assistant", "plan"))
    code_node = g.append(root, ran("print(await launch_agent('work'))", "1"))
    worker = launch(g, code_node, "work", name="worker")
    think = g.append(worker, Node("assistant", "think"))
    done = finish(g, worker, 1)
    awaited(g, code_node, worker)
    reply = g.append(root, Node("assistant", "answer"))
    final = finish(g, root, 1)
    return SimpleNamespace(**locals())


def records(g: Graph) -> str:
    return json.dumps(to_records(g), sort_keys=True)


def call(code_node: Node, tool: str, index: int = 0) -> int:
    """Where the ``index``-th call to ``tool`` sits in the code_node's saved calls."""
    positions = [i for i, entry in enumerate(code_node.calls) if entry["tool"] == tool]
    return positions[index]


# -- rules ----------------------------------------------------------------------


def test_rules_name_what_each_engine_edge_does_when_its_source_drops_or_changes():
    assert RULES == {
        SPAWN: ("remove_root", "match_calls"),
        FORK: ("remove_root", "restart_root"),
        RETURN: ("rerun_target", "rerun_target"),
        WRITE: ("retract_target", "match_calls"),
    }


def test_spawn_dropped_removes_the_child():
    d = delegation()
    inv = d.g.delete(d.code_node)

    assert d.worker.id not in d.g and inv.removed == [d.worker.id]
    assert d.g.edges(kind=SPAWN) == [] and d.g.children(d.root) == []


def test_spawn_changed_keeps_the_child_until_the_rerun_matches_calls():
    d = delegation()
    inv = d.g.edit(d.code_node, code="print(await launch_agent('work'), 2)")

    assert inv.rerun == [d.code_node.id] and inv.removed == []
    assert d.worker.id in d.g and d.g.children(d.root) == [d.worker]
    assert d.g.status(d.worker) == "done"
    [edge] = d.g.edges(kind=SPAWN)
    assert (edge.source, edge.target) == (d.code_node.id, d.worker.id)


def test_fork_dropped_removes_the_fork():
    d = delegation()
    fork = d.g.fork(d.code_node, name="retry")
    inv = d.g.delete(d.code_node)

    assert fork.id not in d.g and fork.id in inv.removed


def test_fork_changed_restarts_the_fork_from_its_base():
    g, root = agent_graph("root")
    plan = g.append(root, Node("assistant", "plan"))
    fork = g.fork(plan, name="branch")
    mine = g.append(fork, ran("x = 2"))

    inv = g.edit(plan, text="other plan")

    assert fork.id in g and own(fork) == [] and g.base(fork) is plan
    assert g.history(fork) == [root, plan, fork]
    assert fork.id in inv.rewound and inv.dropped == [mine.id]


def test_return_dropped_reruns_the_code_that_awaited_it():
    d = delegation()
    inv = d.g.delete(d.done)

    assert own(d.worker) == [d.think] and d.g.status(d.worker) == "running"
    assert inv.rerun == [d.code_node.id] and d.code_node.state == READY
    assert own(d.root) == [d.plan, d.code_node] and d.g.edges(kind=RETURN) == []


def test_return_changed_reruns_the_code_that_awaited_it():
    d = delegation()
    inv = d.g.edit(d.done, result=2)

    assert d.done.result == 2 and d.g.result(d.worker) == 2
    assert inv.rerun == [d.code_node.id]
    assert inv.rewound == [d.root.id, d.worker.id]
    assert inv.dropped == [d.final.id, d.reply.id] and inv.removed == []
    assert own(d.root) == [d.plan, d.code_node] and d.code_node.state == READY
    assert own(d.worker) == [d.think, d.done] and d.g.status(d.worker) == "done"
    assert d.g.edges(kind=RETURN) == []
    assert d.g.tail(d.root) is d.code_node and d.g.status(d.root) == "running"


def test_return_changed_takes_back_a_delivered_result_and_redelivers_it():
    g, root = agent_graph("root")
    code_node = g.append(root, ran("launch_agent('work')"))
    worker = launch(g, code_node, "work")
    done = finish(g, worker, 1)
    flow = Flow()
    message = flow.deliver_results(g, root)
    later = g.append(root, Node("assistant", "got it"))
    assert [(e.source, e.target) for e in g.edges(kind=RETURN)] == [(done.id, message.id)]

    inv = g.edit(done, result=2)

    assert own(root) == [code_node] and message.id not in g
    assert set(inv.dropped) == {message.id, later.id} and inv.rerun == []
    assert unseen_results(g, root) == [(worker, done)]
    again = flow.deliver_results(g, root)
    assert list(again.input.values()) == [2]
    assert g.edges(kind=RETURN) == g.edges(kind=RETURN, target=again.id)


def test_write_dropped_takes_the_message_back():
    g, root = agent_graph("root")
    peer = g.add_agent("peer", name="peer")
    before = g.append(peer, ran("b = 0"))
    code_node = g.append(root, ran("send"))
    message = g.send(peer, "read me", by=code_node)
    after = g.append(peer, Node("assistant", "replying"))

    inv = g.delete(code_node)

    assert own(peer) == [before] and message.id not in g
    assert set(inv.rewound) == {root.id, peer.id}
    assert set(inv.dropped) == {code_node.id, message.id, after.id}
    assert g.edges(kind=WRITE) == []


def test_write_changed_keeps_the_message_until_the_rerun_matches_calls():
    g, root = agent_graph("root")
    peer = g.add_agent("peer", name="peer")
    code_node = g.append(root, ran("send"))
    message = g.send(peer, "read me", by=code_node)

    inv = g.edit(code_node, code="send again")

    assert inv.rerun == [code_node.id] and peer.id not in inv.rewound
    assert own(peer) == [message]
    [edge] = g.edges(kind=WRITE)
    assert (edge.source, edge.target) == (code_node.id, message.id)


# -- edit -----------------------------------------------------------------------


def test_edit_changes_the_node_and_rewinds_the_owner_to_it():
    d = delegation()
    g = d.g
    inv = g.edit(d.plan, text="new plan")

    assert d.plan.text == "new plan"
    assert own(d.root) == [d.plan]
    assert inv.rewound == [d.root.id]
    assert inv.removed == [d.worker.id]
    assert set(inv.dropped) == {
        d.code_node.id,
        d.reply.id,
        d.final.id,
        d.think.id,
        d.done.id,
        d.worker.id,
    }
    assert inv.rerun == []
    assert d.worker.id not in g and g.edges() == []
    assert g.status(d.root) == "running"


def test_editing_a_node_bumps_its_version():
    d = delegation()
    plan, code_node, done = d.plan.version, d.code_node.version, d.done.version

    d.g.edit(d.done, result=2)
    assert d.done.version == done + 1
    assert d.code_node.version > code_node
    d.g.edit(d.plan, text="again")
    assert d.plan.version == plan + 1


def test_editing_the_last_message_only_changes_it():
    g, root = agent_graph("root")
    reply = g.append(root, Node("assistant", "hi"))
    message = g.send(root, "a")

    inv = g.edit(message, text="b")

    assert inv.dropped == [] and inv.removed == [] and inv.rerun == []
    assert message.text == "b" and own(root) == [reply, message]


def test_editing_a_root_task_updates_the_agent():
    g, root = agent_graph("old task")
    step = g.append(root, ran("x = 1"))

    inv = g.edit(root, text="new task")

    assert root.root_data.task == "new task" and own(root) == []
    assert inv.dropped == [step.id]


def test_edit_reruns_a_fork_based_at_the_node_and_removes_forks_based_later():
    g, root = agent_graph("root")
    plan = g.append(root, Node("assistant", "plan"))
    code_node = g.append(root, ran("x = 1"))
    at_plan = g.fork(plan, name="at_plan")
    mine = g.append(at_plan, ran("x = 2"))
    at_code = g.fork(code_node, name="at_code")

    inv = g.edit(plan, text="other plan")

    assert at_plan.id in g and own(at_plan) == [] and g.base(at_plan) is plan
    assert at_code.id not in g
    assert set(inv.rewound) == {root.id, at_plan.id}
    assert inv.removed == [at_code.id]
    assert {code_node.id, mine.id} <= set(inv.dropped)
    assert [a for a in g.agents() if g.base(a) is not None] == [at_plan]


def test_a_rerun_restarts_forks_taken_at_the_code():
    d = delegation()
    fork = d.g.fork(d.code_node, name="retry")
    mine = d.g.append(fork, Node("assistant", "try again"))

    inv = d.g.edit(d.done, result=2)

    assert fork.id in d.g and own(fork) == [] and mine.id in inv.dropped


def test_edit_rejects_bad_fields_and_unknown_nodes():
    d = delegation()
    g = d.g
    before = records(g)
    with pytest.raises(TypeError, match="must be a string"):
        g.edit(d.plan, text=object())
    with pytest.raises(TypeError, match="edit takes"):
        g.edit(d.plan, output="x")
    with pytest.raises(KeyError, match="not in this graph"):
        g.edit(Node("assistant", "stray"), text="x")
    assert records(g) == before


def test_an_edited_await_makes_the_code_ready_and_drops_its_final():
    g, root = agent_graph("root")
    g.append(root, Node("assistant", "plan"))
    code_node = g.append(root, ran("launch"))
    worker = launch(g, code_node, "work")
    done = finish(g, worker, 1)
    submitting = g.append(root, ran("submit(await w)", result=1, terminal=True, seen=3))
    awaited(g, submitting, worker)
    assert g.status(root) == "done" and g.result(root) == 1

    inv = g.edit(done, result=2)

    assert inv.rerun == [submitting.id] and submitting.state == READY
    assert submitting.output == "" and submitting.result is None
    assert "terminal" not in submitting.data
    assert g.tail(root) is submitting
    assert g.status(root) == "running" and g.result(root) is None


# -- rerunning an edited code_node -----------------------------------------------------


def test_rerunning_an_edited_code_reuses_an_unchanged_launch_and_removes_a_changed_one(
    tmp_path,
):
    async def scenario():
        flow = Flow(runtime=TestRuntime(tmp_path))
        g, root = agent_graph("root")
        code_node = g.append(root, Node("assistant", code="h = launch_agent('work')"))
        await flow.run_node(g, root, code_node)
        [child] = g.children(root)

        g.edit(code_node, code="h = launch_agent('work')\nprint('again')")
        await flow.run_node(g, root, code_node)
        assert code_node.output.strip() == "again"
        assert g.children(root) == [child] and calls.launched(code_node) == [child.id]

        g.edit(code_node, code="h = launch_agent('other')")
        await flow.run_node(g, root, code_node)
        [other] = g.children(root)
        assert other is not child and child.id not in g
        assert other.root_data.task == "other" and calls.launched(code_node) == [other.id]
        await flow.aclose()

    asyncio.run(scenario())


# -- delete and remove ------------------------------------------------------------


def test_delete_drops_the_node_and_everything_after_it():
    d = delegation()
    g = d.g
    inv = g.delete(d.reply)

    assert own(d.root) == [d.plan, d.code_node]
    assert inv.dropped == [d.final.id, d.reply.id]
    assert inv.rewound == [d.root.id] and inv.removed == [] and inv.rerun == []
    assert d.worker.id in g and len(g.edges(kind=RETURN)) == 1


def test_deleting_a_sent_message_takes_it_back():
    g, root = agent_graph("root")
    reply = g.append(root, Node("assistant", "hi"))
    sender = g.add_agent("sender", name="sender")
    code_node = g.append(sender, ran("send"))
    message = g.send(root, "hi", by=code_node)

    inv = g.delete(message)

    assert own(root) == [reply] and message.id not in g
    assert inv.dropped == [message.id] and inv.rewound == [root.id]
    assert g.edges(kind=WRITE) == [] and own(sender) == [code_node]


def test_deleting_a_code_removes_its_children_and_takes_back_its_messages():
    g, root = agent_graph("root")
    plan = g.append(root, Node("assistant", "plan"))
    code_node = g.append(root, ran("launch and send"))
    worker = launch(g, code_node, "work", name="worker")
    nested = g.add_agent("nested", parent=worker, name="nested")
    queued_peer = g.add_agent("queued", name="queued")
    queued = g.send(queued_peer, "for later", by=code_node)
    read_peer = g.add_agent("read", name="read")
    before = g.append(read_peer, ran("b0"))
    read = g.send(read_peer, "read me", by=code_node)
    after = g.append(read_peer, Node("assistant", "replying"))

    inv = g.delete(code_node)

    assert own(root) == [plan]
    assert set(inv.removed) == {worker.id, nested.id}
    assert own(queued_peer) == [] and queued.id not in g
    assert own(read_peer) == [before]
    assert set(inv.rewound) == {root.id, queued_peer.id, read_peer.id}
    assert {code_node.id, queued.id, read.id, after.id} <= set(inv.dropped)
    assert g.edges(kind=SPAWN) == [] and g.edges(kind=WRITE) == []
    assert g.agents() == [root, queued_peer, read_peer]


def test_deleting_a_root_removes_its_agent():
    d = delegation()
    inv = d.g.delete(d.worker)

    assert d.worker.id not in d.g and inv.removed == [d.worker.id]
    assert inv.rerun == [d.code_node.id]


def test_remove_takes_out_the_subtree_and_what_it_caused():
    d = delegation()
    g = d.g
    nested = g.add_agent("nested", parent=d.worker, name="nested")
    peer = g.add_agent("peer", name="peer")
    note = g.send(peer, "from the worker", by=d.think)

    inv = g.remove(d.worker)

    assert set(inv.removed) == {d.worker.id, nested.id}
    assert d.worker.id not in g and nested.id not in g
    assert inv.rerun == [d.code_node.id] and d.code_node.state == READY
    assert own(d.root) == [d.plan, d.code_node]
    assert own(peer) == [] and note.id not in g
    assert g.children(d.root) == [] and g.edges(kind=SPAWN) == []
    assert d.root.id in inv.rewound


def test_remove_an_agent_removes_forks_based_on_its_nodes():
    g, root = agent_graph("root")
    plan = g.append(root, Node("assistant", "plan"))
    fork = g.fork(plan, name="branch")
    g.append(fork, Node("assistant", "own"))
    restart = g.fork(root, name="restart")
    peer = g.add_agent("peer", name="peer")

    inv = g.remove(root)

    assert g.agents() == [peer]
    assert set(inv.removed) == {root.id, fork.id, restart.id}


# -- replace ------------------------------------------------------------------------


def test_replace_a_tool_call_with_a_value():
    g, root = agent_graph("root")
    code_node = g.append(root, ran("a = await llm_query('x'); b = await llm_query('y')"))
    calls.record(code_node, "llm_query", "x", value="old")
    calls.record(code_node, "llm_query", "y", value="keep")
    reply = g.append(root, Node("assistant", "next"))
    final = finish(g, root, "old")

    inv = g.replace(code_node, call(code_node, "llm_query"), "new")

    assert code_node.calls == [
        {"tool": "llm_query", "key": "x", "value": "new"},
        {"tool": "llm_query", "key": "y", "value": "keep"},
    ]
    assert code_node.state == READY
    assert inv.rerun == [code_node.id] and inv.rewound == [root.id]
    assert inv.dropped == [final.id, reply.id] and inv.removed == []
    assert own(root) == [code_node]


def test_replace_an_await_with_a_value_keeps_the_old_result_from_being_delivered():
    d = delegation()
    g = d.g
    inv = g.replace(d.code_node, call(d.code_node, calls.AWAIT), 5)

    assert d.code_node.calls[call(d.code_node, calls.AWAIT)] == {
        "tool": "await",
        "key": "await-0",
        "value": 5,
    }
    assert inv.rerun == [d.code_node.id] and inv.removed == []
    assert d.worker.id in g and g.status(d.worker) == "done"
    [edge] = g.edges(kind=RETURN)
    assert (edge.source, edge.target, edge.data) == (d.done.id, d.code_node.id, {"replaced": True})
    assert unseen_results(g, d.root) == []


def test_replace_an_await_with_another_agent():
    d = delegation()
    g = d.g
    other = g.add_agent("other", name="other")

    inv = g.replace(d.code_node, call(d.code_node, calls.AWAIT), agent=other)

    assert d.code_node.calls[call(d.code_node, calls.AWAIT)] == {
        "tool": "await",
        "key": "await-0",
        "agent": other.id,
    }
    assert inv.rerun == [d.code_node.id]


def test_replace_a_launch_with_another_agent_removes_the_old_child():
    d = delegation()
    g = d.g
    other = g.add_agent("prepared", name="prepared")
    finish(g, other, 9)

    inv = g.replace(d.code_node, call(d.code_node, calls.LAUNCH), agent=other)

    assert calls.launched(d.code_node) == [other.id]
    assert inv.removed == [d.worker.id] and d.worker.id not in g
    assert inv.rerun == [d.code_node.id]
    assert g.parent(other) is d.root and g.children(d.root) == [other]
    assert g.roots() == [d.root]
    [edge] = g.edges(kind=SPAWN)
    assert (edge.source, edge.target, edge.data) == (d.code_node.id, other.id, {})


def test_replace_leaves_no_saved_call_pointing_at_a_removed_agent():
    d = delegation()
    g = d.g
    other = g.add_agent("prepared", name="prepared")
    finish(g, other, 9)

    g.replace(d.code_node, call(d.code_node, calls.LAUNCH), agent=other)

    assert all(entry["agent"] in g for entry in d.code_node.calls if "agent" in entry)


def test_replace_refuses_mismatched_replacements():
    d = delegation()
    g = d.g
    with pytest.raises(GraphError, match="only be replaced by another agent"):
        g.replace(d.code_node, call(d.code_node, calls.LAUNCH), "value")
    calls.record(d.code_node, "llm_query", "q", value="answer")
    before = records(g)
    with pytest.raises(GraphError, match="only be replaced by a value"):
        g.replace(d.code_node, call(d.code_node, "llm_query"), agent=d.root)
    with pytest.raises(GraphError, match="saved no call 5"):
        g.replace(d.code_node, 5, "x")
    with pytest.raises(TypeError, match="exactly one"):
        g.replace(d.code_node, call(d.code_node, "llm_query"))
    with pytest.raises(TypeError, match="exactly one"):
        g.replace(d.code_node, call(d.code_node, "llm_query"), "x", agent=d.root)
    with pytest.raises(TypeError, match="strict JSON"):
        g.replace(d.code_node, call(d.code_node, "llm_query"), object())
    assert records(g) == before


# -- every edit -----------------------------------------------------------------------

#: One of each edit on ``delegation()``; ``d.other`` is a finished top-level agent.
EDITS = {
    "edit_done": lambda d, **kw: d.g.edit(d.done, result=2, **kw),
    "edit_plan": lambda d, **kw: d.g.edit(d.plan, text="x", **kw),
    "edit_code": lambda d, **kw: d.g.edit(d.code_node, code="print(2)", **kw),
    "delete": lambda d, **kw: d.g.delete(d.code_node, **kw),
    "remove": lambda d, **kw: d.g.remove(d.worker, **kw),
    "replace_value": lambda d, **kw: d.g.replace(
        d.code_node, call(d.code_node, calls.AWAIT), 5, **kw
    ),
    "replace_agent": lambda d, **kw: d.g.replace(
        d.code_node, call(d.code_node, calls.LAUNCH), agent=d.other, **kw
    ),
}


def delegation_with_other() -> SimpleNamespace:
    d = delegation()
    d.other = d.g.add_agent("other", name="other")
    finish(d.g, d.other, 9)
    return d


@pytest.mark.parametrize("name", sorted(EDITS))
def test_dry_run_computes_the_invalidation_and_leaves_the_graph_untouched(name):
    d = delegation_with_other()
    g = d.g
    before = records(g)
    versions = {node.id: node.version for node in g.nodes()}

    planned = EDITS[name](d, dry_run=True)

    assert isinstance(planned, Invalidation) and planned
    assert records(g) == before
    assert {node.id: node.version for node in g.nodes()} == versions
    assert d.code_node.state == DONE and own(d.root)[-1] is d.final

    applied = EDITS[name](d)
    assert applied == planned
    assert records(g) != before


def test_an_edited_graph_saves_and_loads_with_the_code_ready(tmp_path):
    d = delegation()
    g = d.g
    g.edit(d.done, result=2)
    g.replace(d.code_node, call(d.code_node, calls.AWAIT), 7)

    loaded = Graph.load(g.save(tmp_path / "run"))

    code_node = loaded.node(d.code_node.id)
    assert code_node.state == READY and code_node.version == d.code_node.version
    assert code_node.calls[call(code_node, calls.AWAIT)] == {
        "tool": "await",
        "key": "await-0",
        "value": 7,
    }
    assert loaded.status(loaded.agent(d.root.id)) == "running"


def test_a_code_cut_off_mid_run_keeps_what_it_launched_through_save_and_load(tmp_path):
    g, root = agent_graph("root")
    g.append(root, Node("assistant", "plan"))
    code_node = g.append(root, Node("assistant", code="launch and send", state=READY))
    child = launch(g, code_node, "child", name="child")
    peer = g.add_agent("peer", name="peer")
    g.send(peer, "for later", by=code_node)

    loaded = Graph.load(g.save(tmp_path / "run"))

    again = loaded.node(code_node.id)
    assert again.state == READY and loaded.status(loaded.agent(root.id)) == "running"
    assert loaded.parent(loaded.agent(child.id)).id == root.id
    assert calls.launched(again) == [child.id]
    assert [m.text for m in own(loaded.agent(peer.id))] == ["for later"]


def test_load_refuses_a_dangling_edge(tmp_path):
    d = delegation()
    run = d.g.save(tmp_path / "run")
    document = json.loads((run / "graph.json").read_text())
    [edge] = [e for e in document["edges"] if e["kind"] == RETURN]
    edge["source"] = "node_missing"
    (run / "graph.json").write_text(json.dumps(document))

    with pytest.raises(GraphError, match="points at missing"):
        Graph.load(run)


# -- edits on a live run --------------------------------------------------------------


def test_flow_edit_cancels_the_affected_agents_task_before_applying():
    d = delegation_with_other()
    g = d.g
    seen: dict[str, object] = {}

    async def turn(agent: Node, node: Node):
        node.state = RUNNING
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            seen[agent.id] = (node.id in g, node.result)
            raise

    async def scenario():
        flow = Flow()
        running = asyncio.ensure_future(turn(d.worker, d.done))
        unaffected = asyncio.ensure_future(turn(d.other, g.tail(d.other)))
        flow.tasks.update({d.worker.id: running, d.other.id: unaffected})
        await asyncio.sleep(0)

        inv = await flow.edit(g, "edit", d.done, result=2)

        assert running.cancelled() and d.worker.id not in flow.tasks
        assert not unaffected.done() and d.other.id in flow.tasks
        assert d.worker.id in inv.rewound and d.done.result == 2
        unaffected.cancel()

    asyncio.run(scenario())
    # The task stopped while the graph still held the old result.
    assert seen[d.worker.id] == (True, 1)


def test_flow_edit_refuses_an_unknown_op():
    d = delegation()
    with pytest.raises(ValueError, match="unknown edit"):
        asyncio.run(Flow().edit(d.g, "rewrite", d.plan))


# -- timeline -------------------------------------------------------------------------


def small_run() -> Graph:
    g, root = agent_graph("root")
    g.append(root, Node("assistant", "plan"))
    code_node = g.append(root, ran("launch"))
    child = launch(g, code_node, "work", name="worker")
    finish(g, child, 1)
    awaited(g, code_node, child)
    g.fork(code_node, name="retry")
    finish(g, root, 1)
    return g


def test_timeline_lists_agents_as_their_own_steps():
    g = small_run()

    ordered = steps(g)

    assert [step.kind for step in ordered] == [
        "agent",
        "reply",
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
    assert [len(s.nodes()) for s in snaps] == list(range(1, total + 1))
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
    assert own(cut.agent(worker.id)) == [] and cut.parent(cut.agent(worker.id)).id == root.id
    assert cut.agent(root.id).root_data.config == root.root_data.config
    assert cut is not g and len(g.agents()) == 3


def test_replay_works_on_agents_no_flow_has_stepped():
    g = Graph()
    root = g.add_agent("root")
    g.append(root, Node("assistant", "plan"))

    assert [step.kind for step in steps(g)] == ["agent", "reply"]
    assert [len(s.nodes()) for s in replay(g)] == [1, 2]
