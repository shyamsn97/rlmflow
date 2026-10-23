"""The graph store, its invariants, and the pure ``reads`` over it."""

from __future__ import annotations

import pytest

from rlmflow.graph import ops, reads
from rlmflow.graph.config import AgentConfig, ContextPolicy, Input
from rlmflow.graph.graph import (
    DEPENDS_ON,
    FROM,
    NEXT,
    Edge,
    Graph,
    InvariantError,
    StaleFrontier,
    check_invariants,
)
from rlmflow.graph.nodes import (
    AgentStart,
    AppendChild,
    DoneOutput,
    ExecAction,
    ExecOutput,
    LLMOutput,
    UserQuery,
)


def hand_written(*items) -> Graph:
    """A graph holding exactly ``items``, as a corrupted saved log would."""
    g = Graph()
    g._append_unchecked(*items)
    return g


def test_writes_assign_consecutive_seq_and_log_in_order():
    g = Graph()
    a = ops.spawn(g, "task")
    llm = ops.append(g, a, LLMOutput(content="hi"))

    items = g.items()
    assert [item.seq for item in items] == [0, 1, 2]
    assert items[0] is a and items[1] is llm
    assert isinstance(items[2], Edge) and items[2].kind == NEXT
    assert g.items(after=1) == items[2:]


def test_a_failed_write_leaves_the_graph_unchanged():
    g = Graph()
    a = ops.spawn(g, "task")
    stranger = AgentStart(content="not in g")
    with pytest.raises(InvariantError, match="not in the graph"):
        ops.send(g, a, "hi", sender=stranger)
    assert g.items() == [a] and reads.inbox(g, a) == []


def test_ids_are_unique():
    g = Graph()
    a = ops.spawn(g, "task")
    node = ops.append(g, a, LLMOutput())
    with pytest.raises(InvariantError, match="duplicate node"):
        ops.append(g, a, node)
    edge = g.add_edge(Edge(kind="peer", source=a.id, target=a.id))
    with pytest.raises(InvariantError, match="duplicate edge"):
        g.add_edge(Edge(kind="peer", source=a.id, target=a.id, id=edge.id))


def test_append_must_follow_the_agents_last_node():
    g = Graph()
    a = ops.spawn(g, "task")
    first = ops.append(g, a, LLMOutput(content="one"))
    ops.append(g, a, ExecAction(code=""))
    with pytest.raises(StaleFrontier):
        ops.append(g, a, ExecOutput(), after=first)
    assert reads.frontier(g, a) is not first


def test_application_edges_need_known_endpoints_and_a_non_structural_kind():
    g = Graph()
    a = ops.spawn(g, "task")
    with pytest.raises(InvariantError, match="not in the graph"):
        g.add_edge(Edge(kind="peer", source=a.id, target="node_missing"))
    with pytest.raises(InvariantError, match="written by rlmflow.graph.ops"):
        g.add_edge(Edge(kind=DEPENDS_ON, source=a.id, target=a.id))
    assert g.items() == [a]


def test_undelivered_user_query_waits_in_the_inbox():
    g = Graph()
    a = ops.spawn(g, "task")
    done = ops.append(g, a, DoneOutput(result=1))
    message = ops.send(g, a, "again")

    assert reads.inbox(g, a) == [message]
    assert reads.frontier(g, a) is done
    assert reads.ready(g) == [a]

    assert ops.deliver(g, a) == [message]
    assert reads.inbox(g, a) == []
    assert reads.frontier(g, a) is message


def test_attach_refuses_an_agent_with_a_launcher_or_the_parents_ancestor():
    g = Graph()
    parent = ops.spawn(g, "parent")
    action = ops.append(g, parent, ExecAction(code="launch"))
    child = ops.spawn(g, "child", by=action)
    other = ops.spawn(g, "other")

    with pytest.raises(ValueError, match="already has a launcher"):
        ops.attach(g, other, child)
    with pytest.raises(InvariantError, match="cannot launch itself"):
        ops.attach(g, child, parent)
    with pytest.raises(InvariantError, match="cannot launch itself"):
        ops.attach(g, parent, parent)

    assert reads.launcher(g, child) is parent
    assert reads.children(g, parent) == [child]
    assert reads.depth(g, child) == 1


def test_spawn_needs_producers_in_the_graph():
    g = Graph()
    with pytest.raises(InvariantError, match="not in the graph"):
        ops.spawn(g, "consumer", depends_on=[AgentStart(content="missing")])
    producer = ops.spawn(g, "producer")
    llm = ops.append(g, producer, LLMOutput())
    with pytest.raises(InvariantError, match="expected an AgentStart"):
        ops.spawn(g, "consumer", inputs={"data": llm})


def test_depends_on_blocks_ready_until_the_producer_is_terminal():
    g = Graph()
    producer = ops.spawn(g, "p")
    consumer = ops.spawn(g, "c", inputs={"data": producer})
    assert reads.ready(g) == [producer]

    ops.append(g, producer, DoneOutput(result={"n": 1}))
    assert reads.ready(g) == [consumer]
    assert reads.inputs(g, consumer) == {"data": {"n": 1}}


def test_structural_edges_are_never_removed_and_removal_is_logged():
    g = Graph()
    a, b = ops.spawn(g, "a"), ops.spawn(g, "b")
    node = ops.append(g, a, LLMOutput())
    with pytest.raises(InvariantError, match="cannot be removed"):
        g.remove_edge(g.edges(kind=NEXT, target=node.id)[0])

    peer = g.add_edge(Edge(kind="peer", source=a.id, target=b.id))
    removed = g.remove_edge(peer)
    assert removed.removed_seq == len(g.items()) - 1
    assert g.items()[-1] is removed and not removed.active
    assert g.edges(kind="peer") == []
    assert g.edges(kind="peer", active=False) == [removed]
    with pytest.raises(InvariantError, match="already removed"):
        g.remove_edge(peer.id)


def test_fork_history_shares_the_prefix_and_an_empty_fork_stands_at_its_base():
    g = Graph()
    base = ops.spawn(g, "task")
    llm = ops.append(g, base, LLMOutput(content="plan"))
    ops.append(g, base, ExecAction(code="x"))
    ops.append(g, base, ExecOutput(content="out"))

    fork = ops.fork(g, llm)
    assert reads.frontier(g, fork) is llm
    assert reads.history(g, fork) == [base, llm, fork]
    assert not reads.terminal(g, fork)

    output = ops.append(g, fork, ExecOutput(content="y"))
    assert reads.frontier(g, fork) is output
    assert reads.history(g, fork) == [base, llm, fork, output]
    assert [type(node) for node in reads.history(g, base)] == [
        AgentStart,
        LLMOutput,
        ExecAction,
        ExecOutput,
    ]

    nested = ops.fork(g, output)
    assert reads.history(g, nested) == [base, llm, fork, output, nested]


def test_fork_needs_a_landed_node():
    g = Graph()
    base = ops.spawn(g, "task")
    queued = ops.send(g, base, "later")
    with pytest.raises(InvariantError, match="undelivered"):
        ops.fork(g, queued)
    assert g.nodes(AgentStart) == [base]


def test_paths_derive_from_launchers():
    g = Graph()
    root = ops.spawn(g, "root")
    action = ops.append(g, root, ExecAction(code=""))
    child = ops.spawn(g, "child", by=action, name="worker")

    assert reads.path(g, root) == "root"
    assert reads.path(g, child) == "root.worker"
    assert reads.lineage(g, child) == [root, child]


def test_merge_moves_another_graph_in_log_order_and_refuses_shared_ids():
    side = Graph()
    a, b = ops.spawn(side, "a"), ops.spawn(side, "b")
    ops.append(side, a, DoneOutput(result=1))
    side.remove_edge(side.add_edge(Edge(kind="peer", source=a.id, target=b.id)))

    g = Graph()
    root = ops.spawn(g, "root")
    g.merge(side)
    assert [
        item.removed_seq if isinstance(item, Edge) and not item.active else item.seq
        for item in g.items()
    ] == list(range(len(g.items())))
    assert g.node(a.id) is a and reads.result(g, a) == 1
    assert g.edges(kind="peer") == [] and len(g.edges(kind="peer", active=False)) == 1
    check_invariants(g)
    with pytest.raises(InvariantError, match="share ids"):
        g.merge(side)
    assert reads.frontier(g, root) is root


def test_as_of_copies_the_graph_as_it_stood_at_a_log_position():
    g = Graph()
    a = ops.spawn(g, "task")
    llm = ops.append(g, a, LLMOutput(content="hi"))
    peer = g.add_edge(Edge(kind="peer", source=a.id, target=llm.id))
    g.remove_edge(peer)
    ops.append(g, a, DoneOutput(result=1))

    past = g.as_of(peer.seq)
    assert [node.id for node in reads.history(past, past.node(a.id))] == [a.id, llm.id]
    assert past.node(llm.id) is not llm
    assert len(past.edges(kind="peer")) == 1
    assert past.as_of(-1).items() == []
    assert g.as_of(len(g) - 1).edges(kind="peer") == []


# -- check_invariants: the whole-graph walk run on load -------------------------


def test_check_invariants_accepts_every_graph_the_writes_produce():
    g = Graph()
    root = ops.spawn(g, "root")
    action = ops.append(g, root, ExecAction(code="launch"))
    child = ops.spawn(g, "child", by=action)
    ops.spawn(g, "consumer", depends_on=[child])
    ops.fork(g, ops.append(g, root, ExecOutput(content="launched")))
    ops.send(g, root, "queued")
    check_invariants(g)


def test_check_invariants_needs_each_node_reached_by_next():
    a = AgentStart()
    orphan = ExecOutput(agent_id=a.id)
    with pytest.raises(InvariantError, match="no incoming next"):
        check_invariants(hand_written(a, orphan))


def test_check_invariants_needs_a_known_owner_and_one_predecessor():
    with pytest.raises(InvariantError, match="unknown agent"):
        check_invariants(hand_written(UserQuery(content="lost", agent_id="agent_missing")))
    a = AgentStart()
    llm = LLMOutput(agent_id=a.id)
    twice = hand_written(
        a,
        llm,
        Edge(kind=NEXT, source=a.id, target=llm.id),
        Edge(kind=NEXT, source=a.id, target=llm.id),
    )
    with pytest.raises(InvariantError, match="already has a predecessor"):
        check_invariants(twice)


def test_check_invariants_needs_one_history_chain_per_agent():
    a = AgentStart()
    one, two = LLMOutput(agent_id=a.id), LLMOutput(agent_id=a.id)
    branched = hand_written(
        a,
        one,
        two,
        Edge(kind=NEXT, source=a.id, target=one.id),
        Edge(kind=NEXT, source=a.id, target=two.id),
    )
    with pytest.raises(InvariantError, match="two successors"):
        check_invariants(branched)
    b = AgentStart()
    looped_a, looped_b = LLMOutput(agent_id=b.id), ExecOutput(agent_id=b.id)
    loop = hand_written(
        b,
        looped_a,
        looped_b,
        Edge(kind=NEXT, source=looped_a.id, target=looped_b.id),
        Edge(kind=NEXT, source=looped_b.id, target=looped_a.id),
    )
    with pytest.raises(InvariantError, match="not in its agent's history"):
        check_invariants(loop)


def test_check_invariants_needs_an_acyclic_launcher_tree():
    parent, child = AgentStart(), AgentStart()
    action, child_action = ExecAction(agent_id=parent.id), ExecAction(agent_id=child.id)
    history = [
        parent,
        child,
        action,
        child_action,
        Edge(kind=NEXT, source=parent.id, target=action.id),
        Edge(kind=NEXT, source=child.id, target=child_action.id),
        Edge(kind=FROM, source=child.id, target=action.id),
    ]
    with pytest.raises(InvariantError, match="cannot launch itself"):
        check_invariants(
            hand_written(*history, Edge(kind=FROM, source=parent.id, target=child_action.id))
        )
    with pytest.raises(InvariantError, match="already has a launcher"):
        check_invariants(
            hand_written(*history, Edge(kind=FROM, source=child.id, target=child_action.id))
        )
    with pytest.raises(InvariantError, match="AgentStart or UserQuery"):
        check_invariants(
            hand_written(*history, Edge(kind=FROM, source=action.id, target=child_action.id))
        )


def test_check_invariants_needs_acyclic_depends_on_declared_before_the_first_step():
    a, b = AgentStart(), AgentStart()
    cycle = hand_written(
        a,
        b,
        Edge(kind=DEPENDS_ON, source=a.id, target=b.id),
        Edge(kind=DEPENDS_ON, source=b.id, target=a.id),
    )
    with pytest.raises(InvariantError, match="cycle"):
        check_invariants(cycle)
    llm = LLMOutput(agent_id=a.id)
    late = hand_written(
        a,
        b,
        llm,
        Edge(kind=NEXT, source=a.id, target=llm.id),
        Edge(kind=DEPENDS_ON, source=a.id, target=b.id),
    )
    with pytest.raises(InvariantError, match="already started"):
        check_invariants(late)
    wrong = hand_written(
        a,
        llm,
        Edge(kind=NEXT, source=a.id, target=llm.id),
        Edge(kind=DEPENDS_ON, source=a.id, target=llm.id),
    )
    with pytest.raises(InvariantError, match="connect AgentStart"):
        check_invariants(wrong)


def test_check_invariants_needs_forks_at_landed_nodes():
    a = AgentStart()
    queued = UserQuery(content="later", agent_id=a.id)
    fork = AgentStart()
    with pytest.raises(InvariantError, match="undelivered"):
        check_invariants(
            hand_written(a, queued, fork, Edge(kind=NEXT, source=queued.id, target=fork.id))
        )


# -- ops ----------------------------------------------------------------------


def written(g: Graph, before: int) -> list:
    return g.items(after=before - 1)


def test_spawn_writes_the_agent_and_its_edges():
    g = Graph()
    root = ops.spawn(g, "root task")
    action = ops.append(g, root, ExecAction(code="launch"))
    source = ops.spawn(g, "source", by=action, name="source")
    before = len(g)
    consumer = ops.spawn(
        g,
        "consume",
        by=action,
        name="consumer",
        depends_on=[root],
        inputs={"data": Input(source, description="Parsed rows.")},
        limit=3,
    )

    items = written(g, before)
    assert items[0] is consumer and consumer.args == {"limit": 3}
    assert [(e.kind, e.target, dict(e.data)) for e in items[1:]] == [
        (FROM, action.id, {}),
        (DEPENDS_ON, root.id, {}),
        (DEPENDS_ON, source.id, {"as": "data", "description": "Parsed rows."}),
    ]
    assert reads.path(g, consumer) == "root.consumer"
    assert reads.producers(g, consumer) == [root, source]


def test_append_guard_rejects_a_stale_frontier():
    g = Graph()
    agent = ops.spawn(g)
    first = ops.append(g, agent, LLMOutput(content="one"), after=agent)
    with pytest.raises(StaleFrontier):
        ops.append(g, agent, LLMOutput(content="stale"), after=agent)
    ops.append(g, agent, ExecAction(code=""), after=first)


def test_send_queues_and_deliver_links_in_order_at_a_boundary():
    g = Graph()
    agent = ops.spawn(g)
    llm = ops.append(g, agent, LLMOutput(content="code"))
    before = len(g)
    first = ops.send(g, agent, "one", sender=llm, kind="note")
    second = ops.send(g, agent, "two")
    assert written(g, before)[:2] == [first, g.edges(kind=FROM, source=first.id)[0]]
    assert first.metadata == {"kind": "note"}
    assert reads.inbox(g, agent) == [first, second]

    assert ops.deliver(g, agent) == []  # mid-turn: the LLMOutput's code must run first
    ops.append(g, agent, ExecAction(code=""))
    out = ops.append(g, agent, ExecOutput(content="ran"))
    assert ops.deliver(g, agent) == [first, second]
    assert reads.history(g, agent)[-3:] == [out, first, second]
    assert reads.inbox(g, agent) == []


def test_fork_writes_two_or_three_items_and_shares_history():
    g = Graph()
    base = ops.spawn(g, "task", config=AgentConfig(model="m"))
    llm = ops.append(g, base, LLMOutput(content="plan", code="x = 1"))
    action = ops.append(g, base, ExecAction(code="x = 1"))
    ops.append(g, base, ExecOutput(content="ran"))
    before = len(g)

    fork = ops.fork(g, llm, name="retry", model="other")
    items = written(g, before)
    assert items[0] is fork and [e.kind for e in items[1:]] == [NEXT]
    assert (fork.config.name, fork.config.model, base.config.model) == ("retry", "other", "m")
    assert reads.frontier(g, fork) is llm
    assert reads.history(g, fork)[:2] == [base, llm]
    assert reads.frontier(g, base) is not llm

    with pytest.raises(ValueError, match="cannot fork at exec_action"):
        ops.fork(g, action)

    before = len(g)
    judged = ops.fork(g, llm, by=action)
    assert len(written(g, before)) == 3
    assert reads.launcher(g, judged) is base

    fresh = ops.append(g, fork, ExecAction(code="x = 2"), after=llm)
    assert reads.history(g, fork) == [base, llm, fork, fresh]


def test_fork_at_done_is_terminal_until_a_message_reopens_it():
    g = Graph()
    base = ops.spawn(g)
    done = ops.append(g, base, DoneOutput(result=1))
    fork = ops.fork(g, done)
    assert reads.terminal(g, fork) and fork not in reads.ready(g)

    query = ops.send(g, fork, "Try again.")
    assert fork in reads.ready(g)
    assert ops.deliver(g, fork) == [query]
    assert reads.history(g, fork) == [base, done, fork, query]
    assert reads.terminal(g, base)


def test_attach_writes_one_action_and_from_edges():
    g = Graph()
    judge = ops.spawn(g, "judge")
    done_branch = ops.spawn(g, "a", name="branch_a")
    ops.append(g, done_branch, DoneOutput(result="A"))
    open_branch = ops.spawn(g, "b")
    before = len(g)

    action = ops.attach(g, judge, done_branch, open_branch)
    items = written(g, before)
    assert items[0] is action and isinstance(action, AppendChild)
    assert [e.kind for e in items[1:]] == [NEXT, FROM, FROM]
    assert action.code == (
        "_handles = [\n    launch_agent(task=''),\n    launch_agent(task=''),\n]\n"
        "print(await asyncio.gather(*_handles))"
    )
    assert action.tool_calls("launch_agent") == {
        0: {"$rlmflow_agent": done_branch.id},
        1: {"$rlmflow_agent": open_branch.id},
    }
    assert reads.children(g, judge) == [done_branch, open_branch]
    assert reads.path(g, open_branch) == "root.child1"
    assert reads.frontier(g, judge) is action
    with pytest.raises(ValueError, match="already has a launcher"):
        ops.attach(g, judge, open_branch)


def test_import_a_saved_graph_and_attach_its_top_level_agents():
    other = Graph()
    sub = ops.spawn(other, "prepared", name="prepared")
    ops.append(other, sub, DoneOutput(result="ready"))

    g = Graph()
    parent = ops.spawn(g, "parent")
    g.merge(other)
    ops.attach(g, parent, *[a for a in other.nodes(AgentStart) if reads.launcher(other, a) is None])
    assert reads.result(g, g.node(sub.id)) == "ready"
    assert reads.path(g, sub) == "root.prepared"


def test_context_policy_is_part_of_the_config():
    assert AgentConfig().context_policy is ContextPolicy.RESIDENT
    assert AgentConfig(context_policy="ephemeral").context_policy is ContextPolicy.EPHEMERAL


# -- persistence ----------------------------------------------------------------

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from rlmflow.graph import persistence  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RUNS = Path(__file__).resolve().parents[1] / "examples" / "_runs"


def test_load_rejects_a_log_that_breaks_an_invariant(tmp_path):
    g = Graph()
    root = ops.spawn(g, "task")
    llm = ops.append(g, root, LLMOutput(content="hi"))
    run = g.save(tmp_path / "run")
    data = json.loads((run / "graph.json").read_text())
    data["items"] = [item for item in data["items"] if item.get("target") != llm.id]
    for position, item in enumerate(data["items"]):
        item["seq"] = position
    (run / "graph.json").write_text(json.dumps(data))

    with pytest.raises(InvariantError, match="invalid graph log: .*no incoming next"):
        Graph.load(run)


def test_save_and_load_round_trip_the_log_with_removed_edges(tmp_path):
    g = Graph()
    root = ops.spawn(g, "task", doc="x" * 70_000, config=AgentConfig(context_policy="ephemeral"))
    peer = ops.spawn(g, "peer", name="peer")
    edge = g.add_edge(Edge(kind="peer", source=root.id, target=peer.id, data={"w": 1}))
    ops.append(g, root, LLMOutput(content="hi", code="print(1)"))
    g.remove_edge(edge)
    ops.send(g, peer, "queued")

    run = g.save(tmp_path / "run")
    assert sorted(path.name for path in run.iterdir()) == ["blobs", "graph.json"]
    loaded = Graph.load(run)

    assert loaded.id == g.id
    assert [
        item.removed_seq if isinstance(item, Edge) and not item.active else item.seq
        for item in loaded.items()
    ] == list(range(len(g.items())))
    assert [type(item).__name__ for item in loaded.items()] == [
        type(item).__name__ for item in g.items()
    ]
    loaded_root = loaded.node(root.id)
    assert loaded_root.args == {"doc": "x" * 70_000}
    assert loaded_root.config.context_policy is ContextPolicy.EPHEMERAL
    assert loaded.edges(kind="peer") == []
    assert loaded.edges(kind="peer", active=False)[0].data == {"w": 1}
    assert [q.content for q in reads.inbox(loaded, loaded.node(peer.id))] == ["queued"]
    assert reads.frontier(loaded, loaded_root).content == "hi"


def test_load_enforces_the_invariants(tmp_path):
    g = Graph()
    agent = ops.spawn(g)
    ops.append(g, agent, LLMOutput(content="one"))
    run = g.save(tmp_path / "run")
    data = json.loads((run / "graph.json").read_text())
    data["items"] = [item for item in data["items"] if item.get("kind") != NEXT]
    for position, item in enumerate(data["items"]):
        item["seq"] = position
    (run / "graph.json").write_text(json.dumps(data))

    with pytest.raises(ValueError, match="no incoming next"):
        Graph.load(run)


@pytest.mark.parametrize(
    "run",
    [FIXTURES / "v1_tree_run", RUNS / "needle-filesystem", RUNS / "shared-board-audit" / "peer-1"],
)
def test_upgrade_v1_converts_a_saved_tree_run(run):
    graph = persistence.upgrade_v1(run)
    agents = graph.nodes(AgentStart)
    top = [agent for agent in agents if reads.launcher(graph, agent) is None]

    assert len(top) == 1
    root = top[0]
    old = json.loads((run / "graph.json").read_text())
    assert len(graph.nodes()) == old["node_count"]
    assert {reads.path(graph, agent) for agent in agents} == {
        record["agent_id"] for record in old["nodes"] if record["type"] == "agent_start"
    }
    assert reads.terminal(graph, root) and Graph.load(run).id == graph.id
    for agent in agents:
        expected = [r["id"] for r in old["nodes"] if r["agent_id"] == reads.path(graph, agent)]
        assert [node.id for node in reads.history(graph, agent)] == expected


def test_upgrade_v1_turns_handle_arguments_into_depends_on_edges():
    graph = persistence.upgrade_v1(FIXTURES / "v1_tree_run")
    by_path = {reads.path(graph, agent): agent for agent in graph.nodes(AgentStart)}
    root, producer, summer = by_path["root"], by_path["root.producer"], by_path["root.summer"]

    assert summer.args == {}
    assert [(e.target, dict(e.data)) for e in graph.edges(kind=DEPENDS_ON)] == [
        (producer.id, {"as": "rows", "description": "Rows to add."})
    ]
    assert reads.inputs(graph, summer) == {"rows": [1, 2, 3]}
    assert reads.children(graph, root) == [producer, summer]
    assert reads.result(graph, root) == "total=6"
