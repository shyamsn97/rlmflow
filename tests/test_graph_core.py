"""The graph store, its invariants, and its agent queries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlmflow.graph import persistence
from rlmflow.graph.config import AgentConfig, Input
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
from rlmflow.graph.nodes import AGENT_START, CODE, DONE, LLM_OUTPUT, MESSAGE, Node

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def agent(**data) -> Node:
    """A bare ``agent_start``, as a hand-written log would hold."""
    return Node(AGENT_START, task="", name="a", args={}, **data)


def owned(node_type: str, owner: Node, **data) -> Node:
    node = Node(node_type, **data)
    node.agent_id = owner.id
    return node


def hand_written(*items) -> Graph:
    """A graph holding exactly ``items``, as a corrupted saved log would."""
    g = Graph()
    g._append_unchecked(*items)
    return g


def test_writes_assign_consecutive_seq_and_log_in_order():
    g = Graph()
    a = g.spawn("task")
    llm = g.append(a, Node(LLM_OUTPUT, text="hi"))

    items = g.items()
    assert [item.seq for item in items] == [0, 1, 2]
    assert items[0] is a and items[1] is llm
    assert isinstance(items[2], Edge) and items[2].kind == NEXT
    assert g.items(after=1) == items[2:]


def test_a_failed_write_leaves_the_graph_unchanged():
    g = Graph()
    a = g.spawn("task")
    stranger = agent()
    with pytest.raises(InvariantError, match="not in the graph"):
        g.send(a, "hi", sender=stranger)
    assert g.items() == [a] and g.inbox(a) == []


def test_ids_are_unique():
    g = Graph()
    a = g.spawn("task")
    node = g.append(a, Node(LLM_OUTPUT))
    with pytest.raises(InvariantError, match="duplicate node"):
        g.append(a, node)
    edge = g.add_edge(Edge(kind="peer", source=a.id, target=a.id))
    with pytest.raises(InvariantError, match="duplicate edge"):
        g.add_edge(Edge(kind="peer", source=a.id, target=a.id, id=edge.id))


def test_append_refuses_agent_start_and_message_nodes():
    g = Graph()
    a = g.spawn("task")
    with pytest.raises(InvariantError, match="spawn, fork or send"):
        g.append(a, Node(MESSAGE, text="hi"))


def test_append_must_follow_the_agents_last_node():
    g = Graph()
    a = g.spawn("task")
    first = g.append(a, Node(LLM_OUTPUT, text="one"))
    g.append(a, Node(CODE, code=""))
    with pytest.raises(StaleFrontier):
        g.append(a, Node(CODE), after=first)
    assert g.frontier(a) is not first


def test_application_edges_need_known_endpoints_and_a_non_structural_kind():
    g = Graph()
    a = g.spawn("task")
    with pytest.raises(InvariantError, match="not in the graph"):
        g.add_edge(Edge(kind="peer", source=a.id, target="node_missing"))
    with pytest.raises(InvariantError, match="written by the graph.s history writes"):
        g.add_edge(Edge(kind=DEPENDS_ON, source=a.id, target=a.id))
    assert g.items() == [a]


def test_undelivered_message_waits_in_the_inbox():
    g = Graph()
    a = g.spawn("task")
    done = g.append(a, Node(DONE, result=1))
    message = g.send(a, "again")

    assert g.inbox(a) == [message]
    assert g.frontier(a) is done
    assert g.ready() == [a]

    assert g.deliver(a) == [message]
    assert g.inbox(a) == []
    assert g.frontier(a) is message


def test_status_is_done_waiting_or_running():
    g = Graph()
    parent = g.spawn("parent")
    child = g.spawn("child", by=g.append(parent, Node(CODE, code="")))
    assert g.status(parent) == "running"
    g.append(parent, Node("waiting", on=[child.id]))
    assert g.status(parent) == "waiting" and parent not in g.ready()
    g.send(parent, "nudge")
    assert g.status(parent) == "running" and parent in g.ready()
    g.append(child, Node(DONE, result=1))
    assert g.status(child) == "done"


def test_attach_refuses_an_agent_with_a_launcher_or_the_parents_ancestor():
    g = Graph()
    parent = g.spawn("parent")
    cell = g.append(parent, Node(CODE, code="launch"))
    child = g.spawn("child", by=cell)
    other = g.spawn("other")

    with pytest.raises(ValueError, match="already has a launcher"):
        g.attach(other, child)
    with pytest.raises(InvariantError, match="cannot launch itself"):
        g.attach(child, parent)
    with pytest.raises(InvariantError, match="cannot launch itself"):
        g.attach(parent, parent)

    assert g.launcher(child) is parent
    assert g.children(parent) == [child]
    assert g.depth(child) == 1


def test_spawn_needs_producers_in_the_graph():
    g = Graph()
    with pytest.raises(InvariantError, match="not in the graph"):
        g.spawn("consumer", depends_on=[agent()])
    producer = g.spawn("producer")
    llm = g.append(producer, Node(LLM_OUTPUT))
    with pytest.raises(InvariantError, match="expected an agent_start"):
        g.spawn("consumer", inputs={"data": llm})


def test_depends_on_blocks_ready_until_the_producer_is_terminal():
    g = Graph()
    producer = g.spawn("p")
    consumer = g.spawn("c", inputs={"data": producer})
    assert g.ready() == [producer]

    g.append(producer, Node(DONE, result={"n": 1}))
    assert g.ready() == [consumer]
    assert g.inputs(consumer) == {"data": {"n": 1}}


def test_structural_edges_are_never_removed_and_removal_is_logged():
    g = Graph()
    a, b = g.spawn("a"), g.spawn("b")
    node = g.append(a, Node(LLM_OUTPUT))
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
    base = g.spawn("task")
    llm = g.append(base, Node(LLM_OUTPUT, text="plan"))
    g.append(base, Node(CODE, code="x", output="out"))

    fork = g.fork(llm)
    assert g.frontier(fork) is llm
    assert g.history(fork) == [base, llm, fork]
    assert not g.terminal(fork)

    cell = g.append(fork, Node(CODE, code="y"))
    assert g.frontier(fork) is cell
    assert g.history(fork) == [base, llm, fork, cell]
    assert g.own_history(fork) == [fork, cell]
    assert [node.type for node in g.history(base)] == [AGENT_START, LLM_OUTPUT, CODE]

    nested = g.fork(cell)
    assert g.history(nested) == [base, llm, fork, cell, nested]


def test_fork_needs_a_landed_node():
    g = Graph()
    base = g.spawn("task")
    queued = g.send(base, "later")
    with pytest.raises(InvariantError, match="undelivered"):
        g.fork(queued)
    assert g.agents() == [base]


def test_names_and_paths_derive_from_configs_and_launchers():
    g = Graph()
    root = g.spawn("root")
    cell = g.append(root, Node(CODE, code=""))
    first = g.spawn("a", by=cell, config="worker")
    second = g.spawn("b", by=cell, config="worker")
    named = g.spawn("c", by=cell, name="checker")

    assert [a.name for a in (root, first, second, named)] == [
        "root",
        "worker-0",
        "worker-1",
        "checker",
    ]
    assert g.path(second) == "root.worker-1"
    assert g.lineage(second) == [root, second]
    assert (root.config_name, second.config_name, named.config_name) == (None, "worker", "default")
    assert g.spawn("again").name == "root-0"
    top = g.spawn("solo", config="worker")
    assert (top.name, top.config_name, g.launcher(top)) == ("worker", "worker", None)
    assert g.spawn("more", config="worker").name == "worker-0"


def test_create_agent_makes_a_root_whose_keywords_are_repl_args():
    g = Graph()
    lead = g.create_agent("x", name="lead", limit=3)
    assert (lead.name, lead.config_name, lead.args) == ("lead", None, {"limit": 3})
    with pytest.raises(AttributeError, match="until a Flow steps it"):
        lead.config
    lead.bind_config(AgentConfig(model="big"))
    assert lead.config.model == "big" and lead.metadata["config"]["model"] == "big"
    assert g.create_agent("y").name == "root"


def test_merge_moves_another_graph_in_log_order_and_refuses_shared_ids():
    side = Graph()
    a, b = side.spawn("a"), side.spawn("b")
    side.append(a, Node(DONE, result=1))
    side.remove_edge(side.add_edge(Edge(kind="peer", source=a.id, target=b.id)))

    g = Graph()
    root = g.spawn("root")
    g.merge(side)
    assert [
        item.removed_seq if isinstance(item, Edge) and not item.active else item.seq
        for item in g.items()
    ] == list(range(len(g.items())))
    assert g.node(a.id) is a and g.result(a) == 1
    assert g.edges(kind="peer") == [] and len(g.edges(kind="peer", active=False)) == 1
    check_invariants(g)
    with pytest.raises(InvariantError, match="share ids"):
        g.merge(side)
    assert g.frontier(root) is root


def test_as_of_copies_the_graph_as_it_stood_at_a_log_position():
    g = Graph()
    a = g.spawn("task")
    llm = g.append(a, Node(LLM_OUTPUT, text="hi"))
    peer = g.add_edge(Edge(kind="peer", source=a.id, target=llm.id))
    g.remove_edge(peer)
    g.append(a, Node(DONE, result=1))

    past = g.as_of(peer.seq)
    assert [node.id for node in past.history(past.node(a.id))] == [a.id, llm.id]
    assert past.node(llm.id) is not llm
    assert len(past.edges(kind="peer")) == 1
    assert past.as_of(-1).items() == []
    assert g.as_of(len(g) - 1).edges(kind="peer") == []


def test_usage_sums_every_node_that_reports_it():
    g = Graph()
    a = g.spawn("task")
    g.append(a, Node(LLM_OUTPUT, text="", usage={"input_tokens": 3, "output_tokens": 2}))
    g.append(a, Node(DONE, result=1, usage={"input_tokens": 10, "output_tokens": 1}))
    assert g.usage().total == 16


# -- check_invariants: the whole-graph walk run on load -------------------------


def test_check_invariants_accepts_every_graph_the_writes_produce():
    g = Graph()
    root = g.spawn("root")
    cell = g.append(root, Node(CODE, code="launch"))
    child = g.spawn("child", by=cell)
    g.spawn("consumer", depends_on=[child])
    g.fork(g.append(root, Node(LLM_OUTPUT, text="launched")))
    g.send(root, "queued")
    g.attach(root, g.spawn("orphan"))
    check_invariants(g)


def test_check_invariants_needs_each_node_reached_by_next():
    a = agent()
    orphan = owned(CODE, a)
    with pytest.raises(InvariantError, match="no incoming next"):
        check_invariants(hand_written(a, orphan))


def test_check_invariants_needs_a_known_owner_and_one_predecessor():
    lost = Node(MESSAGE, text="lost")
    lost.agent_id = "agent_missing"
    with pytest.raises(InvariantError, match="unknown agent"):
        check_invariants(hand_written(lost))
    a = agent()
    llm = owned(LLM_OUTPUT, a)
    twice = hand_written(
        a,
        llm,
        Edge(kind=NEXT, source=a.id, target=llm.id),
        Edge(kind=NEXT, source=a.id, target=llm.id),
    )
    with pytest.raises(InvariantError, match="already has a predecessor"):
        check_invariants(twice)


def test_check_invariants_needs_one_history_chain_per_agent():
    a = agent()
    one, two = owned(LLM_OUTPUT, a), owned(LLM_OUTPUT, a)
    branched = hand_written(
        a,
        one,
        two,
        Edge(kind=NEXT, source=a.id, target=one.id),
        Edge(kind=NEXT, source=a.id, target=two.id),
    )
    with pytest.raises(InvariantError, match="two successors"):
        check_invariants(branched)
    b = agent()
    looped_a, looped_b = owned(LLM_OUTPUT, b), owned(CODE, b)
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
    parent, child = agent(), agent()
    cell, child_cell = owned(CODE, parent), owned(CODE, child)
    history = [
        parent,
        child,
        cell,
        child_cell,
        Edge(kind=NEXT, source=parent.id, target=cell.id),
        Edge(kind=NEXT, source=child.id, target=child_cell.id),
        Edge(kind=FROM, source=child.id, target=cell.id),
    ]
    with pytest.raises(InvariantError, match="cannot launch itself"):
        check_invariants(
            hand_written(*history, Edge(kind=FROM, source=parent.id, target=child_cell.id))
        )
    with pytest.raises(InvariantError, match="already has a launcher"):
        check_invariants(
            hand_written(*history, Edge(kind=FROM, source=child.id, target=child_cell.id))
        )
    with pytest.raises(InvariantError, match="agent_start or message"):
        check_invariants(
            hand_written(*history, Edge(kind=FROM, source=cell.id, target=child_cell.id))
        )


def test_check_invariants_needs_acyclic_depends_on_declared_before_the_first_step():
    a, b = agent(), agent()
    cycle = hand_written(
        a,
        b,
        Edge(kind=DEPENDS_ON, source=a.id, target=b.id),
        Edge(kind=DEPENDS_ON, source=b.id, target=a.id),
    )
    with pytest.raises(InvariantError, match="cycle"):
        check_invariants(cycle)
    llm = owned(LLM_OUTPUT, a)
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
    with pytest.raises(InvariantError, match="connect agent_start"):
        check_invariants(wrong)


def test_check_invariants_needs_forks_at_landed_nodes():
    a = agent()
    queued = owned(MESSAGE, a, text="later")
    fork = agent()
    with pytest.raises(InvariantError, match="undelivered"):
        check_invariants(
            hand_written(a, queued, fork, Edge(kind=NEXT, source=queued.id, target=fork.id))
        )


# -- ops ----------------------------------------------------------------------


def written(g: Graph, before: int) -> list:
    return g.items(after=before - 1)


def test_spawn_writes_the_agent_and_its_edges():
    g = Graph()
    root = g.spawn("root task")
    cell = g.append(root, Node(CODE, code="launch"))
    source = g.spawn("source", by=cell, name="source")
    before = len(g)
    consumer = g.spawn(
        "consume",
        by=cell,
        name="consumer",
        depends_on=[root],
        inputs={"data": Input(source, description="Parsed rows.")},
        limit=3,
    )

    items = written(g, before)
    assert items[0] is consumer and consumer.args == {"limit": 3}
    assert [(e.kind, e.target, dict(e.data)) for e in items[1:]] == [
        (FROM, cell.id, {}),
        (DEPENDS_ON, root.id, {}),
        (DEPENDS_ON, source.id, {"as": "data", "description": "Parsed rows."}),
    ]
    assert g.path(consumer) == "root.consumer"
    assert g.producers(consumer) == [root, source]


def test_append_guard_rejects_a_stale_frontier():
    g = Graph()
    a = g.spawn()
    first = g.append(a, Node(LLM_OUTPUT, text="one"), after=a)
    with pytest.raises(StaleFrontier):
        g.append(a, Node(LLM_OUTPUT, text="stale"), after=a)
    g.append(a, Node(CODE, code=""), after=first)


def test_send_queues_and_deliver_links_in_order_at_a_boundary():
    g = Graph()
    a = g.spawn()
    llm = g.append(a, Node(LLM_OUTPUT, text="code"))
    before = len(g)
    first = g.send(a, "one", sender=llm, kind="note")
    second = g.send(a, "two")
    assert written(g, before)[:2] == [first, g.edges(kind=FROM, source=first.id)[0]]
    assert first.data["kind"] == "note" and first.data["sender"] == a.id
    assert second.data["sender"] == "user"
    assert g.inbox(a) == [first, second]

    assert g.deliver(a) == []  # mid-turn: the llm_output's code must run first
    out = g.append(a, Node(CODE, code="", output="ran"))
    assert g.deliver(a) == [first, second]
    assert g.history(a)[-3:] == [out, first, second]
    assert g.inbox(a) == []


def test_fork_writes_two_or_three_items_and_shares_history():
    g = Graph()
    base = g.spawn("task", config="m")
    llm = g.append(base, Node(LLM_OUTPUT, text="plan"))
    cell = g.append(base, Node(CODE, code="x = 1", output="ran"))
    before = len(g)

    fork = g.fork(llm, name="retry", config="other")
    items = written(g, before)
    assert items[0] is fork and [e.kind for e in items[1:]] == [NEXT]
    assert (fork.name, fork.config_name, base.config_name) == ("retry", "other", "m")
    assert g.frontier(fork) is llm
    assert g.history(fork)[:2] == [base, llm]
    assert g.frontier(base) is not llm
    second = g.fork(cell)
    assert (second.name, second.config_name) == ("m-fork1", "m")

    before = len(g)
    judged = g.fork(llm, by=cell)
    assert len(written(g, before)) == 3
    assert g.launcher(judged) is base

    fresh = g.append(fork, Node(CODE, code="x = 2"), after=llm)
    assert g.history(fork) == [base, llm, fork, fresh]


def test_fork_at_done_is_terminal_until_a_message_reopens_it():
    g = Graph()
    base = g.spawn()
    done = g.append(base, Node(DONE, result=1))
    fork = g.fork(done)
    assert g.terminal(fork) and fork not in g.ready()

    message = g.send(fork, "Try again.")
    assert fork in g.ready()
    assert g.deliver(fork) == [message]
    assert g.history(fork) == [base, done, fork, message]
    assert g.terminal(base)


def test_attach_writes_a_message_and_from_edges():
    g = Graph()
    judge = g.spawn("judge")
    done_branch = g.spawn("a", name="branch_a")
    g.append(done_branch, Node(DONE, result="A"))
    open_branch = g.spawn("b")
    before = len(g)

    message = g.attach(judge, done_branch, open_branch)
    items = written(g, before)
    assert items[0] is message and message.type == MESSAGE and message.data["sender"] == "rlmflow"
    assert [e.kind for e in items[1:]] == [FROM, FROM]
    assert "branch_a" in message.data["text"]
    assert g.children(judge) == [done_branch, open_branch]
    assert g.path(open_branch) == "root.root-0"
    assert g.inbox(judge) == [message]
    with pytest.raises(ValueError, match="already has a launcher"):
        g.attach(judge, open_branch)


def test_import_a_saved_graph_and_attach_its_top_level_agents():
    other = Graph()
    sub = other.spawn("prepared", name="prepared")
    other.append(sub, Node(DONE, result="ready"))

    g = Graph()
    parent = g.spawn("parent")
    g.merge(other)
    g.attach(parent, *[a for a in other.agents() if other.launcher(a) is None])
    assert g.result(g.node(sub.id)) == "ready"
    assert g.path(sub) == "root.prepared"


# -- persistence ----------------------------------------------------------------


def test_load_rejects_a_log_that_breaks_an_invariant(tmp_path):
    g = Graph()
    root = g.spawn("task")
    llm = g.append(root, Node(LLM_OUTPUT, text="hi"))
    run = g.save(tmp_path / "run")
    data = json.loads((run / "graph.json").read_text())
    data["items"] = [item for item in data["items"] if item.get("target") != llm.id]
    for position, item in enumerate(data["items"]):
        item["seq"] = position
    (run / "graph.json").write_text(json.dumps(data))

    with pytest.raises(InvariantError, match="invalid graph log: .*no incoming next"):
        Graph.load(run)


def test_save_and_load_round_trip_the_log_configs_and_blobs(tmp_path):
    g = Graph()
    root = g.create_agent("task", doc="x" * 70_000)
    root.bind_config(AgentConfig(max_budget=500))
    peer = g.spawn("peer", name="peer", config="worker")
    edge = g.add_edge(Edge(kind="peer", source=root.id, target=peer.id, data={"w": 1}))
    g.append(root, Node(LLM_OUTPUT, text="hi", usage={"input_tokens": 1}))
    g.remove_edge(edge)
    g.send(peer, "queued")

    run = g.save(tmp_path / "run")
    assert sorted(path.name for path in run.iterdir()) == ["blobs", "graph.json"]
    document = json.loads((run / "graph.json").read_text())
    assert document["format"] == persistence.FORMAT_VERSION and "configs" not in document
    loaded = Graph.load(run)

    assert loaded.id == g.id
    assert [
        item.removed_seq if isinstance(item, Edge) and not item.active else item.seq
        for item in loaded.items()
    ] == list(range(len(g.items())))
    assert [(type(item).__name__, getattr(item, "type", None)) for item in loaded.items()] == [
        (type(item).__name__, getattr(item, "type", None)) for item in g.items()
    ]
    loaded_root = loaded.node(root.id)
    assert loaded_root.args == {"doc": "x" * 70_000}
    assert loaded_root.config_name is None and loaded_root.config.max_budget == 500
    assert loaded.node(peer.id).config_name == "worker"
    with pytest.raises(AttributeError, match="until a Flow steps it"):
        loaded.node(peer.id).config
    assert loaded.edges(kind="peer") == []
    assert loaded.edges(kind="peer", active=False)[0].data == {"w": 1}
    assert [m.data["text"] for m in loaded.inbox(loaded.node(peer.id))] == ["queued"]
    assert loaded.frontier(loaded_root).data == {"text": "hi", "usage": {"input_tokens": 1}}


def test_load_converts_a_format_2_graph_that_saved_whole_configs(tmp_path):
    g = Graph()
    root = g.create_agent("task")
    cell = g.append(root, Node(CODE, code="launch"))
    child = g.spawn("child", by=cell, name="child")
    top = g.spawn("top", name="top")
    run = g.save(tmp_path / "run")
    saved = {
        root.id: AgentConfig(max_iters=4),
        child.id: AgentConfig(name="worker", model="fast"),
        top.id: AgentConfig(name="lead", max_budget=9),
    }
    document = json.loads((run / "graph.json").read_text())
    document["format"] = 2
    document["configs"] = {"worker": saved[child.id].to_dict(), "lead": saved[top.id].to_dict()}
    for item in document["items"]:
        if item.get("type") == AGENT_START:
            item["data"]["config"] = saved[item["id"]].to_dict()
    (run / "graph.json").write_text(json.dumps(document))

    loaded = Graph.load(run)

    for agent_id, config in saved.items():
        assert loaded.node(agent_id).metadata["config"] == config.to_dict()
        assert loaded.node(agent_id).config == config
    assert loaded.node(root.id).config_name is None
    assert loaded.node(child.id).config_name == "worker"
    assert loaded.node(top.id).config_name is None and "config" not in loaded.node(top.id).data
    assert loaded.path(loaded.node(child.id)) == "root.child"


def test_load_refuses_runs_saved_before_the_harness_refactor():
    with pytest.raises(ValueError, match="run format 1"):
        Graph.load(FIXTURES / "v1_tree_run")
