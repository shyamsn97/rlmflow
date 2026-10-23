"""A ``DoneOutput`` is delivered as a ``UserQuery`` to every agent it reports to."""

from __future__ import annotations

import asyncio

from helpers import StubLLM, agent_graph, next_node

from rlmflow import FROM, AgentConfig, Flow, Graph, ops, reads
from rlmflow.engine import handoff
from rlmflow.graph.graph import REPORTS_TO, Edge
from rlmflow.graph.nodes import (
    WAITED_AGENTS_KEY,
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    LLMOutput,
    ReplDead,
    UserQuery,
)

CHILD_GOAL = "Return alpha."
PARENT_QUERY = "Coordinate the child."


def user_text(messages):
    return "\n".join(message["content"] for message in messages if message["role"] == "user")


def handoff_queries(g: Graph, agent: AgentStart) -> list[UserQuery]:
    return [
        node
        for node in reads.history(g, agent)
        if type(node) is UserQuery and "[handoff " in node.content
    ]


def run_parent(flow: Flow, **config) -> tuple[Graph, AgentStart]:
    g = Graph()
    root = g.create_agent(PARENT_QUERY, config=AgentConfig(**config))
    asyncio.run(asyncio.wait_for(flow.arun(g), timeout=5))
    return g, root


def launch_print():
    return (
        f"```repl\nchild = launch_subagent(task={CHILD_GOAL!r})\nprint('queued:', child.name)\n```"
    )


def launch_wait_print():
    return (
        "```repl\n"
        f"child = launch_subagent(task={CHILD_GOAL!r})\n"
        "value = await child\n"
        "print('got', value)\n"
        "```"
    )


def launch_and_finish_early():
    return f"```repl\nchild = launch_subagent(task={CHILD_GOAL!r})\nfinish('early')\n```"


def launch_then_error():
    return f"```repl\nchild = launch_subagent(task={CHILD_GOAL!r})\nprint(1 / 0)\n```"


def child_reply(messages):
    if "working" in user_text(messages):
        return "```repl\nfinish('alpha')\n```"
    return "```repl\nprint('working')\n```"


def finish_child(_messages):
    return "```repl\nfinish('alpha')\n```"


def parent_then_child(*, parent, child=finish_child):
    def reply(messages):
        if CHILD_GOAL in user_text(messages):
            return child(messages)
        return parent(messages)

    return reply


def finished_child(*, parent_frontier: str = "exec") -> tuple[Graph, AgentStart, AgentStart]:
    """A parent whose launched child has already answered."""
    g, root = agent_graph(PARENT_QUERY, max_depth=1)
    ops.append(g, root, LLMOutput(content="launch", code="print('queued')"))
    action = ops.append(g, root, ExecAction(code="print('queued')"))
    child = ops.spawn(g, CHILD_GOAL, by=action, name="alpha")
    ops.append(g, child, DoneOutput(result="alpha"))
    if parent_frontier in {"exec", "llm"}:
        ops.append(g, root, ExecOutput(content="queued: alpha"))
    if parent_frontier == "llm":
        ops.append(g, root, LLMOutput(content="still", code="print('still')"))
    return g, root, child


def test_dependency_routed_result_does_not_create_a_parent_handoff():
    g, root = agent_graph(PARENT_QUERY)
    action = ops.append(g, root, ExecAction(code="launch"))
    source = ops.spawn(g, "source", by=action, name="source")
    ops.spawn(g, "consumer", by=action, name="consumer", inputs={"source": source})
    ops.append(g, source, DoneOutput(result="ready"))
    ops.append(g, root, ExecOutput(content="queued"))

    assert handoff.pending(g, root) == []
    assert handoff.react(g, root) is None


def test_background_spawn_flushes_user_query_then_calls_model():
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nfinish('got-alpha')\n```"
        if "queued:" in text:
            return "```repl\nprint('waiting')\n```"
        return launch_print()

    g, root = run_parent(Flow(StubLLM(parent_then_child(parent=parent))), max_depth=1, max_iters=10)

    [query] = handoff_queries(g, root)
    assert isinstance(reads.prev(g, query), ExecOutput)
    assert isinstance(next_node(g, query), LLMOutput)
    assert "alpha" in query.content
    assert "[handoff root.child0]" in query.content
    assert reads.result(g, root) == "got-alpha"


def test_two_children_batch_into_one_user_query():
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nfinish('both')\n```"
        if "queued:" in text:
            return "```repl\nprint('waiting')\n```"
        return (
            "```repl\n"
            "a = launch_subagent(task='Return alpha.')\n"
            "b = launch_subagent(task='Return beta.')\n"
            "await asyncio.gather(a, b)\n"
            "print('queued:', a.name, b.name)\n"
            "```"
        )

    def reply(messages):
        text = user_text(messages)
        if "Return alpha." in text:
            return "```repl\nfinish('alpha')\n```"
        if "Return beta." in text:
            return "```repl\nfinish('beta')\n```"
        return parent(messages)

    g, root = run_parent(Flow(StubLLM(reply), workers=3), max_depth=1, max_iters=6)

    [query] = handoff_queries(g, root)
    assert "[handoff root.child0]" in query.content
    assert "[handoff root.child1]" in query.content
    assert reads.result(g, root) == "both"


def test_child_done_during_parent_complete_flushes_after_exec():
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nfinish('got-alpha')\n```"
        if "queued:" in text:
            return "```repl\nprint('still-here')\n```"
        return launch_print()

    flow = Flow(StubLLM(parent_then_child(parent=parent, child=child_reply)))
    g, root = run_parent(flow, max_depth=1, max_iters=8)

    queries = handoff_queries(g, root)
    assert queries
    assert all(isinstance(reads.prev(g, query), ExecOutput) for query in queries)
    assert reads.result(g, root) == "got-alpha"


def test_handoff_is_sent_once_with_from_edges_to_the_child_result():
    g, root, child = finished_child()

    query = handoff.react(g, root)

    assert query is not None
    assert reads.inbox(g, root) == [query]
    assert [edge.target for edge in g.edges(kind=FROM, source=query.id)] == [
        reads.frontier(g, child).id
    ]
    assert handoff.react(g, root) is None
    assert ops.deliver(g, root) == [query]
    assert handoff_queries(g, root) == [query]
    assert "[handoff root.alpha]" in query.content


def test_handoff_waits_until_the_parent_turn_lands():
    g, root, _child = finished_child(parent_frontier="llm")

    assert handoff.react(g, root) is None
    assert handoff.pending(g, root)

    ops.append(g, root, ExecAction(code="print('still')"))
    assert handoff.react(g, root) is None
    ops.append(g, root, ExecOutput(content="still"))
    assert handoff.react(g, root) is not None


def test_mapping_result_renders_as_fields():
    g, root, child = finished_child()
    reads.frontier(g, child).result = {"n": 41, "note": "ok"}

    text = handoff.react(g, root).content

    assert "Each block is that child's submit() value" in text
    assert "n: 41" in text
    assert 'note: "ok"' in text
    assert "result: {" not in text


def test_scalar_result_stays_labeled_json():
    g, root, _child = finished_child()
    assert 'result: "alpha"' in handoff.react(g, root).content


def test_error_and_repl_dead_do_not_flush():
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nfinish('got-alpha')\n```"
        if "ZeroDivision" in text or "recovered" in text:
            return "```repl\nprint('recovered')\n```"
        return launch_then_error()

    g, root = run_parent(Flow(StubLLM(parent_then_child(parent=parent))), max_depth=1, max_iters=8)

    queries = handoff_queries(g, root)
    assert queries
    assert all(isinstance(reads.prev(g, query), ExecOutput) for query in queries)
    assert reads.result(g, root) == "got-alpha"


def test_error_and_repl_dead_frontiers_hold_the_handoff():
    for observation in (ErrorOutput(content="boom"), ReplDead()):
        g, root, _child = finished_child(parent_frontier="action")
        ops.append(g, root, observation)
        assert handoff.react(g, root) is None
        assert reads.frontier(g, root) is observation


def test_parent_finish_with_flag_wakes_on_child_done():
    def parent(messages):
        if "[handoff " in user_text(messages):
            return "```repl\nfinish('woke')\n```"
        return launch_and_finish_early()

    flow = Flow(StubLLM(parent_then_child(parent=parent, child=child_reply)))
    g, root = run_parent(flow, max_depth=1, max_iters=8, resume_on_child_done=True)

    queries = handoff_queries(g, root)
    assert queries
    woken = reads.prev(g, queries[0])
    assert isinstance(woken, DoneOutput)
    assert woken.result == "early"
    assert reads.result(g, root) == "woke"


def test_parent_finish_without_flag_stays_terminal():
    flow = Flow(StubLLM(parent_then_child(parent=lambda _m: launch_and_finish_early())))
    g, root = run_parent(flow, max_depth=1, max_iters=4, resume_on_child_done=False)

    assert reads.result(g, root) == "early"
    assert not handoff_queries(g, root)
    assert isinstance(reads.frontier(g, root), DoneOutput)


def test_finished_parent_is_not_woken_by_a_child_that_finished_first():
    g, root, _child = finished_child(parent_frontier="action")
    ops.append(g, root, DoneOutput(result="early"))

    assert handoff.react(g, root) is None


def test_wait_for_result_does_not_deadlock_and_still_handoffs():
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nfinish('got-alpha')\n```"
        return launch_wait_print()

    g, root = run_parent(Flow(StubLLM(parent_then_child(parent=parent))), max_depth=1, max_iters=6)

    queries = handoff_queries(g, root)
    assert queries
    observed = reads.prev(g, queries[0])
    assert isinstance(observed, ExecOutput)
    assert "got" in observed.content
    assert reads.result(g, root) == "got-alpha"
    [child] = reads.children(g, root)
    action = reads.prev(g, observed)
    assert action.metadata[WAITED_AGENTS_KEY] == [child.id]


def test_loaded_graph_sends_a_missed_handoff_once(tmp_path):
    g, root, _child = finished_child()
    g.save(tmp_path / "idle-parent")
    loaded = Graph.load(tmp_path / "idle-parent")
    parent = loaded.node(root.id)

    assert not handoff_queries(loaded, parent)
    handoff.react(loaded, parent)
    ops.deliver(loaded, parent)
    [query] = handoff_queries(loaded, parent)
    assert isinstance(reads.prev(loaded, query), ExecOutput)
    assert "alpha" in query.content


def test_replay_does_not_append_a_second_handoff(tmp_path):
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nfinish('got-alpha')\n```"
        if "queued:" in text:
            return "```repl\nprint('waiting')\n```"
        return launch_print()

    g, root = run_parent(Flow(StubLLM(parent_then_child(parent=parent))), max_depth=1, max_iters=6)
    assert len(handoff_queries(g, root)) == 1

    g.save(tmp_path / "done")
    loaded = Graph.load(tmp_path / "done")
    asyncio.run(Flow(StubLLM(parent_then_child(parent=parent))).arun(loaded))

    assert len(handoff_queries(loaded, loaded.node(root.id))) == 1
    assert reads.result(loaded, loaded.node(root.id)) == "got-alpha"


def test_user_query_after_exec_or_done_calls_model_directly():
    flow = Flow(StubLLM(lambda _messages: "reply"))
    for observation in (
        ExecOutput(content="observed"),
        DoneOutput(content="done", result="early"),
    ):
        g, root = agent_graph(PARENT_QUERY)
        ops.append(g, root, observation)
        ops.append(g, root, UserQuery(content="[handoff root.alpha]\nresult: 1"))
        assert isinstance(asyncio.run(flow.step(g, root)), LLMOutput)


def test_child_config_inherits_resume_on_child_done():
    assert AgentConfig(resume_on_child_done=True).child("worker").resume_on_child_done is True
    assert AgentConfig().resume_on_child_done is True
    assert AgentConfig().child("worker").resume_on_child_done is True
    opted_out = AgentConfig(resume_on_child_done=False)
    assert opted_out.child("worker").resume_on_child_done is False


def test_top_level_agents_have_no_launcher():
    g, root = agent_graph(PARENT_QUERY, max_depth=1)
    action = ops.append(g, root, ExecAction(code="launch"))
    child = ops.spawn(g, CHILD_GOAL, by=action, name="alpha")

    assert reads.launcher(g, root) is None
    assert reads.launcher(g, child) is root


def test_reports_to_edge_delivers_a_peer_result():
    g = Graph()
    worker = g.create_agent("work", config=AgentConfig(name="worker"))
    listener = g.create_agent("listen", config=AgentConfig(name="listener"))
    g.add_edge(Edge(kind=REPORTS_TO, source=worker.id, target=listener.id))
    ops.append(g, listener, ExecOutput(content="waiting"))
    done = ops.append(g, worker, DoneOutput(result=5))

    assert handoff.recipients(g, worker) == [listener]
    query = handoff.react(g, listener)

    assert query is not None
    assert query.content.startswith("Agent results are ready.")
    assert "[handoff worker]" in query.content
    assert [edge.target for edge in g.edges(kind=FROM, source=query.id)] == [done.id]
    assert handoff.react(g, listener) is None


def test_child_result_goes_to_launcher_and_reports_to_peer_once_each():
    g, root, child = finished_child()
    peer = g.create_agent("listen", config=AgentConfig(name="peer"))
    ops.append(g, peer, ExecOutput(content="waiting"))
    g.add_edge(Edge(kind=REPORTS_TO, source=child.id, target=peer.id))

    assert handoff.recipients(g, child) == [root, peer]
    to_root = handoff.react(g, root)
    to_peer = handoff.react(g, peer)

    assert to_root is not None and to_root.content.startswith("Child results are ready.")
    assert to_peer is not None and "[handoff root.alpha]" in to_peer.content
    assert handoff.react(g, root) is None
    assert handoff.react(g, peer) is None


def test_removed_reports_to_edge_stops_delivery():
    g = Graph()
    worker = g.create_agent("work", config=AgentConfig(name="worker"))
    listener = g.create_agent("listen", config=AgentConfig(name="listener"))
    edge = Edge(kind=REPORTS_TO, source=worker.id, target=listener.id)
    g.add_edge(edge)
    g.remove_edge(edge.id)
    ops.append(g, listener, ExecOutput(content="waiting"))
    ops.append(g, worker, DoneOutput(result=5))

    assert handoff.react(g, listener) is None


def test_finished_peer_wakes_on_a_later_reported_result():
    def reply(messages):
        text = user_text(messages)
        if "[handoff worker]" in text:
            return "```repl\nfinish('heard 5')\n```"
        if "listen" in text:
            return "```repl\nfinish('nothing yet')\n```"
        return "```repl\nfinish(5)\n```"

    g = Graph()
    listener = g.create_agent("listen", config=AgentConfig(name="listener", max_iters=4))
    worker = g.create_agent("work", config=AgentConfig(name="worker"), depends_on=(listener,))
    g.add_edge(Edge(kind=REPORTS_TO, source=worker.id, target=listener.id))

    asyncio.run(asyncio.wait_for(Flow(StubLLM(reply)).arun(g), timeout=5))

    assert reads.result(g, worker) == 5
    assert reads.result(g, listener) == "heard 5"
