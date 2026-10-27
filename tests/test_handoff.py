"""A finished agent's result is delivered as a ``message`` to every agent it reports to."""

from __future__ import annotations

import asyncio

from helpers import StubLLM, agent_graph, next_node

from rlmflow import CODE, DONE, FROM, LLM_OUTPUT, WAITING, AgentConfig, Flow, Graph, Node
from rlmflow.engine import handoff
from rlmflow.graph.graph import REPORTS_TO, Edge
from rlmflow.graph.nodes import AGENT_REF_KEY, LAUNCH_TOOLS

CHILD_GOAL = "Return alpha."
PARENT_QUERY = "Coordinate the child."


def user_text(messages):
    return "\n".join(message["content"] for message in messages if message["role"] == "user")


def handoff_messages(g: Graph, agent: Node) -> list[Node]:
    return [
        node
        for node in g.history(agent)
        if node.type == "message" and "[handoff " in node.data["text"]
    ]


def run_parent(flow: Flow, **config) -> tuple[Graph, Node]:
    g = Graph()
    root = g.create_agent(PARENT_QUERY, config=AgentConfig(**config))
    asyncio.run(asyncio.wait_for(flow.arun(g), timeout=5))
    return g, root


def launch_print():
    return f"```repl\nchild = launch_agent({CHILD_GOAL!r})\nprint('queued:', child.name)\n```"


def launch_wait_print():
    return f"```repl\nchild = launch_agent({CHILD_GOAL!r})\nvalue = await child\nprint('got', value)\n```"


def launch_and_finish_early():
    return f"```repl\nchild = launch_agent({CHILD_GOAL!r})\nsubmit('early')\n```"


def launch_then_error():
    return f"```repl\nchild = launch_agent({CHILD_GOAL!r})\nprint(1 / 0)\n```"


def child_reply(messages):
    if "working" in user_text(messages):
        return "```repl\nsubmit('alpha')\n```"
    return "```repl\nprint('working')\n```"


def finish_child(_messages):
    return "```repl\nsubmit('alpha')\n```"


def parent_then_child(*, parent, child=finish_child):
    def reply(messages):
        if CHILD_GOAL in user_text(messages):
            return child(messages)
        return parent(messages)

    return reply


def finished_child(*, parent_frontier: str = "code") -> tuple[Graph, Node, Node]:
    """A parent whose launched child has already answered."""
    g, root = agent_graph(PARENT_QUERY, max_depth=1)
    g.append(root, Node(LLM_OUTPUT, text="launch"))
    cell = g.append(root, Node(CODE, code="launch", output="queued: alpha"))
    child = g.spawn(CHILD_GOAL, by=cell, name="alpha")
    g.append(child, Node(DONE, result="alpha"))
    if parent_frontier == "llm":
        g.append(root, Node(LLM_OUTPUT, text="still"))
    return g, root, child


def test_dependency_routed_result_does_not_create_a_parent_handoff():
    g, root = agent_graph(PARENT_QUERY)
    cell = g.append(root, Node(CODE, code="launch"))
    source = g.spawn("source", by=cell, name="source")
    g.spawn("consumer", by=cell, name="consumer", inputs={"source": source})
    g.append(source, Node(DONE, result="ready"))

    assert handoff.pending(g, root) == []
    assert handoff.react(g, root) is None


def test_background_spawn_delivers_a_message_then_calls_model():
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nsubmit('got-alpha')\n```"
        if "queued:" in text:
            return "```repl\nprint('waiting')\n```"
        return launch_print()

    g, root = run_parent(Flow(StubLLM(parent_then_child(parent=parent))), max_depth=1, max_iters=10)

    [message] = handoff_messages(g, root)
    assert g.prev(message).type == CODE
    assert next_node(g, message).type == LLM_OUTPUT
    assert "alpha" in message.data["text"]
    assert "[handoff default.default-0]" in message.data["text"]
    assert g.result(root) == "got-alpha"


def test_both_children_results_are_delivered():
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nsubmit('both')\n```"
        if "queued:" in text:
            return "```repl\nprint('waiting')\n```"
        return (
            "```repl\n"
            "a = launch_agent('Return alpha.')\n"
            "b = launch_agent('Return beta.')\n"
            "print('queued:', a.name, b.name)\n"
            "```"
        )

    def reply(messages):
        text = user_text(messages)
        if "Return alpha." in text:
            return "```repl\nsubmit('alpha')\n```"
        if "Return beta." in text:
            return "```repl\nsubmit('beta')\n```"
        return parent(messages)

    g = Graph()
    root = g.create_agent(PARENT_QUERY, config=AgentConfig(max_depth=1, max_iters=6))
    flow = Flow(StubLLM(reply), workers=3)
    asyncio.run(asyncio.wait_for(flow.arun(g), timeout=5))

    messages = handoff_messages(g, root)
    text = "\n".join(m.data["text"] for m in messages)
    assert "[handoff default.default-0]" in text
    assert "[handoff default.default-1]" in text
    assert g.result(root) == "both"


def test_child_done_during_parent_model_call_is_delivered_after_its_cell():
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nsubmit('got-alpha')\n```"
        if "queued:" in text:
            return "```repl\nprint('still-here')\n```"
        return launch_print()

    flow = Flow(StubLLM(parent_then_child(parent=parent, child=child_reply)))
    g, root = run_parent(flow, max_depth=1, max_iters=8)

    messages = handoff_messages(g, root)
    assert messages
    assert all(g.prev(message).type == CODE for message in messages)
    assert g.result(root) == "got-alpha"


def test_handoff_is_sent_once_with_from_edges_to_the_child_result():
    g, root, child = finished_child()

    message = handoff.react(g, root)

    assert message is not None
    assert g.inbox(root) == [message]
    assert [edge.target for edge in g.edges(kind=FROM, source=message.id)] == [
        g.frontier(child).id
    ]
    assert handoff.react(g, root) is None
    assert g.deliver(root) == [message]
    assert handoff_messages(g, root) == [message]
    assert "[handoff default.alpha]" in message.data["text"]
    assert message.data["sender"] == child.id


def test_a_fork_after_a_launch_waits_for_and_receives_the_shared_child():
    g, root = agent_graph(PARENT_QUERY)
    cell = g.append(root, Node(CODE, code="child = launch_agent('x')", output=""))
    child = g.spawn(CHILD_GOAL, by=cell, name="alpha")
    fork = g.fork(cell, name="fork")
    g.append(root, Node(DONE, result="superseded"))
    g.append(fork, Node(WAITING, on=[child.id]))

    assert g.children(fork) == []
    assert g.launched(fork) == [child]
    assert [agent.id for agent in handoff.recipients(g, child)] == [root.id, fork.id]
    assert handoff.react(g, fork) is None

    g.append(child, Node(DONE, result="alpha"))

    message = handoff.react(g, fork)
    assert message is not None and "[handoff default.alpha]" in message.data["text"]
    assert handoff.react(g, root) is None


def test_a_fork_before_the_launching_cell_does_not_get_its_launches():
    g, root = agent_graph(PARENT_QUERY)
    first = g.append(root, Node(CODE, code="notes = 'plan'", output=""))
    # An external harness's cell launches from the node before it, then lands.
    child = g.spawn(CHILD_GOAL, by=first, name="alpha")
    second = Node(CODE, code="child = launch_agent('x')", output="")
    second.record_call(LAUNCH_TOOLS[0], 0, {AGENT_REF_KEY: child.id})
    g.append(root, second)
    fork = g.fork(first, name="fork")

    assert g.launched(root) == [child]
    assert g.launched(fork) == []
    assert [agent.id for agent in handoff.recipients(g, child)] == [root.id]


def test_handoff_waits_until_the_parent_cell_lands():
    g, root, _child = finished_child(parent_frontier="llm")

    assert handoff.react(g, root) is None
    assert handoff.pending(g, root)

    g.append(root, Node(CODE, code="print('still')", output="still"))
    assert handoff.react(g, root) is not None


def test_mapping_result_renders_as_fields():
    g, root, child = finished_child()
    g.frontier(child).data["result"] = {"n": 41, "note": "ok"}

    text = handoff.react(g, root).data["text"]

    assert "Each block is that agent's result" in text
    assert "n: 41" in text
    assert 'note: "ok"' in text
    assert "result: {" not in text


def test_scalar_result_stays_labeled_json():
    g, root, _child = finished_child()
    text = handoff.react(g, root).data["text"]
    assert 'result: "alpha"' in text
    assert "config: default" in text


def test_a_failed_cell_still_takes_the_handoff():
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nsubmit('got-alpha')\n```"
        if "ZeroDivision" in text or "recovered" in text:
            return "```repl\nprint('recovered')\n```"
        return launch_then_error()

    g, root = run_parent(Flow(StubLLM(parent_then_child(parent=parent))), max_depth=1, max_iters=8)

    messages = handoff_messages(g, root)
    assert messages
    assert all(g.prev(message).type == CODE for message in messages)
    assert g.result(root) == "got-alpha"


def test_waiting_takes_results_only_once_everything_it_waits_on_is_done():
    g, root, child = finished_child()
    cell = g.frontier(root)
    other = g.spawn("Return beta.", by=cell, name="beta")
    g.append(root, Node(WAITING, on=[child.id, other.id]))

    assert handoff.react(g, root) is None
    g.append(other, Node(DONE, result="beta"))
    message = handoff.react(g, root)
    assert "[handoff default.alpha]" in message.data["text"]
    assert "[handoff default.beta]" in message.data["text"]


def test_finished_parent_is_never_reopened_by_child_results():
    flow = Flow(StubLLM(parent_then_child(parent=lambda _m: launch_and_finish_early(), child=child_reply)))
    g, root = run_parent(flow, max_depth=1, max_iters=4)

    assert g.result(root) == "early"
    assert not handoff_messages(g, root)
    assert g.frontier(root).type == DONE
    [child] = g.children(root)
    assert g.result(child) == "alpha"


def test_finished_parent_is_not_woken_by_a_child_that_finished_first():
    g, root, _child = finished_child()
    g.append(root, Node(DONE, result="early"))

    assert handoff.react(g, root) is None


def test_awaited_result_is_not_handed_off_again():
    def parent(messages):
        text = user_text(messages)
        if "got alpha" in text:
            return "```repl\nsubmit('got-alpha')\n```"
        return launch_wait_print()

    g, root = run_parent(Flow(StubLLM(parent_then_child(parent=parent))), max_depth=1, max_iters=6)

    assert not handoff_messages(g, root)
    assert g.result(root) == "got-alpha"
    [child] = g.children(root)
    [cell] = [n for n in g.history(root) if n.type == CODE and "await" in n.data["code"]]
    assert cell.data[handoff.AWAITED_KEY] == [child.id]
    assert "got alpha" in cell.data["output"]


def test_loaded_graph_sends_a_missed_handoff_once(tmp_path):
    g, root, _child = finished_child()
    g.save(tmp_path / "idle-parent")
    loaded = Graph.load(tmp_path / "idle-parent")
    parent = loaded.node(root.id)

    assert not handoff_messages(loaded, parent)
    handoff.react(loaded, parent)
    loaded.deliver(parent)
    [message] = handoff_messages(loaded, parent)
    assert loaded.prev(message).type == CODE
    assert "alpha" in message.data["text"]


def test_rerunning_a_finished_graph_does_not_append_a_second_handoff(tmp_path):
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nsubmit('got-alpha')\n```"
        if "queued:" in text:
            return "```repl\nprint('waiting')\n```"
        return launch_print()

    g, root = run_parent(Flow(StubLLM(parent_then_child(parent=parent))), max_depth=1, max_iters=6)
    assert len(handoff_messages(g, root)) == 1

    g.save(tmp_path / "done")
    loaded = Graph.load(tmp_path / "done")
    asyncio.run(Flow(StubLLM(parent_then_child(parent=parent))).arun(loaded))

    assert len(handoff_messages(loaded, loaded.node(root.id))) == 1
    assert loaded.result(loaded.node(root.id)) == "got-alpha"


def test_message_after_code_or_done_calls_model_directly():
    flow = Flow(StubLLM(lambda _messages: "reply"))
    for observation in (Node(CODE, code="", output="observed"), Node(DONE, result="early")):
        g, root = agent_graph(PARENT_QUERY)
        g.append(root, observation)
        g.send(root, "[handoff default.alpha]\nresult: 1")
        assert asyncio.run(flow.step(g, root)).type == LLM_OUTPUT


def test_top_level_agents_have_no_launcher():
    g, root = agent_graph(PARENT_QUERY, max_depth=1)
    cell = g.append(root, Node(CODE, code="launch"))
    child = g.spawn(CHILD_GOAL, by=cell, name="alpha")

    assert g.launcher(root) is None
    assert g.launcher(child) is root


def test_reports_to_edge_delivers_a_peer_result():
    g = Graph()
    worker = g.create_agent("work", config=AgentConfig(name="worker"))
    listener = g.create_agent("listen", config=AgentConfig(name="listener"))
    g.add_edge(Edge(kind=REPORTS_TO, source=worker.id, target=listener.id))
    g.append(listener, Node(CODE, code="", output="waiting"))
    done = g.append(worker, Node(DONE, result=5))

    assert handoff.recipients(g, worker) == [listener]
    message = handoff.react(g, listener)

    assert message is not None
    assert message.data["text"].startswith("Agent results are ready.")
    assert "[handoff worker]" in message.data["text"]
    assert [edge.target for edge in g.edges(kind=FROM, source=message.id)] == [done.id]
    assert handoff.react(g, listener) is None


def test_child_result_goes_to_launcher_and_reports_to_peer_once_each():
    g, root, child = finished_child()
    peer = g.create_agent("listen", config=AgentConfig(name="peer"))
    g.append(peer, Node(CODE, code="", output="waiting"))
    g.add_edge(Edge(kind=REPORTS_TO, source=child.id, target=peer.id))

    assert handoff.recipients(g, child) == [root, peer]
    to_root = handoff.react(g, root)
    to_peer = handoff.react(g, peer)

    assert to_root is not None and to_root.data["text"].startswith("Agents you started have finished.")
    assert to_peer is not None and "[handoff default.alpha]" in to_peer.data["text"]
    assert handoff.react(g, root) is None
    assert handoff.react(g, peer) is None


def test_removed_reports_to_edge_stops_delivery():
    g = Graph()
    worker = g.create_agent("work", config=AgentConfig(name="worker"))
    listener = g.create_agent("listen", config=AgentConfig(name="listener"))
    edge = Edge(kind=REPORTS_TO, source=worker.id, target=listener.id)
    g.add_edge(edge)
    g.remove_edge(edge.id)
    g.append(listener, Node(CODE, code="", output="waiting"))
    g.append(worker, Node(DONE, result=5))

    assert handoff.react(g, listener) is None


def test_finished_peer_is_not_woken_by_a_later_reported_result():
    def reply(messages):
        text = user_text(messages)
        if "[handoff worker]" in text:
            return "```repl\nsubmit('heard 5')\n```"
        if "listen" in text:
            return "```repl\nsubmit('nothing yet')\n```"
        return "```repl\nsubmit(5)\n```"

    g = Graph()
    listener = g.create_agent("listen", config=AgentConfig(name="listener", max_iters=4))
    worker = g.create_agent("work", config=AgentConfig(name="worker"), depends_on=(listener,))
    g.add_edge(Edge(kind=REPORTS_TO, source=worker.id, target=listener.id))

    asyncio.run(asyncio.wait_for(Flow(StubLLM(reply)).arun(g), timeout=5))

    assert g.result(worker) == 5
    assert g.result(listener) == "nothing yet"
