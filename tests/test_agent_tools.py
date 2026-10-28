"""Agent-facing graph tools: launch, send, ask, wait and AGENTS, plus REPL rebuilds."""

from __future__ import annotations

import asyncio

import pytest
from helpers import ScriptedFlow, trailing_user_text
from helpers import repl as block

from rlmflow import CODE, DONE, AgentConfig, Graph, Node, Step
from rlmflow.agent_tools import tools as agent_tools
from rlmflow.engine.handoff import recipients


class Scripted:
    """Replies by matching the trailing user messages against needles, in order."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        last = trailing_user_text(messages)
        for needle, reply in self.script:
            if needle in last:
                return reply
        raise AssertionError(f"no scripted reply for {last!r}")


def agent(flow, g, task, name, **config):
    node = g.spawn(task, name=name, config=flow.add_config(AgentConfig(name=name, **config)).name)
    node.bind_config(flow.config(name))
    return node


def test_send_agent_queues_a_message_that_reopens_a_finished_peer():
    llm = Scripted(
        [
            ("hello from a", block("submit('b heard')")),
            ("listen", block("submit('b idle')")),
            (
                "speak",
                block("await AGENTS.get('b')\nsend_agent('b', 'hello from a')\nsubmit('sent')"),
            ),
        ]
    )
    flow = ScriptedFlow(llm)
    g = Graph()
    a = agent(flow, g, "speak", "a")
    b = agent(flow, g, "listen", "b")

    flow.run(g)

    assert g.result(a) == "sent"
    assert g.result(b) == "b heard"
    message = next(n for n in g.history(b) if n.data.get("text") == "hello from a")
    assert message.data["sender"] == a.id
    before = g.prev(message)
    assert before.type == DONE and before.data["result"] == "b idle"


def test_ask_agent_waits_for_the_answer_to_its_own_question():
    llm = Scripted(
        [
            ("ping", block("submit('pong')")),
            ("be b", block("submit('ready')")),
            ("be a", block("submit(await ask_agent('b', 'ping'))")),
        ]
    )
    flow = ScriptedFlow(llm)
    g = Graph()
    a = agent(flow, g, "be a", "a")
    agent(flow, g, "be b", "b")

    flow.run(g)

    assert g.result(a) == "pong"
    [cell] = g.nodes(DONE, agent=a.id)
    assert cell.calls("ask_agent") == {0: "pong"}


def test_connect_agent_draws_edges_and_disconnect_agent_removes_them():
    llm = Scripted(
        [
            ("be b", block("submit('ready')")),
            (
                "be a",
                block(
                    "edge = connect_agent('b', 'reports_to')\n"
                    "note = connect_agent('b', 'peer', why='checks')\n"
                    "disconnect_agent(note)\n"
                    "submit(edge)"
                ),
            ),
        ]
    )
    flow = ScriptedFlow(llm)
    g = Graph()
    a = agent(flow, g, "be a", "a")
    b = agent(flow, g, "be b", "b")

    flow.run(g)

    [edge] = [edge for edge in g.edges() if not edge.structural]
    assert edge.id == g.result(a)
    assert (edge.kind, edge.source, edge.target) == ("reports_to", a.id, b.id)
    assert [peer.id for peer in recipients(g, a)] == [b.id]


def test_ask_agent_refuses_a_wait_that_would_deadlock():
    llm = Scripted(
        [
            ("deadlock", block("submit('refused')")),
            ("child task", block("await ask_agent(AGENTS.get_parent(), 'help')")),
            ("parent task", block("submit(await launch_agent('child task'))")),
        ]
    )
    flow = ScriptedFlow(llm)
    g = Graph()
    root = g.create_agent("parent task")

    flow.run(g)

    assert g.result(root) == "refused"
    (child,) = g.children(root)
    assert any(
        node.type == CODE and node.data["error"] and "deadlock" in node.data["output"]
        for node in g.nodes(agent=child.id)
    )


def test_a_fresh_flow_rebuilds_the_repl_from_recorded_cells(tmp_path):
    llm = Scripted(
        [
            ("again", block("submit(x + 1)")),
            ("once", block("x = 41\nprint('set')")),
            ("set", block("submit('first')")),
        ]
    )
    g = Graph()
    worker = g.spawn("once", config="worker")
    configs = [AgentConfig(name="worker")]
    ScriptedFlow(llm, configs=configs).run(g)
    assert g.result(worker) == "first"

    g.save(tmp_path / "run")
    loaded = Graph.load(tmp_path / "run")
    loaded.send(loaded.node(worker.id), "again")
    ScriptedFlow(llm, configs=configs).run(loaded)
    assert loaded.result(loaded.node(worker.id)) == 42


def test_rebuilt_agent_tools_write_nothing():
    llm = Scripted(
        [
            ("hello", block("submit('heard')")),
            ("be b", block("submit('ready')")),
            ("be a", block("h = launch_agent('child')\nsend_agent('b', 'hello')\nprint('wrote')")),
            ("child", block("submit('kid')")),
            ("wrote", block("submit('ok')")),
        ]
    )
    g = Graph()
    flow = ScriptedFlow(llm)
    a = agent(flow, g, "be a", "a")
    agent(flow, g, "be b", "b")
    flow.run(g)
    before = len(g)
    agents_before = len(g.agents())

    g.send(a, "one more")
    ScriptedFlow(
        Scripted([("one more", block("print(h.name)")), ("default-0", block("submit(1)"))]),
        configs=list(flow.configs.values()),
    ).run(g)

    assert g.result(a) == 1
    assert len(g.agents()) == agents_before
    assert sum(node.data.get("text") == "hello" for node in g.nodes()) == 1
    assert all(getattr(item, "type", None) != "agent_start" for item in g.items(after=before - 1))


def test_send_agent_rejects_unknown_targets():
    llm = Scripted(
        [
            ("unknown agent", block("submit('refused')")),
            ("go", block("send_agent('nobody', 'hi')")),
        ]
    )
    flow = ScriptedFlow(llm)
    g = Graph()
    root = g.create_agent("go")

    flow.run(g)

    assert g.result(root) == "refused"


def test_launch_agent_takes_a_config_name_from_the_flow():
    llm = Scripted(
        [
            ("small job", block("submit('small')")),
            ("go", block("h = launch_agent('small job', config='cheap')\nsubmit(await h)")),
        ]
    )
    g = Graph()
    root = g.create_agent("go")

    ScriptedFlow(llm, configs=[AgentConfig(name="cheap", description="for small jobs")]).run(g)

    assert g.result(root) == "small"
    [child] = g.children(root)
    assert (child.name, child.config.name) == ("cheap-0", "cheap")


# -- the agent_tools functions, called directly ----------------------------------


def stepping_at(g: Graph, agent_node: Node) -> Step:
    return Step(g, agent_node, g.frontier(agent_node))


def test_spawn_checks_depth_task_length_and_config_names():
    flow = ScriptedFlow(Scripted([]))
    g = Graph()
    leaf = agent(flow, g, "x", "leaf", max_depth=0)
    with pytest.raises(ValueError, match="max depth"):
        agent_tools.spawn(flow, stepping_at(g, leaf), "child")
    root = agent(flow, g, "x", "root", max_query_chars=5)
    with pytest.raises(ValueError, match="exceeds 5"):
        agent_tools.spawn(flow, stepping_at(g, root), "too long a task")
    with pytest.raises(KeyError, match="elsewhere"):
        agent_tools.spawn(flow, stepping_at(g, root), "ok", config="elsewhere")
    with pytest.raises(TypeError, match="config name"):
        agent_tools.spawn(flow, stepping_at(g, root), "ok", config=AgentConfig())
    child = agent_tools.spawn(flow, stepping_at(g, root), "ok")
    assert g.launcher(child) is root


def test_wait_returns_results_or_running_after_the_timeout():
    flow = ScriptedFlow(Scripted([]))
    g = Graph()
    root = agent(flow, g, "x", "root")
    cell = g.append(root, Node(CODE, code=""))
    done = g.spawn("a", by=cell)
    g.append(done, Node(DONE, result=7))
    running = g.spawn("b", by=cell)

    result = asyncio.run(
        agent_tools.wait(flow, stepping_at(g, root), [done.id, running.name], timeout=0.01)
    )
    assert result == {done.id: 7, running.id: "running"}


def test_done_checks_the_output_schema():
    flow = ScriptedFlow(Scripted([]))
    g = Graph()
    typed = agent(flow, g, "x", "typed", output_schema={"type": "integer"})
    assert agent_tools.done(flow, stepping_at(g, typed), "3") == 3
    with pytest.raises(Exception, match="integer"):
        agent_tools.done(flow, stepping_at(g, typed), "three")
