"""Agent-facing graph tools: launch, send, wait and AGENTS, plus REPL rebuilds."""

from __future__ import annotations

import pytest
from helpers import ScriptedFlow, trailing_user_text
from helpers import repl as block

from rlmflow import CODE, DONE, LLM, MESSAGE, RETURN, Agent, AgentConfig, Graph, Step
from rlmflow.agent_tools import tools as agent_tools
from rlmflow.graph import calls


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
    record = g.add_agent(
        task, name=name, config=flow.add_config(AgentConfig(name=name, **config)).name
    )
    record.bind_config(flow.config(name))
    return record


def test_a_sibling_messages_a_finished_peer_and_reopens_it():
    llm = Scripted(
        [
            ("hello from a", block("submit('b heard')")),
            ("listen", block("submit('b idle')")),
            ("speak", block("AGENTS['b'].send('hello from a')\nsubmit('sent')")),
            (
                "parent",
                block(
                    "b = launch_agent('listen', name='b')\nawait b\n"
                    "a = launch_agent('speak', name='a')\nsubmit(await a)"
                ),
            ),
        ]
    )
    g = Graph()
    root = g.add_agent("parent")

    ScriptedFlow(llm).run(g)

    assert g.result(root) == "sent"
    a, b = (next(c for c in g.children(root) if c.name == name) for name in "ab")
    assert g.parent(a) is g.parent(b) is root
    assert g.result(a) == "sent" and g.result(b) == "b heard"
    assert [n.type for n in b.nodes] == [LLM, CODE, DONE, MESSAGE, LLM, CODE, DONE]
    message = b.nodes[3]
    assert message.data["text"] == "hello from a" and message.data["sender"] == a.id
    assert b.nodes[2].data["result"] == "b idle"
    [edge] = g.edges(kind=MESSAGE, target=message.id)
    sending_cell = g.node(edge.source)
    assert g.owner(sending_cell) is a and "AGENTS['b'].send" in sending_cell.data["code"]


def test_awaiting_send_to_a_descendant_returns_its_reply():
    llm = Scripted(
        [
            ("ping", block("submit('pong')")),
            ("be b", block("submit('ready')")),
            (
                "be a",
                block("b = launch_agent('be b', name='b')\nawait b\nsubmit(await b.send('ping'))"),
            ),
        ]
    )
    g = Graph()
    a = g.add_agent("be a")

    ScriptedFlow(llm).run(g)

    assert g.result(a) == "pong"
    [b] = g.children(a)
    assert [n.type for n in b.nodes] == [LLM, CODE, DONE, MESSAGE, LLM, CODE, DONE]
    [cell] = g.nodes(CODE, agent=a.id)
    saved = cell.data["calls"]
    assert [e["agent"] for e in saved if e["tool"] == calls.SEND] == [b.id]
    assert [e["agent"] for e in saved if e["tool"] == calls.AWAIT] == [b.id, b.id]
    assert b.nodes[-1].id in {e.source for e in g.edges(kind=RETURN, target=cell.id)}
    [edge] = g.edges(kind=MESSAGE, target=b.nodes[3].id)
    assert edge.source == cell.id


def test_a_rebuilt_repl_gives_each_await_the_result_it_first_returned():
    llm = Scripted(
        [
            ("again", block("submit(first)")),
            ("ping", block("submit('pong')")),
            ("be b", block("submit('ready')")),
            (
                "be a",
                block(
                    "b = launch_agent('be b', name='b')\nfirst = await b\n"
                    "second = await b.send('ping')\nprint('got', first, second)"
                ),
            ),
            ("got", block("submit('ok')")),
        ]
    )
    g = Graph()
    a = g.add_agent("be a")
    ScriptedFlow(llm).run(g)
    cell = next(n for n in a.nodes if n.type == CODE)
    assert cell.data["output"].strip() == "got ready pong"
    [b] = g.children(a)
    assert {e.source for e in g.edges(kind=RETURN, target=cell.id)} == {
        n.id for n in b.nodes if n.type == DONE
    }

    g.send(a, "again")
    ScriptedFlow(llm).run(g)
    assert g.result(a) == "ready"


def test_send_refuses_to_message_itself():
    llm = Scripted(
        [
            ("itself", block("submit('refused')")),
            ("be a", block("AGENTS.self.send('hi')")),
        ]
    )
    flow = ScriptedFlow(llm)
    g = Graph()
    a = agent(flow, g, "be a", "a")

    flow.run(g)

    assert g.result(a) == "refused"


def test_an_agent_reads_another_through_agents():
    llm = Scripted(
        [
            ("child job", block("submit('kid done')")),
            (
                "launched",
                block(
                    "kid = AGENTS['kid']\n"
                    "submit([[a.name for a in AGENTS], [a.name for a in AGENTS.children], "
                    "kid.status, kid.result(), kid.parent.name, kid.task, kid.config, "
                    "'kid' in AGENTS, 'root.kid' in AGENTS, AGENTS[kid.id].name])"
                ),
            ),
            ("parent", block("await launch_agent('child job', name='kid')\nprint('launched')")),
        ]
    )
    g = Graph()
    root = g.add_agent("parent")
    ScriptedFlow(llm).run(g)
    assert g.result(root) == [
        ["root", "kid"],
        ["kid"],
        "done",
        "kid done",
        "root",
        "child job",
        "default",
        True,
        True,
        "kid",
    ]


def test_awaiting_an_agent_that_is_not_a_descendant_raises_not_your_agent():
    llm = Scripted(
        [
            ("you can only await agents you launched", block("submit('refused')")),
            ("child task", block("await AGENTS.parent")),
            ("parent task", block("submit(await launch_agent('child task'))")),
        ]
    )
    g = Graph()
    root = g.add_agent("parent task")

    ScriptedFlow(llm).run(g)

    assert g.result(root) == "refused"
    (child,) = g.children(root)
    failed = next(n for n in child.nodes if n.type == CODE)
    assert failed.data["error"] and "NotYourAgent" in failed.data["output"]


def test_a_fresh_flow_rebuilds_the_repl_from_recorded_cells(tmp_path):
    llm = Scripted(
        [
            ("again", block("submit(x + 1)")),
            ("once", block("x = 41\nprint('set')")),
            ("set", block("submit('first')")),
        ]
    )
    g = Graph()
    worker = g.add_agent("once", config="worker")
    configs = [AgentConfig(name="worker")]
    ScriptedFlow(llm, configs=configs).run(g)
    assert g.result(worker) == "first"

    g.save(tmp_path / "run")
    loaded = Graph.load(tmp_path / "run")
    loaded.send(loaded.agent(worker.id), "again")
    ScriptedFlow(llm, configs=configs).run(loaded)
    assert loaded.result(loaded.agent(worker.id)) == 42


def test_rebuilt_agent_tools_write_nothing():
    llm = Scripted(
        [
            ("hello", block("submit('heard')")),
            ("be b", block("submit('ready')")),
            ("be a", block("h = launch_agent('child')\nAGENTS['b'].send('hello')\nprint('wrote')")),
            ("child", block("submit('kid')")),
            ("wrote", block("submit(await h)")),
        ]
    )
    g = Graph()
    flow = ScriptedFlow(llm)
    a = agent(flow, g, "be a", "a")
    agent(flow, g, "be b", "b")
    flow.run(g)
    assert g.result(a) == "kid"
    agents_before, edges_before = len(g.agents()), len(g.edges())

    g.send(a, "one more")
    nodes_before = {node.id for node in g.nodes()}
    ScriptedFlow(
        Scripted([("one more", block("print(h.name)")), ("default-0", block("submit(1)"))]),
        configs=list(flow.configs.values()),
    ).run(g)

    assert g.result(a) == 1
    assert len(g.agents()) == agents_before
    assert len(g.edges()) == edges_before
    assert sum(node.data.get("text") == "hello" for node in g.nodes()) == 1
    new = [node for node in g.nodes() if node.id not in nodes_before]
    assert [n.type for n in new] == [LLM, CODE, LLM, CODE, DONE]
    assert all(g.owner(n) is a for n in new)


def test_unknown_agents_are_a_key_error():
    llm = Scripted(
        [
            ("unknown agent", block("submit('refused')")),
            ("go", block("AGENTS['nobody'].send('hi')")),
        ]
    )
    flow = ScriptedFlow(llm)
    g = Graph()
    root = g.add_agent("go")

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
    root = g.add_agent("go")

    ScriptedFlow(llm, configs=[AgentConfig(name="cheap", description="for small jobs")]).run(g)

    assert g.result(root) == "small"
    [child] = g.children(root)
    assert (child.name, child.config.name) == ("cheap-0", "cheap")


# -- the agent_tools functions, called directly ----------------------------------


def stepping_at(g: Graph, record: Agent) -> Step:
    return Step.latest(g, record)


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
    assert g.parent(child) is root


def test_done_checks_the_output_schema():
    flow = ScriptedFlow(Scripted([]))
    g = Graph()
    typed = agent(flow, g, "x", "typed", output_schema={"type": "integer"})
    assert agent_tools.done(flow, stepping_at(g, typed), "3") == 3
    with pytest.raises(Exception, match="integer"):
        agent_tools.done(flow, stepping_at(g, typed), "three")
