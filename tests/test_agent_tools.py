"""Agent-facing graph tools: fork, send, ask, connect, disconnect, and AGENTS."""

from __future__ import annotations

import pytest
from helpers import repl as block
from helpers import trailing_user_text

from rlmflow import AgentConfig, ContextPolicy, ExecAction, Flow, Graph, reads
from rlmflow.graph import ops
from rlmflow.graph.nodes import DoneOutput, ErrorOutput


class Scripted:
    """Replies by matching the trailing user messages against needles, in order."""

    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages):
        last = trailing_user_text(messages)
        for needle, reply in self.script:
            if needle in last:
                return reply
        raise AssertionError(f"no scripted reply for {last!r}")


def agent(flow, g, task, name, **config):
    return g.create_agent(task, config=AgentConfig(name=name, **config))


def test_fork_agent_forks_self_into_independent_attempts():
    llm = Scripted(
        [
            ("Try A", block("submit('A')")),
            ("Try B", block("submit('B')")),
            ("['A', 'B']", block("submit('best of two')")),
            (
                "pick",
                block(
                    "tries = [fork_agent(message=f'Try {k}.') for k in 'AB']\n"
                    "print(await asyncio.gather(*tries))"
                ),
            ),
        ]
    )
    flow = Flow(llm)
    g = Graph()
    root = agent(flow, g, "pick", "root", max_depth=1)

    flow.run(g)

    assert reads.result(g, root) == "best of two"
    forks = reads.children(g, root)
    assert [fork.config.name for fork in forks] == ["root_fork0", "root_fork1"]
    assert [reads.result(g, fork) for fork in forks] == ["A", "B"]
    action = next(node for node in g.nodes(ExecAction, agent=root) if "fork_agent" in node.code)
    for fork in forks:
        history = reads.history(g, reads.frontier(g, fork))
        # The shared prefix ends at the turn that forked, then the fork's own message.
        assert reads.prev(g, action) in history and action not in history
    assert set(action.tool_calls("fork_agent")) == {0, 1}


def test_fork_agent_of_a_finished_child_reopens_it_with_a_message():
    llm = Scripted(
        [
            ("draft", block("submit('v1')")),
            ("Improve it", block("submit('v2')")),
            ("v2", block("submit('done')")),
            (
                "write",
                block(
                    "first = launch_agent(name='writer', task='draft')\n"
                    "await first\n"
                    "second = fork_agent(first, message='Improve it.')\n"
                    "print(await second)"
                ),
            ),
        ]
    )
    flow = Flow(llm)
    g = Graph()
    root = agent(flow, g, "write", "root", max_depth=1)

    flow.run(g)

    writer, fork = reads.children(g, root)
    assert reads.result(g, writer) == "v1"
    assert reads.result(g, fork) == "v2"
    assert fork.config.name == "writer_fork1"


def test_fork_agent_requires_a_message_to_fork_self():
    llm = Scripted(
        [
            ("needs message=", block("submit('refused')")),
            ("go", block("fork_agent()")),
        ]
    )
    flow = Flow(llm)
    g = Graph()
    root = agent(flow, g, "go", "root", max_depth=1)

    flow.run(g)

    assert reads.result(g, root) == "refused"
    assert reads.children(g, root) == []


def test_send_agent_queues_a_message_that_reopens_a_finished_peer():
    llm = Scripted(
        [
            ("hello from a", block("submit('b heard')")),
            ("listen", block("submit('b idle')")),
            (
                "speak",
                block(
                    "await AGENTS.get('b').wait_for_result()\nsend_agent('b', 'hello from a')\nsubmit('sent')"
                ),
            ),
        ]
    )
    flow = Flow(llm, use_agent_tree=True)
    g = Graph()
    a = agent(flow, g, "speak", "a")
    b = agent(flow, g, "listen", "b")

    flow.run(g)

    assert reads.result(g, a) == "sent"
    assert reads.result(g, b) == "b heard"
    message = next(q for q in reads.history(g, reads.frontier(g, b)) if q.content == "hello from a")
    sender = reads.prev(g, message)
    assert isinstance(sender, DoneOutput) and sender.result == "b idle"


def test_ask_agent_waits_for_the_answer_to_its_own_question():
    llm = Scripted(
        [
            ("ping", block("submit('pong')")),
            ("be b", block("submit('ready')")),
            ("be a", block("submit(await ask_agent('b', 'ping'))")),
        ]
    )
    flow = Flow(llm, use_agent_tree=True)
    g = Graph()
    a = agent(flow, g, "be a", "a")
    agent(flow, g, "be b", "b")

    flow.run(g)

    assert reads.result(g, a) == "pong"
    action = next(iter(g.nodes(ExecAction, agent=a)))
    assert action.tool_calls("ask_agent") == {0: "pong"}


def test_ask_agent_schema_applies_only_to_the_asked_turn():
    llm = Scripted(
        [
            ("count", block("submit({'n': 3})")),
            ("be b", block("submit('free text')")),
            (
                "be a",
                block(
                    "schema = {'type': 'object', 'properties': {'n': {'type': 'integer'}},"
                    " 'required': ['n'], 'additionalProperties': False}\n"
                    "submit(await ask_agent('b', 'count', schema=schema))"
                ),
            ),
        ]
    )
    flow = Flow(llm, use_agent_tree=True)
    g = Graph()
    a = agent(flow, g, "be a", "a")
    b = agent(flow, g, "be b", "b")

    flow.run(g)

    assert reads.result(g, a) == {"n": 3}
    assert b.config.output_schema is None


def test_ask_agent_refuses_a_wait_that_would_deadlock():
    llm = Scripted(
        [
            ("deadlock", block("submit('refused')")),
            ("child task", block("await ask_agent(AGENTS.get_parent(), 'help')")),
            ("parent task", block("submit(await launch_agent(task='child task'))")),
        ]
    )
    flow = Flow(llm, use_agent_tree=True)
    g = Graph()
    root = agent(flow, g, "parent task", "root", max_depth=1)

    flow.run(g)

    assert reads.result(g, root) == "refused"
    (child,) = reads.children(g, root)
    assert any(
        isinstance(node, ErrorOutput) and "deadlock" in node.content
        for node in g.nodes(agent=child)
    )


def test_ephemeral_agents_refuse_follow_ups_and_drop_their_repl():
    llm = Scripted(
        [
            ("ephemeral", block("submit('refused')")),
            ("once", block("x = 1\nsubmit('done')")),
            (
                "talk",
                block("await AGENTS.get('once').wait_for_result()\nsend_agent('once', 'again')"),
            ),
        ]
    )
    flow = Flow(llm, use_agent_tree=True)
    g = Graph()
    talker = agent(flow, g, "talk", "talker")
    once = agent(flow, g, "once", "once", context_policy=ContextPolicy.EPHEMERAL)

    flow.run(g)

    assert reads.result(g, talker) == "refused"
    assert reads.inbox(g, once) == []
    assert flow.runtime.get(once) is None


def test_rehydratable_agent_drops_its_repl_and_replays_on_follow_up():
    llm = Scripted([("again", block("submit(x + 1)")), ("once", block("x = 41\nsubmit('first')"))])
    flow = Flow(llm)
    g = Graph()
    worker = agent(flow, g, "once", "worker", context_policy=ContextPolicy.REHYDRATABLE)

    flow.run(g)
    assert reads.result(g, worker) == "first"
    assert flow.runtime.get(worker) is None

    ops.send(g, worker, "again")
    flow.run(g)
    assert reads.result(g, worker) == 42


def test_connect_and_disconnect_edit_application_edges_visible_in_agents():
    llm = Scripted(
        [
            ("after", block("submit([e.kind for e in AGENTS.edges(source=AGENTS.get())])")),
            (
                "linked",
                block(
                    "print([e.target == AGENTS.get('b').agent_id for e in AGENTS.edges(kind='peer')])"
                ),
            ),
            ("link", block("connect_agent('b', 'peer', weight=2)\nprint('linked')")),
            ("[True]", block("disconnect_agent(AGENTS.edges(kind='peer')[0].id)\nprint('after')")),
            ("be b", block("submit('b')")),
        ]
    )
    flow = Flow(llm, use_agent_tree=True)
    g = Graph()
    a = agent(flow, g, "link", "a")
    b = agent(flow, g, "be b", "b")

    flow.run(g)

    assert reads.result(g, a) == []
    (edge,) = g.edges(kind="peer", active=None)
    assert (edge.source, edge.target, dict(edge.data)) == (a.id, b.id, {"weight": 2})
    assert not edge.active


def test_replayed_agent_tools_write_nothing():
    llm = Scripted(
        [
            ("hello", block("submit('heard')")),
            ("be b", block("submit('ready')")),
            (
                "be a",
                block("send_agent('b', 'hello')\nconnect_agent('b', 'peer')\nprint('wrote')"),
            ),
            ("wrote", block("submit('ok')")),
        ]
    )
    flow = Flow(llm, use_agent_tree=True)
    g = Graph()
    a = agent(flow, g, "be a", "a")
    agent(flow, g, "be b", "b")
    flow.run(g)
    before = len(g)

    ops.send(g, a, "one more")
    replaying = Flow(
        Scripted([("one more", block("print('x')")), ("x", block("submit(1)"))]),
        use_agent_tree=True,
    )
    replaying.run(g)

    new = g.items(after=before - 1)
    assert not any(getattr(item, "kind", None) == "peer" for item in new)
    assert sum(getattr(item, "content", None) == "hello" for item in g.nodes()) == 1


def test_send_agent_rejects_unknown_targets():
    llm = Scripted(
        [
            ("unknown agent", block("submit('refused')")),
            ("go", block("send_agent('nobody', 'hi')")),
        ]
    )
    flow = Flow(llm, use_agent_tree=True)
    g = Graph()
    root = agent(flow, g, "go", "root")

    flow.run(g)

    assert reads.result(g, root) == "refused"


@pytest.mark.parametrize("kind", ["next", "from", "depends_on", ""])
def test_connect_agent_rejects_structural_kinds(kind):
    llm = Scripted(
        [
            ("application kind", block("submit('refused')")),
            ("go", block(f"connect_agent('root', {kind!r})")),
        ]
    )
    flow = Flow(llm, use_agent_tree=True)
    g = Graph()
    root = agent(flow, g, "go", "root")

    flow.run(g)

    assert reads.result(g, root) == "refused"
