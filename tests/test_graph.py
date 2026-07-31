from rlmflow import (
    AgentStart,
    DoneOutput,
    ExecOutput,
    SupervisingOutput,
    start,
)


def test_agent_owns_an_ordered_step_list():
    agent = start("query")
    first = agent.submit(ExecOutput(content="one"))
    second = first.submit(DoneOutput(result="ok"))

    assert agent.transcript() == (agent, first, second)
    assert agent.tail() is second
    assert first.agent is agent
    assert agent.root is agent
    assert agent.finished()
    assert agent.result() == "ok"


def test_only_supervisors_own_child_agents():
    root = start("parent")
    supervisor = root.submit(SupervisingOutput())
    assert isinstance(supervisor, SupervisingOutput)
    child = AgentStart(
        content="child",
        config=root.config.child("worker"),
    )

    supervisor.spawn(child)

    assert child.supervisor is supervisor
    assert child.root is root
    assert supervisor.children == [child]
    assert [agent.config.path for agent in root.agents()] == ["root", "root.worker"]


def test_duplicate_child_names_are_rejected():
    root = start("parent")
    supervisor = root.submit(SupervisingOutput())
    assert isinstance(supervisor, SupervisingOutput)
    supervisor.spawn(AgentStart(config=root.config.child("worker")))

    try:
        supervisor.spawn(AgentStart(config=root.config.child("worker")))
    except ValueError as exc:
        assert "duplicate child name" in str(exc)
    else:
        raise AssertionError("duplicate child name was accepted")


def test_detached_nodes_cannot_be_attached_twice():
    first = start("first")
    second = start("second")
    node = ExecOutput(content="output")
    first.submit(node)

    try:
        second.submit(node)
    except ValueError as exc:
        assert "already attached" in str(exc)
    else:
        raise AssertionError("attached node was reused")
