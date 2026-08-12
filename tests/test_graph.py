import pytest

from rlmflow import (
    AgentStart,
    AppendChild,
    DoneOutput,
    ExecAction,
    ExecOutput,
    start,
)


def test_agent_owns_an_ordered_transcript():
    agent = start("query")
    first = agent.append(ExecOutput(content="one"))
    second = first.append(DoneOutput(result="ok"))

    assert agent.transcript() == [agent, first, second]
    assert [node.seq for node in agent.transcript()] == [0, 1, 2]
    assert agent.frontier is second
    assert first.parent_agent is agent
    assert agent.root is agent
    assert agent.terminal
    assert agent.result() == "ok"


def test_a_child_agent_branches_off_the_node_that_launched_it():
    root = start("parent")
    action = root.append(ExecAction(code="launch"))
    child = AgentStart(content="child", config=root.config.child("worker"))

    action.append(child)

    assert child.parent is action
    assert child.root is root
    assert child.parent_agent is child  # a child starts a transcript of its own
    assert root.sub_agents == [child]
    assert [agent.config.path for agent in (root, *root.sub_agents)] == ["root", "root.worker"]
    # Launching moved neither agent on: the parent is still mid-step at the action,
    # and the child has nothing but its own start.
    assert root.frontier is action and action.next is None
    assert child.transcript() == [child]


def test_duplicate_child_names_are_rejected():
    root = start("parent")
    root.append(AgentStart(config=root.config.child("worker")))

    with pytest.raises(ValueError, match="duplicate child name"):
        root.append(AgentStart(config=root.config.child("worker")))


def test_append_child_rehomes_a_complete_subtree():
    subtree = start("copied worker")
    nested_action = subtree.append(ExecAction(code="launch"))
    nested = nested_action.append(
        AgentStart(content="nested", config=subtree.config.child("nested"))
    )
    nested.append(DoneOutput(result="done"))

    root = start("shepherd")
    root.append_child(subtree, name="branch0")
    action = root.frontier

    assert isinstance(action, AppendChild)
    assert action.child_agents == [subtree]
    assert subtree.parent is action
    assert root.sub_agents == [subtree]
    assert all(node.root is root for node in subtree.walk())
    assert subtree.config.path == "root.branch0"
    assert nested.config.path == "root.branch0.nested"
    assert action.code == (
        "_handle = await launch_subagent('', name='branch0')\n"
        "print(await _handle.wait_for_result())"
    )


def test_append_child_reuses_one_launch_action():
    root = start("shepherd")

    first = root.append_child(start("one"), name="one")
    action = root.frontier
    second = root.append_child(start("two"), name="two")

    assert isinstance(action, AppendChild)
    assert action.child_agents == [first, second]
    assert action.code == (
        "_handles = [\n"
        "    await launch_subagent('', name='one'),\n"
        "    await launch_subagent('', name='two'),\n"
        "]\n"
        "print([await h.wait_for_result() for h in _handles])"
    )
