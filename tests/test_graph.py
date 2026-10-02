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
    first = ExecOutput(content="one")
    assert agent.append(first) is None
    second = DoneOutput(result="ok")
    first.append(second)

    assert agent.transcript() == [agent, first, second]
    assert [node.seq for node in agent.transcript()] == [0, 1, 2]
    assert agent.frontier is second
    assert first.parent_agent is agent
    assert agent.root is agent
    assert agent.terminal
    assert agent.result() == "ok"


def test_a_child_agent_branches_off_the_node_that_launched_it():
    root = start("parent")
    action = ExecAction(code="launch")
    root.append(action)
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


def test_fork_remaps_and_prunes_launch_receipts():
    root = start("parent")
    action = ExecAction(code="launch")
    root.append(action)
    child = AgentStart(content="child", config=root.config.child("child"))
    action.append(child)
    action.record_agent_launch(0, child)
    action.mark_agent_retrieved(child.id)
    child.append(DoneOutput(result="done"))

    complete = child.frontier.fork()
    copied_action = next(node for node in complete.walk() if isinstance(node, ExecAction))
    copied_child = complete.sub_agents[0]
    assert copied_action.launched_agent_ids == {0: copied_child.id}
    assert copied_action.retrieved_agent_ids == [copied_child.id]
    assert copied_child.id != child.id

    pruned = action.fork()
    pruned_action = next(node for node in pruned.walk() if isinstance(node, ExecAction))
    assert pruned.sub_agents == []
    assert pruned_action.launched_agent_ids == {}
    assert pruned_action.retrieved_agent_ids == []


def test_fork_remaps_dependency_agent_ids():
    root = start("parent")
    action = ExecAction(code="launch")
    root.append(action)
    source = AgentStart(content="source", config=root.config.child("source"))
    action.append(source)
    action.record_agent_launch(0, source)
    source.append(DoneOutput(result="source"))
    consumer = AgentStart(
        content="consumer",
        args={"source": {"$rlmflow_agent": source.id}},
        config=root.config.child("consumer"),
    )
    action.append(consumer)
    action.record_agent_launch(1, consumer)

    copied = consumer.fork()
    copied_source, copied_consumer = copied.sub_agents

    assert copied_consumer.args == {"source": {"$rlmflow_agent": copied_source.id}}
    assert copied_consumer.argument_references == (("source", copied_source.id),)


def test_append_child_rehomes_a_complete_subtree():
    subtree = start("copied worker")
    nested_action = ExecAction(code="launch")
    subtree.append(nested_action)
    nested = AgentStart(content="nested", config=subtree.config.child("nested"))
    nested_action.append(nested)
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
        "_handle = launch_subagent(task='')\n"
        "print(await _handle)"
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
        "    launch_subagent(task=''),\n"
        "    launch_subagent(task=''),\n"
        "]\n"
        "print(await asyncio.gather(*_handles))"
    )
