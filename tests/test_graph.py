import pytest
from helpers import agent_graph

from rlmflow import (
    AppendChild,
    DoneOutput,
    ExecAction,
    ExecOutput,
    Graph,
    Input,
    ops,
    reads,
)


@pytest.mark.parametrize("description", ["", "   ", None])
def test_input_requires_a_description(description):
    with pytest.raises(ValueError, match="non-empty"):
        Input("value", description=description)


def test_input_like_user_dictionary_remains_plain_data_after_roundtrip(tmp_path):
    value = {
        "$rlmflow_input": {"nested": "value"},
        "$rlmflow_description": "ordinary user data",
    }
    g, root = agent_graph("query", args={"payload": value})

    g.save(tmp_path / "run")
    loaded = Graph.load(tmp_path / "run").node(root.id)

    assert root.resolved_args() == {"payload": value}
    assert loaded.resolved_args() == {"payload": value}
    assert loaded.argument_descriptions() == {}


def test_agent_owns_an_ordered_history():
    g, agent = agent_graph("query")
    first = ops.append(g, agent, ExecOutput(content="one"))
    second = ops.append(g, agent, DoneOutput(result="ok"))

    assert reads.history(g, agent) == [agent, first, second]
    assert agent.seq < first.seq < second.seq
    assert reads.frontier(g, agent) is second
    assert first.agent_id == agent.id
    assert reads.lineage(g, agent) == [agent]
    assert reads.terminal(g, agent)
    assert reads.result(g, agent) == "ok"


def test_a_child_agent_branches_off_the_node_that_launched_it():
    g, root = agent_graph("parent")
    action = ops.append(g, root, ExecAction(code="launch"))
    child = ops.spawn(g, "child", by=action, name="worker")

    assert reads.launcher(g, child) is root
    assert reads.lineage(g, child) == [root, child]
    assert reads.children(g, root) == [child]
    assert [reads.path(g, agent) for agent in (root, child)] == ["root", "root.worker"]
    # Launching moved neither agent on: the parent is still mid-step at the action,
    # and the child has nothing but its own start.
    assert reads.frontier(g, root) is action
    assert reads.history(g, child) == [child]


def test_duplicate_attached_child_names_are_rejected():
    g, root = agent_graph("parent")
    ops.attach(g, root, ops.spawn(g, "one"), names=["worker"])

    with pytest.raises(ValueError, match="duplicate child name"):
        ops.attach(g, root, ops.spawn(g, "two"), names=["worker"])


def test_fork_after_a_launch_shares_its_children_instead_of_copying_them():
    g, root = agent_graph("parent")
    action = ops.append(g, root, ExecAction(code="launch"))
    child = ops.spawn(g, "child", by=action, name="child")
    action.record_tool_call("launch_agent", 0, {"$rlmflow_agent": child.id})
    ops.append(g, child, DoneOutput(result="done"))
    output = ops.append(g, root, ExecOutput(content="done"))

    after = ops.fork(g, output)
    before = ops.fork(g, root)

    assert action in reads.history(g, after)
    assert action.launch_agent_ids() == [child.id]
    assert len(g.nodes(type(child))) == 4  # root, child, and two empty forks
    assert action not in reads.history(g, before)


def test_dependency_edges_name_the_producer_and_its_description():
    g, root = agent_graph("parent")
    action = ops.append(g, root, ExecAction(code="launch"))
    source = ops.spawn(g, "source", by=action, name="source")
    ops.append(g, source, DoneOutput(result="artifact"))
    consumer = ops.spawn(
        g,
        "consumer",
        by=action,
        name="consumer",
        inputs={"source": Input(source, description="The completed source artifact.")},
    )

    assert reads.producers(g, consumer) == [source]
    assert reads.inputs(g, consumer) == {"source": "artifact"}
    [edge] = g.edges(source=consumer.id, target=source.id)
    assert edge.data == {"as": "source", "description": "The completed source artifact."}


def test_attach_rehomes_a_complete_subtree():
    g, root = agent_graph("shepherd")
    subtree = ops.spawn(g, "copied worker")
    nested_action = ops.append(g, subtree, ExecAction(code="launch"))
    nested = ops.spawn(g, "nested", by=nested_action, name="nested")
    ops.append(g, nested, DoneOutput(result="done"))

    action = ops.attach(g, root, subtree, names=["branch0"])

    assert isinstance(action, AppendChild)
    assert reads.frontier(g, root) is action
    assert reads.children(g, root) == [subtree]
    assert reads.lineage(g, nested) == [root, subtree, nested]
    assert reads.path(g, subtree) == "root.branch0"
    assert reads.path(g, nested) == "root.branch0.nested"


def test_attach_launches_several_agents_from_one_action():
    g, root = agent_graph("shepherd")
    one, two = ops.spawn(g, "one"), ops.spawn(g, "two")

    action = ops.attach(g, root, one, two, names=["one", "two"])

    assert reads.children(g, root) == [one, two]
    assert action.launch_agent_ids() == [one.id, two.id]
    assert action.code == (
        "_handles = [\n"
        "    launch_agent(task=''),\n"
        "    launch_agent(task=''),\n"
        "]\n"
        "print(await asyncio.gather(*_handles))"
    )
