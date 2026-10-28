import pytest
from helpers import agent_graph

from rlmflow import CODE, DONE, Graph, Input, Node


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
    first = g.append(agent, Node(CODE, code="", output="one"))
    second = g.append(agent, Node(DONE, result="ok"))

    assert g.history(agent) == [agent, first, second]
    assert agent.seq < first.seq < second.seq
    assert g.frontier(agent) is second
    assert first.agent_id == agent.id
    assert g.lineage(agent) == [agent]
    assert g.terminal(agent)
    assert g.result(agent) == "ok"


def test_a_child_agent_branches_off_the_node_that_launched_it():
    g, root = agent_graph("parent")
    cell = g.append(root, Node(CODE, code="launch"))
    child = g.spawn("child", by=cell, name="worker")

    assert g.launcher(child) is root
    assert g.lineage(child) == [root, child]
    assert g.children(root) == [child]
    assert [g.path(agent) for agent in (root, child)] == ["root", "root.worker"]
    # Launching moved neither agent on: the parent still stands at the cell,
    # and the child has nothing but its own start.
    assert g.frontier(root) is cell
    assert g.history(child) == [child]


def test_fork_after_a_launch_shares_its_children_instead_of_copying_them():
    g, root = agent_graph("parent")
    cell = g.append(root, Node(CODE, code="launch"))
    child = g.spawn("child", by=cell, name="child")
    cell.record_call("launch_agent", 0, {"$rlmflow_agent": child.id})
    g.append(child, Node(DONE, result="done"))
    later = g.append(root, Node(CODE, code="print(1)", output="1"))

    after = g.fork(later)
    before = g.fork(root)

    assert cell in g.history(after)
    assert cell.launched() == [child.id]
    assert len(g.agents()) == 4  # root, child, and two empty forks
    assert cell not in g.history(before)


def test_dependency_edges_name_the_producer_and_its_description():
    g, root = agent_graph("parent")
    cell = g.append(root, Node(CODE, code="launch"))
    source = g.spawn("source", by=cell, name="source")
    g.append(source, Node(DONE, result="artifact"))
    consumer = g.spawn(
        "consumer",
        by=cell,
        name="consumer",
        inputs={"source": Input(source, description="The completed source artifact.")},
    )

    assert g.producers(consumer) == [source]
    assert g.inputs(consumer) == {"source": "artifact"}
    [edge] = g.edges(source=consumer.id, target=source.id)
    assert edge.data == {"as": "source", "description": "The completed source artifact."}


def test_attach_rehomes_a_complete_subtree():
    g, root = agent_graph("shepherd")
    subtree = g.spawn("copied worker", name="branch0")
    nested_cell = g.append(subtree, Node(CODE, code="launch"))
    nested = g.spawn("nested", by=nested_cell, name="nested")
    g.append(nested, Node(DONE, result="done"))

    message = g.attach(root, subtree)

    assert g.inbox(root) == [message]
    assert g.children(root) == [subtree]
    assert g.lineage(nested) == [root, subtree, nested]
    assert g.path(subtree) == "root.branch0"
    assert g.path(nested) == "root.branch0.nested"


def test_attach_adds_several_agents_with_one_message():
    g, root = agent_graph("shepherd")
    one, two = g.spawn("one", name="one"), g.spawn("two", name="two")

    message = g.attach(root, one, two)

    assert g.children(root) == [one, two]
    assert [e.source for e in g.edges(kind="from", target=message.id)] == [one.id, two.id]
