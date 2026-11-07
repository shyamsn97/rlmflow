import pytest
from helpers import agent_graph

from rlmflow import CODE, DONE, Graph, Input, Node
from rlmflow.graph import calls


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
    loaded = Graph.load(tmp_path / "run").agent(root.id)

    assert root.resolved_args() == {"payload": value}
    assert loaded.resolved_args() == {"payload": value}
    assert loaded.argument_descriptions() == {}


def test_input_wrapped_arguments_resolve_and_keep_their_description(tmp_path):
    g, root = agent_graph("query", args={"rows": Input([1, 2], description="Parsed rows.")})

    g.save(tmp_path / "run")
    loaded = Graph.load(tmp_path / "run").agent(root.id)

    assert loaded.resolved_args() == {"rows": [1, 2]}
    assert loaded.argument_descriptions() == {"rows": "Parsed rows."}


def test_agent_owns_an_ordered_history():
    g, agent = agent_graph("query")
    first = g.append(agent, Node(CODE, code="", output="one"))
    second = g.append(agent, Node(DONE, result="ok"))

    assert g.history(agent) == [first, second]
    assert agent.nodes == [first, second]
    assert agent.seq < first.seq < second.seq
    assert g.history(agent)[-1] is second
    assert first.agent_id == agent.id
    assert g.lineage(agent) == [agent]
    assert g.status(agent) == "done"
    assert g.result(agent) == "ok"


def test_a_child_agent_branches_off_the_node_that_launched_it():
    g, root = agent_graph("parent")
    cell = g.append(root, Node(CODE, code="launch"))
    child = g.add_agent("child", parent=cell, name="worker")

    assert g.parent(child) is root
    assert g.lineage(child) == [root, child]
    assert g.children(root) == [child]
    assert [g.path(agent) for agent in (root, child)] == ["root", "root.worker"]
    # Launching moved neither agent on: the parent still stands at the cell,
    # and the child has no history yet.
    assert g.history(root)[-1] is cell
    assert g.history(child) == []


def test_fork_after_a_launch_shares_its_children_instead_of_copying_them():
    g, root = agent_graph("parent")
    cell = g.append(root, Node(CODE, code="launch"))
    child = g.add_agent("child", parent=cell, name="child")
    calls.record(cell, calls.LAUNCH, 0, agent=child.id)
    g.append(child, Node(DONE, result="done"))
    later = g.append(root, Node(CODE, code="print(1)", output="1"))

    after = g.fork(later)
    before = g.fork(root)

    assert cell in g.history(after)
    assert calls.launched(cell) == [child.id]
    assert len(g.agents()) == 4  # root, child, and two empty forks
    assert g.children(after) == [] and g.children(root) == [child]
    assert cell not in g.history(before)
