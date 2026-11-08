import pytest
from helpers import agent_graph, finish, own, ran

from rlmflow import Graph, Input
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

    assert root.input == {"task": "query", "payload": value}
    assert loaded.input == {"task": "query", "payload": value}
    assert loaded.root_data.arg_descriptions == {}


def test_input_wrapped_arguments_resolve_and_keep_their_description(tmp_path):
    g, root = agent_graph("query", args={"rows": Input([1, 2], description="Parsed rows.")})

    g.save(tmp_path / "run")
    loaded = Graph.load(tmp_path / "run").agent(root.id)

    assert loaded.input == {"task": "query", "rows": [1, 2]}
    assert loaded.root_data.arg_descriptions == {"rows": "Parsed rows."}


def test_an_agent_is_its_root_node_and_owns_an_ordered_history():
    g, agent = agent_graph("query")
    first = g.append(agent, ran("x = 1", "one"))
    second = finish(g, agent, "ok")

    assert g.history(agent) == [agent, first, second]
    assert own(agent) == [first, second]
    assert agent.seq < first.seq < second.seq
    assert first.owner == agent.id and agent.owner == agent.id
    assert g.lineage(agent) == [agent]
    assert g.status(agent) == "done"
    assert g.result(agent) == "ok"


def test_a_child_agent_branches_off_the_node_that_launched_it():
    g, root = agent_graph("parent")
    turn = g.append(root, ran("launch"))
    child = g.add_agent("child", parent=turn, name="worker")

    assert g.parent(child) is root
    assert g.lineage(child) == [root, child]
    assert g.children(root) == [child]
    assert [g.path(agent) for agent in (root, child)] == ["root", "root.worker"]
    # Launching moved neither agent on: the parent still stands at the turn,
    # and the child has only its root.
    assert g.tail(root) is turn
    assert g.history(child) == [child]


def test_fork_after_a_launch_shares_its_children_instead_of_copying_them():
    g, root = agent_graph("parent")
    turn = g.append(root, ran("launch"))
    child = g.add_agent("child", parent=turn, name="child")
    calls.record(turn, calls.LAUNCH, "k", agent=child.id)
    finish(g, child, "done")
    later = g.append(root, ran("print(1)", "1"))

    after = g.fork(later)
    before = g.fork(root)

    assert turn in g.history(after)
    assert calls.launched(turn) == [child.id]
    assert len(g.agents()) == 4  # root, child, and two forks
    assert g.children(after) == [] and g.children(root) == [child]
    assert turn not in g.history(before)
    assert g.history(before)[0] is root


def test_status_follows_the_last_turn_and_what_arrived_after_it():
    g, root = agent_graph("q")
    assert g.status(root) == "running"
    finish(g, root, "first")
    assert g.status(root) == "done"

    g.send(root, "one more thing", n=2)
    assert g.status(root) == "running"
    assert g.result(root) is None
    finish(g, root, "second")
    assert (g.status(root), g.result(root)) == ("done", "second")
