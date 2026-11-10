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
    assert loaded.arg_descriptions == {}


def test_input_wrapped_arguments_resolve_and_keep_their_description(tmp_path):
    g, root = agent_graph("query", args={"rows": Input([1, 2], description="Parsed rows.")})

    g.save(tmp_path / "run")
    loaded = Graph.load(tmp_path / "run").agent(root.id)

    assert loaded.input == {"task": "query", "rows": [1, 2]}
    assert loaded.arg_descriptions == {"rows": "Parsed rows."}


def test_an_agent_is_its_first_node_and_owns_its_nodes_in_order():
    g, agent = agent_graph("query")
    first = g.append(agent, ran("x = 1", "one"))
    second = finish(g, agent, "ok")

    assert agent.nodes() == [agent, first, second]
    assert own(agent) == [first, second]
    assert agent.seq < first.seq < second.seq
    assert first.owner == agent.id and agent.owner == agent.id
    assert agent.lineage() == [agent]
    assert agent.status() == "done"
    assert agent.final_result() == "ok"


def test_a_child_agent_branches_off_the_node_that_launched_it():
    g, root = agent_graph("parent")
    turn = g.append(root, ran("launch"))
    child = g.start("child", parent=turn, name="worker")

    assert child.parent() is root
    assert child.lineage() == [root, child]
    assert root.children() == [child]
    assert [agent.path() for agent in (root, child)] == ["root", "root.worker"]
    # Launching moved neither agent on: the parent still stands at the turn,
    # and the child has only its root.
    assert root.nodes()[-1] is turn
    assert child.nodes() == [child]


def test_fork_after_a_launch_copies_its_children_and_before_it_drops_them():
    g, root = agent_graph("parent")
    turn = g.append(root, ran("launch"))
    child = g.start("child", parent=turn, name="child")
    calls.record(turn, calls.LAUNCH, "k", agent=child.id)
    finish(g, child, "done")
    later = g.append(root, ran("print(1)", "1"))

    after = g.fork(later)
    before = g.fork(root)

    assert [a.id for a in after.agents()] == [root.id, child.id]
    assert after.agent(child.id).final_result() == "done"
    assert calls.launched(after.node(turn.id)) == [child.id]
    assert [a.id for a in before.agents()] == [root.id] and turn.id not in before
    assert len(g.agents()) == 2 and root.children() == [child]


def test_status_follows_the_last_turn_and_what_arrived_after_it():
    g, root = agent_graph("q")
    assert root.status() == "running"
    finish(g, root, "first")
    assert root.status() == "done"

    g.send(root, "one more thing", n=2)
    assert root.status() == "running"
    assert root.final_result() is None
    finish(g, root, "second")
    assert (root.status(), root.final_result()) == ("done", "second")


def test_a_finished_agent_takes_another_done_only_after_something_reopens_it():
    from rlmflow.graph.graph import GraphError

    g = Graph()
    root = g.start("count")
    finish(g, root, 1)
    with pytest.raises(GraphError, match="already finished"):
        finish(g, root, 2)
    g.send(root, "count again")
    finish(g, root, 2)
    assert root.final_result() == 2
