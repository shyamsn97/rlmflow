"""The graph figure's layout."""

from helpers import finish, ran

from rlmflow import RETURN, AgentConfig, Flow, Graph, Node
from rlmflow.graph import calls
from rlmflow.view.figure import graph_svg, handoffs, parent_ids, tidy_tree
from rlmflow.view.timeline import timeline


def test_an_agents_chain_is_one_column_with_launches_and_forks_to_its_right():
    g = Graph()
    root = g.add_agent("solve")
    reply = g.append(root, Node("assistant", "plan", seen=1, model="m"))
    turn = g.append(root, ran("x = 1"))
    kids = [g.add_agent(f"part {i}", parent=turn) for i in range(3)]
    for kid in kids:
        finish(g, kid, 1)
    fork = g.fork(reply, name="alt")
    finish(g, fork, 2)
    done = finish(g, root, 3)

    pos = tidy_tree(timeline(g), parent_ids(g))

    column = {pos[item.id][0] for item in (root, reply, turn, done)}
    assert len(column) == 1
    [x] = column
    assert all(pos[agent.id][0] > x for agent in (*kids, fork))
    assert pos[done.id][1] > pos[turn.id][1] > pos[reply.id][1] > pos[root.id][1]


def test_a_launch_from_the_agent_itself_keeps_the_chain_under_the_agent():
    g = Graph()
    root = g.add_agent("lead")
    kids = [g.add_agent(f"part {i}", parent=root) for i in range(4)]
    turn = g.append(root, ran("launch"))
    done = finish(g, root, "ok")

    pos = tidy_tree(timeline(g), parent_ids(g))

    assert pos[root.id][0] == pos[turn.id][0] == pos[done.id][0]
    assert all(pos[kid.id][0] > pos[root.id][0] for kid in kids)


def test_results_and_messages_coming_back_are_drawn_from_their_source():
    g = Graph()
    root = g.add_agent("lead")
    root.root_data.bind_config(AgentConfig())
    launch = g.append(root, ran("launch"))
    kids = [g.add_agent(f"part {i}", parent=launch) for i in range(2)]
    for kid in kids:
        kid.root_data.bind_config(AgentConfig())
    note_turn = g.append(kids[1], ran("send"))
    note = g.send(root, "heads up", by=note_turn)
    dones = [finish(g, kid, i) for i, kid in enumerate(kids)]
    calls.record(launch, calls.AWAIT, "k", agent=kids[0].id, done=dones[0].id)
    g.link(RETURN, dones[0], launch)
    results = Flow().deliver_results(g, root)

    assert sorted(handoffs(g)) == sorted(
        [(dones[0].id, launch.id), (note_turn.id, note.id), (dones[1].id, results.id)]
    )
    assert graph_svg(g).count('stroke-dasharray="3 3"') == 3
