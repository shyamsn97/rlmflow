"""The graph figure's layout."""

from rlmflow import CODE, DONE, LLM, AgentConfig, Flow, Graph, Node
from rlmflow.graph import calls
from rlmflow.view.figure import graph_svg, handoffs, parent_ids, tidy_tree
from rlmflow.view.timeline import timeline


def test_an_agents_chain_is_one_column_with_launches_and_forks_to_its_right():
    g = Graph()
    root = g.add_agent("solve")
    reply = g.append(root, Node(LLM, text="plan"))
    cell = g.append(root, Node(CODE, code="x = 1"))
    kids = [g.add_agent(f"part {i}", parent=cell) for i in range(3)]
    for kid in kids:
        g.append(kid, Node(DONE, result=1))
    fork = g.fork(reply, name="alt")
    g.append(fork, Node(DONE, result=2))
    done = g.append(root, Node(DONE, result=3))

    pos = tidy_tree(timeline(g), parent_ids(g))

    column = {pos[item.id][0] for item in (root, reply, cell, done)}
    assert len(column) == 1
    [x] = column
    assert all(pos[agent.id][0] > x for agent in (*kids, fork))
    assert pos[done.id][1] > pos[cell.id][1] > pos[reply.id][1] > pos[root.id][1]


def test_a_launch_from_the_agent_itself_keeps_the_chain_under_the_agent():
    g = Graph()
    root = g.add_agent("lead")
    kids = [g.add_agent(f"part {i}", parent=root) for i in range(4)]
    cell = g.append(root, Node(CODE, code="launch"))
    done = g.append(root, Node(DONE, result="ok"))

    pos = tidy_tree(timeline(g), parent_ids(g))

    assert pos[root.id][0] == pos[cell.id][0] == pos[done.id][0]
    assert all(pos[kid.id][0] > pos[root.id][0] for kid in kids)


def test_results_and_messages_coming_back_are_drawn_from_their_source():
    g = Graph()
    root = g.add_agent("lead")
    root.bind_config(AgentConfig())
    launch = g.append(root, Node(CODE, code="launch"))
    kids = [g.add_agent(f"part {i}", parent=launch) for i in range(2)]
    note_cell = g.append(kids[1], Node(CODE, code="send"))
    note = g.send(root, "heads up", by=note_cell)
    dones = [g.append(kid, Node(DONE, result=i)) for i, kid in enumerate(kids)]
    calls.record(launch, calls.AWAIT, 0, agent=kids[0].id, done=dones[0].id)
    g.link_return(dones[0], launch)
    flow = Flow()
    results = flow.deliver_results(g, root)
    g.deliver(root)

    assert sorted(handoffs(g)) == sorted(
        [(dones[0].id, launch.id), (note_cell.id, note.id), (dones[1].id, results.id)]
    )
    assert graph_svg(g).count('stroke-dasharray="3 3"') == 3
