import asyncio

from rflow import (
    AppendNode,
    DoneOutput,
    ExecAction,
    Flow,
    Graph,
    LLMOutput,
    SupervisingOutput,
    UserQuery,
)
from rflow.consumers.ui import LiveGraphTree, LiveTreeRenderer, render_tree
from rflow.utils.viewer import _topology, _visible_nodes

from helpers import (
    StubLLM,
    first_user,
)


def test_minimal_graph_agent_tree_is_high_level():
    def reply(messages):
        task = first_user(messages)
        if task == "child task":
            return '```repl\ndone("child result")\n```'
        return (
            "```repl\n"
            'results = await launch_subagents([{"name": "auth", "query": "child task"}])\n'
            'done("root saw " + results[0])\n'
            "```"
        )

    flow = Flow(StubLLM(reply), max_depth=1)
    graph = Graph(query="root query")
    flow.run(graph=graph)

    tree = render_tree(graph)
    assert tree.startswith("root query")
    assert "root: done root saw child result" in tree
    assert "root.auth: done child result" in tree
    assert "provider response" not in tree
    assert "tool call:" not in tree


def test_minimal_live_tree_renderer_consumes_events(capsys):
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="root query")

    renderer = LiveTreeRenderer(clear=False)

    async def collect():
        async for event in flow.run_streaming(graph=graph):
            renderer.handle(event, graph)

    asyncio.run(collect())
    output = capsys.readouterr().out
    assert "root:" in output
    assert "done ok" in output


def test_render_tree_shows_children_running_active_total():
    graph = Graph(query="root query")
    graph.append(SupervisingOutput(output="", waiting_on=["root.left", "root.right"]))
    left = Graph(
        agent_id="root.left",
        graph_id=graph.graph_id,
        query="left task",
        depth=1,
        parent_agent_id="root",
    )
    right = Graph(
        agent_id="root.right",
        graph_id=graph.graph_id,
        query="right task",
        depth=1,
        parent_agent_id="root",
    )
    left.append(DoneOutput(result="left result"))
    # right still pending
    graph.children[left.agent_id] = left
    graph.children[right.agent_id] = right

    tree = render_tree(graph)
    assert "children running 1/2" in tree
    assert "root.left: done left result" in tree
    assert "root.right: pending" in tree


def test_render_tree_mid_turn_agent_reads_working_not_pending():
    # A live agent between turns sits on the per-turn UserQuery it just committed
    # while it awaits the model. Once it has taken turns that is "working", not the
    # "pending" reserved for a fresh (0-turn) agent — otherwise a busy branch that
    # has run many turns looks stuck.
    graph = Graph(query="task")
    graph.append(LLMOutput(content="plan", code="pass"))
    graph.append(UserQuery(content="next board"))

    line = render_tree(graph).splitlines()[-1]
    assert "working" in line and "1 turns" in line
    assert "pending" not in line


def test_live_graph_tree_consumer_tracks_forest(capsys):
    left = Graph(query="left query")
    right = Graph(query="right query")
    left.append(DoneOutput(result="L"))
    right.append(SupervisingOutput(output="", waiting_on=["right.child"]))
    child = Graph(
        agent_id="right.child",
        graph_id=right.graph_id,
        query="child",
        depth=1,
        parent_agent_id="right",
    )
    right.children[child.agent_id] = child

    tree = LiveGraphTree(rich=False, clear=False, every_s=0)
    tree.label(left.graph_id, "branch-a")
    tree.label(right.graph_id, "branch-b")
    tree.handle(AppendNode(type="append_node", agent_id=left.agent_id, node=left.nodes[-1]), left)
    tree.handle(
        AppendNode(type="append_node", agent_id=right.agent_id, node=right.nodes[-1]),
        right,
    )
    text = tree.render()
    assert "[branch-a]" in text
    assert "[branch-b]" in text
    assert "children running 1/1" in text
    tree.close()
    output = capsys.readouterr().out
    assert "children running 1/1" in output


def test_viewer_topology_fans_children_out_and_collapses_bookkeeping():
    graph = Graph(query="root query")
    graph.append(ExecAction(code="print('x')"))
    graph.append(SupervisingOutput(output="", waiting_on=["root.left", "root.right"]))
    for name in ("left", "right"):
        child = Graph(
            agent_id=f"root.{name}",
            graph_id=graph.graph_id,
            query=f"{name} task",
            depth=1,
            parent_agent_id="root",
        )
        child.append(DoneOutput(result=f"{name} result"))
        graph.children[child.agent_id] = child

    # Bookkeeping exec_action collapses into its supervising observation.
    assert not any(isinstance(node, ExecAction) for node in _visible_nodes(graph))

    positions, _chain, spawn_edges, _nodes = _topology(graph)
    supervising = next(n for n in graph.nodes if isinstance(n, SupervisingOutput))
    left_first = graph["root.left"].nodes[0].id
    right_first = graph["root.right"].nodes[0].id
    assert (supervising.id, left_first) in spawn_edges
    assert (supervising.id, right_first) in spawn_edges
    # Children fan out to distinct columns below the supervisor.
    assert positions[left_first][0] != positions[right_first][0]
    assert positions[left_first][1] < positions[supervising.id][1]
