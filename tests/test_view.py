import asyncio

from rflow import (
    Flow,
    Graph,
)
from rflow.consumers.ui import LiveTreeRenderer, render_tree

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

