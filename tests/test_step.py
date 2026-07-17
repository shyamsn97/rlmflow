"""Async stepping: global (next/idle) steps, bounded boundaries, reactive control.

These pin the behaviour the old streaming stepper lacked: a step actually *halts*
the run at a boundary (the background driver does not race ahead), and edits made
between steps are seen by the run when it resumes.
"""

import asyncio

from rflow import Flow, Graph

from helpers import StubLLM, counting_replies, first_user


async def collect(flow, **kwargs):
    return [event async for event in flow.run_streaming(**kwargs)]


def test_step_node_boundary_does_not_race_ahead_of_the_graph():
    reply = counting_replies(
        "```repl\nprint('a')\n```",
        "```repl\ndone('ok')\n```",
    )
    flow = Flow(StubLLM(reply))
    graph = Graph(query="q")

    async def main():
        return await collect(flow, graph=graph, until="next")

    events = asyncio.run(main())
    appended = [e for e in events if e.type == "append_node"]
    assert len(appended) == 1
    # The run halted exactly at the boundary: the graph had the seeded query,
    # then this step added exactly one node.
    assert [node.type for node in graph.nodes] == ["user_query", "llm_output"]


def test_step_next_advances_frontier_one_node_per_step():
    reply = counting_replies(
        "```repl\nprint('a')\n```",
        "```repl\ndone('ok')\n```",
    )
    flow = Flow(StubLLM(reply))
    graph = Graph(query="q")

    async def main():
        prev = len(graph.nodes)
        steps = 0
        while not graph.finished and steps < 20:
            events = await collect(flow, graph=graph, until="next")
            appended = [e for e in events if e.type == "append_node"]
            # Graph grew by exactly the nodes handed back — no runaway driver.
            assert len(graph.nodes) == prev + len(appended)
            prev = len(graph.nodes)
            steps += 1

    asyncio.run(main())
    assert graph.result() == "ok"


def test_step_idle_heals_error_and_settles_at_clean_output():
    reply = counting_replies(
        "no repl here",                       # -> error_output (not a rest point)
        "```repl\nprint('recovered')\n```",   # -> exec_output (rest)
        "```repl\ndone('ok')\n```",
    )
    flow = Flow(StubLLM(reply))
    graph = Graph(query="q")

    async def main():
        return await collect(flow, graph=graph, until="idle")

    events = asyncio.run(main())
    types = [e.node_type for e in events if e.type == "append_node"]
    # A single idle step blew past the error and healed to a clean exec_output.
    assert "error_output" in types
    assert types[-1] == "exec_output"
    assert graph.nodes[-1].type == "exec_output"


def test_step_next_surfaces_error_without_healing():
    reply = counting_replies(
        "no repl here",
        "```repl\ndone('ok')\n```",
    )
    flow = Flow(StubLLM(reply))
    graph = Graph(query="q")

    async def main():
        halted_on_error = False
        steps = 0
        while not graph.finished and steps < 20:
            await collect(flow, graph=graph, until="next")
            if graph.nodes and graph.nodes[-1].type == "error_output":
                halted_on_error = True
            steps += 1
        return halted_on_error

    halted_on_error = asyncio.run(main())
    # next stops *on* the error (surfacing it) rather than healing past it.
    assert halted_on_error
    assert graph.result() == "ok"


def test_step_idle_advances_parent_and_children_to_completion():
    def reply(messages):
        if first_user(messages) == "parent":
            return (
                "```repl\n"
                "results = await launch_subagents([{'name': 'c', 'query': 'child'}])\n"
                "done('p:' + results[0])\n"
                "```"
            )
        return "```repl\ndone('c')\n```"

    flow = Flow(StubLLM(reply), max_depth=1)
    graph = Graph(query="parent")

    async def main():
        steps = 0
        while not graph.finished and steps < 30:
            await collect(flow, graph=graph, until="idle")
            steps += 1

    asyncio.run(main())
    assert graph.result() == "p:c"
    assert graph["root.c"].result() == "c"


def test_step_delegation_supervising_and_child_boundaries_under_concurrency():
    def reply(messages):
        task = first_user(messages)
        if task == "parent":
            return (
                "```repl\n"
                "results = await launch_subagents(["
                "{'name': 'slow', 'query': 'child'}, "
                "{'name': 'fast', 'query': 'child'}])\n"
                "done(','.join(results))\n"
                "```"
            )
        return "```repl\ndone('c')\n```"

    flow = Flow(StubLLM(reply), max_depth=1, max_iters=8, workers=2)
    graph = Graph(query="parent")

    def first_child_done(event, _graph):
        return (
            event.type == "append_node"
            and event.agent_id != "root"
            and event.node_type == "done_output"
        )

    async def main():
        await collect(flow, graph=graph, until="next")
        supervising = await collect(flow, graph=graph, until="supervising")
        assert any(
            event.type == "append_node" and event.node_type == "supervising_output"
            for event in supervising
        )
        assert any(
            child.result() == "c" for child in graph.children.values()
        )
        assert any(first_child_done(event, graph) for event in supervising)

    asyncio.run(asyncio.wait_for(main(), timeout=10))
    assert graph.result() == "c,c"


def test_step_then_inject_is_seen_by_the_resumed_run():
    def reply(messages):
        users = [m["content"] for m in messages if m["role"] == "user"]
        if any("STOP NOW" in u for u in users):
            return '```repl\ndone("stopped")\n```'
        return "```repl\nprint('thinking')\n```"

    flow = Flow(StubLLM(reply), max_iters=10)
    graph = Graph(query="work")

    async def main():
        await collect(flow, graph=graph, until="next")  # llm_output
        await collect(flow, graph=graph, until="idle")  # settle at a clean exec_output
        assert not graph.finished
        # Inject a control instruction between steps; the paused run must see it.
        graph.inject("STOP NOW")
        await collect(flow, graph=graph, until="done")
        return graph.result()

    assert asyncio.run(main()) == "stopped"
