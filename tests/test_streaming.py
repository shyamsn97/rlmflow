"""Public streaming API semantics: event order, boundaries, and resume."""

import asyncio

import pytest

from rlmflow import Flow, Graph

from helpers import StubLLM, counting_replies, first_user


async def collect(flow: Flow, **kwargs):
    return [event async for event in flow.run_streaming(**kwargs)]


def node_events(events):
    return [event for event in events if event.type == "append_node"]


def node_types(events):
    return [event.node_type for event in node_events(events)]


def test_run_streaming_full_run_emits_graph_created_and_all_nodes_in_order():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))

    events = asyncio.run(collect(flow, query="q"))

    graph = events[0].graph
    assert events[0].type == "graph_created"
    assert node_types(events) == [
        "llm_output",
        "exec_action",
        "done_output",
    ]
    assert [node.type for node in graph.nodes] == ["user_query", *node_types(events)]
    assert all(
        event.graph_id == graph.graph_id
        for event in events
        if event.type != "graph_created"
    )
    assert graph.result() == "ok"


def test_run_streaming_next_n_advances_exactly_n_global_node_steps():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")

    first = asyncio.run(collect(flow, graph=graph, until="next", n=2))

    assert node_types(first) == ["llm_output", "exec_action"]
    assert [node.type for node in graph.nodes] == ["user_query", *node_types(first)]
    assert not graph.finished

    rest = asyncio.run(collect(flow, graph=graph, until="done"))
    assert node_types(rest) == ["done_output"]
    assert graph.result() == "ok"


def test_run_streaming_idle_heals_error_but_next_surfaces_it():
    reply = counting_replies(
        "no repl block",                     # -> error_output
        "```repl\nprint('fixed')\n```",      # -> exec_output
        "```repl\ndone('ok')\n```",
    )
    flow = Flow(StubLLM(reply))
    graph = Graph(query="q")

    while not graph.nodes or graph.nodes[-1].type != "error_output":
        asyncio.run(collect(flow, graph=graph, until="next"))

    assert graph.nodes[-1].type == "error_output"

    healed = asyncio.run(collect(flow, graph=graph, until="idle"))
    assert "error_output" not in node_types(healed)
    assert node_types(healed)[-1] == "exec_output"
    assert graph.nodes[-1].type == "exec_output"


def test_run_streaming_callable_boundary_observes_child_event():
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

    def child_done(event, graph):
        return (
            event.type == "append_node"
            and event.agent_id != graph.agent_id
            and event.node_type == "done_output"
        )

    events = asyncio.run(collect(flow, graph=graph, until=child_done))

    assert any(child_done(event, graph) for event in events)
    assert graph["root.c"].result() == "c"
    assert graph.result() == "p:c"


def test_run_streaming_resume_after_injection_reads_graph_edit():
    def reply(messages):
        text = "\n".join(message["content"] for message in messages)
        if "STOP NOW" in text:
            return '```repl\ndone("stopped")\n```'
        return "```repl\nprint('thinking')\n```"

    flow = Flow(StubLLM(reply), max_iters=8)
    graph = Graph(query="work")

    asyncio.run(collect(flow, graph=graph, until="idle"))
    assert not graph.finished

    graph.inject("STOP NOW")
    asyncio.run(collect(flow, graph=graph, until="done"))

    assert graph.result() == "stopped"


def test_run_streaming_drives_two_graphs_concurrently_on_one_flow():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    a = Graph(query="a")
    b = Graph(query="b")

    async def main():
        # Interleave two independent runs on the same Flow. Each keeps its own
        # scheduler/until, so stepping one does not advance the other.
        stream_a = flow.run_streaming(graph=a, until="next")
        stream_b = flow.run_streaming(graph=b, until="next")

        # Streams are lazy: b's run does not exist until b is iterated.
        first_a = await stream_a.__anext__()
        assert set(flow.runs) == {a.graph_id}
        assert [node.type for node in b.nodes] == ["user_query"]

        # Both runs now coexist on the one Flow with independent schedulers.
        first_b = await stream_b.__anext__()
        assert set(flow.runs) == {a.graph_id, b.graph_id}

        events_a = [first_a, *[event async for event in stream_a]]
        events_b = [first_b, *[event async for event in stream_b]]
        return events_a, events_b

    events_a, events_b = asyncio.run(main())

    assert node_types(events_a)[0] == "llm_output"
    assert node_types(events_b)[0] == "llm_output"
    assert not a.finished and not b.finished

    asyncio.run(collect(flow, graph=a, until="done"))
    asyncio.run(collect(flow, graph=b, until="done"))

    assert a.result() == "ok"
    assert b.result() == "ok"
    assert flow.runs == {}


def test_run_streaming_rejects_two_consumers_on_the_same_graph():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")

    async def main():
        first = flow.run_streaming(graph=graph, until="next")
        await first.__anext__()  # first consumer now owns the run

        second = flow.run_streaming(graph=graph, until="next")
        with pytest.raises(RuntimeError, match="already being streamed"):
            await second.__anext__()

        # Draining the first consumer releases the run for a later call.
        async for _event in first:
            pass
        assert not flow.runs[graph.graph_id].streaming

    asyncio.run(main())


def test_run_streaming_parallel_runs_each_delegate_children():
    def reply(messages):
        user = first_user(messages)
        if user in ("A", "B"):
            return (
                "```repl\n"
                "results = await launch_subagents([{'name': 'c', 'query': 'child'}])\n"
                f"done({user!r} + ':' + results[0])\n"
                "```"
            )
        return "```repl\ndone('c')\n```"

    flow = Flow(StubLLM(reply), max_depth=1, workers=2)
    a = Graph(query="A")
    b = Graph(query="B")

    async def main():
        await asyncio.gather(collect(flow, graph=a), collect(flow, graph=b))

    asyncio.run(main())

    assert a.result() == "A:c"
    assert b.result() == "B:c"
    assert a["root.c"].result() == "c"
    assert b["root.c"].result() == "c"
    assert flow.runs == {}


def test_run_streaming_query_appends_new_turn_and_redrives_finished_graph():
    def reply(messages):
        last_user = [m["content"] for m in messages if m["role"] == "user"][-1]
        return f'```repl\ndone({last_user!r})\n```'

    flow = Flow(StubLLM(reply))
    graph = Graph(query="one")

    asyncio.run(collect(flow, graph=graph))
    assert graph.finished and graph.result() == "one"

    # A new turn on the finished graph re-drives the same trajectory. The turn
    # itself is emitted as a user_query append so consumers/checkpointers see it.
    events = asyncio.run(collect(flow, graph=graph, query="two"))
    appended = [event for event in node_events(events) if event.node_type == "user_query"]
    assert [event.node.content for event in appended] == ["two"]
    assert node_types(events)[-1] == "done_output"
    assert graph.result() == "two"
    assert [node.type for node in graph.nodes].count("user_query") == 2


def test_run_streaming_resume_with_inputs_syncs_repl_inputs_namespace():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone(INPUTS["x"])\n```'))
    graph = Graph(query="q", inputs={"x": "first"})

    asyncio.run(collect(flow, graph=graph))
    assert graph.result() == "first"
    # REPL is kept warm across turns (default close_repls=False).
    assert any(key[0] == graph.graph_id for key in flow.repls)

    # Resuming with new inputs re-injects INPUTS into the warm REPL so the next
    # turn observes the updated values.
    asyncio.run(collect(flow, graph=graph, query="again", inputs={"x": "second"}))
    assert graph.result() == "second"


def test_run_streaming_merge_inputs_false_replaces_existing_inputs():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone(sorted(INPUTS))\n```'))
    graph = Graph(query="q", inputs={"a": "1", "b": "2"})

    asyncio.run(collect(flow, graph=graph))
    assert graph.result() == "['a', 'b']"

    # Default merges into existing inputs.
    asyncio.run(collect(flow, graph=graph, query="t2", inputs={"c": "3"}))
    assert graph.inputs == {"a": "1", "b": "2", "c": "3"}

    # merge_inputs=False replaces the whole dict.
    asyncio.run(
        collect(flow, graph=graph, query="t3", inputs={"d": "4"}, merge_inputs=False)
    )
    assert graph.inputs == {"d": "4"}


def test_run_streaming_close_repls_controls_repl_teardown():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))

    keep = Graph(query="keep")
    asyncio.run(collect(flow, graph=keep))
    assert any(key[0] == keep.graph_id for key in flow.repls)

    drop = Graph(query="drop")
    asyncio.run(collect(flow, graph=drop, close_repls=True))
    assert not any(key[0] == drop.graph_id for key in flow.repls)
