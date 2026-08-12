import asyncio

from helpers import StubLLM, counting_replies, first_user

from rlmflow import Flow, start


async def collect(flow, root, **kwargs):
    return [node async for node in flow.run_streaming(root, **kwargs)]


def test_full_stream_yields_every_step_in_order():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    root = start("query")

    events = asyncio.run(collect(flow, root))

    assert [node.type for node in events] == [
        "llm_output",
        "exec_action",
        "done_output",
    ]
    assert root.result() == "ok"


def test_next_stops_before_starting_another_step():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    root = start("query")

    first = asyncio.run(collect(flow, root, until="next"))
    second = asyncio.run(collect(flow, root, until="next"))

    assert [node.type for node in first] == ["llm_output"]
    assert [node.type for node in second] == ["exec_action"]
    assert not root.terminal


def test_idle_heals_an_error_and_stops_at_exec_output():
    flow = Flow(
        StubLLM(
            counting_replies(
                "no repl",
                "```repl\nprint('recovered')\n```",
                "```repl\ndone('ok')\n```",
            )
        )
    )
    root = start("query")

    events = asyncio.run(collect(flow, root, until="idle"))

    assert "error_output" in [node.type for node in events]
    assert events[-1].type == "exec_output"
    assert root.frontier.type == "exec_output"


def test_error_boundary_stops_on_error():
    flow = Flow(StubLLM(lambda _messages: "no repl"))
    root = start("query")

    events = asyncio.run(collect(flow, root, until="error"))

    assert events[-1].type == "error_output"


def test_callable_boundary_sees_child_events():
    def reply(messages):
        if first_user(messages) == "parent":
            return (
                "```repl\n"
                "child = await launch_subagent('child', name='c')\n"
                "value = await child.wait_for_result()\n"
                "done('p:' + value)\n"
                "```"
            )
        return "```repl\ndone('c')\n```"

    flow = Flow(StubLLM(reply))
    root = start("parent", max_depth=1)

    def child_done(node, _root):
        return node.type == "done_output" and node.parent_agent.config.path == "root.c"

    events = asyncio.run(collect(flow, root, until=child_done))

    assert child_done(events[-1], root)
    assert root.sub_agents[0].result() == "c"
    # The boundary landed inside the parent's step, which was cancelled with the
    # stream, so the parent never got to read the answer it was waiting on.
    assert root.result() is None


def test_a_string_query_gets_a_root_of_its_own():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))

    events = asyncio.run(collect(flow, "query", until="next"))

    # A root is where a stream starts rather than something it lands, so the only
    # handle a string caller gets on it is through the node it produced.
    assert [node.type for node in events] == ["llm_output"]
    assert events[0].root.content == "query"
