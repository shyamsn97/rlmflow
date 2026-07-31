import asyncio

import pytest
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
    assert not root.finished()


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
    assert root.tail().type == "exec_output"


def test_error_boundary_stops_on_error():
    flow = Flow(StubLLM(lambda _messages: "no repl"))
    root = start("query")

    events = asyncio.run(collect(flow, root, until="error"))

    assert events[-1].type == "error_output"


def test_callable_boundary_sees_child_events_but_parent_step_settles():
    def reply(messages):
        if first_user(messages) == "parent":
            return (
                "```repl\n"
                "values = await launch_subagents([{'name': 'c', 'query': 'child'}])\n"
                "done('p:' + values[0])\n"
                "```"
            )
        return "```repl\ndone('c')\n```"

    flow = Flow(StubLLM(reply), max_depth=1)
    root = start("parent", max_depth=1)

    def child_done(node, _root):
        return node.agent_id == "root.c" and node.type == "done_output"

    events = asyncio.run(collect(flow, root, until=child_done))

    assert any(child_done(node, root) for node in events)
    assert root.result() == "p:c"
    assert root.find_agent("root.c").result() == "c"


def test_string_stream_yields_the_new_root_first():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))

    events = asyncio.run(collect(flow, "query", until="next"))

    assert [node.type for node in events] == ["agent_start"]


def test_same_agent_cannot_be_streamed_twice():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    root = start("query")

    async def main():
        first = flow.run_streaming(root, until="next")
        await anext(first)
        second = flow.run_streaming(root, until="next")
        with pytest.raises(RuntimeError, match="already streaming"):
            await anext(second)
        await first.aclose()

    asyncio.run(main())
