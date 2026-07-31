import asyncio

from helpers import StubLLM, counting_replies, first_user

from rlmflow import Flow, start, tool


def test_run_executes_model_code_until_done():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    root = start("query")

    assert flow.run(root) == "ok"
    assert [node.type for node in root.steps] == [
        "llm_output",
        "exec_action",
        "done_output",
    ]


def test_repl_state_survives_between_turns():
    replies = counting_replies(
        "```repl\nvalue = 41\nprint('stored')\n```",
        "```repl\ndone(str(value + 1))\n```",
    )
    flow = Flow(StubLLM(replies))

    assert flow.run(start("query")) == "42"


def test_execution_errors_are_visible_to_the_next_model_turn():
    def reply(messages):
        if any("NameError" in message["content"] for message in messages):
            return "```repl\ndone('recovered')\n```"
        return "```repl\nmissing_name\n```"

    flow = Flow(StubLLM(reply))

    assert flow.run(start("query")) == "recovered"


def test_custom_tools_are_seeded_into_the_repl():
    @tool("Return a stable value.")
    def answer():
        return "tool-result"

    flow = Flow(
        StubLLM(lambda _messages: "```repl\ndone(answer())\n```"),
        tools=[answer],
    )

    assert flow.run(start("query")) == "tool-result"


def test_structured_child_results_return_live_python_values():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }

    def reply(messages):
        if first_user(messages) == "parent":
            return (
                "```repl\n"
                f"r = await launch_subagents([{{'name':'c','query':'child',"
                f"'output_schema':{schema!r}}}])\n"
                "done(str(r[0]['answer']))\n"
                "```"
            )
        return "```repl\ndone({'answer': 3})\n```"

    flow = Flow(StubLLM(reply), max_depth=1)
    root = start("parent", max_depth=1)

    assert flow.run(root) == "3"


def test_aclose_cancels_tracked_work_and_closes_repls():
    async def main():
        flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
        root = start("query")
        await flow.arun(root)
        assert flow.repls
        await flow.aclose()
        assert not flow.tasks.tasks
        assert not flow.repls

    asyncio.run(main())
