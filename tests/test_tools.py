import pytest
from helpers import StubLLM

from rlmflow import Flow, format_tool_line, start, tool


def test_registered_tool_runs_inside_agent_repl():
    @tool("Return a value.")
    def custom():
        return "custom"

    flow = Flow(
        StubLLM(lambda _messages: "```repl\ndone(custom())\n```"),
        tools=[custom],
    )

    assert flow.run(start("query")) == "custom"


def test_builtin_names_are_reserved():
    flow = Flow(StubLLM(lambda _messages: "unused"))

    with pytest.raises(ValueError, match="reserved"):
        flow.inject("done", object())
    with pytest.raises(ValueError, match="reserved"):
        flow.inject("launch_subagents", object())
    with pytest.raises(ValueError, match="reserved"):
        flow.inject("INPUTS", object())


def test_builtins_override_custom_dictionary_values():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    flow.tools["done"] = "wrong"
    agent = start("query")

    namespace = flow.build_tools(agent)

    assert callable(namespace["done"])
    assert callable(namespace["launch_subagents"])
    assert namespace["INPUTS"] == {}


def test_tool_metadata_is_available_for_prompt_rendering():
    @tool("Describe this tool.")
    def described(value: str) -> str:
        return value

    line = format_tool_line(described)

    assert "described" in line
    assert "Describe this tool." in line
