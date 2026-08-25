import pytest
from helpers import StubLLM

from rlmflow import Flow, format_tool_line, start, tool


def test_builtin_names_are_reserved():
    flow = Flow(StubLLM(lambda _messages: "unused"))

    for name in ("finish", "launch_subagent", "asyncio", "INPUTS", "ENV"):
        with pytest.raises(ValueError, match="reserved"):
            flow.inject(name, object())


def test_builtins_override_only_their_reserved_names():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    flow.tools["finish"] = "wrong"
    flow.tools["done"] = "custom"
    root = start("query")

    namespace = flow.build_tools(root)

    assert callable(namespace["finish"])
    assert namespace["done"] == "custom"
    assert callable(namespace["launch_subagent"])
    assert namespace["asyncio"].gather
    assert namespace["INPUTS"] == {}


def test_tool_metadata_is_available_for_prompt_rendering():
    @tool("Describe this tool.")
    def described(value: str) -> str:
        return value

    assert format_tool_line(described) == "- `described(value: str) -> str`: Describe this tool."
