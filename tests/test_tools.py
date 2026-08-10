import pytest
from helpers import StubLLM

from rlmflow import Flow, format_tool_line, start, tool


def test_builtin_names_are_reserved():
    flow = Flow(StubLLM(lambda _messages: "unused"))

    for name in ("finish", "done", "launch_subagents", "Subagent", "INPUTS", "ENV"):
        with pytest.raises(ValueError, match="reserved"):
            flow.inject(name, object())


def test_builtins_override_custom_dictionary_values():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    flow.tools["finish"] = "wrong"
    flow.tools["done"] = "wrong"
    root = start("query")

    namespace = flow.build_tools(root)

    assert callable(namespace["finish"])
    assert callable(namespace["done"])
    assert namespace["done"] is namespace["finish"]
    assert callable(namespace["launch_subagents"])
    assert namespace["INPUTS"] == {}


def test_tool_metadata_is_available_for_prompt_rendering():
    @tool("Describe this tool.")
    def described(value: str) -> str:
        return value

    assert format_tool_line(described) == "- `described(value: str) -> str`: Describe this tool."
