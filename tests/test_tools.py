import pytest
from helpers import StubLLM

from rlmflow import FILE_TOOLS, FileTools, Flow, format_tool_line, start, tool, toolset
from rlmflow.graph.nodes import ExecOutput


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


def test_a_toolset_flattens_methods_and_not_the_class():
    @toolset("Probe")
    class Probe:
        @tool("Return the marker.")
        def mark(self, value: str) -> str:
            return f"marked:{value}"

    flow = Flow(StubLLM(lambda _messages: "unused"), tools=Probe())
    root = start("query")
    namespace = flow.build_tools(root)

    assert "Probe" not in namespace
    assert namespace["mark"]("x") == "marked:x"


def test_two_toolsets_cannot_share_a_name():
    @toolset("A")
    class Alpha:
        @tool("one")
        def ping(self) -> str:
            return "a"

    @toolset("B")
    class Beta:
        @tool("two")
        def ping(self) -> str:
            return "b"

    flow = Flow(StubLLM(lambda _messages: "unused"), tools=Alpha())
    with pytest.raises(ValueError, match="ping"):
        flow.add_tool(Beta())


def test_tools_list_mixes_toolsets_and_functions():
    @tool("extra")
    def extra() -> str:
        return "x"

    assert isinstance(FILE_TOOLS, FileTools)
    flow = Flow(StubLLM(lambda _messages: "unused"), tools=[FILE_TOOLS, extra])
    root = start("query")
    namespace = flow.build_tools(root)
    prompt = flow.build_messages(root.frontier)[0]["content"]

    assert namespace["extra"]() == "x"
    assert callable(namespace["grep"])
    assert "### File tools" in prompt
    assert "`extra()" in prompt or "`extra()`" in prompt


def test_a_bare_tool_is_the_same_as_a_one_item_list():
    @tool("one")
    def one() -> str:
        return "1"

    as_item = Flow(StubLLM(lambda _messages: "unused"), tools=one)
    as_list = Flow(StubLLM(lambda _messages: "unused"), tools=[one])
    assert "one" in as_item.tools and "one" in as_list.tools
    assert "grep" in Flow(StubLLM(lambda _messages: "unused"), tools=FILE_TOOLS).tools



def test_toolset_description_sees_the_frontier_node_not_just_the_agent():
    seen = []

    @toolset("Seen")
    class Seen:
        @tool("noop")
        def noop(self) -> str:
            return "ok"

        def description(self, flow, node) -> str:
            seen.append(type(node).__name__)
            return "conditional"

    flow = Flow(StubLLM(lambda _messages: "unused"), tools=Seen())
    root = start("query")
    root.append(ExecOutput(content="observed"))
    flow.build_messages(root.frontier)
    assert seen == ["ExecOutput"]
