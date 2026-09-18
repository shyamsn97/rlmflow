import pytest
from helpers import StubLLM

from rlmflow import FILE_TOOLS, FileTools, Flow, at_leaf, format_tool_line, start, tool, toolset
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


def test_toolsets_get_sections_and_standalone_tools_share_a_section():
    @toolset("Probe tools")
    class Probe:
        @tool("Return one marker.")
        def first(self) -> str:
            return "first"

        @tool("Return another marker.")
        def second(self) -> str:
            return "second"

        def guidance(self, flow, node) -> str:
            return "Call `first()` before `second()`."

    @tool("Return an extra marker.")
    def extra() -> str:
        return "extra"

    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        tools=[Probe(), extra],
    )
    prompt = flow.build_messages(start("query").frontier)[0]["content"]

    assert "### Probe tools" in prompt
    assert prompt.count("`first() -> str`") == 1
    assert prompt.count("`second() -> str`") == 1
    assert "Call `first()` before `second()`." in prompt
    assert "### Tools\n- `extra() -> str`: Return an extra marker." in prompt


def test_toolset_section_and_guidance_disappear_when_its_tools_are_removed():
    @toolset("Temporary tools")
    class Temporary:
        @tool("Return a temporary value.")
        def temporary(self) -> str:
            return "temporary"

        def guidance(self, flow, node) -> str:
            return "Temporary guidance."

    flow = Flow(StubLLM(lambda _messages: "unused"), tools=Temporary())
    flow.remove_tool("temporary")
    prompt = flow.build_messages(start("query").frontier)[0]["content"]

    assert "### Temporary tools" not in prompt
    assert "Temporary guidance." not in prompt
    assert "`temporary(" not in prompt


def test_a_bare_tool_is_the_same_as_a_one_item_list():
    @tool("one")
    def one() -> str:
        return "1"

    as_item = Flow(StubLLM(lambda _messages: "unused"), tools=one)
    as_list = Flow(StubLLM(lambda _messages: "unused"), tools=[one])
    assert "one" in as_item.tools and "one" in as_list.tools
    assert "grep" in Flow(StubLLM(lambda _messages: "unused"), tools=FILE_TOOLS).tools


def test_toolset_guidance_sees_the_frontier_node_not_just_the_agent():
    seen = []

    @toolset("Seen")
    class Seen:
        @tool("noop")
        def noop(self) -> str:
            return "ok"

        def guidance(self, flow, node) -> str:
            seen.append(type(node).__name__)
            return "conditional"

    flow = Flow(StubLLM(lambda _messages: "unused"), tools=Seen())
    root = start("query")
    root.append(ExecOutput(content="observed"))
    flow.build_messages(root.frontier)
    assert seen == ["ExecOutput"]


def test_when_hides_a_tool_from_prompt_and_namespace():
    @tool("leaf only", when=lambda _flow, node: at_leaf(None, node))
    def secret() -> str:
        return "secret"

    flow = Flow(StubLLM(lambda _messages: "unused"), tools=secret)
    root = start("query", max_depth=1)
    leaf = start("leaf", config=root.config.child("worker"))

    assert "secret" in flow.tools
    assert "secret" not in flow.build_tools(root)
    assert flow.build_tools(leaf)["secret"]() == "secret"
    assert "`secret()" not in flow.build_messages(root.frontier)[0]["content"]
    assert "`secret()" in flow.build_messages(leaf.frontier)[0]["content"]


def test_file_writes_bind_only_on_leaves():
    flow = Flow(StubLLM(lambda _messages: "unused"), tools=FILE_TOOLS)
    root = start("query", max_depth=1)
    leaf = start("leaf", config=root.config.child("worker"))

    planner = flow.build_tools(root)
    worker = flow.build_tools(leaf)
    assert "grep" in planner and "write_file" not in planner
    assert "write_file" in worker and "edit_file" in worker
    assert "launch_subagent" in planner
    assert "launch_subagent" not in worker


def test_file_writes_bind_on_a_root_that_cannot_spawn():
    flow = Flow(StubLLM(lambda _messages: "unused"), tools=FILE_TOOLS)
    root = start("query", max_depth=0)

    namespace = flow.build_tools(root)
    assert "write_file" in namespace
    assert "launch_subagent" not in namespace


def test_file_writes_stay_off_while_an_agent_can_still_spawn():
    flow = Flow(StubLLM(lambda _messages: "unused"), tools=FILE_TOOLS)
    root = start("query", max_depth=2)
    mid = start("mid", config=root.config.child("mid"))
    leaf = start("leaf", config=mid.config.child("leaf"))

    assert "write_file" not in flow.build_tools(root)
    assert "write_file" not in flow.build_tools(mid)
    assert "write_file" in flow.build_tools(leaf)


def test_visible_tools_and_when_both_filter_the_namespace():
    @toolset("Gated")
    class Probe:
        @tool("always")
        def ping(self) -> str:
            return "ping"

        @tool("leaf only", when=lambda _flow, node: at_leaf(None, node))
        def poke(self) -> str:
            return "poke"

        def visible_tools(self, flow, node) -> set[str]:
            names = {"ping", "poke"}
            if not at_leaf(flow, node):
                names.discard("poke")
            return names

    flow = Flow(StubLLM(lambda _messages: "unused"), tools=Probe())
    root = start("query", max_depth=1)
    leaf = start("leaf", config=root.config.child("worker"))

    assert set(flow.build_tools(root)).issuperset({"ping"})
    assert "poke" not in flow.build_tools(root)
    assert flow.build_tools(leaf)["poke"]() == "poke"
