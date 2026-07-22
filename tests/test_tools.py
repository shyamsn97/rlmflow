import asyncio
import sys

from rlmflow import (
    Flow,
    Graph,
    tool,
)
from rlmflow.runtime import PopenConnection
from rlmflow.runtime.repl import DoneSignal
from rlmflow.runtime.repl_client import RemoteRepl

from helpers import (
    StubLLM,
    first_user,
)


def test_minimal_add_tool_injects_into_live_repl_and_prompt():
    @tool("Return a fixed greeting.")
    def greet() -> str:
        return "hi from tool"

    flow = Flow(StubLLM(lambda _messages: "```repl\npass\n```"))
    graph = Graph(query="q")
    repl = flow.repl_for(graph)
    assert "greet" not in repl.namespace

    flow.add_tool(greet)

    assert repl.namespace["greet"] is greet
    assert flow.tools["greet"] is greet
    # Per-turn prompt reflects the newly registered tool.
    assert "greet()" in flow.build_system_prompt(graph)
    # ...and it is immediately callable inside the running REPL.
    out = asyncio.run(repl.run("print(greet())"))
    assert "hi from tool" in out



def test_minimal_add_tool_reaches_agents_spawned_later():
    @tool("Return the answer.")
    def answer() -> int:
        return 42

    flow = Flow(StubLLM(lambda _messages: "```repl\ndone(str(answer()))\n```"))
    flow.add_tool(answer)

    assert flow.run(graph=Graph(query="q")) == "42"



def test_minimal_injected_tools_are_available_to_subagents():
    """Objects injected via ``Flow(tools=[...])`` — functions *and* classes —
    are seeded into every agent's REPL, including subagents spawned via
    ``launch_subagents`` (child REPLs rebuild the same tool namespace)."""

    def shout(text: str) -> str:
        return text.upper()

    class Box:
        def __init__(self, value: int) -> None:
            self.value = value

        def label(self) -> str:
            return f"box:{self.value}"

    def reply(messages):
        if first_user(messages) == "parent":
            return (
                "```repl\n"
                "results = await launch_subagents("
                "[{'name': 'a', 'query': 'child'}])\n"
                "done(results[0])\n"
                "```"
            )
        # The child uses both the injected function and the injected class.
        return '```repl\ndone(shout("hi") + "|" + Box(7).label())\n```'

    flow = Flow(StubLLM(reply), tools=[shout, Box], max_depth=1)

    assert asyncio.run(flow.arun(graph=Graph(query="parent"))) == "HI|box:7"


def test_minimal_flow_inject_binds_global_everywhere_without_advertising():
    """``flow.inject`` puts an object in every REPL (now and later, including
    subagents) as an ambient global — usable in code but never advertised as a
    tool in the system prompt."""

    class Sokoban:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def render(self) -> str:
            return f"board:{self.tag}"

    def reply(messages):
        if first_user(messages) == "parent":
            return (
                "```repl\n"
                "results = await launch_subagents("
                "[{'name': 'a', 'query': 'child'}])\n"
                "done(results[0])\n"
                "```"
            )
        return '```repl\ndone(Sokoban("x").render())\n```'

    flow = Flow(StubLLM(reply), max_depth=1)
    flow.inject("Sokoban", Sokoban)

    graph = Graph(query="parent")
    # Tracked as a binding, present in the namespace, but NOT advertised in the
    # prompt: advertising is gated on @tool metadata, not on being a binding.
    assert flow.tools["Sokoban"] is Sokoban
    assert flow.repl_for(graph).namespace["Sokoban"] is Sokoban
    assert "Sokoban" not in flow.build_system_prompt(graph)
    # Reaches a subagent spawned later, and is usable in code.
    assert asyncio.run(flow.arun(graph=graph)) == "board:x"


def test_minimal_flow_inject_reaches_forked_repl():
    """An injected global is seeded into a fork's rebuilt REPL, so construction
    code that references it works on the branch (not just the original)."""

    class Widget:
        def __init__(self, n: int) -> None:
            self.n = n

    flow = Flow(StubLLM(lambda _messages: "```repl\npass\n```"))
    flow.inject("Widget", Widget)
    graph = Graph(query="q")

    async def scenario():
        flow.repl_for(graph)  # open the original REPL
        fork = await flow.fork(graph)  # fork rebuilds a fresh REPL from tools/globals
        return await flow.repl_for(fork).run("print(Widget(9).n)")

    assert "9" in asyncio.run(scenario())


def test_minimal_flow_inject_rejects_reserved_names():
    flow = Flow(StubLLM(lambda _messages: "```repl\npass\n```"))
    for name in ("done", "launch_subagents", "INPUTS"):
        try:
            flow.inject(name, object())
        except ValueError:
            pass
        else:
            raise AssertionError(f"inject should reject reserved name {name!r}")


def test_minimal_add_tool_reaches_subagents_spawned_later():
    """A tool injected at runtime via ``add_tool`` reaches a child spawned after
    it was registered (build_tools re-seeds ``flow.tools`` for every new REPL)."""

    def stamp() -> str:
        return "stamped"

    def reply(messages):
        if first_user(messages) == "parent":
            return (
                "```repl\n"
                "results = await launch_subagents("
                "[{'name': 'a', 'query': 'child'}])\n"
                "done(results[0])\n"
                "```"
            )
        return "```repl\ndone(stamp())\n```"

    flow = Flow(StubLLM(reply), max_depth=1)
    flow.add_tool(stamp)

    assert asyncio.run(flow.arun(graph=Graph(query="parent"))) == "stamped"


def test_minimal_remove_tool_drops_from_live_repl_and_flow():
    @tool("Return a fixed greeting.")
    def greet() -> str:
        return "hi"

    flow = Flow(StubLLM(lambda _messages: "```repl\npass\n```"), tools=[greet])
    graph = Graph(query="q")
    repl = flow.repl_for(graph)
    assert "greet" in repl.namespace

    removed = flow.remove_tool("greet")

    assert removed is greet
    assert "greet" not in flow.tools
    assert "greet" not in repl.namespace



def test_minimal_add_remove_tool_reject_reserved_names():
    flow = Flow(StubLLM(lambda _messages: "```repl\npass\n```"))
    for name in ("done", "launch_subagents", "INPUTS"):
        try:
            flow.add_tool(lambda: None, name=name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"add_tool should reject reserved name {name!r}")
    try:
        flow.remove_tool("done")
    except ValueError:
        pass
    else:
        raise AssertionError("remove_tool should reject reserved names")



def test_minimal_repl_client_inject_and_remove_tool():
    repl = RemoteRepl(
        PopenConnection(
            [sys.executable, "-u", "-m", "rlmflow.runtime.repl_server"],
            label="test minimal remote REPL",
            repl_timeout=5,
        )
    )

    def done(answer):
        repl.done_result = str(answer)
        raise DoneSignal()

    @tool("Echo the argument back.", proxy=True)
    def echo(value):
        return value

    async def run():
        try:
            repl.seed({"done": done}, {})
            repl.inject("echo", echo)
            injected = await repl.run('print(echo("hey"))')
            repl.remove_tool("echo")
            removed = await repl.run(
                'print("gone" if "echo" not in dir() else echo("x"))'
            )
            return injected, removed
        finally:
            repl.close()

    injected, removed = asyncio.run(run())

    assert "hey" in injected
    assert "gone" in removed



def test_minimal_tool_metadata_marks_async_and_renders_await():
    from rlmflow.tools import format_tool_line, get_tool_metadata

    @tool("Do async work.")
    async def worker(x):
        return x

    @tool("Do sync work.")
    def helper(x):
        return x

    assert get_tool_metadata(worker).is_async is True
    assert get_tool_metadata(helper).is_async is False
    assert format_tool_line(worker).startswith("- `await worker(")
    assert format_tool_line(helper).startswith("- `helper(")



def test_minimal_repl_client_async_proxy_tool_is_awaitable():
    repl = RemoteRepl(
        PopenConnection(
            [sys.executable, "-u", "-m", "rlmflow.runtime.repl_server"],
            label="test minimal remote REPL",
            repl_timeout=5,
        )
    )

    def done(answer):
        repl.done_result = str(answer)
        raise DoneSignal()

    @tool("Double asynchronously.", proxy=True)
    async def adouble(value):
        return value * 2

    async def run():
        try:
            repl.seed({"done": done}, {})
            repl.inject("adouble", adouble)
            return await repl.run("v = await adouble(21)\nprint(v)")
        finally:
            repl.close()

    out = asyncio.run(run())

    assert "42" in out

