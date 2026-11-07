"""The agent-tools server and ``rlmflow`` CLI, driven by a scripted external harness."""

import asyncio
import os
import shlex

import pytest
from helpers import ScriptedFlow, StubLLM
from helpers import repl as block

from rlmflow import CODE, DONE, IDLE, MESSAGE, AgentConfig, Flow, Graph, Node
from rlmflow.graph import calls, queries
from rlmflow.graph.nodes import Step


async def sh(command: str, env: dict[str, str]) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_shell(
        command,
        env={**os.environ, **env},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await process.communicate()
    return process.returncode, out.decode(), err.decode()


async def shell(flow, step):
    """A stand-in for Claude Code: each turn runs its scripted shell commands."""
    g, agent = step.graph, step.agent
    turns = agent.config.harness_options["turns"]
    index = sum(1 for node in g.history(agent) if node.type == IDLE)
    transcript = []
    async with flow.agent_tools.turn(step) as turn:
        for command in turns[index]:
            transcript.append(await sh(command, turn.env))
    text = "\n".join(out + err for _, out, err in transcript)
    exits = [code for code, _, _ in transcript]
    if turn.submitted:
        return Node(DONE, result=turn.result, text=text, exits=exits)
    if queries.unseen_results(g, agent) or not all(
        g.status(a) == "done" for a in g.children(agent)
    ):
        return Node(IDLE, text=text, exits=exits)
    return Node(DONE, result=text, exits=exits)


def shell_config(*turns, **config):
    return AgentConfig(harness=shell, harness_options={"turns": [list(t) for t in turns]}, **config)


def test_repl_cells_land_code_nodes_and_keep_state():
    g = Graph()
    root = g.add_agent("count", doc="one two three")
    config = shell_config(
        [
            "rlmflow repl 'n = len(doc)'",
            "rlmflow repl <<'EOF'\nwords = doc.split()\nprint(n, len(words))\nEOF",
            "rlmflow repl '1/0'",
            "rlmflow repl 'submit(n * 2)'",
        ],
    )
    flow = Flow()

    async def run():
        return [step.node.type async for step in flow.run_streaming(g, root_config=config)]

    streamed = asyncio.run(run())
    asyncio.run(flow.aclose())

    assert g.result(root) == 26
    cells = [node for node in g.history(root) if node.type == CODE]
    assert [c.data["code"] for c in cells] == [
        "n = len(doc)",
        "words = doc.split()\nprint(n, len(words))\n",
        "1/0",
        "submit(n * 2)",
    ]
    assert cells[1].data["output"].strip() == "13 3"
    assert cells[2].data["error"] and "ZeroDivisionError" in cells[2].data["output"]
    assert streamed == [CODE, CODE, CODE, CODE, DONE]
    assert g.history(root)[-1].data["exits"] == [0, 0, 1, 0]


def test_ending_a_turn_with_running_children_waits_then_resumes():
    g = Graph()
    root = g.add_agent("delegate")
    config = shell_config(
        [
            "rlmflow repl \"h = launch_agent('add', config='worker', x=2)\"",
            "rlmflow repl 'AGENTS.print_graph()'",
        ],
        ["rlmflow repl 'print(h.result())'", "rlmflow repl 'submit({\"total\": 5})'"],
    )
    flow = ScriptedFlow(
        StubLLM(lambda messages: "```repl\nawait asyncio.sleep(1)\nsubmit(x + 3)\n```"),
        configs=[AgentConfig(name="worker", description="adds")],
    )
    asyncio.run(flow.arun(g, root_config=config))
    asyncio.run(flow.aclose())

    assert g.result(root) == {"total": 5}
    [child] = g.children(root)
    assert child.config.name == "worker" and g.result(child) == 5
    types = [node.type for node in g.history(root)]
    assert types == [CODE, CODE, IDLE, MESSAGE, CODE, CODE, DONE]
    [cell] = [n for n in g.history(root) if n.type == CODE and "launch_agent" in n.data["code"]]
    assert calls.launched(cell) == [child.id]
    assert "worker" in g.history(root)[1].data["output"]
    assert g.history(root)[4].data["output"].strip() == "5"
    [delivered] = g.edges(kind="return", target=g.history(root)[3].id)
    assert delivered.source == child.nodes[-1].id


DELEGATE = "h = launch_agent('add', config='worker', x=2)\nAGENTS['peer'].send('hi')\nprint('ok')"


def edge_shapes(g):
    """Each edge as ``(kind, source, target)``, endpoints named by owner and what they are."""

    def end(item_id):
        if item_id in {a.id for a in g.agents()}:
            return g.agent(item_id).name, "agent"
        node = g.node(item_id)
        return g.owner(node).name, "cell" if node.data.get("code") else node.type

    return sorted((e.kind, end(e.source), end(e.target)) for e in g.edges())


def delegating_reply(messages):
    """The lead delegates, then submits its child's result; the others answer at once."""
    task, last = messages[1]["content"], messages[-1]["content"]
    if task == "add":
        return block("submit(x + 3)")
    if task == "listen":
        return block("submit('heard')" if "hi" in last else "submit('ready')")
    return block(DELEGATE) if len(messages) == 2 else block("submit(await h)")


def test_server_cells_write_the_same_edges_as_in_process_cells():
    def run(root_config):
        g = Graph()
        root = g.add_agent("delegate", name="lead")
        g.add_agent("listen", name="peer", config="listener")
        flow = ScriptedFlow(
            StubLLM(delegating_reply),
            configs=[AgentConfig(name="worker"), AgentConfig(name="listener")],
        )
        asyncio.run(flow.arun(g, root_config=root_config))
        asyncio.run(flow.aclose())
        return g, root

    native, native_root = run(None)
    served, served_root = run(
        AgentConfig(
            harness=shell,
            harness_options={
                "turns": [
                    [f"rlmflow repl {shlex.quote(DELEGATE)}", "rlmflow repl 'submit(await h)'"]
                ]
            },
        )
    )
    assert native.result(native_root) == 5
    assert served.result(served_root) == 5
    assert edge_shapes(served) == edge_shapes(native)
    for g, root in ((native, native_root), (served, served_root)):
        [child] = g.children(root)
        [spawn] = g.edges(kind="spawn", target=child.id)
        assert g.owner(g.node(spawn.source)) is root
        [message] = g.edges(kind=MESSAGE)
        assert g.node(message.source) is g.node(spawn.source)
        assert g.node(message.target).data["text"] == "hi"


def test_a_token_stops_working_when_its_turn_ends():
    g = Graph()
    root = g.add_agent("x")
    root.bind_config(AgentConfig())
    flow = Flow()
    seen = {}

    async def run():
        async with flow.agent_tools.turn(Step(g, root, root)) as turn:
            seen["inside"] = await sh("rlmflow repl 'print(1)'", turn.env)
        seen["after"] = await sh("rlmflow repl 'print(2)'", turn.env)
        outside = {**turn.env, "RLMFLOW_URL": "", "RLMFLOW_TOKEN": ""}
        seen["outside"] = await sh("rlmflow repl 'print(3)'", outside)
        await flow.aclose()

    asyncio.run(run())

    assert seen["inside"][:2] == (0, "1\n")
    assert seen["after"][0] == 1 and "turn has ended" in seen["after"][2]
    assert seen["outside"][0] == 1 and "not inside an rlmflow agent turn" in seen["outside"][2]
    assert [n.data["code"] for n in g.nodes(CODE)] == ["print(1)"]


def test_concurrent_cells_from_one_agent_run_one_at_a_time():
    g = Graph()
    root = g.add_agent("race")
    root.bind_config(shell_config([]))
    flow = Flow()

    async def run():
        async with flow.agent_tools.turn(Step(g, root, root)) as turn:
            results = await asyncio.gather(
                *(
                    sh(
                        f"rlmflow repl 'import time; time.sleep(0.2); seen = {i}; print(seen)'",
                        turn.env,
                    )
                    for i in range(3)
                )
            )
        await flow.aclose()
        return results

    results = asyncio.run(run())
    assert sorted(out for _, out, _ in results) == ["0\n", "1\n", "2\n"]
    cells = [n for n in g.history(root) if n.type == CODE]
    assert len(cells) == 3  # all three landed in one chain, none dropped as stale


def test_mcp_exposes_only_the_repl_as_python():
    pytest.importorskip("mcp")
    from rlmflow.agent_tools.mcp import build_server, mcp_config

    tools = asyncio.run(build_server().list_tools())
    [python] = tools
    assert python.name == "python"
    schema = getattr(python, "input_schema", None) or python.inputSchema
    assert list(schema["properties"]) == ["code"]
    server = mcp_config({"RLMFLOW_TOKEN": "t"})["mcpServers"]["rlmflow"]
    assert server["args"] == ["-m", "rlmflow.agent_tools.mcp"] and server["env"] == {
        "RLMFLOW_TOKEN": "t"
    }
