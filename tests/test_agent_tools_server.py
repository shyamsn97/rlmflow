"""The agent-tools server and ``rlmflow`` CLI, driven by a scripted external harness."""

import asyncio
import os

import pytest

from rlmflow import CODE, DONE, MESSAGE, WAITING, AgentConfig, Flow, Graph, Node
from rlmflow.agent_tools.cli import json_or_text


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
    index = sum(1 for node in g.history(agent) if node.type == WAITING)
    transcript = []
    async with flow.agent_tools.turn(step) as turn:
        for command in turns[index]:
            transcript.append(await sh(command, turn.env))
    text = "\n".join(out + err for _, out, err in transcript)
    exits = [code for code, _, _ in transcript]
    if turn.submitted:
        return Node(DONE, result=turn.result, text=text, exits=exits)
    running = [a.id for a in g.children(agent) if not g.terminal(a)]
    if running:
        return Node(WAITING, on=running, text=text, exits=exits)
    return Node(DONE, result=text, exits=exits)


def shell_config(*turns, **config):
    return AgentConfig(harness=shell, harness_options={"turns": [list(t) for t in turns]}, **config)


def test_repl_cells_land_code_nodes_and_keep_state():
    g = Graph()
    root = g.create_agent("count", doc="one two three")
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
    assert cells[3].data["submitted"] and cells[3].data["result"] == 26
    assert streamed == [CODE, CODE, CODE, CODE, DONE]
    assert g.frontier(root).data["exits"] == [0, 0, 1, 0]


def test_ending_a_turn_with_running_children_waits_then_resumes():
    g = Graph()
    root = g.create_agent("delegate")
    config = shell_config(
        ["rlmflow repl \"h = launch_agent('add', config='worker', x=2)\"", "rlmflow agents"],
        ["rlmflow repl 'print(h.result())'", "rlmflow done '{\"total\": 5}'"],
    )
    from helpers import StubLLM

    flow = Flow(
        StubLLM(lambda messages: "```repl\nawait asyncio.sleep(1)\nsubmit(x + 3)\n```"),
        configs=[AgentConfig(name="worker", description="adds")],
    )
    asyncio.run(flow.arun(g, root_config=config))
    asyncio.run(flow.aclose())

    assert g.result(root) == {"total": 5}
    [child] = g.children(root)
    assert child.config.name == "worker" and g.result(child) == 5
    types = [node.type for node in g.history(root)]
    assert types == ["agent_start", CODE, WAITING, MESSAGE, CODE, DONE]
    [cell] = [n for n in g.history(root) if n.type == CODE and "launch_agent" in n.data["code"]]
    assert cell.launched() == [child.id]
    assert "worker" in g.history(root)[2].data["text"]
    assert g.history(root)[4].data["output"].strip() == "5"


def test_spawn_wait_and_send_from_the_shell():
    g = Graph()
    root = g.create_agent("fan out")
    config = shell_config(
        [
            "rlmflow spawn --config worker --name a --arg x=40 'add two'",
            "rlmflow wait a",
            "rlmflow wait a",
            "rlmflow wait a",
            "rlmflow wait a",
            "rlmflow send nobody hi",
            "rlmflow done finished",
        ],
    )
    from helpers import StubLLM

    flow = Flow(
        StubLLM(lambda messages: "```repl\nsubmit(x + 2)\n```"),
        configs=[AgentConfig(name="worker")],
    )
    asyncio.run(flow.arun(g, root_config=config))
    asyncio.run(flow.aclose())

    assert g.result(root) == "finished"
    [child] = g.children(root)
    assert g.result(child) == 42
    lines = g.frontier(root).data["text"].split("\n")
    text = "\n".join(lines)
    assert child.id in text
    assert '"' + child.id + '": 42' in text
    assert text.count("Still running. End your turn") >= 1  # the fourth wait is refused
    assert "unknown agent" in text or "no agent" in text


def test_a_token_stops_working_when_its_turn_ends():
    g = Graph()
    root = g.create_agent("x")
    root.bind_config(AgentConfig())
    flow = Flow()
    seen = {}

    async def run():
        from rlmflow.graph.nodes import Step

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
    root = g.create_agent("race")
    root.bind_config(shell_config([]))
    flow = Flow()

    async def run():
        from rlmflow.graph.nodes import Step

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


def test_mcp_exposes_the_same_tools_with_repl_as_python():
    pytest.importorskip("mcp")
    from rlmflow.agent_tools.mcp import build_server, mcp_config

    tools = asyncio.run(build_server().list_tools())
    assert sorted(tool.name for tool in tools) == [
        "agents",
        "done",
        "python",
        "send",
        "spawn",
        "wait",
    ]
    [python] = [tool for tool in tools if tool.name == "python"]
    schema = getattr(python, "input_schema", None) or python.inputSchema
    assert list(schema["properties"]) == ["code"]
    server = mcp_config({"RLMFLOW_TOKEN": "t"})["mcpServers"]["rlmflow"]
    assert server["args"] == ["-m", "rlmflow.agent_tools.mcp"] and server["env"] == {
        "RLMFLOW_TOKEN": "t"
    }


@pytest.mark.parametrize(
    ("raw", "value"), [('{"a": 1}', {"a": 1}), ("42", 42), ("plain words", "plain words")]
)
def test_done_and_args_take_json_or_text(raw, value):
    assert json_or_text(raw) == value
