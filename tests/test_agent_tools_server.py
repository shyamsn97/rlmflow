"""The agent-tools server and ``rlmflow`` CLI, driven by a scripted external harness."""

import asyncio
import os
import shlex

import pytest
from helpers import ScriptedFlow, StubLLM, code_nodes, turns
from helpers import repl as block

from rlmflow import WRITE, AgentConfig, Flow, Graph
from rlmflow.graph import calls
from rlmflow.graph.nodes import Agent, Step
from rlmflow.harnesses.sessions import SessionHarness, TurnResult


async def sh(command: str, env: dict[str, str]) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_shell(
        command,
        env={**os.environ, **env},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await process.communicate()
    return process.returncode, out.decode(), err.decode()


class Shell(SessionHarness):
    """A stand-in for Claude Code: each turn runs its scripted shell commands."""

    name = "shell"

    async def run_turn(self, flow, step, turn):
        commands = step.config.harness_options["turns"][len(turns(step.graph, step.agent))]
        transcript = [await sh(command, turn.env) for command in commands]
        self.exits = [code for code, _, _ in transcript]
        return TurnResult("shell", "\n".join(out + err for _, out, err in transcript))

    async def step(self, flow, step):
        node = await super().step(flow, step)
        node.data["exits"] = self.exits
        return node


def shell_config(*turns, **config):
    return AgentConfig(harness=Shell, harness_options={"turns": [list(t) for t in turns]}, **config)


def shape(node):
    """What a node is: the root, a code_node, a turn end, or a node from someone else."""
    if isinstance(node, Agent):
        return "root"
    if node.runs_code:
        return "code_node"
    if node.data.get("terminal"):
        return "done"
    if node.data.get("idle"):
        return "idle"
    return node.role


def test_repl_code_nodes_land_code_nodes_and_keep_state():
    g = Graph()
    root = g.start("count", doc="one two three")
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
        return [shape(step.node) async for step in flow.run_streaming(g, root_config=config)]

    streamed = asyncio.run(run())
    asyncio.run(flow.aclose())

    assert root.final_result() == 26
    ran = code_nodes(g, root)
    assert [c.code for c in ran] == [
        "n = len(doc)",
        "words = doc.split()\nprint(n, len(words))\n",
        "1/0",
        "submit(n * 2)",
    ]
    assert ran[1].output.strip() == "13 3"
    assert ran[2].error and "ZeroDivisionError" in ran[2].output
    assert streamed == ["code_node", "code_node", "code_node", "code_node", "done"]
    assert ran[3].result == 26 and root.nodes()[-1].data["terminal"]
    assert [n for n in root.nodes() if n.data.get("terminal")] == [root.nodes()[-1]]
    assert root.nodes()[-1].data["exits"] == [0, 0, 1, 0]


def test_ending_a_turn_with_running_children_waits_then_resumes():
    g = Graph()
    root = g.start("delegate")
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

    assert root.final_result() == {"total": 5}
    [child] = root.children()
    assert child.config.name == "worker" and child.final_result() == 5
    nodes = root.nodes()
    assert [shape(n) for n in nodes] == [
        "root",
        "code_node",
        "code_node",
        "idle",
        "user",
        "code_node",
        "code_node",
        "done",
    ]
    assert calls.launched(nodes[1]) == [child.id]
    assert "worker" in nodes[2].output
    assert nodes[5].output.strip() == "5"
    [delivered] = g.edges(kind="return", target=nodes[4].id)
    assert delivered.source == child.nodes()[-1].id


DELEGATE = "h = launch_agent('add', config='worker', x=2)\nAGENTS['peer'].send('hi')\nprint('ok')"


def edge_shapes(g):
    """Each edge as ``(kind, source, target)``, endpoints named by owner and what they are."""

    def end(item_id):
        node = g.node(item_id)
        return node.agent.name, shape(node)

    return sorted((e.kind, end(e.source), end(e.target)) for e in g.edges())


def delegating_reply(messages):
    """The lead delegates, then submits its child's result; the others answer at once."""
    task, last = messages[1]["content"], messages[-1]["content"]
    if task == "add":
        return block("submit(x + 3)")
    if task == "listen":
        return block("submit('heard')" if "hi" in last else "submit('ready')")
    return block(DELEGATE) if len(messages) == 2 else block("submit(await h)")


def test_server_code_nodes_write_the_same_edges_as_in_process_code_nodes():
    def run(root_config):
        g = Graph()
        root = g.start("delegate", name="lead")
        g.start("listen", name="peer", config="listener")
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
            harness=Shell,
            harness_options={
                "turns": [
                    [f"rlmflow repl {shlex.quote(DELEGATE)}", "rlmflow repl 'submit(await h)'"]
                ]
            },
        )
    )
    assert native_root.final_result() == 5
    assert served_root.final_result() == 5
    assert edge_shapes(served) == edge_shapes(native)
    for g, root in ((native, native_root), (served, served_root)):
        [child] = root.children()
        [spawn] = g.edges(kind="spawn", target=child.id)
        assert g.node(spawn.source).agent is root
        [message] = g.edges(kind=WRITE)
        assert g.node(message.source) is g.node(spawn.source)
        assert g.node(message.target).text == "From lead: hi"


def test_a_token_stops_working_when_its_turn_ends():
    g = Graph()
    root = g.start("x")
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
    assert [n.code for n in code_nodes(g, root)] == ["print(1)"]


def test_concurrent_code_nodes_from_one_agent_run_one_at_a_time():
    g = Graph()
    root = g.start("race")
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
    assert len(code_nodes(g, root)) == 3


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
