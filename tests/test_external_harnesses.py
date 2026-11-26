"""Claude Code and pi harnesses, run against stand-in executables."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from helpers import ScriptedFlow, StubLLM, code_nodes, name_root
from helpers import reasoning_repl as block

from rlmflow import Agent, ClaudeCode, Flow, Graph, LocalRuntime, Node, Pi, tool
from rlmflow.agents import last_session
from rlmflow.agents import pi as pi_module
from rlmflow.graph import calls
from rlmflow.tools.files import FILE_TOOLS

FAKE_CLAUDE = Path(__file__).parent / "fixtures" / "fake_claude.py"


@pytest.fixture
def claude(tmp_path, monkeypatch):
    """Install the fake ``claude``; returns ``(script(entries), calls())``."""
    shim = tmp_path / "claude"
    shim.write_text(f'#!/bin/sh\nexec {sys.executable} {FAKE_CLAUDE} "$@"\n')
    shim.chmod(0o755)
    log = tmp_path / "claude.log"
    log.write_text("")
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))

    def script(*entries):
        monkeypatch.setenv("FAKE_CLAUDE_SCRIPT", json.dumps([list(e) for e in entries]))

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()]

    return SimpleClaude(str(shim), script, calls)


class SimpleClaude:
    def __init__(self, command, script, calls):
        self.command, self.script, self.calls = command, script, calls

    def agent(self, *, options=None, **shared):
        return ClaudeCode(options={"cli_path": self.command, **(options or {})}, **shared)


def flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def run(flow, g, agent=None):
    name_root(flow, g, agent)

    async def go():
        try:
            await flow.arun(g)
        finally:
            await flow.aclose()

    asyncio.run(go())


WORKER = block("submit(chunk.count('x'))")


def test_a_claude_lead_delegates_from_code_nodes_waits_and_resumes_its_session(claude):
    claude.script(
        (
            "Count the x's",
            [
                "rlmflow repl 'chunks = [doc[:4], doc[4:]]\nprint(len(doc))'",
                "rlmflow repl <<'EOF'\nhandles = [launch_agent('count', agent='worker', chunk=c) for c in chunks]\nnotes = 'launched two counters'\nEOF",
            ],
            "Launched the counters; ending my turn. {outputs}",
        ),
        (
            "Agents you started have finished",
            ["rlmflow repl 'submit(sum(h.result() for h in handles))'"],
            "Done.",
        ),
    )
    g = Graph()
    root = g.start("Count the x's", doc="xxaaxbxx")
    worker = Agent(description="Counts x's in one chunk.")
    flow = ScriptedFlow(StubLLM(lambda messages: WORKER), agents={"worker": worker})
    run(flow, g, claude.agent())

    assert root.final_result() == 5
    _, *code_before, idle, results, last_code, done = root.nodes()
    assert [c.runs_code for c in code_before] == [True, True]
    assert idle.data["idle"] and results.role == "user" and last_code.runs_code
    assert done.data["terminal"] and done.data["seen"] == 5
    first, second = claude.calls()
    assert idle.data["session"] == flag(first, "--session-id")
    assert idle.text.startswith("Launched the counters; ending my turn. 8 |")
    assert flag(second, "--resume") == idle.data["session"] and "--fork-session" not in second
    assert {e.source for e in g.edges(kind="return", target=results.id)} == {
        c.nodes()[-1].id for c in root.children()
    }
    assert "Agents you started have finished" in flag(second, "-p")

    assert flag(first, "-p") == "Count the x's"
    system = flag(first, "--append-system-prompt")
    assert "rlmflow repl <<'EOF'" in system and "`doc`: str, 8 chars" in system
    assert "`worker` (Agent, model `gpt-6-sol`; Counts x's in one chunk.)" in system
    assert "keep a `notes` string" in system
    assert flag(first, "--allowedTools").split(",")[0] == "Bash(rlmflow:*)"
    assert g.usage().input_tokens >= 20  # both turns' usage counts toward budgets
    [code_node] = [n for n in code_nodes(g, root) if calls.launched(n)]
    assert len(calls.launched(code_node)) == 2


def test_forking_a_claude_agent_resumes_at_turn_ends_and_restarts_mid_turn(claude):
    claude.script(
        ("Continue", ["rlmflow repl 'submit(notes)'"], "ok"),
        ("[you]", ["rlmflow repl 'submit(notes + \"!\")'"], "ok"),
        ("Agents you started have finished", ["rlmflow repl 'submit(len(handles))'"], "ok"),
        (
            "Plan",
            [
                "rlmflow repl \"notes = 'plan: count'\"",
                "rlmflow repl \"handles = [launch_agent('count', agent='worker', chunk='xx')]\"",
            ],
            "waiting",
        ),
    )
    g = Graph()
    root = g.start("Plan")
    agents = {"worker": Agent()}
    run(ScriptedFlow(StubLLM(lambda messages: WORKER), agents=agents), g, claude.agent())
    assert root.final_result() == 1

    nodes = root.nodes()
    idle = next(n for n in nodes if n.data.get("idle"))
    first_code = code_nodes(g, root)[0]
    at_turn_end = g.fork(idle)
    at_turn_end.send(at_turn_end.agent_root(root.id), "Continue")
    mid_turn = g.fork(first_code)
    calls_before = len(claude.calls())
    for branch in (at_turn_end, mid_turn):
        run(ScriptedFlow(StubLLM(lambda messages: WORKER), agents=agents), branch, claude.agent())

    assert at_turn_end.agent_root(root.id).final_result() == "plan: count"
    assert mid_turn.agent_root(root.id).final_result() == "plan: count!"
    resumed, fresh = claude.calls()[calls_before:]
    assert flag(resumed, "--resume") == idle.data["session"] and "--fork-session" in resumed
    prompt = flag(fresh, "-p")
    assert "--session-id" in fresh and prompt.startswith("Plan\n\n[you]\n")
    assert "notes = 'plan: count'" in prompt
    # The turn-end copy keeps the worker; the mid-turn one is cut before the launch.
    workers = [
        [a for a in branch.agent_roots() if a.agent == "worker"]
        for branch in (g, at_turn_end, mid_turn)
    ]
    assert [len(found) for found in workers] == [1, 1, 0]


def test_a_message_reopens_a_finished_claude_agent_in_the_same_session(claude):
    claude.script(
        ("and now?", ["rlmflow repl \"submit('second')\""], "ok"),
        ("first", ["rlmflow repl \"submit('first')\""], "ok"),
    )
    g = Graph()
    root = g.start("first")
    run(Flow(), g, claude.agent())
    assert root.final_result() == "first"

    g.send(root, "and now?")
    run(Flow(), g, claude.agent())
    assert root.final_result() == "second"
    first, second = claude.calls()
    assert flag(second, "--resume") == flag(first, "--session-id")


def test_claude_output_schema_and_plain_text_results(claude):
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    claude.script(("structured", [], '{"n": 3}'), ("plain", [], "just text"))
    sdk = {"permission_mode": "acceptEdits", "allowed_tools": ["Edit"]}
    flow = Flow(
        agents={
            "structured": claude.agent(output_schema=schema),
            "plain": claude.agent(options=sdk),
        }
    )
    g = Graph()
    structured = g.start("structured", agent="structured")
    plain = g.start("plain", agent="plain")
    run(flow, g)

    assert structured.final_result() == {"n": 3}
    assert plain.final_result() == "just text"
    by_task = {flag(argv, "-p"): argv for argv in claude.calls()}
    assert json.loads(flag(by_task["structured"], "--json-schema")) == schema
    assert flag(by_task["structured"], "--permission-mode") is None
    assert flag(by_task["plain"], "--permission-mode") == "acceptEdits"
    assert flag(by_task["plain"], "--allowedTools").split(",") == ["Edit", "Bash(rlmflow:*)"]


def sdk_options(agent, g=None):
    pytest.importorskip("claude_agent_sdk")
    g = g or Graph()
    root = g.top_level_roots()[0] if g.top_level_roots() else g.start("t")
    flow = Flow(agents={"default": agent})
    repl = flow.repl(root)

    async def build():
        await flow.agent_tools.start()
        try:
            return agent.sdk_options("SYSTEM", repl, last_session(root.nodes()[-1]))
        finally:
            await flow.aclose()

    return asyncio.run(build())


def test_claude_disallows_its_own_messaging_on_top_of_the_users_list():
    assert sdk_options(ClaudeCode()).disallowed_tools == ["SendMessage", "ListAgents"]
    options = sdk_options(ClaudeCode(options={"disallowed_tools": ["WebFetch"]}))
    assert options.disallowed_tools == ["WebFetch", "SendMessage", "ListAgents"]


def test_claude_allowed_tools_add_rlmflow_to_the_users_own():
    assert sdk_options(ClaudeCode()).allowed_tools == ["Bash(rlmflow:*)"]
    assert sdk_options(ClaudeCode(mcp=True)).allowed_tools == ["Bash(rlmflow:*)", "mcp__rlmflow"]
    options = sdk_options(ClaudeCode(mcp=True, options={"allowed_tools": ["Read", "Edit"]}))
    assert options.allowed_tools == ["Read", "Edit", "Bash(rlmflow:*)", "mcp__rlmflow"]


def test_claude_starts_resumes_or_forks_its_session():
    g = Graph()
    root = g.start("t")
    fresh = sdk_options(ClaudeCode(), g)
    assert fresh.session_id and fresh.resume is None
    assert fresh.system_prompt == {"type": "preset", "preset": "claude_code", "append": "SYSTEM"}

    end = g.append(root, Node("assistant", data={"session": "s1", "seen": 1, "idle": True}))
    resumed = sdk_options(ClaudeCode(), g)
    assert (resumed.resume, resumed.fork_session) == ("s1", False)
    copy = g.fork(end)
    forked = sdk_options(ClaudeCode(), copy)
    assert (forked.resume, forked.fork_session) == ("s1", True)
    same = g.fork(end, repl="same")
    assert sdk_options(ClaudeCode(), same).fork_session is False
    g.append(root, Node("assistant", code="x = 1"))
    assert sdk_options(ClaudeCode(), g).resume is None


def test_unknown_claude_options_are_an_error():
    pytest.importorskip("claude_agent_sdk")
    with pytest.raises(TypeError, match="tools_loose"):
        sdk_options(ClaudeCode(options={"tools_loose": True}))


class RemoteRuntime(LocalRuntime):
    @property
    def location(self):
        return "a Docker container"


@tool("Look up a word.", where="repl")
def define(word: str) -> str:
    return word


def system_prompt(agent, flow=None):
    g = Graph()
    root = g.start("t")
    flow = flow or Flow()
    flow.agents["default"] = agent
    return agent.system_prompt(root, flow.repl(root))


def test_a_cli_prompt_lists_the_flows_tools_and_where_the_repl_runs():
    local = system_prompt(Pi(), Flow(tools=[FILE_TOOLS, define]))
    assert "Also in the REPL:" in local
    assert "- `read_file(" in local and "- `define(word: str) -> str`: Look up a word." in local
    assert "Your REPL runs in" not in local

    remote = system_prompt(Pi(), Flow(runtime=RemoteRuntime()))
    assert "Your REPL runs in a Docker container." in remote
    assert "Also in the REPL:" not in remote


def test_prompt_replaces_drops_and_adds_sections():
    prompt = system_prompt(
        Pi(prompt={"state": "KEEP NOTES.", "verify": None, "style": "BE TERSE."})
    )
    assert "KEEP NOTES." in prompt and "REPL variables, not only" not in prompt
    assert "Before you submit" not in prompt
    assert prompt.endswith("BE TERSE.")


def test_every_harness_gets_the_same_guidance():
    shared = ("verify", "state", "orchestrate", "arguments", "schema")
    agents = [Agent(), Pi(), ClaudeCode()]
    g = Graph()
    root = g.start("t")
    flow = Flow()
    found = []
    for agent in agents:
        flow.agents["default"] = agent
        sections = agent.sections(root, flow.repl(root))
        found.append({key: sections[key] for key in shared})
    assert found[0]["orchestrate"] and found[0] == found[1] == found[2]


def test_a_prompt_section_can_be_a_function_of_the_step():
    board = Agent(prompt={"state": None, "style": lambda node, repl: f"BOARD for {node.text}."})
    for agent in (board, Pi(prompt=board.prompt), ClaudeCode(prompt=board.prompt)):
        prompt = system_prompt(agent)
        assert prompt.endswith("BOARD for t.") and "REPL variables, not only" not in prompt


def test_pi_reads_the_final_assistant_text_and_usage():
    events = [
        {"type": "session"},
        {
            "type": "agent_end",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "toolCall"}],
                    "usage": {"input": 3, "output": 1},
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "done"}],
                    "usage": {"input": 5, "output": 2},
                },
            ],
        },
    ]
    assert pi_module.final_reply(events) == ("done", {"input_tokens": 8, "output_tokens": 3})
    failed = [
        {
            "type": "agent_end",
            "messages": [{"role": "assistant", "stopReason": "error", "errorMessage": "401"}],
        }
    ]
    with pytest.raises(RuntimeError, match="401"):
        pi_module.final_reply(failed)


def test_pi_session_files_resume_in_place_and_fork_by_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(pi_module, "SESSION_DIR", tmp_path)
    g = Graph()
    root = g.start("t")
    flow = Flow(agents={"default": Pi()})
    fresh = pi_module.session_file(None, flow.repl(root))
    assert fresh.parent == tmp_path and fresh.suffix == ".jsonl"
    fresh.write_text('{"type": "session"}\n')
    end = g.append(root, Node("assistant", data={"session": str(fresh), "seen": 1, "idle": True}))
    assert pi_module.session_file(end, flow.repl(root)) == fresh
    copy = g.fork(end)
    forked = pi_module.session_file(end, flow.repl(copy.agent_root(root.id)))
    assert forked != fresh and forked.read_text() == fresh.read_text()


FAKE_PI = """\
import json, sys
open(sys.argv[0] + ".log", "w").write(json.dumps(sys.argv[1:]))
reply = {"role": "assistant", "content": [{"type": "text", "text": "done"}], "usage": {"input": 2, "output": 1}}
print(json.dumps({"type": "agent_end", "messages": [reply]}))
"""


def test_pi_args_are_appended_to_its_command_line(tmp_path, monkeypatch):
    monkeypatch.setattr(pi_module, "SESSION_DIR", tmp_path / "sessions")
    script = tmp_path / "pi.py"
    script.write_text(FAKE_PI)
    shim = tmp_path / "pi"
    shim.write_text(f'#!/bin/sh\nexec {sys.executable} {script} "$@"\n')
    shim.chmod(0o755)
    g = Graph()
    root = g.start("t")
    run(Flow(), g, Pi(command=str(shim), args=["--thinking", "high"]))

    assert root.final_result() == "done"
    argv = json.loads((tmp_path / "pi.py.log").read_text())
    assert argv[:2] == ["-p", "t"] and argv[-2:] == ["--thinking", "high"]


def test_a_person_launches_default_agents_unless_it_names_one():
    from types import SimpleNamespace

    from rlmflow.agents.person import Person
    from rlmflow.tools.builtins import child_agent

    def repl(agent, own):
        return SimpleNamespace(agent=agent, root=SimpleNamespace(agent=own))

    assert child_agent(repl(Person(), "person")) == "default"
    assert child_agent(repl(Agent(), "worker")) == "worker"
