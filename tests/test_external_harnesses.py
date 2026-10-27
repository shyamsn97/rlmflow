"""Claude Code and pi harnesses, run against stand-in executables."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from helpers import StubLLM
from helpers import reasoning_repl as block

from rlmflow import CODE, DONE, MESSAGE, WAITING, AgentConfig, Flow, Graph
from rlmflow.graph.nodes import Step
from rlmflow.harnesses.claude_code import ClaudeCodeOptions
from rlmflow.harnesses.pi import PiHarness
from rlmflow.harnesses.sessions import Session

FAKE_CLAUDE = Path(__file__).parent / "fixtures" / "fake_claude.py"


@pytest.fixture
def claude(tmp_path, monkeypatch):
    """Install the fake ``claude``; returns ``(script(entries), calls())``."""
    shim = tmp_path / "claude"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE_CLAUDE} \"$@\"\n")
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

    def config(self, **fields):
        options = {"command": self.command, **fields.pop("harness_options", {})}
        return AgentConfig(harness="claude_code", harness_options=options, **fields)


def flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def run(flow, g):
    async def go():
        try:
            await flow.arun(g)
        finally:
            await flow.aclose()

    asyncio.run(go())


WORKER = block("submit(chunk.count('x'))")


def test_a_claude_lead_delegates_from_cells_waits_and_resumes_its_session(claude):
    claude.script(
        (
            "Count the x's",
            [
                "rlmflow repl 'chunks = [doc[:4], doc[4:]]\nprint(len(doc))'",
                "rlmflow repl <<'EOF'\nhandles = [launch_agent('count', config='worker', chunk=c) for c in chunks]\nnotes = 'launched two counters'\nEOF",
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
    g.add_config(AgentConfig(name="worker", description="Counts x's in one chunk."))
    root = g.create_agent("Count the x's", config=claude.config(max_depth=1), doc="xxaaxbxx")
    run(Flow(StubLLM(lambda messages: WORKER)), g)

    assert g.result(root) == 5
    types = [n.type for n in g.history(root)]
    assert types == ["agent_start", CODE, CODE, WAITING, MESSAGE, CODE, DONE]
    waiting = g.history(root)[3]
    first, second = claude.calls()
    assert waiting.data["session_id"] == flag(first, "--session-id")
    assert waiting.data["on"] == [c.id for c in g.children(root)]
    assert waiting.data["text"].startswith("Launched the counters; ending my turn. 8 |")
    assert flag(second, "--resume") == waiting.data["session_id"] and "--fork-session" not in second
    assert "Agents you started have finished" in flag(second, "-p")

    prompt = flag(first, "-p")
    assert prompt.startswith("Count the x's") and "`doc`: str, 8 chars" in prompt
    system = flag(first, "--append-system-prompt")
    assert "rlmflow repl <<'EOF'" in system and "`task`, `doc`" in system
    assert "`worker` (Counts x's in one chunk.)" in system and "keep a `notes` string" in system
    assert flag(first, "--allowedTools").split(",")[0] == "Bash(rlmflow:*)"
    assert g.usage().input_tokens >= 20  # both turns' usage counts toward budgets
    [cell] = [n for n in g.history(root) if n.type == CODE and n.launched()]
    assert len(cell.launched()) == 2


def test_forking_a_claude_agent_resumes_at_turn_ends_and_restarts_mid_turn(claude):
    claude.script(
        ("Continue", ["rlmflow repl 'submit(notes)'"], "ok"),
        ("You are continuing an earlier session", ["rlmflow repl 'submit(notes + \"!\")'"], "ok"),
        ("Agents you started have finished", ["rlmflow repl 'submit(len(handles))'"], "ok"),
        (
            "Plan",
            [
                "rlmflow repl \"notes = 'plan: count'\"",
                "rlmflow repl \"handles = [launch_agent('count', config='worker', chunk='xx')]\"",
            ],
            "waiting",
        ),
    )
    g = Graph()
    g.add_config(AgentConfig(name="worker"))
    root = g.create_agent("Plan", config=claude.config(max_depth=1))
    run(Flow(StubLLM(lambda messages: WORKER)), g)
    assert g.result(root) == 1

    history = g.history(root)
    waiting = next(n for n in history if n.type == WAITING)
    first_cell = next(n for n in history if n.type == CODE)
    at_turn_end = g.fork(waiting, name="at-turn-end")
    g.send(at_turn_end, "Continue")
    mid_turn = g.fork(first_cell, name="mid-turn")
    calls_before = len(claude.calls())
    run(Flow(StubLLM(lambda messages: WORKER)), g)

    assert g.result(at_turn_end) == "plan: count"
    assert g.result(mid_turn) == "plan: count!"
    resumed, fresh = sorted(claude.calls()[calls_before:], key=lambda argv: "--resume" not in argv)
    assert flag(resumed, "--resume") == waiting.data["session_id"] and "--fork-session" in resumed
    prompt = flag(fresh, "-p")
    assert "--session-id" in fresh and prompt.startswith("Your task: Plan")
    assert "[cell 1] notes = 'plan: count'" in prompt
    assert "notes (str, 11 chars)" in prompt
    # Neither fork launched the worker again.
    assert len([a for a in g.agents() if a.config.name == "worker"]) == 1


def test_a_message_reopens_a_finished_claude_agent_in_the_same_session(claude):
    claude.script(
        ("and now?", ["rlmflow done second"], "ok"),
        ("first", ["rlmflow done first"], "ok"),
    )
    g = Graph()
    root = g.create_agent("first", config=claude.config())
    run(Flow(), g)
    assert g.result(root) == "first"

    g.send(root, "and now?")
    run(Flow(), g)
    assert g.result(root) == "second"
    first, second = claude.calls()
    assert flag(second, "--resume") == flag(first, "--session-id")


def test_claude_output_schema_and_plain_text_results(claude):
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    claude.script(("structured", [], '{"n": 3}'), ("plain", [], "just text"))
    g = Graph()
    structured = g.create_agent("structured", config=claude.config(output_schema=schema))
    plain = g.create_agent("plain", config=claude.config(harness_options={"tools": "loose"}))
    run(Flow(), g)

    assert g.result(structured) == {"n": 3}
    assert g.result(plain) == "just text"
    by_task = {flag(argv, "-p"): argv for argv in claude.calls()}
    assert json.loads(flag(by_task["structured"], "--json-schema")) == schema
    assert flag(by_task["plain"], "--permission-mode") == "acceptEdits"
    assert "Edit" in flag(by_task["plain"], "--allowedTools").split(",")


def test_tool_modes():
    strict = ClaudeCodeOptions(mcp=True).permissions()
    assert strict["allowed_tools"][0] == "Bash(rlmflow:*)" and "mcp__rlmflow" in strict["allowed_tools"]
    assert "Edit" in strict["disallowed_tools"]
    assert ClaudeCodeOptions(tools="loose").permissions()["permission_mode"] == "acceptEdits"
    with pytest.raises(ValueError, match="strict"):
        ClaudeCodeOptions(tools="anything")


def test_unknown_harness_options_are_an_error(claude):
    claude.script(("t", [], "ok"))
    g = Graph()
    g.create_agent("t", config=claude.config(harness_options={"tool": "loose"}))
    with pytest.raises(ValueError, match="invalid claude_code harness_options"):
        run(Flow(), g)


def test_a_session_starts_fresh_resumes_or_forks():
    g = Graph()
    agent = g.create_agent("t")
    start = Session.for_step(Step(g, agent, agent))
    assert start.resume is False and start == Session.for_step(Step(g, agent, agent))

    from rlmflow import Node

    end = g.append(agent, Node(WAITING, on=[], session_id="s1"))
    assert Session.for_step(Step(g, agent, end)) == Session("s1", resume=True, fork=False)
    fork = g.fork(end)
    assert Session.for_step(Step(g, fork, end)).fork is True
    cell = g.append(agent, Node(CODE, code="x = 1", output=""))
    assert Session.for_step(Step(g, agent, cell)).resume is False


def test_pi_reads_the_final_assistant_text_and_usage():
    events = [
        {"type": "session"},
        {
            "type": "agent_end",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "toolCall"}], "usage": {"input": 3, "output": 1}},
                {"role": "assistant", "content": [{"type": "text", "text": "done"}], "usage": {"input": 5, "output": 2}},
            ],
        },
    ]
    assert PiHarness.final_reply(events) == ("done", {"input_tokens": 8, "output_tokens": 3})
    failed = [{"type": "agent_end", "messages": [{"role": "assistant", "stopReason": "error", "errorMessage": "401"}]}]
    with pytest.raises(RuntimeError, match="401"):
        PiHarness.final_reply(failed)


def test_pi_session_files_resume_in_place_and_fork_by_copy(tmp_path):
    g = Graph()
    agent = g.create_agent("t", config=AgentConfig(harness="pi", harness_options={"session_dir": str(tmp_path)}))
    step, pi = Step(g, agent, agent), PiHarness()
    fresh = pi.session_file(step, Session("abc"))
    assert fresh == tmp_path / "abc.jsonl"
    fresh.write_text('{"type": "session"}\n')
    assert pi.session_file(step, Session(str(fresh), resume=True)) == fresh
    copy = pi.session_file(step, Session(str(fresh), resume=True, fork=True))
    assert copy != fresh and copy.read_text() == fresh.read_text()
