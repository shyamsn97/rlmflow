"""Phase 3 — tools, builtins, history helpers, and the ported prompt builder."""

from __future__ import annotations

import asyncio

import pytest

from rflow import (
    BaseFlow,
    BaseOutputParser,
    DEFAULT_BUILDER,
    FILE_TOOLS,
    Flow,
    PromptBuilder,
    SYSTEM_PROMPT,
    get_tool_metadata,
    tool,
)
from rflow.prompts.default import MAX_STATIC_PROMPT_CHARS
from rflow.graph import ChildHandle, WaitRequest
from rflow.integrations.structured import StructuredOutputError
from rflow.runtime.context import EngineContext
from rflow.runtime.repl import DoneSignal
from rflow.tools import format_tool_line
from rflow.tools.builtins import (
    History,
    make_done,
    make_history,
    make_launch_subagents,
    make_show_vars,
)
from rflow.tools.filesystem import (
    append_file,
    edit_file,
    grep,
    line_count,
    ls,
    read_file,
    read_lines,
    write_file,
)
from rflow.tools.registry import HIDDEN_REPL_TOOL_NAMES, partition_repl_namespace

from .helpers import StubLLM, make_flow, run_to_completion

DONE_OK = '```repl\ndone("ok")\n```'


# ── @tool / metadata ────────────────────────────────────────────────────


def test_tool_decorator_and_metadata():
    @tool("Add two ints.", name="adder")
    def add(a: int, b: int) -> int:
        return a + b

    meta = get_tool_metadata(add)
    assert meta is not None
    assert meta.name == "adder"
    assert meta.description == "Add two ints."
    assert get_tool_metadata(lambda: None) is None


def test_tool_default_name_strips_prefix():
    @tool("x")
    def tool_search() -> None: ...

    assert get_tool_metadata(tool_search).name == "search"


def test_format_tool_line_renders_signature():
    @tool("Read a file.")
    def read_file_(path: str) -> str:
        return path

    line = format_tool_line(read_file_)
    assert line.startswith("- `read_file_(")
    assert line.endswith("`: Read a file.")
    assert "path" in line
    assert format_tool_line(lambda: None) == ""


def test_get_tool_metadata_unwraps_bound_method():
    flow = make_flow()
    # llm_query_batched is a @tool-decorated method
    meta = get_tool_metadata(flow.llm_query_batched)
    assert meta is not None and meta.name == "llm_query_batched"


# ── filesystem tools ────────────────────────────────────────────────────


def test_filesystem_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert write_file("sub/a.txt", "hello\nworld\n").startswith("Wrote")
    assert read_file("sub/a.txt") == "hello\nworld\n"
    assert line_count("sub/a.txt") == 2
    assert read_lines("sub/a.txt", 0, 1) == "hello"
    edit_file("sub/a.txt", ("world", "there"))
    assert "there" in read_file("sub/a.txt")
    assert "sub/a.txt" in ls("sub")
    hits = grep("hello", "sub")
    assert "hello" in hits


def test_file_tools_collection_decorated():
    names = {get_tool_metadata(fn).name for fn in FILE_TOOLS}
    assert {"read_file", "write_file", "grep", "ls"} <= names


# ── registry partition ──────────────────────────────────────────────────


def test_partition_repl_namespace_hides_control_tools():
    flow = make_flow(include_llm_query=False)
    flow.start("q")
    namespace = flow.build_tools()
    visible, hidden = partition_repl_namespace(namespace)
    assert set(hidden) == set(HIDDEN_REPL_TOOL_NAMES)
    assert "done" in visible and "launch_subagents" in visible
    assert "get_subagent_result" in visible
    assert "llm_query_batched" not in visible
    assert "flow_delegate" not in visible
    assert "flow_wait" not in visible


def test_partition_skips_private_and_noncallable():
    visible, hidden = partition_repl_namespace(
        {"_x": lambda: None, "data": "str", "SHOW_VARS": lambda: None, "tool_a": lambda: None}
    )
    assert set(visible) == {"tool_a"}
    assert not hidden


# ── builtins factories ──────────────────────────────────────────────────


def _graph_wait(child_results: list | None = None, requests: list | None = None):
    async def wait(request: WaitRequest):
        if requests is not None:
            requests.append(request)
        return child_results or []

    return wait


def _run_launch(coro):
    return asyncio.run(coro)


def test_make_launch_subagents_spec_validation():
    def fake_spawn(**spec):
        return ChildHandle(spec["name"])

    launch = make_launch_subagents(
        fake_spawn, graph_wait=_graph_wait(["a", "b"]), max_query_chars=2_000
    )
    with pytest.raises(TypeError, match="list of dict"):
        asyncio.run(launch("notalist"))
    with pytest.raises(TypeError, match="every spec to be a dict"):
        asyncio.run(launch([123]))
    with pytest.raises(KeyError, match="query"):
        asyncio.run(launch([{"name": "x"}]))
    with pytest.raises(TypeError, match="'query' must be a str"):
        asyncio.run(launch([{"name": "x", "query": 123}]))
    with pytest.raises(ValueError, match="'query' is too long"):
        asyncio.run(launch([{"name": "x", "query": "x" * 2_001}]))
    with pytest.raises(ValueError, match="inputs must not contain reserved key 'query'"):
        asyncio.run(launch([{"name": "x", "query": "q", "inputs": {"query": "bad"}}]))
    with pytest.raises(ValueError, match="duplicate child name"):
        asyncio.run(launch([{"name": "x", "query": "q"}, {"name": "x", "query": "q"}]))
    out = _run_launch(launch([{"name": "a", "query": "q"}, {"name": "b", "query": "q"}]))
    assert out == ["a", "b"]


def test_make_done_plain_and_structured():
    flow = make_flow()
    # plain: no schema in engine context
    context = EngineContext()
    done = make_done(flow, context)
    with pytest.raises(DoneSignal):
        done("  hi  ")
    assert context.done_result == "hi"

    # structured: schema stashed in context (as repl_for does per agent)
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    context2 = EngineContext(output_schema=schema)
    done2 = make_done(flow, context2)
    with pytest.raises(DoneSignal):
        done2({"x": 5})
    assert context2.done_result == '{"x": 5}'
    with pytest.raises(StructuredOutputError):
        done2({"x": "nope"})


# ── HISTORY ─────────────────────────────────────────────────────────────


def test_history_over_finished_run():
    flow = make_flow(DONE_OK)
    graph = run_to_completion(flow, "summarize this")
    hist = History(lambda: graph)
    msgs = hist.messages()
    assert len(hist) == len(msgs) >= 2
    assert msgs[0]["role"] == "user"
    assert any(m["role"] == "assistant" for m in msgs)
    # no system message in HISTORY
    assert all(m["role"] != "system" for m in msgs)
    assert hist.last(1) == [msgs[-1]]
    assert hist.read(0) == msgs[0]
    assert "summarize this" in hist.text()  # query is delivered as the message
    assert hist.grep("summarize")


def test_history_not_injected_into_repl_namespace_by_default():
    flow = make_flow(DONE_OK)
    agent = flow.start("q")
    repl = flow.repl_for(agent)
    assert "HISTORY" not in repl.namespace
    assert "SHOW_VARS" not in repl.namespace  # off by default


def test_history_reflects_live_trajectory():
    flow = make_flow(DONE_OK)
    flow.start("q")
    # Resolver reads the live run, so it tracks the graph across steps/copies.
    hist = History(lambda: flow.graph)
    before = len(hist)
    flow.step()  # appends llm_action + llm_output
    assert len(hist) > before


def test_make_history_tracks_live_graph_across_set_graph():
    # make_history resolves the agent by id, so adopting an override graph is
    # reflected with no rebinding of the REPL's HISTORY.
    flow = make_flow(DONE_OK)
    graph = flow.start("q")
    hist = make_history(flow, EngineContext(agent_id=graph.agent_id))
    before = len(hist)
    graph = flow.step(graph)
    assert len(hist) > before


# ── prompt builder + build_system_prompt ────────────────────────────────


def test_static_system_prompt_content():
    assert "CONTEXT" not in SYSTEM_PROMPT
    assert "llm_query_batched" not in SYSTEM_PROMPT
    assert "HISTORY" not in SYSTEM_PROMPT
    assert "launch_subagents" in SYSTEM_PROMPT
    assert "async clients and helpers can be used directly" in SYSTEM_PROMPT
    assert "not inside a function" not in SYSTEM_PROMPT
    assert "an over-long `query` is rejected" in SYSTEM_PROMPT
    assert "Never dump large `INPUTS` values" in SYSTEM_PROMPT
    assert 'INPUTS["query"][:' not in SYSTEM_PROMPT
    assert "Use exactly one block per assistant" in SYSTEM_PROMPT
    assert "never include a second ```repl fence" in SYSTEM_PROMPT
    assert "act as an orchestrator, not a solver" in SYSTEM_PROMPT
    assert "observe -> plan -> delegate independent branches -> integrate outputs -> verify" in SYSTEM_PROMPT
    assert "delegate those branches" in SYSTEM_PROMPT
    assert "keep the root focused on preparing inputs" in SYSTEM_PROMPT
    assert "Keep an orchestration mindset throughout the run" in SYSTEM_PROMPT
    assert "Before doing substantial work inline" in SYSTEM_PROMPT
    assert "Use the root agent for small local steps" in SYSTEM_PROMPT
    assert "act directly for trivial work" in SYSTEM_PROMPT
    assert "Failed checks are not a final answer" in SYSTEM_PROMPT
    assert 'done({"status": "failed", ...})' in SYSTEM_PROMPT
    assert "fix them or delegate a repair, then re-run the checks" in SYSTEM_PROMPT
    assert "With non-empty `INPUTS`, turn 1 is an inspection-only observation turn" in SYSTEM_PROMPT
    assert "Do not call `done(...)`, `launch_subagents(...)`, or effectful tools in that first block" in SYSTEM_PROMPT
    # The default prompt has clean examples that teach inspection and parallelization.
    assert "**Example 1 -- observe inputs before acting" in SYSTEM_PROMPT
    assert "**Example 2 -- fan out slices after observation" in SYSTEM_PROMPT
    assert "**Example 3 -- verify, repair, re-verify" in SYSTEM_PROMPT
    # The fanout example covers every batch, not a capped sample.
    assert "for i, batch in enumerate(batches)" in SYSTEM_PROMPT
    assert "launch_subagents" in SYSTEM_PROMPT
    assert "Delegate everything else" in SYSTEM_PROMPT
    assert "Push every long-context operation" in SYSTEM_PROMPT


def test_static_system_prompt_can_include_llm_query():
    # llm_query_batched is now a plain builtin tool: it adds no bespoke prompt
    # prose. It surfaces only as a Tools-section line in the live prompt (covered
    # by test_build_system_prompt_can_include_llm_query_tool); the static builder
    # has no agent graph, so its Tools section is empty regardless.
    prompt = DEFAULT_BUILDER.build(make_flow(include_llm_query=True))
    assert "Route by capability, not habit" not in prompt
    assert "llm_query_batched(" not in prompt


def test_flow_implements_base_contract():
    flow = make_flow()
    assert isinstance(flow, BaseFlow)
    assert isinstance(flow.output_parser, BaseOutputParser)
    assert flow.llm_clients["default"] is flow.llm


def test_static_system_prompt_length_ceiling():
    raw = DEFAULT_BUILDER.build()
    assert SYSTEM_PROMPT == raw
    assert len(raw) < MAX_STATIC_PROMPT_CHARS


def test_enable_structured_output_false_omits_schema_prompt():
    flow = make_flow(enable_structured_output=False)
    agent = flow.start("q")
    prompt = flow.build_system_prompt(agent)
    assert "## Structured Output" not in prompt


def test_enable_structured_output_false_skips_schema_section():
    flow = make_flow(enable_structured_output=False)
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    agent = flow.start("q", output_schema=schema)
    prompt = flow.build_system_prompt(agent)
    assert "## Structured Output" in prompt


def test_build_system_prompt_has_dynamic_tool_docs():
    flow = make_flow(include_llm_query=False)
    agent = flow.start("q")
    prompt = flow.build_system_prompt(agent)
    assert "\n## Tools\n\n" in prompt
    assert "`done(" in prompt
    assert "launch_subagents" in prompt
    assert "`get_subagent_result(" in prompt
    assert "llm_query_batched" not in prompt
    # hidden control tools are not advertised
    assert "`flow_delegate(" not in prompt
    assert "`flow_wait(" not in prompt
    # recursion status
    assert "recursion depth" in prompt


def test_tool_docs_use_agent_context_before_repl_exists():
    class ContextualToolFlow(Flow):
        def build_tools(self, engine_context=None):
            tools = super().build_tools(engine_context)
            agent_id = getattr(engine_context, "agent_id", "")

            @tool("Only visible to the root agent.")
            def root_only() -> None:
                return None

            @tool("Only visible to child agents.")
            def child_only() -> None:
                return None

            if agent_id == "root":
                tools["root_only"] = root_only
            else:
                tools["child_only"] = child_only
            return tools

    flow = ContextualToolFlow(StubLLM(), max_depth=1)
    root = flow.start("q")
    root_prompt = flow.build_system_prompt(root)
    handle = flow.spawn_child("root", "kid", "child task")
    child_prompt = flow.build_system_prompt(flow.graph[handle.agent_id])

    assert "`root_only(" in root_prompt
    assert "`child_only(" not in root_prompt
    assert "`child_only(" in child_prompt
    assert "`root_only(" not in child_prompt


def test_build_system_prompt_can_include_llm_query_tool():
    flow = make_flow(include_llm_query=True)
    agent = flow.start("q")
    prompt = flow.build_system_prompt(agent)
    assert "\n## Tools\n\n" in prompt
    tools_section = prompt.split("\n## Tools\n\n", 1)[1].split("\n\n## ", 1)[0]
    # Surfaced as a plain builtin tool line, not bespoke prompt prose.
    assert "- `llm_query_batched(" in tools_section


def test_build_system_prompt_structured_section():
    flow = make_flow()
    agent = flow.start("q", output_schema={"type": "object", "properties": {"x": {"type": "integer"}}})
    prompt = flow.build_system_prompt(agent)
    assert "JSON Schema" in prompt
    assert "Structured Output" in prompt


def test_system_prompt_is_rendered_once_and_stored_on_graph():
    flow = make_flow()
    agent = flow.start("q")
    # builder path stores the full rendered prompt on the graph (not "")
    assert agent.system_prompt
    assert "launch_subagents" in agent.system_prompt
    # survives serialization round-trip
    from rflow import Graph

    assert Graph.from_dict(agent.to_dict()).system_prompt == agent.system_prompt


def test_explicit_system_prompt_is_escape_hatch():
    flow = make_flow(system_prompt="STATIC PROMPT")
    agent = flow.start("q")
    assert flow.build_system_prompt(agent) == "STATIC PROMPT"


def test_explicit_system_prompt_appends_schema():
    flow = make_flow(system_prompt="STATIC")
    agent = flow.start("q", output_schema={"type": "object", "properties": {"x": {"type": "integer"}}})
    prompt = flow.build_system_prompt(agent)
    assert prompt.startswith("STATIC")
    assert "JSON Schema" in prompt


def test_prompt_builder_update_is_immutable():
    custom = DEFAULT_BUILDER.update("role", "You are a security auditor.")
    assert "security auditor" in custom.build()
    assert "security auditor" not in DEFAULT_BUILDER.build()
    assert isinstance(custom, PromptBuilder)


def test_tool_section_lists_models_when_multimodel():
    flow = Flow(StubLLM(DONE_OK), llm_clients={"fast": StubLLM(DONE_OK)})
    agent = flow.start("q")
    prompt = flow.build_system_prompt(agent)
    assert "Available models" in prompt
    assert "`fast`" in prompt


# ── first_prompt / depth notes / no-code error ──────────────────────────


def test_first_prompt_has_query_message_manifest_and_depth_note():
    flow = make_flow()
    msg = flow.first_prompt("do a thing", {"doc": "abc"}, depth=0)
    assert "Your REPL INPUTS contain:" in msg
    assert "query: str" not in msg  # query is the message, not an input
    assert "doc: str, 3 chars" in msg
    assert "Total input chars: 3" in msg
    assert "`INPUTS`" in msg
    assert "recursion depth 0" in msg
    assert "do a thing" in msg  # query is delivered as the message now
    assert "abc" not in msg  # input values are never dumped
    assert "inspection-only observation turn" in msg
    assert "Wait for that REPL output before planning" in msg
    assert "Think step-by-step" in msg


def test_first_prompt_without_inputs_omits_manifest():
    flow = make_flow()
    msg = flow.first_prompt("just do it", {}, depth=0)
    assert "Your REPL INPUTS contain:" not in msg
    assert "just do it" in msg


def test_followup_prompt_wraps_new_task_without_first_turn_language():
    flow = make_flow(max_depth=3)
    msg = flow.followup_prompt("convert stuff into python", depth=0)

    assert "New user task:" in msg
    assert "convert stuff into python" in msg
    assert "Continue using the REPL environment" in msg
    assert "delegate them with `await launch_subagents" in msg
    assert "recursion depth 0" in msg
    assert "full recursion budget" in msg
    assert "You have not interacted with the REPL environment" not in msg
    assert "Your REPL INPUTS contain:" not in msg
    assert "inspection-only observation turn" not in msg


def test_large_first_prompt_warns_not_to_print_contents():
    flow = make_flow()
    msg = flow.first_prompt("do a thing", {"doc": "x" * 50_001}, depth=0)
    assert "task-relevant windows" in msg
    assert "later turns or subagents" in msg


def test_first_prompt_delegation_nudge_is_depth_aware():
    flow = make_flow(max_depth=3)
    root_msg = flow.first_prompt("do a thing", {}, depth=0)
    child_msg = flow.first_prompt("do a thing", {}, depth=1)
    near_limit_msg = flow.first_prompt("do a thing", {}, depth=2)

    root = " ".join(root_msg.split())
    assert "root coordinator" in root
    assert "act directly for simple work" in root
    assert "make the launch block after the observation turn" in root
    assert "preparing focused inputs" in root

    child = " ".join(child_msg.split())
    assert "Own your assigned task" in child
    assert "solve simple work directly" in child
    assert "clearly separable subtasks" in child
    assert "root coordinator" not in child

    near_limit = " ".join(near_limit_msg.split())
    assert "Own your assigned task" in near_limit
    assert "near the recursion limit" in near_limit
    assert "clearly bounded leaf subtask" in near_limit
    assert "root coordinator" not in near_limit


def test_first_prompt_at_depth_limit_omits_delegation_nudge():
    flow = make_flow(max_depth=1)
    msg = flow.first_prompt("child task", {}, depth=1)
    assert "work can proceed independently" not in msg
    assert "cannot spawn sub-agents" in msg


def test_repl_output_truncated_by_default():
    flow = make_flow('```repl\nprint("x" * 5000)\n```')
    graph = flow.start("print a lot")
    flow.step()
    flow.step()
    out = graph.current().output
    assert len(out) < 4_100
    assert "...<truncated" in out
    assert "keep full data in variables" in out


def test_first_prompt_depth_limit_note():
    flow = make_flow(max_depth=2)
    at_limit = flow.first_prompt("q", {}, depth=2)
    assert "cannot spawn sub-agents" in at_limit


def test_no_code_block_message_used():
    flow = make_flow("I will not write a code block.")
    flow.start("q")
    flow.step()  # CallLLM -> LLMOutput with empty code
    flow.step()  # Exec -> no_code_block error
    errors = flow.graph.all_nodes.errors()
    assert errors and errors[0].error == "no_code_block"
    assert "repl" in errors[0].content


# ── reserved names + show_vars ──────────────────────────────────────────


def test_reserved_names_include_query_history_and_tools():
    for name in ("query", "HISTORY", "SHOW_VARS", "llm_query_batched", "done"):
        assert name in Flow._RESERVED
    flow = make_flow()
    with pytest.raises(ValueError, match="reserved"):
        flow.start("q", {"HISTORY": "x"})
    with pytest.raises(ValueError, match="reserved"):
        flow.start("q", {"query": "x"})


def test_show_vars_opt_in():
    flow = make_flow(DONE_OK, show_vars=True)
    agent = flow.start("q", {"doc": "hello"})
    repl = flow.repl_for(agent)
    assert "SHOW_VARS" in repl.namespace
    # Inputs live under INPUTS now, not as a top-level `doc` variable.
    # The query is the first message, not mirrored into INPUTS.
    assert repl.namespace["INPUTS"] == {"doc": "hello"}
    assert "doc" not in repl.namespace
    show = repl.namespace["SHOW_VARS"]
    out = show()
    assert out.get("INPUTS") == "dict"
    assert "done" not in out  # tools excluded
    # role section advertises it when enabled
    assert "SHOW_VARS()" in flow.build_system_prompt(agent)


def test_make_show_vars_filters_tools_and_hidden():
    @tool("a tool")
    def done():
        return None

    ns = {"x": 5, "_p": 1, "done": done, "_rflow_spawn_child": (lambda: None)}
    get = make_show_vars(ns)
    out = get()
    assert out == {"x": "int"}  # tool metadata + private names are filtered


# ── FileFlow integration ────────────────────────────────────────────────


class FileFlow(Flow):
    def build_tools(self, engine_context=None):
        file_tools = {get_tool_metadata(fn).name: fn for fn in FILE_TOOLS}
        return super().build_tools(engine_context) | file_tools


def test_file_flow_writes_and_reads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reply = (
        "```repl\n"
        'write_file("out.txt", "generated")\n'
        'done(read_file("out.txt"))\n'
        "```"
    )
    flow = FileFlow(StubLLM(reply), max_iters=3)
    result = flow.run("write a file")
    assert result == "generated"
    assert (tmp_path / "out.txt").read_text() == "generated"
    # file tools advertised in the prompt
    agent = flow.graph
    prompt = flow.build_system_prompt(agent)
    assert "\n## Tools\n\n" in prompt
    tools_section = prompt.split("\n## Tools\n\n", 1)[1].split("\n\n## ", 1)[0]
    assert "`write_file(" in tools_section
    assert "act as an orchestrator, not a solver" in prompt
    assert "`done(" in prompt
    assert "- `done(" not in tools_section
    assert "- `launch_subagents(" not in tools_section
    assert "- `llm_query_batched(" not in tools_section
    assert "- `flow_delegate(" not in tools_section
    assert "- `flow_wait(" not in tools_section
    assert "- `get_subagent_result(" in tools_section


# ── filesystem tools: edge cases (ported from legacy test_filesystem_tools) ─


def test_ls_returns_workspace_relative_paths_for_relative_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "item.txt").write_text("x")
    assert ls(".") == ["nested"]
    assert ls("nested") == ["nested/item.txt"]
    assert ls("nested/item.txt") == ["nested/item.txt"]


def test_ls_preserves_absolute_paths_for_absolute_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "item.txt").write_text("x")
    assert ls(str(tmp_path / "nested")) == [str(tmp_path / "nested" / "item.txt")]


def test_append_file_creates_dirs_and_appends(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    msg = append_file("logs/run.txt", "first\n")
    assert msg == "Appended 6 bytes to logs/run.txt"
    append_file("logs/run.txt", "second\n")
    assert read_file("logs/run.txt") == "first\nsecond\n"


def test_edit_file_reports_partial_and_no_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_file("f.txt", "alpha beta alpha")
    # only the first occurrence of a matched edit is replaced; misses are counted.
    msg = edit_file("f.txt", ("alpha", "X"), ("missing", "Y"))
    assert msg == "Applied 1/2 edits to f.txt"
    assert read_file("f.txt") == "X beta alpha"


def test_read_lines_and_line_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_file("f.txt", "l0\nl1\nl2\nl3")
    assert line_count("f.txt") == 4
    assert read_lines("f.txt", 0, 1) == "l0"  # 0-indexed, end-exclusive
    assert read_lines("f.txt", 1, 3) == "l1\nl2"


def test_line_count_and_read_lines_on_empty_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_file("empty.txt", "")
    assert line_count("empty.txt") == 0
    assert read_lines("empty.txt", 0, 5) == ""


def test_grep_searches_directory_and_caps_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_file("a.txt", "needle here\nno match\nneedle again")
    write_file("b.txt", "nothing")
    out = grep("needle", ".")
    lines = out.splitlines()
    assert len(lines) == 2
    assert all(line.startswith("a.txt:") and "needle" in line for line in lines)
    # max_results truncates
    assert len(grep("needle", ".", max_results=1).splitlines()) == 1
    assert grep("absent-pattern", ".") == ""


def test_file_tools_collection_is_complete():
    names = {get_tool_metadata(fn).name for fn in FILE_TOOLS}
    assert names == {
        "read_file",
        "write_file",
        "append_file",
        "edit_file",
        "ls",
        "read_lines",
        "line_count",
        "grep",
    }


def test_launch_subagents_mixes_refusal_strings_and_handles():
    calls = iter([ChildHandle("root.ok"), "refused: max depth"])

    def fake_spawn(**spec):
        return next(calls)

    launch = make_launch_subagents(
        fake_spawn,
        graph_wait=_graph_wait(["result:root.ok"]),
        max_query_chars=2_000,
    )
    out = _run_launch(launch([{"name": "ok", "query": "a"}, {"name": "refused", "query": "b"}]))
    # handle slot gets the awaited result; refusal slot keeps the string.
    assert out == ["result:root.ok", "refused: max depth"]


# ── PromptBuilder structure (ported from legacy test_prompt_rendering) ──


def test_default_builder_section_order():
    assert DEFAULT_BUILDER.names == [
        "role",
        "strategy",
        "format",
        "examples",
        "final",
        "structured-output",
        "tools",
        "status",
    ]


def test_prompt_builder_anchors_and_immutability():
    base = PromptBuilder().section("a", "A", title="A").section("c", "C", title="C")
    inserted = base.section("b", "B", title="B", before="c")
    assert inserted.names == ["a", "b", "c"]
    after = base.section("z", "Z", title="Z", after="a")
    assert after.names == ["a", "z", "c"]
    # original never mutated
    assert base.names == ["a", "c"]


def test_prompt_builder_callable_sections_and_overrides():
    seen = {}

    def body(engine, graph):
        seen["args"] = (engine, graph)
        return "callable body"

    builder = PromptBuilder().section("dyn", body, title="Dyn")
    out = builder.build("ENG", "GRAPH")
    assert "callable body" in out and seen["args"] == ("ENG", "GRAPH")
    # keyword override wins over the callable for one render
    assert "overridden" in builder.build("ENG", "GRAPH", dyn="overridden")


def test_default_prompt_skips_structured_section_without_schema():
    flow = make_flow()
    agent = flow.start("q")
    assert agent.output_schema is None
    assert "## Structured Output" not in flow.build_system_prompt(agent)
