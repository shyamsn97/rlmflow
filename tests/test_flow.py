import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from rlmflow import (
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Flow,
    LLMOutput,
    SequentialPool,
    UserQuery,
    persistence,
    start,
)
from rlmflow.prompts import PromptProfile
from rlmflow.prompts.messages import UserPromptBuilder
from rlmflow.runtime import LocalRuntime
from rlmflow.tools import tool


class ScriptedLLM:
    """Replies by matching the last user turn against a list of (needle, reply)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        last = messages[-1]["content"]
        for needle, reply in self.script:
            if needle in last:
                if isinstance(reply, Exception):
                    raise reply
                return reply
        raise AssertionError(f"no scripted reply for {last!r}")


def block(code):
    return f"thinking\n```python\n{code}\n```"


def _walk(node):
    """Every node of a saved graph, which nests children the way the tree does."""
    yield node
    for child in node["children"]:
        yield from _walk(child)


def test_done():
    llm = ScriptedLLM([("count", block("x = 2\nprint(x)")), ("2", block("done(x * 3)"))])
    flow = Flow(llm)
    assert flow.run("count for me") == "6"
    assert llm.calls[0][0]["role"] == "system"


def test_error_then_recovery():
    llm = ScriptedLLM([("boom", block("1 / 0")), ("ZeroDivisionError", block("done('recovered')"))])
    root = start("boom")
    assert Flow(llm).run(root) == "recovered"
    types = [type(n).__name__ for n in root.walk()]
    assert "ErrorOutput" in types


def test_max_iters():
    llm = ScriptedLLM([("", block("print('again')"))])
    root = start("loop", max_iters=3)
    assert Flow(llm).run(root) == "[max_iters exceeded]"
    assert sum(isinstance(n, LLMOutput) for n in root.walk()) == 3


def test_inputs():
    llm = ScriptedLLM([("", block("done(INPUTS['doc'].upper())"))])
    root = start("shout", inputs={"doc": "hi"})
    assert Flow(llm).run(root) == "HI"
    assert "- doc: str, 2 chars" in Flow(llm).messages(root.frontier)[0]["content"]


def test_keep_n_messages_keeps_the_tail_without_rendering_the_history():
    class CountingPrompt(UserPromptBuilder):
        renders = 0

        def render_node(self, node):
            type(self).renders += 1
            return super().render_node(node)

    llm = ScriptedLLM([("", block("print('again')"))])
    root = start("loop", max_iters=6, keep_n_messages=3)
    flow = Flow(llm, user_prompt=CountingPrompt())
    assert flow.run(root) == "[max_iters exceeded]"
    assert len(root.transcript()) == 21

    CountingPrompt.renders = 0
    messages = flow.messages(root.frontier)
    assert messages[0]["role"] == "system"
    assert flow.truncation_summary in messages[1]["content"]
    assert messages[-1]["content"] == "again"
    # Four turns rendered to find the three worth keeping, not twenty-one nodes.
    assert CountingPrompt.renders == 6


async def stream(flow, root, until):
    return [node async for node in flow.run_streaming(root, until=until)]


def test_streaming_until_idle():
    llm = ScriptedLLM([("look", block("print('peek')")), ("peek", block("done('finished')"))])
    flow = Flow(llm)
    root = start("look")
    seen = asyncio.run(stream(flow, root, "idle"))
    assert isinstance(seen[-1], ExecOutput) and seen[-1].content.strip() == "peek"
    assert not root.terminal

    rest = asyncio.run(stream(flow, root, "done"))
    assert isinstance(rest[-1], DoneOutput) and root.result() == "finished"


def test_children_run_in_parallel():
    parent = block(
        "results = await launch_subagents("
        "[{'name': 'a', 'query': 'do a'}, {'name': 'b', 'query': 'do b'}])\n"
        "print(results)"
    )
    llm = ScriptedLLM(
        [
            ("split", parent),
            ("do a", block("done('A')")),
            ("do b", block("done('B')")),
            ("['A', 'B']", block("done('AB')")),
        ]
    )
    root = start("split this up")
    flow = Flow(llm)
    assert flow.run(root) == "AB"
    assert [child.config.name for child in root.sub_agents] == ["a", "b"]
    assert [child.result() for child in root.sub_agents] == ["A", "B"]

    own = {id(node) for node in root.transcript()}
    assert all(node.parent_agent is root for node in root.transcript())
    assert own.isdisjoint(id(node) for child in root.sub_agents for node in child.walk())
    child = root.sub_agents[0]
    assert child.transcript()[0] is child
    assert all(node.parent_agent is child for node in child.transcript())

    # Children branch off the action that launched them, so the parent's frontier
    # stayed on that action for the whole step, and its sequel is the step's output.
    action = next(n for n in root.transcript() if isinstance(n, ExecAction))
    assert action.children[:2] == root.sub_agents
    assert action.next is action.children[-1]
    assert isinstance(action.next, ExecOutput)
    assert root.frontier.next is None

    streamed = asyncio.run(stream(Flow(llm), start("split this up"), "done"))
    assert sum(isinstance(n, AgentStart) for n in streamed) == 2


def test_save_writes_a_run_directory(tmp_path):
    parent = block("print(await launch_subagents([{'name': 'a', 'query': 'do a'}]))")
    llm = ScriptedLLM(
        [("split", parent), ("do a", block("done('A')")), ("['A']", block("done('AB')"))]
    )
    root = start("split this up", inputs={"doc": "hi"})
    assert Flow(llm).run(root) == "AB"

    run = root.save(tmp_path / "run")
    graph = json.loads((run / "graph.json").read_text())
    assert (graph["version"], graph["node_count"], graph["metadata"]) == (2, 11, {})
    assert graph["root"]["payload"]["inputs"] == {"doc": "hi"}
    assert [node["order"] for node in graph["root"]["children"]] == [1]

    summary = json.loads((run / "latest.json").read_text())
    assert summary["agent_ids"] == ["root", "root.a"]
    assert (summary["root_agent_id"], summary["node_count"]) == ("root", 11)
    assert (summary["finished"], summary["result"]) == (True, "AB")

    action = next(n for n in root.transcript() if isinstance(n, ExecAction))
    saved = next(node for node in _walk(graph["root"]) if node["id"] == action.id)
    assert [node["type"] for node in saved["children"]] == ["agent_start", "exec_output"]
    assert saved["children"][0]["agent_id"] == "root.a"

    child = root.sub_agents[0]
    child_dir = run / "agents" / "root" / "a"
    agent_json = json.loads((child_dir / "agent.json").read_text())
    assert agent_json["agent_id"] == "root.a"
    assert (agent_json["parent_agent_id"], agent_json["depth"]) == ("root", 1)
    assert agent_json["query"] == "do a"

    lines = (child_dir / "session.jsonl").read_text().splitlines()
    session = [json.loads(line) for line in lines]
    assert [node["id"] for node in session] == [node.id for node in child.transcript()]
    assert all(node["agent_id"] == "root.a" and node["children"] == [] for node in session)
    assert json.loads((child_dir / "latest.json").read_text())["id"] == child.frontier.id


def test_run_records_prompts_seq_and_timing():
    parent = block(
        "print(await launch_subagents("
        "[{'name': 'a', 'query': 'do a'}, {'name': 'b', 'query': 'do b'}]))"
    )
    llm = ScriptedLLM(
        [
            ("split", parent),
            ("do a", block("done('A')")),
            ("do b", block("done('B')")),
            ("['A', 'B']", block("done('AB')")),
        ]
    )
    root = start("split this up")
    flow = Flow(llm)
    assert flow.run(root) == "AB"

    # seq counts within one agent, so every agent starts over at 0. The tree plus
    # these is the whole run: two children racing changes neither.
    for agent in (root, *root.sub_agents):
        assert [node.seq for node in agent.transcript()] == list(range(len(agent.transcript())))

    # Only executed nodes are timed; a node written inside a step is not a step.
    stepped = [node for node in root.walk() if isinstance(node, (LLMOutput, ExecOutput))]
    assert all(node.timing()["duration_ms"] >= 0 for node in stepped)
    written = [node for node in root.walk() if isinstance(node, AgentStart)]
    assert written and all(node.timing() == {} for node in written)

    turns = [node for node in root.transcript() if isinstance(node, LLMOutput)]
    first, last = root.system_prompt_for(turns[0]), root.system_prompt_for(turns[-1])
    assert "Recursive Coding Agent" in first
    assert "## First Turn" in first and "## First Turn" not in last
    assert root.latest_system_prompt() == last
    # The text is stored once per distinct prompt, and turns only keep the id.
    assert sorted(root.system_prompts) == sorted({turn.prompt_id for turn in turns})
    assert turns[0].to_dict()["metadata"]["system_prompt"] == turns[0].prompt_id
    assert turns[0].timing()["duration_ms"] >= 0


def test_appending_off_the_frontier_is_refused():
    llm = ScriptedLLM([("count", block("x = 2\nprint(x)")), ("2", block("done(x * 3)"))])
    root = start("count for me")
    assert Flow(llm).run(root) == "6"

    stale = root.transcript()[1]
    with pytest.raises(ValueError, match="frontier"):
        stale.append(UserQuery(content="too late"))
    assert stale.children == [stale.next]


def test_structured_output_is_stored_parsed():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
    llm = ScriptedLLM([("count", block("done({'n': 41})"))])
    root = start("count", output_schema=schema)
    assert Flow(llm).run(root) == {"n": 41}
    assert root.frontier.to_dict()["payload"]["result"] == {"n": 41}


def test_save_load_roundtrip(tmp_path):
    parent = block("print(await launch_subagents([{'name': 'a', 'query': 'do a'}]))")
    llm = ScriptedLLM(
        [("split", parent), ("do a", block("done('A')")), ("['A']", block("done('AB')"))]
    )
    root = start("split this up", inputs={"doc": "hi"})
    Flow(llm).run(root)

    loaded = AgentStart.load(root.save(tmp_path / "run"))
    assert persistence.to_dict(loaded) == persistence.to_dict(root)
    assert persistence.summary(loaded) == persistence.summary(root)
    assert (loaded.result(), loaded.terminal) == ("AB", True)
    assert [sub.config.path for sub in loaded.sub_agents] == ["root.a"]
    assert loaded.sub_agents[0].result() == "A"
    assert loaded.save(tmp_path / "again") and (tmp_path / "again" / "graph.json").is_file()


def paused_run(tmp_path, name="run"):
    """A saved run that set a variable and stopped before using it."""
    llm = ScriptedLLM([("count", block("secret = 41\nprint('stored')"))])
    root = start("count for me")
    asyncio.run(stream(Flow(llm), root, "idle"))
    assert isinstance(root.frontier, ExecOutput)
    return AgentStart.load(root.save(tmp_path / name))


def test_resuming_a_saved_run_replays_its_repl(tmp_path):
    loaded = paused_run(tmp_path)
    llm = ScriptedLLM([("stored", block("done(secret + 1)"))])
    assert Flow(llm).run(loaded) == "42"
    assert not [node for node in loaded.walk() if isinstance(node, ErrorOutput)]
    assert len(llm.calls) == 1  # the replay itself asks the model nothing


def test_lazy_restore_tells_the_agent_its_repl_is_gone(tmp_path):
    loaded = paused_run(tmp_path)
    llm = ScriptedLLM([("Re-derive", block("done(str(2 + 2))"))])
    flow = Flow(llm, restore="lazy")
    assert flow.run(loaded) == "4"

    notes = [node for node in loaded.transcript() if node.content == flow.cold_repl_note]
    assert len(notes) == 1 and isinstance(notes[0], UserQuery)
    assert "secret" not in flow.runtime.namespace_for(loaded)


def test_replay_reads_recorded_child_answers_instead_of_relaunching(tmp_path):
    parent = block(
        "answers = await launch_subagents([{'name': 'kid', 'query': 'add up'}])\n"
        "print(answers[0])"
    )
    llm = ScriptedLLM([("delegate", parent), ("add up", block("done('20')"))])
    root = start("delegate", max_depth=2)

    def parent_turn_ended(node, root):
        return isinstance(node, ExecOutput) and node.parent_agent is root

    asyncio.run(stream(Flow(llm), root, parent_turn_ended))
    loaded = AgentStart.load(root.save(tmp_path / "run"))

    resumed = ScriptedLLM([("20", block("done('parent saw ' + answers[0])"))])
    assert Flow(resumed).run(loaded) == "parent saw 20"
    # The child ran once, when it was recorded: no second launch, no second model call.
    assert [child.config.name for child in loaded.sub_agents] == ["kid"]
    assert len(resumed.calls) == 1


def test_replaying_agents_can_skip_their_own_unrepeatable_work(tmp_path):
    trace: list[str] = []
    code = block("if ENV['RLMFLOW_REPLAY'] == '0':\n    log('live')\nprint('set up')")
    flow = Flow(ScriptedLLM([("count", code)]))
    flow.inject("log", trace.append)
    root = start("count for me")
    asyncio.run(stream(flow, root, "idle"))
    assert trace == ["live"]

    loaded = AgentStart.load(root.save(tmp_path / "run"))
    resumed = Flow(ScriptedLLM([("set up", block("done('ok')"))]))
    resumed.inject("log", trace.append)
    assert resumed.run(loaded) == "ok"
    assert trace == ["live"]  # the replay saw RLMFLOW_REPLAY=1 and skipped it


def test_a_finished_run_resumes_to_its_recorded_answer(tmp_path):
    llm = ScriptedLLM([("count", block("done('six')"))])
    root = start("count for me")
    assert Flow(llm).run(root) == "six"

    loaded = AgentStart.load(root.save(tmp_path / "run"))
    idle = ScriptedLLM([])
    assert Flow(idle).run(loaded) == "six"
    assert idle.calls == []  # nothing left to step


def test_a_dead_repl_is_an_observation_and_is_replaced():
    class OneDeadRepl(LocalRuntime):
        """Whatever the first agent gets, its REPL dies the way a sandbox does."""

        opened = 0

        def open(self, agent):
            self.opened += 1
            repl = super().open(agent)
            if self.opened == 1:

                async def die(code):
                    raise ConnectionResetError("sandbox went away")

                repl.run = die
            return repl

    llm = ScriptedLLM(
        [("count", block("x = 1")), ("ConnectionResetError", block("done('recovered')"))]
    )
    runtime = OneDeadRepl()
    root = start("count for me")
    assert Flow(llm, runtime=runtime).run(root) == "recovered"

    errors = [node for node in root.walk() if isinstance(node, ErrorOutput)]
    assert len(errors) == 1 and "REPL execution failed" in errors[0].content
    assert runtime.opened == 2  # the dead one was dropped, not handed out again

    # A dead REPL reads differently from code that raised: the agent is told its
    # namespace went with it, and the record says which kind of failure it was.
    assert errors[0].error == "repl"
    assert errors[0].content.endswith(Flow.cold_repl_note)
    assert errors[0].to_dict()["payload"]["error"] == "repl"


def test_a_live_launch_still_refuses_a_reused_child_name():
    parent = block(
        "await launch_subagents([{'name': 'kid', 'query': 'do it'}])\n"
        "print(await launch_subagents([{'name': 'kid', 'query': 'again'}]))"
    )
    llm = ScriptedLLM(
        [
            ("delegate", parent),
            ("do it", block("done('once')")),
            ("duplicate child name", block("done('caught')")),
        ]
    )
    root = start("delegate", max_depth=2)
    assert Flow(llm).run(root) == "caught"
    assert [child.config.name for child in root.sub_agents] == ["kid"]


def test_saving_a_smaller_tree_prunes_stale_agents(tmp_path):
    parent = block("print(await launch_subagents([{'name': 'a', 'query': 'do a'}]))")
    llm = ScriptedLLM(
        [("split", parent), ("do a", block("done('A')")), ("['A']", block("done('AB')"))]
    )
    root = start("split this up")
    Flow(llm).run(root)

    run = root.save(tmp_path / "run")
    assert (run / "agents" / "root" / "a" / "agent.json").is_file()

    root.sub_agents.clear()
    root.save(run)
    assert not (run / "agents" / "root" / "a").exists()


def test_max_depth_refusal():
    llm = ScriptedLLM(
        [
            ("deep", block("print(await launch_subagents([{'query': 'x'}]))")),
            ("refused", block("done('stopped')")),
        ]
    )
    root = start("go deep", max_depth=0)
    assert Flow(llm).run(root) == "stopped"
    assert root.sub_agents == []


def test_grandchildren():
    llm = ScriptedLLM(
        [
            (
                "plan",
                block("print(await launch_subagents([{'name': 'mid', 'query': 'go mid'}]))"),
            ),
            (
                "go mid",
                block("print(await launch_subagents([{'name': 'leaf', 'query': 'go leaf'}]))"),
            ),
            ("go leaf", block("done('leaf')")),
            ("['leaf']", block("done('mid')")),
            ("['mid']", block("done('root')")),
        ]
    )
    root = start("plan the work", max_depth=2)
    assert Flow(llm).run(root) == "root"
    mid = root.sub_agents[0]
    assert [agent.config.path for agent in (mid, mid.sub_agents[0])] == [
        "root.mid",
        "root.mid.leaf",
    ]


class BarrierLLM:
    """Blocks each caller until every expected child has asked for a reply."""

    def __init__(self, parties, replies):
        self.barrier = asyncio.Barrier(parties)
        self.replies = replies

    async def chat(self, messages):
        last = messages[-1]["content"]
        for needle, reply in self.replies:
            if needle in last:
                if needle.startswith("child"):
                    await self.barrier.wait()
                return reply
        raise AssertionError(f"no scripted reply for {last!r}")


def test_children_step_concurrently():
    specs = ", ".join(f"{{'name': 'c{i}', 'query': 'child {i}'}}" for i in range(3))
    llm = BarrierLLM(
        3,
        [
            ("fan out", block(f"print(await launch_subagents([{specs}]))")),
            *[(f"child {i}", block(f"done({i})")) for i in range(3)],
            ("['0', '1', '2']", block("done('all')")),
        ],
    )
    root = start("fan out", max_depth=1)

    async def main():
        # Sequential children would never clear the barrier, so this would hang.
        return await asyncio.wait_for(Flow(llm).arun(root), timeout=5)

    assert asyncio.run(main()) == "all"
    assert [child.result() for child in root.sub_agents] == ["0", "1", "2"]


def test_break_cancels_the_run():
    llm = ScriptedLLM([("", block("print('again')"))])
    flow = Flow(llm)

    async def main():
        stream = flow.run_streaming(start("loop forever"))
        async for _node in stream:
            break
        await stream.aclose()
        assert flow.queue is None
        await flow.aclose()

    asyncio.run(main())


def test_step_failure_reaches_the_caller():
    class BrokenLLM:
        def chat(self, messages):
            raise RuntimeError("model is down")

    try:
        Flow(BrokenLLM()).run("anything")
    except RuntimeError as exc:
        assert str(exc) == "model is down"
    else:
        raise AssertionError("expected the step failure to propagate")


def test_custom_tool_is_injected_and_documented():
    @tool("Upper-case a string.")
    def shout(text: str) -> str:
        return text.upper()

    llm = ScriptedLLM([("", block("done(shout('hi'))"))])
    flow = Flow(llm, tools=[shout])
    root = start("say it")
    assert flow.run(root) == "HI"
    prompt = flow.messages(root.frontier)[0]["content"]
    assert "`shout(text: str) -> str`: Upper-case a string." in prompt


def test_child_output_schema_comes_back_parsed():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "number"}},
        "required": ["n"],
    }
    parent = block(
        "results = await launch_subagents([{'name': 'x', 'query': 'count', "
        f"'output_schema': {schema!r}}}])\n"
        "print(results[0]['n'] + 1)"
    )
    llm = ScriptedLLM(
        [
            ("measure", parent),
            ("count", block("done({'n': 41})")),
            ("42", block("done('42')")),
        ]
    )
    root = start("measure it", max_depth=1)
    assert Flow(llm).run(root) == "42"
    assert root.sub_agents[0].config.output_schema == schema


def test_prompt_profile_picks_a_child_system_prompt():
    parent = block(
        "print(await launch_subagents("
        "[{'name': 'terse', 'query': 'be brief', 'prompt_profile': 'terse'}]))"
    )
    llm = ScriptedLLM(
        [("delegate", parent), ("be brief", block("done('ok')")), ("['ok']", block("done('done')"))]
    )
    profiles = {"terse": PromptProfile(system="Answer in one word.", description="brief")}
    flow = Flow(llm, prompt_profiles=profiles)
    root = start("delegate briefly", max_depth=1)
    assert flow.run(root) == "done"
    child = root.sub_agents[0]
    assert flow.messages(child.frontier)[0]["content"] == "Answer in one word."
    assert "`terse` — brief" in flow.messages(root.frontier)[0]["content"]


def test_a_profile_takes_a_bare_user_prompt_function():
    llm = ScriptedLLM([("the board so far", block("done('solved')"))])
    profiles = {"board": PromptProfile(user=lambda flow, node: "the board so far")}
    root = start("play", prompt_profile="board")

    assert Flow(llm, prompt_profiles=profiles).run(root) == "solved"
    assert [node.content for node in root.transcript() if isinstance(node, UserQuery)] == [
        "the board so far"
    ]


def test_named_model_routes_to_another_client():
    default = ScriptedLLM([("", block("done('wrong client')"))])
    fast = ScriptedLLM([("", block("done('fast client')"))])
    root = start("who answers", model="fast")
    assert Flow(default, llm_clients={"fast": fast}).run(root) == "fast client"
    assert default.calls == []


def test_long_child_query_is_refused():
    llm = ScriptedLLM(
        [
            ("delegate", block("print(await launch_subagents([{'query': 'far too long'}]))")),
            ("refused", block("done('stopped')")),
        ]
    )
    root = start("delegate this", max_depth=1, max_query_chars=4)
    assert Flow(llm).run(root) == "stopped"
    assert root.sub_agents == []


def test_budget_stops_the_run():
    llm = ScriptedLLM([("", block("print('spending')"))])
    llm.last_usage = SimpleNamespace(input_tokens=8, output_tokens=8)
    root = start("spend it all", max_budget=10)
    assert Flow(llm).run(root) == "[budget exceeded]"


def test_child_failure_reaches_the_parent():
    llm = ScriptedLLM(
        [
            ("delegate", block("print(await launch_subagents([{'query': 'boom'}]))")),
            ("boom", RuntimeError("child is down")),
            ("child is down", block("done('recovered')")),
        ]
    )
    root = start("delegate this", max_depth=1)
    assert Flow(llm).run(root) == "recovered"


def test_aclose_drops_the_work_and_the_repls():
    llm = ScriptedLLM([("count", block("done('six')"))])
    flow = Flow(llm)

    async def run_then_close():
        await flow.arun(start("count for me"))
        assert flow.repls
        await flow.aclose()

    asyncio.run(run_then_close())
    assert not flow.repls
    assert flow.queue is None


def test_parallel_roots_use_parallel_stream_not_competing_stream_drivers():
    llm = ScriptedLLM([("one", block("done('first')")), ("two", block("done('second')"))])
    flow = Flow(llm)
    first, second = start("one"), start("two")

    async def both():
        left = flow.run_streaming(first)
        right = flow.run_streaming(second)
        await anext(left)
        with pytest.raises(RuntimeError, match="already driving"):
            await anext(right)
        await left.aclose()
        await right.aclose()

    asyncio.run(both())


class ReplyLLM:
    """Answers from a function of the messages, and records the kwargs it got."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        return self.fn(messages)


def peaked(delay=0.02):
    """A blocking reply, plus the most calls it ever saw running at once."""
    lock = threading.Lock()
    seen = {"active": 0, "peak": 0}

    def reply(messages):
        with lock:
            seen["active"] += 1
            seen["peak"] = max(seen["peak"], seen["active"])
        try:
            time.sleep(delay)
            return messages[-1]["content"].upper()
        finally:
            with lock:
                seen["active"] -= 1

    return reply, seen


def test_llm_query_batched_is_an_opt_in_tool():
    def reply(messages):
        if len(messages) == 1:  # a one-shot query, with no system turn of its own
            return messages[0]["content"].upper()
        return block('done("|".join(await llm_query_batched(["a", "b"])))')

    flow = Flow(ReplyLLM(reply), use_llm_query=True)
    assert flow.run("fan out") == "A|B"

    # Off by default, so an agent cannot call what its prompt never described.
    assert "llm_query_batched" not in Flow(ReplyLLM(reply)).tools


def test_workers_bounds_blocking_model_calls():
    reply, seen = peaked()
    flow = Flow(ReplyLLM(reply), workers=2)

    assert asyncio.run(flow.llm_query_batched(["a", "b", "c", "d"])) == ["A", "B", "C", "D"]
    assert seen["peak"] == 2


def test_a_pool_replaces_the_one_workers_would_have_built():
    reply, seen = peaked()
    flow = Flow(ReplyLLM(reply), pool=SequentialPool())

    assert asyncio.run(flow.llm_query_batched(["a", "b", "c"])) == ["A", "B", "C"]
    assert seen["peak"] == 1

    with pytest.raises(ValueError, match="not both"):
        Flow(ReplyLLM(reply), workers=2, pool=SequentialPool())


def test_llm_request_timeout_gives_up_on_a_slow_call():
    class SlowLLM:
        async def chat(self, messages):
            await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        Flow(SlowLLM(), llm_request_timeout=0.01).run("hang")


def test_llm_request_timeout_is_handed_to_clients_that_take_one():
    llm = ReplyLLM(lambda _messages: block("done('ok')"))
    assert Flow(llm, llm_request_timeout=5).run("go") == "ok"
    assert [call["timeout"] for call in llm.calls] == [5]

    # A client with no timeout parameter is called the way it was written.
    lean = ScriptedLLM([("", block("done('ok')"))])
    assert Flow(lean, llm_request_timeout=5).run("go") == "ok"
