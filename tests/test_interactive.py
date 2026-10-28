"""``rlmflow serve`` and the Claude Code hooks, driven in-process with recorded hook input."""

import asyncio
import json
import os
from pathlib import Path

import pytest
from helpers import StubLLM
from helpers import reasoning_repl as block

from rlmflow import CODE, DONE, MESSAGE, WAITING, AgentConfig, Flow, Graph
from rlmflow.agent_tools import cli, hooks
from rlmflow.cli.serve import INTERACTIVE, Serve, until
from rlmflow.graph.nodes import Step
from rlmflow.harnesses.interactive import read_transcript

SESSION = "11111111-2222-3333-4444-555555555555"
WORKER = block("submit(chunk.count('x'))")


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("RLMFLOW_URL", "RLMFLOW_TOKEN", "RLMFLOW_AGENT"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


class Session:
    """One recorded Claude Code session talking to one in-process serve."""

    def __init__(self, project: Path, serve: Serve, session_id: str = SESSION) -> None:
        self.project, self.serve, self.session_id = project, serve, session_id
        self.transcript = project / f"{session_id}.jsonl"
        self.transcript.touch()

    def hook(self, event: str, **data):
        data = {
            "cwd": str(self.project),
            "session_id": self.session_id,
            "transcript_path": str(self.transcript),
            **data,
        }
        return asyncio.to_thread(hooks.run, event, data)

    def prompt(self, text: str):
        return self.hook("prompt", prompt=text)

    def stop(self, reply: str = "ok"):
        return self.hook("stop", last_assistant_message=reply, stop_hook_active=False)

    def cell(self, code: str):
        return asyncio.to_thread(cli.request, "repl", {"code": code})


class SlowLLM:
    """Answers like the worker, a second and a half later."""

    async def chat(self, messages, **kwargs):
        await asyncio.sleep(1.5)
        return WORKER


async def serving(
    project: Path, graph: Graph | None = None, llm=None, **kwargs
) -> tuple[Serve, asyncio.Task]:
    if graph is None:
        graph = Graph()
    flow = Flow(llm or StubLLM(lambda messages: WORKER), configs=[AgentConfig(name="worker")])
    serve = Serve(flow, graph, graph_dir=project / "run", project=project, **kwargs)
    task = asyncio.create_task(serve.run())
    assert await until(lambda: serve.serve_file.exists() or None, 5)
    return serve, task


async def shut(session: Session, task: asyncio.Task) -> None:
    await session.hook("end", reason="other")
    await asyncio.wait_for(task, 10)


def types(g, agent):
    return [node.type for node in g.history(agent)]


def test_the_first_prompt_creates_a_bound_interactive_root_and_later_prompts_are_messages(project):
    async def main():
        serve, task = await serving(project)
        you = Session(project, serve)
        context = await you.prompt("/rlmflow:run count the x's in xxaxx")
        root = serve.root
        assert root.task == "count the x's in xxaxx" and root.name == "root"
        assert root.config.harness == INTERACTIVE and serve.session_id == SESSION
        assert "rlmflow repl <<'EOF'" in context and "keep a `notes` string" in context
        assert "only through `rlmflow repl`" not in context

        ran = await you.cell("text = 'xxaxx'\nprint(text.count('x'))")
        assert ran["output"].strip() == "4"
        await you.stop("There are 4.")
        assert types(serve.graph, root)[-1] == DONE

        assert await you.prompt("now count the a's") == ""
        await you.cell("submit(text.count('a'))")
        await you.stop("1")
        history = serve.graph.history(root)
        message = next(n for n in history if n.type == MESSAGE)
        assert message.data["text"] == "now count the a's" and message.data["sender"] == "user"
        assert serve.graph.result(root) == 1
        assert json.loads(serve.serve_file.read_text())["session_id"] == SESSION
        await shut(you, task)
        assert not serve.serve_file.exists()
        assert Graph.load(project / "run").result(root) == 1

    asyncio.run(main())


def test_a_cell_with_no_open_turn_is_refused(project):
    async def main():
        serve, task = await serving(project)
        you = Session(project, serve)
        await you.prompt("/rlmflow:run anything")
        await you.stop()
        with pytest.raises(cli.ToolError, match="no open turn"):
            await you.cell("print(1)")
        await shut(you, task)

    asyncio.run(main())


def test_stop_lands_done_with_a_submission_or_the_reply_when_nothing_is_pending(project):
    async def main():
        serve, task = await serving(project)
        you = Session(project, serve)
        await you.prompt("/rlmflow:run say hi")
        assert await you.stop("hi there") == ""
        [done] = [n for n in serve.graph.history(serve.root) if n.type == DONE]
        assert done.data["result"] == "hi there" and done.data["session_id"] == SESSION

        await you.prompt("submit 7")
        await you.cell("submit(7)")
        assert await you.stop("submitted") == ""
        assert serve.graph.result(serve.root) == 7
        await shut(you, task)

    asyncio.run(main())


def test_stop_with_children_lands_waiting_then_blocks_with_their_results(project):
    async def main():
        serve, task = await serving(project)
        you = Session(project, serve)
        await you.prompt("/rlmflow:run count x's in two chunks")
        await you.cell(
            "handles = [launch_agent('count', config='worker', chunk=c) for c in ('xxa', 'xax')]"
        )
        decision = json.loads(await you.stop("Launched two counters."))
        g, root = serve.graph, serve.root
        waiting = next(n for n in g.history(root) if n.type == WAITING)
        assert waiting.data["text"] == "Launched two counters."
        assert decision["decision"] == "block"
        assert decision["reason"].startswith("Agents you started have finished.")
        assert "[handoff root.worker-0]" in decision["reason"] and "result: 2" in decision["reason"]

        await you.cell("submit(sum(h.result() for h in handles))")
        assert await you.stop("4 in all") == ""
        assert types(g, root) == ["agent_start", CODE, WAITING, MESSAGE, CODE, DONE]
        assert g.result(root) == 4
        await shut(you, task)

    asyncio.run(main())


def test_a_stop_that_times_out_ends_the_turn_and_the_next_prompt_carries_the_results(
    project, monkeypatch
):
    monkeypatch.setattr(hooks, "STOP_WAIT", 0.1)

    async def main():
        serve, task = await serving(project, llm=SlowLLM())
        you = Session(project, serve)
        await you.prompt("/rlmflow:run count slowly")
        await you.cell("h = launch_agent('count', config='worker', chunk='xx')")
        assert await you.stop("launched") == ""
        assert types(serve.graph, serve.root)[-1] == WAITING
        assert await until(lambda: serve.flow.agent_tools.open_turn(serve.root), 10)
        context = await you.prompt("what did it find?")
        assert "Agents you started have finished." in context and "result: 2" in context
        await you.cell("submit(h.result())")
        await you.stop()
        assert serve.graph.result(serve.root) == 2
        await shut(you, task)

    asyncio.run(main())


def test_the_turn_end_records_the_session_and_usage_from_the_transcript(project):
    async def main():
        serve, task = await serving(project)
        you = Session(project, serve)
        you.transcript.write_text(
            json.dumps({"type": "user", "message": {"content": "old"}}) + "\n"
        )
        await you.prompt("/rlmflow:run hi")
        entries = [
            {
                "type": "assistant",
                "requestId": "r1",
                "message": {
                    "usage": {
                        "input_tokens": 3,
                        "cache_read_input_tokens": 100,
                        "output_tokens": 9,
                    },
                    "content": [{"type": "thinking"}],
                },
            },
            {
                "type": "assistant",
                "requestId": "r1",
                "message": {
                    "usage": {
                        "input_tokens": 3,
                        "cache_read_input_tokens": 100,
                        "output_tokens": 9,
                    },
                    "content": [{"type": "text", "text": "hello"}],
                },
            },
            {
                "type": "assistant",
                "requestId": "r2",
                "message": {
                    "usage": {
                        "input_tokens": 1,
                        "cache_creation_input_tokens": 50,
                        "output_tokens": 2,
                    },
                    "content": [{"type": "text", "text": "bye"}],
                },
            },
        ]
        with you.transcript.open("a") as handle:
            handle.writelines(json.dumps(entry) + "\n" for entry in entries)
        await you.hook("stop", stop_hook_active=False)
        done = serve.graph.frontier(serve.root)
        assert done.data["session_id"] == SESSION and done.data["result"] == "bye"
        assert done.data["usage"] == {"input_tokens": 154, "output_tokens": 11}
        assert serve.graph.usage().input_tokens == 154
        await shut(you, task)

    asyncio.run(main())
    usage, text, length = read_transcript(project / "missing.jsonl")
    assert usage == {"input_tokens": 0, "output_tokens": 0} and text == "" and length == 0


def test_serve_json_is_found_above_the_cwd_and_the_environment_wins(project, monkeypatch):
    serve_file = project / cli.SERVE_FILE
    serve_file.parent.mkdir()
    serve_file.write_text(
        json.dumps({"url": "http://127.0.0.1:1", "agent_token": "abc", "serve_token": "s"})
    )
    os.chmod(serve_file, 0o600)
    nested = project / "src" / "pkg"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert cli.endpoint() == ("http://127.0.0.1:1", "abc")
    monkeypatch.setenv("RLMFLOW_URL", "http://127.0.0.1:2")
    monkeypatch.setenv("RLMFLOW_TOKEN", "turn")
    assert cli.endpoint() == ("http://127.0.0.1:2", "turn")
    monkeypatch.delenv("RLMFLOW_URL")
    os.chmod(serve_file, 0o644)
    with pytest.raises(cli.ToolError, match="chmod 600"):
        cli.endpoint()


def test_an_agent_token_acts_only_for_its_own_agent(project):
    async def main():
        g = Graph()
        a, b = g.create_agent("a"), g.create_agent("b")
        for agent in (a, b):
            agent.bind_config(AgentConfig(harness=INTERACTIVE))
        flow = Flow(StubLLM(lambda messages: WORKER))
        server = flow.agent_tools
        url = await server.start()
        token_a = server.attach(a)
        try:
            async with server.turn(Step(g, b, b)) as turn_b:
                with pytest.raises(cli.ToolError, match="no open turn"):
                    await asyncio.to_thread(
                        cli.request, "repl", {"code": "print(1)"}, url=url, token=token_a
                    )
                ran = await asyncio.to_thread(
                    cli.request, "repl", {"code": "print('b')"}, url=url, token=turn_b.token
                )
                assert ran["output"].strip() == "b"
            with pytest.raises(cli.ToolError, match="serve token"):
                await asyncio.to_thread(cli.request, "stop", {}, url=url, token=token_a)
        finally:
            await flow.aclose()
        assert [n.type for n in g.history(b)] == ["agent_start", CODE]
        assert [n.type for n in g.history(a)] == ["agent_start"]

    asyncio.run(main())


def test_hooks_without_a_serve_or_from_another_session_do_nothing(project):
    async def main():
        stranger = {"cwd": str(project), "session_id": "other", "prompt": "hello"}
        for event in hooks.EVENTS:
            assert await asyncio.to_thread(hooks.run, event, stranger) == ""

        serve, task = await serving(project)
        you = Session(project, serve)
        await you.prompt("/rlmflow:run hi")
        intruder = Session(project, serve, session_id="other")
        assert await intruder.prompt("hello") == ""
        assert (await intruder.prompt("/rlmflow:run mine")).startswith(
            "rlmflow: this project's serve is bound"
        )
        assert await intruder.stop() == ""
        await intruder.hook("end")
        assert not task.done() and serve.flow.agent_tools.open_turn(serve.root) is not None
        await you.stop()
        await shut(you, task)

    asyncio.run(main())


def test_a_restarted_serve_continues_the_same_root_and_session(project):
    async def main():
        serve, task = await serving(project)
        you = Session(project, serve)
        await you.prompt("/rlmflow:run remember")
        await you.cell("notes = 'kept'")
        await you.stop("noted")
        root_id = serve.root.id
        await shut(you, task)

        again, task = await serving(project, graph=Graph.load(project / "run"))
        assert again.root.id == root_id and again.session_id == SESSION
        you = Session(project, again)
        await you.prompt("what are your notes?")
        await you.cell("submit(notes)")
        await you.stop()
        assert again.graph.result(again.root) == "kept"
        await shut(you, task)

    asyncio.run(main())
