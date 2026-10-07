"""Child ``DoneOutput`` is delivered to the launching parent as a ``UserQuery``."""

from __future__ import annotations

import asyncio

from helpers import StubLLM

from rlmflow import AgentConfig, Flow, start
from rlmflow.engine import handoff
from rlmflow.graph.nodes import (
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    LLMOutput,
    ReplDead,
    UserQuery,
)

CHILD_GOAL = "Return alpha."
PARENT_QUERY = "Coordinate the child."


def user_text(messages):
    return "\n".join(message["content"] for message in messages if message["role"] == "user")


def handoff_queries(agent: AgentStart) -> list[UserQuery]:
    return [
        node
        for node in agent.transcript()
        if type(node) is UserQuery and "[handoff " in node.content
    ]


def test_dependency_routed_result_does_not_create_a_parent_handoff():
    root = start(PARENT_QUERY)
    action = ExecAction(code="launch")
    root.append(action)
    source = AgentStart(content="source", config=root.config.child("source"))
    action.append(source)
    consumer = AgentStart(
        content="consumer",
        args={"source": {"$rlmflow_agent": source.id}},
        config=root.config.child("consumer"),
    )
    action.append(consumer)
    result = DoneOutput(result="ready")
    source.append(result)

    assert handoff.apply(result) == []
    assert root.pending_handoffs == []


def launch_print():
    return (
        "```repl\n"
        f"child = launch_subagent(task={CHILD_GOAL!r})\n"
        "print('queued:', child.name)\n"
        "```"
    )


def launch_wait_print():
    return (
        "```repl\n"
        f"child = launch_subagent(task={CHILD_GOAL!r})\n"
        "value = await child\n"
        "print('got', value)\n"
        "```"
    )


def launch_and_finish_early():
    return (
        "```repl\n"
        f"child = launch_subagent(task={CHILD_GOAL!r})\n"
        "finish('early')\n"
        "```"
    )


def launch_then_error():
    return (
        "```repl\n"
        f"child = launch_subagent(task={CHILD_GOAL!r})\n"
        "print(1 / 0)\n"
        "```"
    )


def child_reply(messages):
    if "working" in user_text(messages):
        return "```repl\nfinish('alpha')\n```"
    return "```repl\nprint('working')\n```"


def finish_child(_messages):
    return "```repl\nfinish('alpha')\n```"


def parent_then_child(*, parent, child=finish_child):
    def reply(messages):
        if CHILD_GOAL in user_text(messages):
            return child(messages)
        return parent(messages)

    return reply


def test_background_spawn_flushes_user_query_then_calls_model():
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nfinish('got-alpha')\n```"
        if "queued:" in text:
            return "```repl\nprint('waiting')\n```"
        return launch_print()

    flow = Flow(StubLLM(parent_then_child(parent=parent)))
    root = flow.start(PARENT_QUERY, config=AgentConfig(max_depth=1, max_iters=10))

    asyncio.run(flow.arun(root))

    queries = handoff_queries(root)
    assert len(queries) == 1
    query = queries[0]
    assert isinstance(query.prev, ExecOutput)
    assert isinstance(query.next, LLMOutput)
    assert "alpha" in query.content
    assert "[handoff root.child0]" in query.content
    assert root.result() == "got-alpha"


def test_two_children_batch_into_one_user_query():
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nfinish('both')\n```"
        if "queued:" in text:
            return "```repl\nprint('waiting')\n```"
        return (
            "```repl\n"
            "a = launch_subagent(task='Return alpha.')\n"
            "b = launch_subagent(task='Return beta.')\n"
            "await asyncio.gather(a, b)\n"
            "print('queued:', a.name, b.name)\n"
            "```"
        )

    def reply(messages):
        text = user_text(messages)
        if "Return alpha." in text:
            return "```repl\nfinish('alpha')\n```"
        if "Return beta." in text:
            return "```repl\nfinish('beta')\n```"
        return parent(messages)

    flow = Flow(StubLLM(reply), workers=3)
    root = flow.start(PARENT_QUERY, config=AgentConfig(max_depth=1, max_iters=6))
    asyncio.run(flow.arun(root))

    queries = handoff_queries(root)
    assert len(queries) == 1
    assert "[handoff root.child0]" in queries[0].content
    assert "[handoff root.child1]" in queries[0].content
    assert root.result() == "both"


def test_child_done_during_parent_complete_does_not_raise_and_flushes_after_exec():
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nfinish('got-alpha')\n```"
        if "queued:" in text:
            return "```repl\nprint('still-here')\n```"
        return launch_print()

    flow = Flow(StubLLM(parent_then_child(parent=parent, child=child_reply)))
    root = flow.start(PARENT_QUERY, config=AgentConfig(max_depth=1, max_iters=8))

    asyncio.run(flow.arun(root))

    queries = handoff_queries(root)
    assert queries
    assert all(isinstance(query.prev, ExecOutput) for query in queries)
    assert root.result() == "got-alpha"


def test_flush_does_not_append_while_parent_step_is_in_flight():
    root = start(PARENT_QUERY)
    output = ExecOutput(content="queued")
    root.append(output)
    root.in_flight = output
    root.pending_handoffs.append(handoff.Handoff("root.alpha", "alpha", "alpha"))

    assert handoff.flush(root) is False
    assert root.frontier is output
    assert root.pending_handoffs

    root.in_flight = None
    assert handoff.flush(root) is True
    assert type(root.frontier) is UserQuery
    assert "[handoff root.alpha]" in root.frontier.content


def test_mapping_result_renders_as_fields():
    root = start(PARENT_QUERY)
    root.append(ExecOutput(content="queued"))
    root.pending_handoffs.append(handoff.Handoff("root.alpha", "alpha", {"n": 41, "note": "ok"}))
    assert handoff.flush(root)
    text = root.frontier.content
    assert "Each block is that child's submit() value" in text
    assert "n: 41" in text
    assert 'note: "ok"' in text
    assert "result: {" not in text


def test_scalar_result_stays_labeled_json():
    root = start(PARENT_QUERY)
    root.append(ExecOutput(content="queued"))
    root.pending_handoffs.append(handoff.Handoff("root.alpha", "alpha", "alpha"))
    assert handoff.flush(root)
    assert 'result: "alpha"' in root.frontier.content


def test_error_and_repl_dead_do_not_flush():
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nfinish('got-alpha')\n```"
        if "ZeroDivision" in text or "recovered" in text:
            return "```repl\nprint('recovered')\n```"
        return launch_then_error()

    flow = Flow(StubLLM(parent_then_child(parent=parent)))
    root = flow.start(PARENT_QUERY, config=AgentConfig(max_depth=1, max_iters=8))
    asyncio.run(flow.arun(root))

    queries = handoff_queries(root)
    assert queries
    assert all(isinstance(query.prev, ExecOutput) for query in queries)
    assert not any(isinstance(query.prev, ErrorOutput) for query in queries)
    assert root.result() == "got-alpha"


def test_flush_rejects_error_and_repl_dead_frontiers():
    for observation in (ErrorOutput(content="boom"), ReplDead()):
        root = start(PARENT_QUERY)
        root.append(observation)
        root.pending_handoffs.append(handoff.Handoff("root.alpha", "alpha", "alpha"))
        assert handoff.flush(root) is False
        assert root.frontier is observation


def test_parent_finish_with_flag_wakes_on_child_done():
    def parent(messages):
        if "[handoff " in user_text(messages):
            return "```repl\nfinish('woke')\n```"
        return launch_and_finish_early()

    flow = Flow(StubLLM(parent_then_child(parent=parent, child=child_reply)))
    root = flow.start(
        PARENT_QUERY,
        config=AgentConfig(
            max_depth=1,
            max_iters=8,
            resume_on_child_done=True,
        ),
    )
    asyncio.run(flow.arun(root))

    queries = handoff_queries(root)
    assert queries
    assert isinstance(queries[0].prev, DoneOutput)
    assert queries[0].prev.result == "early"
    assert root.result() == "woke"


def test_parent_finish_without_flag_stays_terminal():
    flow = Flow(StubLLM(parent_then_child(parent=lambda _m: launch_and_finish_early())))
    root = flow.start(
        PARENT_QUERY,
        config=AgentConfig(
            max_depth=1,
            max_iters=4,
            resume_on_child_done=False,
        ),
    )
    asyncio.run(flow.arun(root))

    assert root.result() == "early"
    assert not handoff_queries(root)
    assert isinstance(root.frontier, DoneOutput)


def test_wait_for_result_does_not_deadlock_and_still_handoffs():
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nfinish('got-alpha')\n```"
        return launch_wait_print()

    flow = Flow(StubLLM(parent_then_child(parent=parent)))
    root = flow.start(PARENT_QUERY, config=AgentConfig(max_depth=1, max_iters=6))
    asyncio.run(asyncio.wait_for(flow.arun(root), timeout=5))

    queries = handoff_queries(root)
    assert queries
    assert isinstance(queries[0].prev, ExecOutput)
    assert "got" in queries[0].prev.content
    assert root.result() == "got-alpha"


def _graph_with_finished_child(*, parent_frontier: str) -> AgentStart:
    root = start(PARENT_QUERY, max_depth=1)
    llm = LLMOutput(content="launch", code="print('queued')")
    root.append(llm)
    action = ExecAction(code="print('queued')")
    llm.append(action)
    child = AgentStart(content=CHILD_GOAL, config=root.config.child("alpha"))
    action.append(child)
    child.append(DoneOutput(result="alpha"))
    if parent_frontier == "exec":
        action.append(ExecOutput(content="queued: alpha"))
    elif parent_frontier == "llm":
        action.append(ExecOutput(content="queued: alpha"))
        root.frontier.append(LLMOutput(content="still", code="print('still')"))
    return root


def test_restore_flushes_when_parent_is_already_on_exec_output(tmp_path):
    root = _graph_with_finished_child(parent_frontier="exec")
    loaded = AgentStart.load(root.save(tmp_path / "idle-parent"))

    assert not handoff_queries(loaded)
    handoff.rebuild_mailboxes(loaded)
    queries = handoff_queries(loaded)
    assert len(queries) == 1
    assert isinstance(queries[0].prev, ExecOutput)
    assert "alpha" in queries[0].content


def test_restore_holds_mailbox_until_parent_lands_exec_output(tmp_path):
    root = _graph_with_finished_child(parent_frontier="llm")
    loaded = AgentStart.load(root.save(tmp_path / "busy-parent"))

    handoff.rebuild_mailboxes(loaded)
    assert not handoff_queries(loaded)
    assert loaded.pending_handoffs

    loaded.frontier.append(ExecAction(code="print('still')"))
    loaded.frontier.append(ExecOutput(content="still"))
    assert handoff.flush(loaded)
    queries = handoff_queries(loaded)
    assert len(queries) == 1
    assert isinstance(queries[0].prev, ExecOutput)


def test_replay_does_not_append_a_second_handoff(tmp_path):
    def parent(messages):
        text = user_text(messages)
        if "[handoff " in text:
            return "```repl\nfinish('got-alpha')\n```"
        if "queued:" in text:
            return "```repl\nprint('waiting')\n```"
        return launch_print()

    flow = Flow(StubLLM(parent_then_child(parent=parent)))
    root = flow.start(PARENT_QUERY, config=AgentConfig(max_depth=1, max_iters=6))
    asyncio.run(flow.arun(root))
    assert len(handoff_queries(root)) == 1

    path = root.save(tmp_path / "done")
    loaded = AgentStart.load(path)
    resumed = Flow(StubLLM(parent_then_child(parent=parent)))
    asyncio.run(resumed.arun(loaded))

    assert len(handoff_queries(loaded)) == 1
    assert loaded.result() == "got-alpha"


def test_user_query_after_exec_or_done_calls_model_directly():
    flow = Flow(StubLLM(lambda _messages: "reply"))
    for observation in (
        ExecOutput(content="observed"),
        DoneOutput(content="done", result="early"),
    ):
        root = start(PARENT_QUERY)
        root.append(observation)
        query = UserQuery(content="[handoff root.alpha]\nresult: 1")
        observation.append(query)
        assert isinstance(asyncio.run(flow.step(query)), LLMOutput)


def test_child_config_inherits_resume_on_child_done():
    parent = start(PARENT_QUERY, resume_on_child_done=True).config
    assert parent.child("worker").resume_on_child_done is True
    assert start(PARENT_QUERY).config.resume_on_child_done is True
    assert start(PARENT_QUERY).config.child("worker").resume_on_child_done is True
    opted_out = start(PARENT_QUERY, resume_on_child_done=False).config
    assert opted_out.resume_on_child_done is False
    assert opted_out.child("worker").resume_on_child_done is False


def test_child_done_does_not_flush_a_parent_exec_output():
    root = start(PARENT_QUERY, max_depth=1)
    output = ExecOutput(content="queued")
    root.append(output)
    child = AgentStart(content=CHILD_GOAL, config=root.config.child("alpha"))
    output.append(child)
    child.append(DoneOutput(result="alpha"))

    extra = handoff.apply(child.frontier)
    assert extra == []
    assert root.frontier is output
    assert root.pending_handoffs
    assert handoff.flush(root)
    assert type(root.frontier) is UserQuery


def test_supervisor_skips_roots():
    root = start(PARENT_QUERY, max_depth=1)
    action = ExecAction(code="launch")
    root.append(action)
    child = AgentStart(content=CHILD_GOAL, config=root.config.child("alpha"))
    action.append(child)

    assert handoff.supervisor(root) is None
    assert handoff.supervisor(child) is root
