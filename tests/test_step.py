import asyncio

from helpers import StubLLM

from rlmflow import (
    ExecAction,
    ExecActionStep,
    ExecOutput,
    Flow,
    LLMChunk,
    LLMOutput,
    LLMRequestStep,
    LLMUsage,
    MessageBuilder,
    PlanQuery,
    ReplRun,
    ReplStatus,
    Runtime,
    WrappedRuntime,
    start,
)


def test_step_advances_exactly_one_state_transition():
    flow = Flow(StubLLM(lambda _messages: '```repl\nfinish("ok")\n```'))
    root = start("query")

    transition = asyncio.run(flow.step(root))
    produced = transition.created

    assert produced.type == "plan_query"
    assert transition.submitted is root
    assert not transition.is_agent_start
    assert transition.error is None
    assert root.frontier is produced
    assert root.transcript() == [root, produced]


def test_a_run_can_be_driven_by_hand_one_step_at_a_time():
    flow = Flow(StubLLM(lambda _messages: '```repl\nfinish("ok")\n```'))
    root = start("query")

    produced = [asyncio.run(flow.step(root.frontier)).created for _ in range(3)]

    assert [node.type for node in produced] == [
        "plan_query",
        "llm_output",
        "exec_action",
    ]
    final = asyncio.run(flow.step(root.frontier)).created
    assert final.type == "done_output"
    assert root.terminal and root.result() == "ok"


def test_llm_request_step_runs_with_plain_fake_primitives():
    seen = []

    class FakeLLM:
        async def stream(self, messages):
            seen.append(messages)
            yield LLMChunk(
                text="reply",
                usage=LLMUsage(1, 2),
            )

    class Messages(MessageBuilder):
        def build(self, node):
            return [
                {"role": "system", "content": "system"},
                *node.project(),
            ]

    class UnusedRuntime(Runtime):
        def open(self, agent):
            raise AssertionError("runtime is not used")

    root = start("query")
    step = LLMRequestStep(
        llm=FakeLLM(),
        messages=Messages(),
        runtime=WrappedRuntime(UnusedRuntime(), lambda _node: {}),
    )

    plan = asyncio.run(step(root))
    landed = asyncio.run(step(plan))

    assert isinstance(plan, PlanQuery)
    assert isinstance(landed, LLMOutput)
    assert landed.usage == LLMUsage(1, 2)
    assert seen[0][-1]["content"] == plan.content


def test_exec_action_step_runs_with_the_same_primitive_abi():
    class UnusedLLM:
        async def stream(self, _messages):
            raise AssertionError("stream is not used")
            yield LLMChunk()

    class FakeRuntime(Runtime):
        def open(self, agent):
            raise AssertionError("open is not used")

        async def execute(self, node, code):
            return ReplRun(output="observed", status=ReplStatus.OK)

    class FakeRepl:
        def seed(self, tools, inputs):
            self.seeded = (tools, inputs)

    root = start("query")
    action = root.append(ExecAction(code="print('observed')"))
    runtime = FakeRuntime()
    repl = FakeRepl()
    runtime.repls[root.id] = repl
    wrapped = WrappedRuntime(runtime, lambda _node: {"tool": "value"})
    step = ExecActionStep(
        llm=UnusedLLM(),
        messages=MessageBuilder(),
        runtime=wrapped,
    )

    landed = asyncio.run(step(action))

    assert isinstance(landed, ExecOutput)
    assert landed.content == "observed"
    assert wrapped.runtime is runtime
    assert repl.seeded == ({"tool": "value"}, {})
