import asyncio

import pytest
from helpers import StubLLM

from rlmflow import (
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    FinalQuery,
    Flow,
    LLMOutput,
    ReplRun,
    ReplStatus,
    StructuredResult,
    TransitionDecision,
    TransitionPolicyError,
    TruncationSummary,
    UserQuery,
    start,
)
from rlmflow.engine.steps import complete, node_for_run
from rlmflow.engine.transitions import TransitionFunction, resolve


def test_transition_metadata_lives_on_the_callable():
    assert isinstance(complete, TransitionFunction)
    assert complete.sources == (AgentStart, UserQuery, ErrorOutput, ExecOutput)
    assert complete.when is None
    assert complete.fn.__name__ == "complete"
    assert (
        complete.id
        == TransitionFunction.create(
            complete.fn,
            source=complete.sources,
        ).id
    )


def test_flow_transition_returns_the_callable_transition():
    flow = Flow(object())

    @flow.transition(source=ErrorOutput)
    def retry(node):
        return UserQuery(content=node.content)

    assert isinstance(retry, TransitionFunction)
    assert retry(ErrorOutput(content="retry")).content == "retry"
    assert retry.node_only is True


def test_resolve_reuses_one_recorded_semantic_choice():
    judged = []

    async def judge(node, candidates):
        judged.append(candidates)
        return TransitionDecision(
            candidates={rule_id: condition for rule_id, _label, condition in candidates},
            source_node_id=node.id,
        )

    semantic = TransitionFunction.create(
        lambda node: UserQuery(content="semantic"),
        source=ErrorOutput,
        when="Needs a different plan",
        node_only=True,
    )
    skipped = TransitionFunction.create(
        lambda node: UserQuery(content="skipped"),
        source=ErrorOutput,
        when=lambda node: False,
        node_only=True,
    )
    fallback = TransitionFunction.create(
        lambda node: UserQuery(content="fallback"),
        source="all",
        node_only=True,
    )
    root = start("query")
    error = ErrorOutput(content="failed")
    root.append(error)

    chosen, anchor, source = asyncio.run(resolve((semantic, skipped, fallback), error, judge))
    assert chosen is fallback
    assert source is error
    assert isinstance(anchor, TransitionDecision)
    assert anchor.prev is error
    assert judged == [((semantic.id, semantic.__name__, "Needs a different plan"),)]

    chosen, again, source = asyncio.run(resolve((semantic, skipped, fallback), anchor, judge))
    assert chosen is fallback
    assert again is anchor
    assert source is error
    assert judged == [((semantic.id, semantic.__name__, "Needs a different plan"),)]


def test_one_turn_budget_starts_on_the_final_turn():
    root = start("query", max_iters=1)

    result = asyncio.run(Flow(object()).step(root))

    assert isinstance(result, FinalQuery)


def test_startup_calls_the_model_with_finish_available():
    root = start("query", args={"context": "material"})
    requests = []

    class Recorder:
        def chat(self, messages):
            requests.append(messages)
            return "reply"

    flow = Flow(Recorder())
    reply = asyncio.run(flow.step(root))

    assert isinstance(reply, LLMOutput)
    assert reply.prev is root
    assert [message["role"] for message in requests[0]] == ["system", "user"]
    assert requests[0][-1]["content"] == "query"
    assert "finish" in flow.build_tools(root)


def test_first_block_may_finish():
    root = start("query")
    reply = LLMOutput(content="```repl\nfinish(42)\n```", code="finish(42)")
    root.append(reply)
    action = ExecAction(code=reply.code)
    reply.append(action)

    done = node_for_run(action, ReplRun(status=ReplStatus.DONE, answer=42))

    assert isinstance(done, DoneOutput)
    assert done.result == 42


def test_queries_and_observations_call_the_model_directly():
    flow = Flow(StubLLM(lambda _messages: "reply"))
    root = start("query")
    query = UserQuery(content="host instruction")
    root.append(query)
    assert isinstance(asyncio.run(flow.step(query)), LLMOutput)

    for observation_type, content in (
        (ExecOutput, "observed"),
        (ErrorOutput, "failed"),
    ):
        root = start("query")
        observation = observation_type(content=content)
        root.append(observation)
        assert isinstance(asyncio.run(flow.step(observation)), LLMOutput)

    for prior in (ExecOutput(content="observed"), DoneOutput(result="child")):
        root = start("query")
        root.append(prior)
        query = UserQuery(content="[handoff root.child]\nresult: 1")
        prior.append(query)
        assert isinstance(asyncio.run(flow.step(query)), LLMOutput)


def test_specialized_queries_still_run_automatic_guards_before_chat():
    root = start("query", keep_n_messages=1)
    plan = UserQuery(content="next")
    root.append(plan)

    result = asyncio.run(Flow(object()).step(plan))

    assert isinstance(result, TruncationSummary)


def test_flow_transition_registers_node_only_sync_and_async_handlers():
    sync_flow = Flow(object())

    @sync_flow.transition(source="all")
    def sync_handler(node):
        assert isinstance(node, type(start("other")))
        return UserQuery()

    assert isinstance(asyncio.run(sync_flow.step(start("query"))), UserQuery)

    async_flow = Flow(object())

    @async_flow.transition(source=ErrorOutput, when=lambda node: node.content == "retry")
    async def async_handler(node):
        return UserQuery(content=node.content)

    root = start("query")
    error = ErrorOutput(content="retry")
    root.append(error)
    created = asyncio.run(async_flow.step(error))
    assert isinstance(created, UserQuery)
    assert created.content == "retry"


def test_transition_source_and_condition_validation():
    with pytest.raises(TransitionPolicyError, match="source"):

        @Flow(object()).transition(source=object)
        def invalid_source(node):
            return node

    with pytest.raises(TransitionPolicyError, match="cannot be empty"):

        @Flow(object()).transition(source="all", when="  ")
        def empty_condition(node):
            return node


class SemanticEngine:
    def __init__(self, *values):
        self.values = list(values)
        self.requests = []

    def supports(self, request):
        return ()

    async def generate(self, request, codec):
        self.requests.append(request)
        value = self.values.pop(0)
        if callable(value):
            value = value(request)
        return StructuredResult(
            value,
            {
                "engine": "semantic-test",
                "model": "judge",
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "reasoning_tokens": 2,
                },
            },
        )


def test_semantic_transition_persists_true_decision_and_uses_original_source(
    tmp_path,
):
    engine = SemanticEngine(lambda request: request.state["candidates"][0]["transition"])
    flow = Flow(
        object(),
        structured_engine=engine,
    )

    @flow.transition(source=ErrorOutput, when="The error needs replanning")
    def replan(node):
        assert isinstance(node, ErrorOutput)
        return UserQuery(content="replanned")

    root = start("query")
    error = ErrorOutput(content="requirements changed")
    root.append(error)
    created = asyncio.run(flow.step(error))

    assert isinstance(created, UserQuery)
    decision = error.next
    assert isinstance(decision, TransitionDecision)
    assert decision.selected_rule_id == replan.id
    assert decision.source_node_id == error.id
    assert decision.next is created
    assert decision.read_set_hash
    assert decision.prompt_hash
    assert decision.model == "judge"
    assert decision.usage.input_tokens == 7
    assert decision.usage.output_tokens == 3
    assert decision.usage.reasoning_tokens == 2
    assert decision.started_at is not None
    assert decision.finished_at is not None
    assert decision.finished_at >= decision.started_at
    assert root.usage == decision.usage

    loaded = type(root).load(root.save(tmp_path / "semantic"))
    loaded_decision = next(node for node in loaded.walk() if isinstance(node, TransitionDecision))
    assert loaded_decision.candidates == {replan.id: "The error needs replanning"}
    assert loaded_decision.selected_rule_id == decision.selected_rule_id
    assert loaded_decision.read_set_node_ids == decision.read_set_node_ids
    assert loaded_decision.usage == decision.usage
    assert loaded.usage == root.usage

    forked = decision.fork()
    request_count = len(engine.requests)
    replayed = asyncio.run(flow.step(forked.frontier))
    assert isinstance(replayed, UserQuery)
    assert len(engine.requests) == request_count


def test_semantic_transitions_are_selected_by_one_structured_choice():
    engine = SemanticEngine(lambda request: request.state["candidates"][1]["transition"])
    flow = Flow(object(), structured_engine=engine)

    @flow.transition(source="all", when="This is an authentication error")
    def authenticate(node):
        return UserQuery(content="authenticate")

    @flow.transition(source="all", when="This can be retried unchanged")
    def retry(node):
        return UserQuery(content="retry")

    @flow.transition(source="all")
    def fallback(node):
        return UserQuery(content="fallback")

    root = start("query")
    error = ErrorOutput(content="retry this")
    root.append(error)
    created = asyncio.run(flow.step(error))

    decision = error.next
    assert isinstance(decision, TransitionDecision)
    assert decision.selected_rule_id == retry.id
    assert decision.candidates == {
        authenticate.id: "This is an authentication error",
        retry.id: "This can be retried unchanged",
    }
    assert decision.next is created
    assert created.content == "retry"
    assert len(engine.requests) == 1
    request = engine.requests[0]
    assert request.schema == {
        "type": "string",
        "enum": ["authenticate", "retry", "continue"],
    }
    assert request.state["candidates"] == [
        {
            "transition": "authenticate",
            "condition": "This is an authentication error",
        },
        {
            "transition": "retry",
            "condition": "This can be retried unchanged",
        },
    ]
