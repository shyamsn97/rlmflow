import asyncio

import pytest
from helpers import StubLLM, agent_graph, next_node, stepping

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
    TurnMode,
    UserQuery,
    ops,
    reads,
)
from rlmflow.engine.steps import TransitionFunction, complete, node_for_run, resolve_transition
from rlmflow.graph import Graph


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
    g, root = agent_graph("query")
    error = ErrorOutput(content="failed")
    ops.append(g, root, error)

    with stepping(g, root):
        chosen, anchor, source = asyncio.run(
            resolve_transition((semantic, skipped, fallback), error, judge)
        )
    assert chosen is fallback
    assert source is error
    assert isinstance(anchor, TransitionDecision)
    assert reads.prev(g, anchor) is error
    assert judged == [((semantic.id, semantic.__name__, "Needs a different plan"),)]

    with stepping(g, root):
        chosen, again, source = asyncio.run(
            resolve_transition((semantic, skipped, fallback), anchor, judge)
        )
    assert chosen is fallback
    assert again is anchor
    assert source is error
    assert judged == [((semantic.id, semantic.__name__, "Needs a different plan"),)]


def test_one_turn_budget_starts_on_the_final_turn():
    g, root = agent_graph("query", max_iters=1)

    result = asyncio.run(Flow(object()).step(g, root))

    assert isinstance(result, FinalQuery)


def test_startup_calls_the_model_with_submit_available():
    g, root = agent_graph("query", args={"context": "material"})
    requests = []

    class Recorder:
        def chat(self, messages):
            requests.append(messages)
            return "reply"

    flow = Flow(Recorder())
    reply = asyncio.run(flow.step(g, root))

    assert isinstance(reply, LLMOutput)
    assert reads.prev(g, reply) is root
    assert [message["role"] for message in requests[0]] == ["system", "user"]
    assert requests[0][-1]["content"] == "query"
    with stepping(g, root):
        assert "submit" in flow.build_tools(root)


def test_first_block_may_submit():
    g, root = agent_graph("query")
    reply = LLMOutput(content="```repl\nsubmit(42)\n```", code="submit(42)")
    ops.append(g, root, reply)
    action = ExecAction(code=reply.code)
    ops.append(g, root, action)

    with stepping(g, root):
        done = node_for_run(action, ReplRun(status=ReplStatus.DONE, answer=42))

    assert isinstance(done, DoneOutput)
    assert done.result == 42


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (UserQuery(content="continue"), TurnMode.ACTION),
        (FinalQuery(), TurnMode.FINAL),
    ],
)
def test_exec_action_stamps_the_source_turn_mode(source, expected):
    g, root = agent_graph("query")
    ops.append(g, root, source)
    reply = LLMOutput(content="```repl\nprint(1)\n```", code="print(1)")
    ops.append(g, root, reply)

    action = asyncio.run(Flow(object()).step(g, root, reply))

    assert isinstance(action, ExecAction)
    assert action.turn_mode is expected


def test_final_action_exposes_only_submit_and_errors_without_it():
    g, root = agent_graph("query")
    action = ExecAction(code="print(1)", turn_mode=TurnMode.FINAL)
    ops.append(g, root, action)
    flow = Flow(object())

    with stepping(g, root):
        assert set(flow.build_tools(action)) == {"submit"}
    with stepping(g, root):
        failed = node_for_run(action, ReplRun(status=ReplStatus.OK, output="1"))
    assert isinstance(failed, ErrorOutput)
    assert failed.content == "1\nMissingSubmitError: the final turn must call submit(answer)"


def test_queries_and_observations_call_the_model_directly():
    flow = Flow(StubLLM(lambda _messages: "reply"))
    g, root = agent_graph("query")
    query = UserQuery(content="host instruction")
    ops.append(g, root, query)
    assert isinstance(asyncio.run(flow.step(g, root, query)), LLMOutput)

    for observation_type, content in (
        (ExecOutput, "observed"),
        (ErrorOutput, "failed"),
    ):
        g, root = agent_graph("query")
        observation = observation_type(content=content)
        ops.append(g, root, observation)
        assert isinstance(asyncio.run(flow.step(g, root, observation)), LLMOutput)

    for prior in (ExecOutput(content="observed"), DoneOutput(result="child")):
        g, root = agent_graph("query")
        ops.append(g, root, prior)
        query = UserQuery(content="[handoff root.child]\nresult: 1")
        ops.append(g, root, query)
        assert isinstance(asyncio.run(flow.step(g, root, query)), LLMOutput)


def test_specialized_queries_still_run_automatic_guards_before_chat():
    g, root = agent_graph("query", keep_n_messages=1)
    plan = UserQuery(content="next")
    ops.append(g, root, plan)

    result = asyncio.run(Flow(object()).step(g, root, plan))

    assert isinstance(result, TruncationSummary)


def test_flow_transition_registers_node_only_sync_and_async_handlers():
    sync_flow = Flow(object())

    @sync_flow.transition(source="all")
    def sync_handler(node):
        assert isinstance(node, AgentStart)
        return UserQuery()

    g, root = agent_graph("query")
    assert isinstance(asyncio.run(sync_flow.step(g, root)), UserQuery)

    async_flow = Flow(object())

    @async_flow.transition(source=ErrorOutput, when=lambda node: node.content == "retry")
    async def async_handler(node):
        return UserQuery(content=node.content)

    g, root = agent_graph("query")
    error = ErrorOutput(content="retry")
    ops.append(g, root, error)
    created = asyncio.run(async_flow.step(g, root, error))
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

    g, root = agent_graph("query")
    error = ErrorOutput(content="requirements changed")
    ops.append(g, root, error)
    created = asyncio.run(flow.step(g, root, error))

    assert isinstance(created, UserQuery)
    decision = next_node(g, error)
    assert isinstance(decision, TransitionDecision)
    assert decision.selected_rule_id == replan.id
    assert decision.source_node_id == error.id
    assert next_node(g, decision) is created
    assert decision.read_set_hash
    assert decision.prompt_hash
    assert decision.model == "judge"
    assert decision.usage.input_tokens == 7
    assert decision.usage.output_tokens == 3
    assert decision.usage.reasoning_tokens == 2
    assert decision.started_at is not None
    assert decision.finished_at is not None
    assert decision.finished_at >= decision.started_at
    assert reads.usage(g) == decision.usage

    g.save(tmp_path / "semantic")
    loaded = Graph.load(tmp_path / "semantic")
    [loaded_decision] = loaded.nodes(TransitionDecision)
    assert loaded_decision.candidates == {replan.id: "The error needs replanning"}
    assert loaded_decision.selected_rule_id == decision.selected_rule_id
    assert loaded_decision.read_set_node_ids == decision.read_set_node_ids
    assert loaded_decision.usage == decision.usage
    assert reads.usage(loaded) == reads.usage(g)

    forked = ops.fork(g, decision)
    request_count = len(engine.requests)
    replayed = asyncio.run(flow.step(g, forked))
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

    g, root = agent_graph("query")
    error = ErrorOutput(content="retry this")
    ops.append(g, root, error)
    created = asyncio.run(flow.step(g, root, error))

    decision = next_node(g, error)
    assert isinstance(decision, TransitionDecision)
    assert decision.selected_rule_id == retry.id
    assert decision.candidates == {
        authenticate.id: "This is an authentication error",
        retry.id: "This can be retried unchanged",
    }
    assert next_node(g, decision) is created
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
