import asyncio
from dataclasses import dataclass
from typing import ClassVar

import pytest
from helpers import StubLLM, TestRuntime, counting_replies

from rlmflow import (
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    FinalQuery,
    Flow,
    InvalidTransitionError,
    LLMOutput,
    LocalRuntime,
    PlanQuery,
    TransitionPolicyError,
    Transitions,
    TruncationSummary,
    UserQuery,
    start,
)
from rlmflow.engine.steps import complete, to_final, to_plan
from rlmflow.prompts import format_transition_footer


@dataclass
class WorkQuery(UserQuery):
    type: ClassVar[str] = "work_query"
    name: ClassVar[str] = "work"
    transition_description: ClassVar[str] = "Continue working from the latest result."
    finish_description: ClassVar[str] = "Submit the current candidate."


@dataclass
class VerifyQuery(UserQuery):
    type: ClassVar[str] = "verify_query"
    name: ClassVar[str] = "verify"
    transition_description: ClassVar[str] = "Choose when a candidate needs verification."
    finish_description: ClassVar[str] = "Submit the verified candidate."


def test_selectable_policy_defines_behavior_without_a_marker_base():
    transitions = Transitions().choices(WorkQuery, VerifyQuery)
    root = start("query")
    work = WorkQuery()
    root.append(work)
    verify = VerifyQuery()
    work.append(verify)

    assert transitions.current_behavior(verify) is verify
    assert transitions.available(verify) == ()


def test_utility_control_does_not_erase_policy_registered_behavior():
    transitions = Transitions().choices(WorkQuery, VerifyQuery).choices(
        VerifyQuery,
        WorkQuery,
    )
    root = start("query")
    work = WorkQuery()
    root.append(work)
    verify = VerifyQuery()
    work.append(verify)
    summary = TruncationSummary()
    verify.append(summary)

    assert transitions.current_behavior(summary) is verify
    assert [option.name for option in transitions.available(summary)] == ["work"]


def test_unregistered_query_does_not_replace_a_custom_behavior():
    transitions = Transitions().choices(WorkQuery, VerifyQuery)
    work = WorkQuery()
    start("query").append(work)
    verify = VerifyQuery()
    work.append(verify)
    plan = PlanQuery()
    verify.append(plan)

    assert transitions.current_behavior(plan) is verify
    assert transitions.available(plan) == ()


def test_selectable_controls_require_explicit_model_facing_metadata():
    @dataclass
    class UtilityQuery(UserQuery):
        type: ClassVar[str] = "utility_query"

    with pytest.raises(TransitionPolicyError, match="no transition name"):
        Transitions().choices(UtilityQuery, VerifyQuery)


def test_selectable_transition_names_are_unique_and_cannot_shadow_finish():
    @dataclass
    class DuplicateVerifyQuery(UserQuery):
        type: ClassVar[str] = "duplicate_verify_query"
        name: ClassVar[str] = "verify"
        transition_description: ClassVar[str] = "Verify another way."

    @dataclass
    class ActQuery(UserQuery):
        type: ClassVar[str] = "act_query"
        name: ClassVar[str] = "act"
        transition_description: ClassVar[str] = "Choose the host-defined act behavior."

    @dataclass
    class FinishQuery(UserQuery):
        type: ClassVar[str] = "finish_query"
        name: ClassVar[str] = "finish"
        transition_description: ClassVar[str] = "Shadow completion."

    with pytest.raises(TransitionPolicyError, match="duplicate transition names"):
        (
            Transitions()
            .choices(WorkQuery, VerifyQuery)
            .choices(WorkQuery, DuplicateVerifyQuery)
        )
    transitions = Transitions().choices(WorkQuery, ActQuery)
    work = WorkQuery()
    start("query").append(work)
    assert [option.name for option in transitions.available(work)] == ["act"]
    with pytest.raises(TransitionPolicyError, match="'finish' is reserved"):
        Transitions().choices(WorkQuery, FinishQuery)


def test_builtin_transition_footer_is_short_and_describes_every_exit():
    plan = PlanQuery()
    start("query").append(plan)
    options = [
        (option.name, option.description)
        for option in Flow.transitions.available(plan)
    ]

    assert format_transition_footer(options) == (
        "End the REPL block normally to continue.\n"
        "- finish(answer) — Submit your final answer."
    )


def test_undeclared_transition_names_do_not_resolve():
    plan = PlanQuery()
    start("query").append(plan)

    options = Flow.transitions.available(plan)

    assert options == ()
    assert Flow.transitions.resolve_choice(plan, "act") is None
    assert Flow.transitions.resolve_choice(plan, "inspect") is None
    assert Flow.transitions.resolve_choice(plan, "plan") is None


def test_current_query_owns_finish_and_selectable_transition_guidance():
    transitions = Transitions().choices(WorkQuery, VerifyQuery).choices(
        VerifyQuery,
        WorkQuery,
    )
    work = WorkQuery()
    start("query").append(work)
    verify = VerifyQuery()
    work.append(verify)
    summary = TruncationSummary()
    verify.append(summary)

    footer = Flow(object(), transitions=transitions).transition_footer(verify)
    utility_footer = Flow(object(), transitions=transitions).transition_footer(
        summary
    )

    assert "End the REPL block normally to continue." in footer
    assert '- transition("work") — Continue working from the latest result.' in footer
    assert footer.endswith("- finish(answer) — Submit the verified candidate.")
    assert utility_footer.endswith("- finish(answer) — Submit the verified candidate.")


def test_transition_lands_the_selected_query_without_an_intermediate_output():
    transitions = Flow.transitions.derive().choices(WorkQuery, VerifyQuery)
    flow = Flow(
        StubLLM(
            counting_replies(
                '```repl\ntransition("verify")\n```',
                '```repl\nfinish("verified")\n```',
            )
        ),
        runtime=TestRuntime(),
        transitions=transitions,
    )
    root = start("query")
    work = WorkQuery()
    root.append(work)

    assert flow.run(root) == "verified"

    transcript = root.transcript()
    transition_action = next(
        node
        for node in transcript
        if isinstance(node, ExecAction) and 'transition("verify")' in node.code
    )
    assert isinstance(transition_action.prev, LLMOutput)
    assert isinstance(transition_action.next, VerifyQuery)
    assert not isinstance(transition_action.next, ExecOutput)


def test_unknown_transition_fails_through_the_running_flow():
    transitions = Flow.transitions.derive().choices(WorkQuery, VerifyQuery)
    flow = Flow(
        StubLLM(lambda _messages: '```repl\ntransition("missing")\n```'),
        runtime=TestRuntime(),
        transitions=transitions,
    )
    root = start("query")
    root.append(WorkQuery())

    with pytest.raises(InvalidTransitionError, match="'missing'.*'verify'"):
        flow.run(root)


def test_worker_proxy_transition_lands_the_selected_query():
    transitions = Flow.transitions.derive().choices(WorkQuery, VerifyQuery)
    flow = Flow(
        StubLLM(
            counting_replies(
                '```repl\ntransition("verify")\n```',
                '```repl\nfinish("verified")\n```',
            )
        ),
        runtime=LocalRuntime(repl_timeout=5),
        transitions=transitions,
    )
    root = start("query")
    root.append(WorkQuery())

    assert flow.run(root, close_repls=True) == "verified"
    assert any(isinstance(node, VerifyQuery) for node in root.transcript())


def test_final_query_owns_its_finish_guidance():
    final = FinalQuery()
    start("query").append(final)

    footer = Flow(object()).transition_footer(final)

    assert footer == "End with finish(answer) — Submit your final answer now."


def test_final_guard_preempts_normal_startup():
    root = start("query", max_iters=1)

    assert Flow.transitions.resolve(root) is to_final


def test_startup_always_enters_a_planning_action():
    with_inputs = start("query", inputs={"context": "material"})
    without_inputs = start("query")

    assert Flow.transitions.resolve(with_inputs) is to_plan
    assert Flow.transitions.resolve(without_inputs) is to_plan


def test_observations_enter_a_fresh_planning_action_without_self_looping():
    root = start("query")
    plan = PlanQuery()
    root.append(plan)

    assert Flow.transitions.resolve(plan) is complete
    for observation_type, content in (
        (ExecOutput, "observed"),
        (ErrorOutput, "failed"),
    ):
        prior = PlanQuery()
        start("query").append(prior)
        observation = observation_type(content=content)
        prior.append(observation)
        assert Flow.transitions.resolve(observation) is to_plan

    for prior in (ExecOutput(content="observed"), DoneOutput(result="child")):
        root = start("query")
        root.append(prior)
        query = UserQuery(content="[handoff root.child]\nresult: 1")
        prior.append(query)
        assert Flow.transitions.resolve(query) is to_plan


def test_specialized_queries_still_run_automatic_guards_before_chat():
    root = start("query", keep_n_messages=1)
    plan = PlanQuery()
    root.append(plan)

    result = asyncio.run(Flow(object()).step(plan))

    assert isinstance(result, TruncationSummary)
