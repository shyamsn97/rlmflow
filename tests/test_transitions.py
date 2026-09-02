import asyncio
from dataclasses import dataclass
from typing import ClassVar

import pytest

from rlmflow import (
    FinalQuery,
    Flow,
    InspectQuery,
    PlanQuery,
    TransitionPolicyError,
    Transitions,
    TruncationSummary,
    UserQuery,
    start,
)
from rlmflow.engine.transitions import DEFAULT_TRANSITIONS
from rlmflow.prompts import format_transition_footer


@dataclass
class VerifyQuery(UserQuery):
    type: ClassVar[str] = "verify_query"
    name: ClassVar[str] = "verify"
    transition_description: ClassVar[str] = "Choose when a candidate needs verification."
    action_description: ClassVar[str] = "Continue verifying the current candidate."
    finish_description: ClassVar[str] = "Submit the verified candidate."


def test_selectable_policy_defines_behavior_without_a_marker_base():
    transitions = Transitions().on(InspectQuery, [VerifyQuery])
    root = start("query")
    inspect = root.append(InspectQuery())
    verify = inspect.append(VerifyQuery())

    assert transitions.current_behavior(verify) is verify
    assert [option.name for option in transitions.available(verify)] == ["act"]


def test_utility_control_does_not_erase_policy_registered_behavior():
    transitions = Transitions().on(InspectQuery, [VerifyQuery]).on(
        VerifyQuery,
        [InspectQuery],
    )
    root = start("query")
    verify = root.append(InspectQuery()).append(VerifyQuery())
    summary = verify.append(TruncationSummary())

    assert transitions.current_behavior(summary) is verify
    assert [option.name for option in transitions.available(summary)] == [
        "act",
        "inspect",
    ]


def test_newer_action_capable_query_replaces_an_older_custom_behavior():
    transitions = Transitions().on(InspectQuery, [VerifyQuery])
    verify = start("query").append(InspectQuery()).append(VerifyQuery())
    plan = verify.append(PlanQuery())

    assert transitions.current_behavior(plan) is plan
    assert [option.name for option in transitions.available(plan)] == ["act"]


def test_selectable_controls_require_explicit_model_facing_metadata():
    @dataclass
    class UtilityQuery(UserQuery):
        type: ClassVar[str] = "utility_query"

    with pytest.raises(TransitionPolicyError, match="no transition name"):
        Transitions().on(UtilityQuery, [InspectQuery])


def test_selectable_transition_names_are_unique_per_owner_and_cannot_shadow_exits():
    @dataclass
    class DuplicateVerifyQuery(UserQuery):
        type: ClassVar[str] = "duplicate_verify_query"
        name: ClassVar[str] = "verify"
        transition_description: ClassVar[str] = "Verify another way."

    @dataclass
    class ActQuery(UserQuery):
        type: ClassVar[str] = "act_query"
        name: ClassVar[str] = "act"
        transition_description: ClassVar[str] = "Shadow the built-in action."

    with pytest.raises(TransitionPolicyError, match="duplicate transition names"):
        (
            Transitions()
            .on(InspectQuery, [VerifyQuery])
            .on(InspectQuery, [DuplicateVerifyQuery])
        )
    with pytest.raises(TransitionPolicyError, match="'act' is reserved"):
        Transitions().on(InspectQuery, [ActQuery])


def test_builtin_transition_footer_is_short_and_describes_every_exit():
    plan = start("query").append(PlanQuery())
    options = [
        (option.name, option.description)
        for option in DEFAULT_TRANSITIONS.available(plan)
    ]

    assert format_transition_footer(options) == (
        "End the REPL block with one:\n"
        '- transition("act") — Continue working from the latest result.\n'
        "- finish(answer) — Submit your final answer."
    )


def test_persisted_inspection_state_continues_as_one_working_state():
    inspect = start("query").append(InspectQuery())

    options = DEFAULT_TRANSITIONS.available(inspect)

    assert [option.name for option in options] == ["act"]
    assert options[0].description == "Continue working from the latest result."
    assert DEFAULT_TRANSITIONS.resolve(inspect, "inspect") == options[0]
    assert DEFAULT_TRANSITIONS.resolve(inspect, "plan") == options[0]


def test_current_query_owns_action_and_finish_guidance():
    transitions = Transitions().on(InspectQuery, [VerifyQuery]).on(
        VerifyQuery,
        [InspectQuery],
    )
    verify = start("query").append(InspectQuery()).append(VerifyQuery())

    footer = Flow(object(), transitions=transitions).transition_footer(verify)
    utility_footer = Flow(object(), transitions=transitions).transition_footer(
        verify.append(TruncationSummary())
    )

    assert '- transition("act") — Continue verifying the current candidate.' in footer
    assert footer.endswith("- finish(answer) — Submit the verified candidate.")
    assert utility_footer.endswith("- finish(answer) — Submit the verified candidate.")


def test_final_query_owns_its_finish_guidance():
    final = start("query").append(FinalQuery())

    footer = Flow(object()).transition_footer(final)

    assert footer == "End with finish(answer) — Submit your final answer now."


def test_final_guard_preempts_normal_startup():
    root = start("query", max_iters=1)

    rule = DEFAULT_TRANSITIONS.resolve_automatic(root)

    assert rule is not None
    assert rule.target is FinalQuery


def test_specialized_queries_still_run_automatic_guards_before_chat():
    root = start("query", keep_n_messages=1)
    plan = root.append(InspectQuery()).append(PlanQuery())

    result = asyncio.run(Flow(object()).step(plan))

    assert isinstance(result.created, TruncationSummary)
