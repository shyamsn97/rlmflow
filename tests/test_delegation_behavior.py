"""Delegation behavior: fan out on substantial work, stay local on trivial work.

The checks that call a real model are marked ``live`` and skipped unless
``RLMFLOW_LIVE_TESTS=1`` and a key for the model is set, because they cost money
and are stochastic. ``make test-live`` runs all of them, boids included: the
routine suite never reaches a live check anyway, so splitting the heavy one out of
the paid run only meant the behavior it measures went untested::

    make test-live         # every scenario
    make test-live-boids   # just boids, the slow one

The scenarios and their grading live in ``examples/behavior/delegation.py``, so
the script you run by hand and the check that gates a prompt change measure the
same thing. Model choice: ``RLMFLOW_LIVE_MODEL``, defaulting to the examples'
own default. ``RLMFLOW_LIVE_ADDITIONAL_MODEL`` optionally exposes another model
that the root may select explicitly for each child. The unmarked tests at the
bottom run in the normal suite and keep the graders themselves honest, since a
broken grader would otherwise stay invisible until someone paid for a live run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from helpers import StubLLM

from rlmflow import AgentConfig, AgentStart

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from behavior.delegation import (  # noqa: E402
    BOIDS_FILES,
    Observed,
    boids_scope_files,
    grade_boids,
    run_scenario,
    scenarios,
    usable_child_result,
)

MODEL = os.environ.get("RLMFLOW_LIVE_MODEL", "gpt-5-mini")
KEY = "ANTHROPIC_API_KEY" if MODEL.startswith("claude") else "OPENAI_API_KEY"
ADDITIONAL_MODEL = os.environ.get("RLMFLOW_LIVE_ADDITIONAL_MODEL")
ADDITIONAL_KEY = (
    "ANTHROPIC_API_KEY"
    if ADDITIONAL_MODEL is not None and ADDITIONAL_MODEL.startswith("claude")
    else "OPENAI_API_KEY"
)

#: Live graphs land in the repo, not in `tmp_path`. These runs cost money, and
#: pytest keeps only its last three temp roots, so a paid trajectory would be
#: deleted a couple of test runs later. Three attempts make the stochastic behavior
#: a pass-rate measurement rather than letting one sample gate a prompt change.
LIVE_OUT = Path(
    os.environ.get("RLMFLOW_LIVE_OUT", EXAMPLES_DIR / "_runs" / "delegation-behavior" / "live")
)
LIVE_ATTEMPTS = 3
LIVE_REQUIRED_PASSES = 2

LIVE_MARKS = (
    pytest.mark.live,
    # The suite-wide 30s cap exists to break deadlocks; a real multi-agent run
    # needs minutes, and a scenario that hangs still fails rather than stalling.
    pytest.mark.timeout(1800),
    pytest.mark.skipif(
        os.environ.get("RLMFLOW_LIVE_TESTS") != "1",
        reason="live model test; set RLMFLOW_LIVE_TESTS=1 to run",
    ),
    pytest.mark.skipif(not os.environ.get(KEY), reason=f"needs {KEY}"),
    pytest.mark.skipif(
        ADDITIONAL_MODEL is not None and not os.environ.get(ADDITIONAL_KEY),
        reason=f"needs {ADDITIONAL_KEY}",
    ),
)


def live(fn):
    for mark in LIVE_MARKS:
        fn = mark(fn)
    return fn


def scenario(name: str):
    found = next((item for item in scenarios() if item.name == name), None)
    assert found is not None, f"unknown scenario {name!r}"
    return found


def check(name: str) -> None:
    results = [
        run_scenario(
            scenario(name),
            model=MODEL,
            additional_model=ADDITIONAL_MODEL,
            out_dir=LIVE_OUT / f"attempt-{attempt}",
        )
        for attempt in range(1, LIVE_ATTEMPTS + 1)
    ]
    for attempt, result in enumerate(results, 1):
        seen = result.observed
        assert seen is not None
        print(
            f"\n{name} #{attempt}: {len(seen.children)} children, "
            f"batch {seen.max_launch_batch}, {seen.root_turns} root turns, "
            f"{seen.tokens:,} tokens, {seen.seconds:.1f}s"
            f"\ngraph: {result.graph}"
            f"\nverdict: {'PASS' if result.ok else 'FAIL'}"
        )
        for failure in result.failures:
            print(f"- {failure}")

    passed = sum(result.ok for result in results)
    failures = [
        f"attempt {attempt}: {'; '.join(result.failures)}"
        for attempt, result in enumerate(results, 1)
        if not result.ok
    ]
    assert passed >= LIVE_REQUIRED_PASSES, f"{passed}/{LIVE_ATTEMPTS} passed; " + " | ".join(
        failures
    )


#: Each test is named for the scenario it runs, so a `-v` line or a deselection
#: says which behavior was measured and which was skipped.
@live
def test_explainers_delegates_substantial_independent_deliverables():
    """Three independent explainers: the root should orchestrate, not write all three."""
    check("explainers")


@live
def test_gateway_port_keeps_a_single_lookup_in_the_parent():
    """One regex-findable fact in a small config buys nothing from a subagent."""
    check("gateway-port")


@live
def test_number_stats_keeps_separable_but_trivial_outputs_in_the_parent():
    """Four statistics over one list look separable; each is one line of Python."""
    check("number-stats")


@live
def test_boids_delegates_the_substantial_modules():
    """The case the strategy section was written against. The slowest of these by
    far, and it writes files, but a live run that skips it measures nothing about
    the behavior the prompt was rewritten for."""
    check("boids")


# --- Offline: the graders, run against canned trajectories ---------------------


def repl(code: str) -> str:
    return f"```repl\n{code}\n```"


CORRECT_STATS = repl(
    """
numbers = [int(part) for part in INPUTS["numbers"].split(",")]
finish(
    {
        "count": len(numbers),
        "sum": sum(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }
)
""".strip()
)

WRONG_STATS = repl('finish({"count": 1, "sum": 2, "min": 3, "max": 4})')


def test_a_correct_local_answer_passes_the_no_delegation_grader(tmp_path):
    result = run_scenario(
        scenario("number-stats"),
        out_dir=tmp_path,
        llm=StubLLM(lambda _messages: CORRECT_STATS),
    )

    assert result.ok, result.failures
    assert result.observed is not None
    assert result.observed.children == []
    assert result.observed.max_launch_batch == 0


def test_concise_structured_child_manifest_is_usable():
    manifest = {
        "created_files": ["rules.js", "species.js"],
        "globals_defined": ["Rules", "SPECIES"],
    }

    assert usable_child_result(manifest, min_chars=400)
    assert not usable_child_result({}, min_chars=200)
    assert not usable_child_result("ok", min_chars=200)


def test_a_wrong_local_answer_still_fails(tmp_path):
    """Otherwise "no subagents" could be earned by answering badly."""
    result = run_scenario(
        scenario("number-stats"),
        out_dir=tmp_path,
        llm=StubLLM(lambda _messages: WRONG_STATS),
    )

    assert not result.ok
    assert "wrong statistics" in result.failures[0]


def test_delegating_a_trivial_scope_is_reported_as_over_delegation(tmp_path):
    """A child that answers in one turn with nothing is the over-delegation shape."""
    child = repl('finish("42")')
    parent = repl(
        """
handle = await launch_subagent(
    name="adder",
    goal="Add the numbers in INPUTS['numbers'].",
    model="default",
)
print(await handle.wait_for_result())
""".strip()
    )
    replies = iter([parent, CORRECT_STATS, child])

    def reply(messages):
        system = messages[0]["content"]
        # The child's prompt has no delegation section, which is how we tell them apart.
        return child if "launch_subagent" not in system else next(replies)

    result = run_scenario(
        scenario("number-stats"),
        out_dir=tmp_path,
        llm=StubLLM(reply),
    )

    assert not result.ok
    assert "delegated 1 scope(s) to adder" in result.failures[0]
    assert "did one turn for a tiny result" in result.failures[0]
    assert result.observed is not None
    assert result.observed.wasted_children == 1
    assert result.observed.max_launch_batch == 1


def test_children_launched_in_one_block_count_as_one_batch(tmp_path):
    """Guards the fan-out metric, which silently read 0 while it looked for the
    `AppendChild` type that only controller-injected launches produce."""
    child = repl(f'finish("{"x" * 500}")')
    parent = repl(
        """
handles = await asyncio.gather(
    launch_subagent(name="first", goal="Do the first half.", model="default"),
    launch_subagent(name="second", goal="Do the second half.", model="default"),
)
print(await asyncio.gather(*(handle.wait_for_result() for handle in handles)))
""".strip()
    )
    replies = iter([parent, CORRECT_STATS])

    def reply(messages):
        return next(replies) if "launch_subagent" in messages[0]["content"] else child

    result = run_scenario(
        scenario("number-stats"),
        out_dir=tmp_path,
        llm=StubLLM(reply),
    )

    assert result.observed is not None
    assert len(result.observed.children) == 2
    assert result.observed.max_launch_batch == 2
    assert result.observed.wasted_children == 0


def boids_child(
    name: str,
    *,
    properties: tuple[str, ...] = (),
    inputs: dict[str, str] | None = None,
) -> AgentStart:
    schema = (
        {
            "type": "object",
            "properties": {filename: {"type": "string"} for filename in properties},
        }
        if properties
        else None
    )
    return AgentStart(
        content=f"Implement {', '.join(properties) or name}.",
        config=AgentConfig(
            name=name,
            path=f"root.{name}",
            inputs=inputs or {},
            output_schema=schema,
        ),
    )


def observed_boids(tmp_path, children: list[AgentStart]) -> Observed:
    return Observed(
        answer="",
        children=children,
        root_turns=1,
        max_launch_batch=len(children),
        wasted_children=0,
        tokens=0,
        seconds=0,
        workdir=tmp_path,
    )


def test_boids_scope_prefers_specific_filename_input_over_broad_context():
    child = boids_child(
        "style",
        inputs={
            "context": f"Every requested file: {', '.join(BOIDS_FILES)}",
            "filename": "style.css",
        },
    )

    assert boids_scope_files(child) == {"style.css"}


def test_boids_grader_rejects_duplicate_artifact_scopes(tmp_path):
    children = [
        boids_child("vec-first", properties=("vec.js",)),
        boids_child("vec-second", properties=("vec.js",)),
    ]

    failures = grade_boids(observed_boids(tmp_path, children))

    assert any("duplicate delegated artifact scopes: vec.js" in failure for failure in failures)


def test_boids_grader_rejects_more_children_than_artifacts(tmp_path):
    children = [
        boids_child(f"child-{index}", properties=(BOIDS_FILES[index % len(BOIDS_FILES)],))
        for index in range(len(BOIDS_FILES) + 1)
    ]

    failures = grade_boids(observed_boids(tmp_path, children))

    assert any(
        "delegation ran beyond the available deliverables" in failure for failure in failures
    )
