"""Shared execution policy and turn-facing graph prompt text.

Distinct from ``rlmflow.prompts``, which assembles the system protocol.
"""

from __future__ import annotations

FINAL_ANSWER_ACTION = """This is your last turn: the run is out of budget to keep working.
Based on the work above, call finish(answer) now with only the final answer, in the
exact form the query requested. Do not investigate further, and do not hold the answer
back for verification — no turn follows this one to read a print, so submit your best
inference rather than ending the run with nothing."""

TRUNCATION_SUMMARY = """[earlier turns omitted to fit the context window; the most recent
turns follow. The REPL kept running, so variables, imports, and helpers defined in those
turns are still bound — reuse them instead of redefining them.]"""

COLD_REPL_NOTE = """[this agent's REPL was restarted, so variables and imports from
earlier turns are gone. Re-derive whatever you need before using it.]"""

USER_PROMPT = "Turn {iter_1}/{max_iter}:"

FIRST_TURN_SAFEGUARD = (
    "You have not interacted with the REPL environment yet. Look at INPUTS if any "
    "are bound, otherwise inspect the workspace; do not provide a final answer yet.\n\n"
)

PLANNING_ACTION = (
    "Before acting, briefly decide:\n"
    "- the next objective;\n"
    "- what stays local and what, if anything, gets its own owner;\n"
    "- which known work products depend on which others.\n"
    "Then execute the next step in one REPL block."
)

EXECUTION_POLICY = """\
Work allocation and graph policy:
- Keep one coherent, tightly coupled deliverable local.
- Give a distinct multi-step investigation, implementation, search, or verification
  its own child when a separate context will help.
- Give every child one bounded, non-overlapping deliverable and the inputs it needs.
- Launch ready independent work together.
- When a known work product depends on another, launch it with the prerequisite
  `AgentHandle` as a named `inputs` value; do not wait before declaring a known edge.
- Do not duplicate delegated work locally or create a child for light synthesis.
- Consume and validate child results before finishing.
- Stop launching when the answer is supported, then finish."""

# Compatibility names for callers that imported the previous split policy.
ORCHESTRATOR_ADDENDUM = EXECUTION_POLICY
WORKER_ADDENDUM = """\
Assigned-work policy:
- Own the requested deliverable and use the provided inputs.
- Do not leave TODOs, stubs, or work for a parent to complete.
- Validate the result, then call `finish(...)` with the actual deliverable."""


def orchestrator_addendum(_flow: object | None = None) -> str:
    """Return the spawn-capable execution policy."""
    return EXECUTION_POLICY


LAST_TURN_ADDENDUM = (
    "This is your last turn. Call `finish(...)` now with your best inference rather "
    "than letting the rollout terminate unsubmitted."
)

#: Last sentence of the inherited protocol; FinalQuery strips it so the last
#: turn does not say "don't finish on turn 1."
TURN_ONE_FINISH_SENTENCE = "Do not call `finish(...)` on turn 1 without first inspecting `INPUTS`."


__all__ = [
    "COLD_REPL_NOTE",
    "FINAL_ANSWER_ACTION",
    "FIRST_TURN_SAFEGUARD",
    "EXECUTION_POLICY",
    "LAST_TURN_ADDENDUM",
    "ORCHESTRATOR_ADDENDUM",
    "PLANNING_ACTION",
    "TRUNCATION_SUMMARY",
    "TURN_ONE_FINISH_SENTENCE",
    "USER_PROMPT",
    "WORKER_ADDENDUM",
    "orchestrator_addendum",
]
