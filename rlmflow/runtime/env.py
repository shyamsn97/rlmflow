"""Public ``RLMFLOW_*`` metadata exposed to agent code.

These are safe, string-valued metadata an agent can read from its per-REPL
``ENV`` mapping (e.g. ``ENV["RLMFLOW_AGENT_ID"]``). Being per-REPL, they stay
isolated across concurrent local agents. Private control state (output schemas,
done results) never travels this way.
"""

from __future__ import annotations

#: Current agent id (for example ``"root"`` or ``"root.worker"``).
RLMFLOW_AGENT_ID = "RLMFLOW_AGENT_ID"
#: Current agent depth in the spawn tree (``"0"`` for root).
RLMFLOW_DEPTH = "RLMFLOW_DEPTH"
#: Parent agent id, or ``""`` for root.
RLMFLOW_PARENT_AGENT_ID = "RLMFLOW_PARENT_AGENT_ID"
#: Recursion bound configured on the flow.
RLMFLOW_MAX_DEPTH = "RLMFLOW_MAX_DEPTH"
#: ``"1"`` for root, ``"0"`` otherwise.
RLMFLOW_IS_ROOT = "RLMFLOW_IS_ROOT"
#: ``"1"`` while recorded code is being re-run to rebuild a namespace, ``"0"`` during
#: a live turn. Code that reads it can skip work it should not repeat.
RLMFLOW_REPLAY = "RLMFLOW_REPLAY"


def agent_process_env(
    *,
    agent_id: str,
    depth: int,
    parent_agent_id: str | None,
    max_depth: int,
) -> dict[str, str]:
    """Return public ``RLMFLOW_*`` environment variables for one agent."""
    return {
        RLMFLOW_AGENT_ID: agent_id,
        RLMFLOW_DEPTH: str(depth),
        RLMFLOW_PARENT_AGENT_ID: parent_agent_id or "",
        RLMFLOW_MAX_DEPTH: str(max_depth),
        RLMFLOW_IS_ROOT: "1" if depth == 0 else "0",
        RLMFLOW_REPLAY: "0",
    }


__all__ = [
    "RLMFLOW_AGENT_ID",
    "RLMFLOW_DEPTH",
    "RLMFLOW_IS_ROOT",
    "RLMFLOW_MAX_DEPTH",
    "RLMFLOW_PARENT_AGENT_ID",
    "RLMFLOW_REPLAY",
    "agent_process_env",
]
