"""Private per-agent engine context for host-side tool closures."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class EngineContext:
    """Trusted control state for one agent's backend.

    This is intentionally separate from process environment variables: schema
    validation and completion signaling are private engine concerns, while
    public agent metadata can be exposed through ``os.environ``.
    """

    agent_id: str = ""
    output_schema: dict[str, Any] | None = None
    done_result: str | None = None
    recovery_launch_id: str | None = None
    launch_children: (
        Callable[[list[dict[str, Any]], list[str]], Awaitable[list[object]]] | None
    ) = None


__all__ = ["EngineContext"]
