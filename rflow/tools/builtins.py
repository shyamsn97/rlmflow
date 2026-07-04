"""Core REPL tools, assembled once per agent by :meth:`rflow.base.BaseFlow.build_tools`.

Each ``make_*`` factory returns a closure bound to the live ``BaseFlow`` and the
agent's :class:`rflow.runtime.context.EngineContext` — *not* its ``Graph``. The
per-agent context is seeded by :meth:`rflow.base.BaseFlow.seed_agent_context` and
read at call time, so a single build serves the agent for its whole life. They
raise the existing :class:`rflow.runtime.repl.DoneSignal` and are
``@tool``-decorated so the prompt's tool list can describe them. ``History`` is
the read-only ``HISTORY`` object an agent uses to re-read its own past turns when
the model's context has been windowed (it does wrap the ``Graph``, since it is a
trajectory view rather than a control tool).
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from hashlib import blake2s
from typing import TYPE_CHECKING, Any

from rflow.base import BaseFlow
from rflow.graph import ChildHandle, WaitRequest
from rflow.runtime.context import EngineContext
from rflow.runtime.repl import DoneSignal
from rflow.tools.registry import HIDDEN_REPL_TOOL_NAMES
from rflow.tools.tools import get_tool_metadata, tool

if TYPE_CHECKING:
    from rflow.graph import Graph

#: Default ceiling (in characters) for a child agent's ``query`` string. The
#: ``query`` is a short task instruction; large payloads belong in ``inputs``.
#: Configurable per run via ``Flow(max_query_chars=...)``.
DEFAULT_MAX_QUERY_CHARS = 2_000


def _launch_id_from_names(names: list[str], agent_ids: list[str]) -> str:
    """Readable, stable-ish launch id from child names plus concrete ids."""
    label = "-".join(names[:3]) if names else "subagents"
    if len(names) > 3:
        label += f"-plus-{len(names) - 3}"
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "subagents"
    digest = blake2s(
        "|".join([*names, *agent_ids]).encode("utf-8"),
        digest_size=3,
    ).hexdigest()
    return f"launch-{label}-{digest}"


def make_done(flow: BaseFlow, engine_context: EngineContext):
    """Build ``done(answer)``; enforces the schema in ``engine_context``.

    Reads context set per-agent by ``seed_agent_context`` so the same factory
    works for any agent — no Graph captured.
    """

    @tool("Finish and return this agent's final answer.", proxy=True)
    def done(answer: Any) -> None:
        schema = engine_context.output_schema
        if schema is not None:
            content = answer if isinstance(answer, str) else json.dumps(answer)
            # Validate against the schema; a mismatch raises StructuredOutputError
            # whose message is the recovery hint, recorded as a retryable error.
            flow.output_parser(content, schema)
            result = content
        else:
            result = str(answer).strip()
        engine_context.done_result = result
        print(f"[done] {result}")
        raise DoneSignal()

    return done


def make_spawn_child(flow: BaseFlow, engine_context: EngineContext):
    """Build the private host-side child-spawn hook used by ``launch_subagents``."""

    @tool("Private rflow transport hook for spawning one child agent.", proxy=True)
    def _rflow_spawn_child(
        *,
        name: str = "subagent",
        query: str,
        inputs: dict[str, str] | None = None,
        model: str = "default",
        output_schema: Any | None = None,
        strict_name: bool = False,
    ) -> ChildHandle | str:
        return flow.spawn_child(
            engine_context.agent_id,
            name,
            query,
            inputs,
            model,
            output_schema,
            strict_name=strict_name,
        )

    return _rflow_spawn_child


def make_launch_subagents(
    launch_children: (
        Callable[[list[dict[str, Any]], list[str]], Awaitable[list[object]]] | None
    ),
    *,
    max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
):
    """Build the public ``launch_subagents(specs)`` launcher.

    ``max_query_chars`` bounds each spec's ``query`` so a model cannot route a
    large payload through the child's task message; oversized queries raise
    before any child is spawned.
    """

    @tool(
        "Launch one or many sub-agents in parallel and wait for all. Await from "
        "normal REPL async code. specs is a list of dicts; each requires a "
        "short 'query' (the child's task; put bulk data in 'inputs', not here) "
        "and may set 'name', 'model', 'inputs', and 'output_schema'. "
        "Returns child results in spec order."
    )
    async def launch_subagents(specs):
        launch_specs, launch_names = _validate_launch_specs(specs, max_query_chars)
        if launch_children is None:
            raise RuntimeError(
                "launch_subagents(...) is not bound to the Flow scheduler"
            )
        return await launch_children(launch_specs, launch_names)

    return launch_subagents


def _validate_launch_specs(
    specs: object, max_query_chars: int
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(specs, list):
        raise TypeError("launch_subagents(...) takes a list of dict specs")
    if not specs:
        raise ValueError("launch_subagents(...) requires at least one spec")
    launch_specs: list[dict[str, Any]] = []
    launch_names: list[str] = []
    seen_names: set[str] = set()
    for spec in specs:
        if not isinstance(spec, dict):
            raise TypeError("launch_subagents(...) requires every spec to be a dict")
        if "query" not in spec:
            raise KeyError("launch_subagents(...) spec missing required 'query'")
        query = spec["query"]
        if not isinstance(query, str):
            raise TypeError(
                "launch_subagents(...) spec 'query' must be a str; got "
                f"{type(query).__name__}"
            )
        if len(query) > max_query_chars:
            raise ValueError(
                f"launch_subagents(...) spec 'query' is too long "
                f"({len(query)} chars > {max_query_chars} limit). 'query' "
                "must be a short instruction. Move the large/helpful payload "
                "(context, data, specs, file contents) into 'inputs' (a "
                "str -> str dict) and have 'query' refer to it by key, e.g. "
                "\"Answer INPUTS['question'] using INPUTS['corpus']\"."
            )
        inputs = spec.get("inputs")
        if isinstance(inputs, dict) and "query" in inputs:
            raise ValueError(
                "launch_subagents(...) spec inputs must not contain reserved "
                "key 'query'; put the child task in the top-level spec "
                "'query' and use another input key for supporting text"
            )
        name = spec.get("name", "subagent")
        if not isinstance(name, str):
            raise TypeError(
                "launch_subagents(...) spec 'name' must be a str; got "
                f"{type(name).__name__}"
            )
        if name in seen_names:
            raise ValueError(
                f"launch_subagents(...) duplicate child name {name!r}; "
                "choose a unique 'name' for each spec"
            )
        seen_names.add(name)
        clean = dict(spec)
        clean["name"] = name
        launch_specs.append(clean)
        launch_names.append(name)
    return launch_specs, launch_names


def make_graph_wait_launch_subagents(
    spawn_child,
    *,
    graph_wait: Callable[[WaitRequest], Awaitable[object]] | None,
    max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
):
    """Legacy remote launcher until remote async tool calls replace graph_wait."""

    @tool(
        "Launch one or many sub-agents in parallel and wait for all. Await from "
        "normal REPL async code. specs is a list of dicts; each requires a "
        "short 'query' (the child's task; put bulk data in 'inputs', not here) "
        "and may set 'name', 'model', 'inputs', and 'output_schema'. "
        "Returns child results in spec order."
    )
    async def launch_subagents(specs):
        clean_specs, clean_names = _validate_launch_specs(specs, max_query_chars)
        if not isinstance(specs, list):
            raise TypeError("launch_subagents(...) takes a list of dict specs")
        if not specs:
            raise ValueError("launch_subagents(...) requires at least one spec")
        results: list = [None] * len(clean_specs)
        agent_ids: list[str] = []
        positions: list[int] = []
        launch_specs: list[dict[str, Any]] = []
        launch_names: list[str] = []
        for i, spec in enumerate(clean_specs):
            name = clean_names[i]
            spawned = spawn_child(
                name=name,
                query=spec["query"],
                inputs=spec.get("inputs"),
                model=spec.get("model", "default"),
                output_schema=spec.get("output_schema"),
                strict_name=True,
            )
            if isinstance(spawned, str):
                results[i] = spawned
            else:
                if not isinstance(spawned, ChildHandle):
                    raise TypeError(
                        "launch_subagents(...) internal spawn hook returned "
                        f"{type(spawned).__name__}; expected ChildHandle or str"
                    )
                agent_ids.append(spawned.agent_id)
                positions.append(i)
                launch_specs.append(dict(spec))
                launch_names.append(str(name))
        if agent_ids:
            if graph_wait is None:
                raise RuntimeError(
                    "launch_subagents(...) is not bound to a graph wait bridge"
                )
            waited = await graph_wait(
                WaitRequest(
                    agent_ids,
                    launch_id=_launch_id_from_names(launch_names, agent_ids),
                    launch_specs=launch_specs,
                    launch_names=launch_names,
                )
            )
            for pos, result in zip(positions, waited):
                results[pos] = result
        return results

    return launch_subagents


def make_get_subagent_result(flow: BaseFlow, engine_context: EngineContext):
    """Build ``get_subagent_result(id=None)`` over durable graph launch state."""

    @tool(
        "Read results for a completed subagent launch by launch id. "
        "Returns one entry per immediate child in launch order.",
        proxy=True,
    )
    def get_subagent_result(
        id: str | None = None,
    ) -> list[dict[str, Any]]:  # noqa: A002
        graph = flow.graph
        if graph is None:
            raise RuntimeError("get_subagent_result(...) needs an active graph")
        agent = graph.agents.get(engine_context.agent_id)
        if agent is None:
            raise RuntimeError(
                f"current agent {engine_context.agent_id!r} is not in the graph"
            )
        target = id or engine_context.recovery_launch_id
        launches = [
            node
            for node in agent.nodes
            if getattr(node, "type", None) == "supervising_output"
        ]
        if target is None:
            if len(launches) == 1:
                launch = launches[0]
            else:
                available = ", ".join(
                    getattr(n, "launch_id", None) or n.id for n in launches
                )
                raise ValueError(
                    "get_subagent_result() needs a launch id; "
                    f"available launches: {available or '<none>'}"
                )
        else:
            launch = next(
                (
                    n
                    for n in launches
                    if getattr(n, "launch_id", None) == target or n.id == target
                ),
                None,
            )
            if launch is None:
                available = ", ".join(
                    getattr(n, "launch_id", None) or n.id for n in launches
                )
                raise KeyError(
                    f"no subagent launch {target!r}; "
                    f"available launches: {available or '<none>'}"
                )

        waiting_on = list(getattr(launch, "waiting_on", []))
        names = list(getattr(launch, "launch_names", []) or [])
        specs = list(getattr(launch, "launch_specs", []) or [])
        out: list[dict[str, Any]] = []
        agents = graph.agents
        for i, child_id in enumerate(waiting_on):
            child = agents.get(child_id)
            name = (
                names[i]
                if i < len(names)
                else child_id.removeprefix(agent.agent_id + ".")
            )
            entry: dict[str, Any] = {
                "agent_id": child_id,
                "name": name,
                "status": "missing",
                "result": None,
                "error": None,
            }
            if i < len(specs):
                entry["spec"] = specs[i]
            if child is None:
                entry["error"] = "child is missing from graph"
                out.append(entry)
                continue
            cur = child.current()
            if child.finished:
                entry["status"] = "done"
                entry["result"] = flow._child_result(child_id)  # noqa: SLF001
            elif cur is not None and getattr(cur, "type", None) == "error_output":
                entry["status"] = "error"
                entry["error"] = getattr(cur, "content", "") or getattr(
                    cur, "error", ""
                )
            else:
                entry["status"] = "pending"
                entry["error"] = "child is not terminal"
            out.append(entry)
        return out

    return get_subagent_result


def make_show_vars(namespace: dict[str, Any]):
    """Build ``SHOW_VARS()`` over the live REPL ``namespace`` (off by default)."""

    @tool("Show current public REPL variable names and their type names.")
    def SHOW_VARS() -> dict[str, str]:
        out: dict[str, str] = {}
        for name, value in namespace.items():
            if name.startswith("_") or name in HIDDEN_REPL_TOOL_NAMES:
                continue
            if name == "SHOW_VARS" or get_tool_metadata(value) is not None:
                continue
            out[name] = type(value).__name__
        return out

    return SHOW_VARS


def make_history(flow: BaseFlow, engine_context: EngineContext) -> "History":
    """Build this agent's ``HISTORY`` view, mirroring ``make_done``.

    The view resolves the agent's *live* graph by id at call time (never captures
    a ``Graph``), so it stays correct across ``set_graph(...)``, ``inject``, and
    ``truncate`` — and ships nothing until a method is actually called. ``HISTORY``
    is host-bound: a remote runtime object-proxies it so the slice computed over
    the full host trajectory is all that crosses the wire.
    """
    return History(lambda: flow.graph.agents.get(engine_context.agent_id))


class History:
    """Read-only view of THIS agent's own message trajectory (untruncated).

    ``build_messages`` may window or truncate what the model sees each turn;
    ``HISTORY`` always exposes the full user/assistant projection so code can
    recover earlier turns. It holds a *resolver* (not a ``Graph``) and reads the
    live trajectory on every call, so it reflects everything recorded so far —
    even after the run's graph has been copied or edited.
    """

    __slots__ = ("_resolve",)

    def __init__(self, resolve: "Callable[[], Graph | None]") -> None:
        self._resolve = resolve

    def messages(self) -> list[dict[str, str]]:
        """The full chat projection of this agent's turns (no system message)."""
        agent = self._resolve()
        return agent.messages("") if agent is not None else []

    def count(self) -> int:
        """Number of projected messages (so ``len`` works over the wire too)."""
        return len(self.messages())

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"HISTORY({self.count()} messages)"

    def read(self, i: int) -> dict[str, str]:
        """One message by index (negative indexes count from the end)."""
        return self.messages()[i]

    def last(self, n: int = 5) -> list[dict[str, str]]:
        """The most recent ``n`` messages."""
        if n <= 0:
            return []
        return self.messages()[-n:]

    def text(self, *, roles: tuple[str, ...] = ("user", "assistant")) -> str:
        """The selected turns flattened to ``[role] content`` blocks."""
        return "\n\n".join(
            f"[{m['role']}] {m['content']}"
            for m in self.messages()
            if m["role"] in roles
        )

    def grep(self, pattern: str, *, max_results: int = 50) -> list[str]:
        """Lines across all turns matching ``pattern`` (``idx[role]: line``)."""
        regex = re.compile(pattern)
        out: list[str] = []
        for idx, message in enumerate(self.messages()):
            for line in message["content"].splitlines():
                if regex.search(line):
                    out.append(f"{idx}[{message['role']}]: {line}")
                    if len(out) >= max_results:
                        return out
        return out


__all__ = [
    "make_get_subagent_result",
    "DEFAULT_MAX_QUERY_CHARS",
    "History",
    "make_done",
    "make_history",
    "make_launch_subagents",
    "make_show_vars",
    "make_spawn_child",
]
