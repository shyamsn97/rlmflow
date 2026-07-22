"""Graph-operation tools: let a *controller* agent fork, run, merge, and discard
other graphs from inside its own REPL.

These are free functions layered on a :class:`~rlmflow.flow.Flow` — no new ``Flow``
methods and no class. The ``Flow`` only exposes two generic seams they build on:
``flow.halts`` (named ``(event, graph) -> bool`` stop predicates) and
``flow.tool_factories`` (``(flow, graph, repl) -> {name: obj}`` callables run per
agent in ``build_tools``). See ``docs/internal/graph_ops_tools_plan.md``.

Authority model:

* **Ambient** — a controller may always operate on its own descendants (agents
  reachable by ``agent_id`` in its graph). Passing ``controller`` alone grants
  this and nothing else.
* **Granted** — ``roots={"worker": worker_graph}`` hands a controller explicit
  authority over foreign graphs it did not spawn (the Shepherd case). The named
  graphs are also injected into the REPL so the agent can refer to them.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable

from rlmflow.graph import Graph
from rlmflow.tools.tools import get_tool_metadata, tool

if TYPE_CHECKING:
    from rlmflow.flow import Flow
    from rlmflow.graph.events import Event
    from rlmflow.runtime.repl import Repl

HaltFn = Callable[["Event", Graph], bool]
ToolFactory = Callable[["Flow", Graph, "Repl"], dict[str, Any]]


def _repl_key(graph: Graph) -> tuple[str, str]:
    # Inlined from ``rlmflow.utils.repl_key`` to keep this module free of the
    # ``rlmflow.utils`` import (which imports back into ``rlmflow.tools``, a cycle).
    return (graph.graph_id, graph.agent_id)


def _tool_name(fn: Any) -> str:
    meta = get_tool_metadata(fn)
    return meta.name if meta is not None else fn.__name__


def register_halt(flow: Flow, name: str, predicate: HaltFn) -> None:
    """Register a named stop predicate agents can pass as ``run(..., until=name)``.

    Names live in ``flow.halts`` so a callable boundary survives crossing the
    tool boundary as a plain string (readable for the LLM, portable to remote
    runtimes) instead of being smuggled as an opaque object.
    """
    flow.halts[name] = predicate


def inject_tools(
    flow: Flow,
    source: ToolFactory | dict[str, Any] | list[Any],
    *,
    only: Graph | None = None,
) -> None:
    """Add a tool *source* to the flow, applying it to every agent's REPL.

    ``source`` is either a factory (``(flow, graph, repl) -> {name: obj}``, run
    per agent so tools can close over that agent's graph/REPL) or a static
    ``dict``/``list`` of tools shared by all agents. Registered factories run in
    ``build_tools`` for future REPLs and are backfilled onto any already-open
    REPLs immediately, so the call is order-independent.

    A bare callable is treated as a factory; pass static tools as a ``list`` or
    ``dict`` to disambiguate. Pass ``only=graph`` to restrict the source to that
    one agent's REPL instead of every agent's.
    """
    factory = source if callable(source) else _const_factory(_as_tool_dict(source))
    if only is not None:
        factory = _scoped_factory(factory, _repl_key(only))
    flow.tool_factories.append(factory)
    for graph, repl in flow.open_repls():
        for name, obj in factory(flow, graph, repl).items():
            repl.inject(name, obj)


def _scoped_factory(factory: ToolFactory, key: tuple[str, str]) -> ToolFactory:
    def scoped(flow: Flow, graph: Graph, repl: Repl) -> dict[str, Any]:
        return factory(flow, graph, repl) if _repl_key(graph) == key else {}

    return scoped


def _as_tool_dict(source: dict[str, Any] | list[Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        return dict(source)
    return {_tool_name(fn): fn for fn in source}


def _const_factory(tools: dict[str, Any]) -> ToolFactory:
    def factory(flow: Flow, graph: Graph, repl: Repl) -> dict[str, Any]:
        return tools

    return factory


def enable_graph_ops(
    flow: Flow,
    controller: Graph | None = None,
    *,
    roots: dict[str, Graph] | None = None,
    default_until: str = "done",
) -> None:
    """Give agents fork/run/merge/discard tools over graphs they have authority on.

    * ``enable_graph_ops(flow)`` — flow-wide: every agent gets the tools with
      ambient authority over its own descendants.
    * ``enable_graph_ops(flow, controller, roots={...})`` — targeted: only
      ``controller`` gets the tools, additionally granted authority over the
      named foreign ``roots``.

    ``default_until`` is the halt name ``run(target)`` uses when the agent omits
    ``until`` (a built-in like ``"done"`` or a name from :func:`register_halt`).
    """
    if controller is None and roots:
        raise ValueError(
            "roots grants authority over foreign graphs to a specific controller; "
            "it is meaningless flow-wide. Pass a controller."
        )
    only = None if controller is None else _repl_key(controller)
    inject_tools(
        flow,
        partial(
            _graph_ops_factory, roots=roots, default_until=default_until, only=only
        ),
    )


def _graph_ops_factory(
    flow: Flow,
    graph: Graph,
    repl: Repl,
    *,
    roots: dict[str, Graph] | None,
    default_until: str,
    only: tuple[str, str] | None,
) -> dict[str, Any]:
    if only is not None and _repl_key(graph) != only:
        return {}
    return _graph_ops(flow, graph, roots=roots, default_until=default_until)


def _graph_ops(
    flow: Flow,
    controller: Graph,
    *,
    roots: dict[str, Graph] | None,
    default_until: str,
) -> dict[str, Any]:
    """Build the fork/run/merge/discard tools bound to one controller graph."""

    def resolve(target: Graph | str) -> Graph:
        if isinstance(target, Graph):
            return target
        if roots and target in roots:
            return roots[target]
        if target in controller:  # a descendant of the controller
            return controller[target]
        raise KeyError(
            f"no authority over graph {target!r}: not a descendant and not a "
            f"granted root"
        )

    @tool("Fork a graph you control into an independent branch; returns the new Graph.")
    async def fork(
        target: Graph | str,
        *,
        from_node_id: str | None = None,
        keep_anchor: bool = False,
        mode: str = "replay",
    ) -> Graph:
        return await flow.fork(
            resolve(target),
            from_node_id=from_node_id,
            keep_anchor=keep_anchor,
            mode=mode,
        )

    @tool(
        "Rewind a graph you control by n decision turns into a fresh branch; returns the new Graph.",
        name="rewind",
    )
    async def rewind_tool(
        target: Graph | str, *, n: int = 1, mode: str = "replay"
    ) -> Graph:
        return await flow.rewind(resolve(target), n=n, mode=mode)

    @tool(
        "Run a graph you control to a named halt ('done' or a registered halt); returns it."
    )
    async def run(target: Graph | str, *, until: str | None = None) -> Graph:
        graph = resolve(target)
        # ``until`` is a halt name (built-in or registered); run_streaming resolves it.
        async for _ in flow.run_streaming(graph=graph, until=until or default_until):
            pass
        return graph

    @tool("Advance a graph you control by exactly one step (one event); returns it.")
    async def step(target: Graph | str) -> Graph:
        graph = resolve(target)
        async for _ in flow.run_streaming(graph=graph, until="next"):
            pass
        return graph

    @tool("Fold a child branch's changes back into a parent graph you control.")
    async def merge(
        parent: Graph | str, child: Graph | str, *, summary: str | None = None
    ) -> None:
        await flow.merge(resolve(parent), resolve(child), summary=summary)

    @tool("Discard branches you no longer need (frees their REPLs).")
    def discard(*targets: Graph | str) -> None:
        flow.discard(*(resolve(t) for t in targets))

    tools: dict[str, Any] = {
        "fork": fork,
        "rewind": rewind_tool,
        "run": run,
        "step": step,
        "merge": merge,
        "discard": discard,
    }
    # Also bind granted roots by name so the agent can refer to them directly.
    if roots:
        tools.update(roots)
    return tools


__all__ = [
    "HaltFn",
    "ToolFactory",
    "enable_graph_ops",
    "inject_tools",
    "register_halt",
]
