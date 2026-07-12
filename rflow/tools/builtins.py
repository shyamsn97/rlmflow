"""Built-in agent tool factories.

These are the framework tools every agent gets (``done`` and
``launch_subagents``). They live here rather than on ``Flow`` so the class stays
a lean orchestrator; each factory closes over the ``Flow`` it belongs to. Not
re-exported from ``tools/__init__`` to keep the package import graph acyclic.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rflow.graph import (
    Graph,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
)
from rflow.graph.events import AddChild
from rflow.runtime.repl import DoneSignal
from rflow.structured import json_schema_for
from rflow.tools.tools import tool

if TYPE_CHECKING:
    from rflow.flow import Flow
    from rflow.runtime.repl import ReplLike


def refuse_max_depth(max_depth: int) -> str:
    return f"[refused: max depth {max_depth}]"


def refuse_query_too_long(length: int, cap: int) -> str:
    return (
        f"[refused: query too long ({length} chars > {cap})] keep query a short "
        "instruction and move bulk payloads into inputs"
    )


def make_done(flow: Flow, graph: Graph, repl: ReplLike):
    @tool("Submit this agent's final answer and end its run.", proxy=True)
    def done(answer: object) -> None:
        if graph.output_schema is not None:
            content = answer if isinstance(answer, str) else json.dumps(answer)
            flow.output_parser(content, graph.output_schema)
            repl.done_result = content
        else:
            repl.done_result = str(answer)
        print(f"[done] {repl.done_result}")
        raise DoneSignal()

    return done


def make_launch_subagents(flow: Flow, parent: Graph, repl: ReplLike):
    @tool(
        "Spawn child agents from a list of specs and await their results.",
        proxy=True,
    )
    async def launch_subagents(specs: list[dict[str, Any]]) -> list[Any]:
        results: list[Any] = []
        child_ids: list[str] = []
        for index, spec in enumerate(specs):
            if parent.depth >= flow.max_depth:
                results.append(refuse_max_depth(flow.max_depth))
                continue
            query = spec["query"]
            if len(query) > flow.max_query_chars:
                results.append(refuse_query_too_long(len(query), flow.max_query_chars))
                continue
            name = spec.get("name", f"child{index}")
            schema = spec.get("output_schema")
            child = Graph(
                agent_id=f"{parent.agent_id}.{name}",
                graph_id=parent.graph_id,
                query=query,
                inputs=dict(spec.get("inputs") or {}),
                model=spec.get("model", "default"),
                output_schema=json_schema_for(schema) if schema is not None else None,
                depth=parent.depth + 1,
                parent_agent_id=parent.agent_id,
            )
            flow.apply_action(
                parent,
                AddChild(
                    type="add_child", parent_agent_id=parent.agent_id, child=child
                ),
            )
            flow.append_node(child, UserQuery(content=query))
            child_ids.append(child.agent_id)
            results.append("")

        await flow.commit(
            parent, SupervisingOutput(output=repl.drain(), waiting_on=child_ids)
        )
        if flow.tasks is None or flow.until in {"done", "finished"}:
            # Tool called directly or full run: gather children directly. During an
            # active stream, suppress scheduler follow-ups for these child ids so
            # the stream observes their events but does not drive them a second time.
            flow.suppressed_agents.update(child_ids)
            try:
                await flow.pool.gather(
                    *(flow.run_agent(parent[cid]) for cid in child_ids)
                )
            finally:
                flow.suppressed_agents.difference_update(child_ids)
        else:
            for cid in child_ids:
                flow.schedule_agent(parent[cid])
            while not all(parent[cid].finished for cid in child_ids):
                await flow.tasks.changed()
        await flow.commit(parent, ResumeAction(resumed_from=child_ids))
        for i, child_id in enumerate(child_ids):
            child = parent[child_id]
            result, schema = child.result(), child.output_schema
            results[i] = (
                flow.output_parser(result, schema) if schema is not None else result
            )
        return results

    return launch_subagents


__all__ = [
    "make_done",
    "make_launch_subagents",
    "refuse_max_depth",
    "refuse_query_too_long",
]
