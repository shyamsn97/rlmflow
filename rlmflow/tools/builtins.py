"""Built-in agent tool factories.

These are the framework tools every agent gets (``done`` and
``launch_subagents``). They live here rather than on ``Flow`` so the class stays
a lean orchestrator; each factory closes over the ``Flow`` it belongs to. Not
re-exported from ``tools/__init__`` to keep the package import graph acyclic.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rlmflow.graph import (
    Graph,
    ResumeAction,
    SupervisingOutput,
)
from rlmflow.graph.events import AddChild
from rlmflow.runtime.repl import DoneSignal
from rlmflow.structured import json_schema_for
from rlmflow.tasks import TaskQueue
from rlmflow.tools.tools import tool

if TYPE_CHECKING:
    from rlmflow.flow import Flow
    from rlmflow.runtime.repl import Repl


def refuse_max_depth(max_depth: int) -> str:
    return f"[refused: max depth {max_depth}]"


def refuse_query_too_long(length: int, cap: int) -> str:
    return (
        f"[refused: query too long ({length} chars > {cap})] keep query a short "
        "instruction and move bulk payloads into inputs"
    )


def make_done(flow: Flow, graph: Graph, repl: Repl):
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


def make_launch_subagents(flow: Flow, parent: Graph, repl: Repl):
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
            name = spec.get("name", f"child{index}")

            # Warm start: a spec may carry a prepared child ``graph`` (a fork/rewind
            # result) instead of a ``query``. Attach it via ``adopt`` and, if given,
            # append the kickoff ``query`` as the next user turn on its trajectory.
            # It then supervises/awaits on the same loop as a cold child.
            prepared = spec.get("graph")
            if prepared is not None:
                child = flow.adopt(parent, prepared, name=name)
                query = spec.get("query")
                if query:
                    child.append_query(query)
                child_ids.append(child.agent_id)
                results.append("")
                continue

            query = spec["query"]
            if len(query) > flow.max_query_chars:
                results.append(refuse_query_too_long(len(query), flow.max_query_chars))
                continue
            schema = spec.get("output_schema")
            child = Graph(
                agent_id=f"{parent.agent_id}.{name}",
                graph_id=parent.graph_id,
                query=query,
                inputs=dict(spec.get("inputs") or {}),
                model=spec.get("model", "default"),
                prompt_profile=spec.get(
                    "prompt_profile", spec.get("prompt", "default")
                ),
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
            child_ids.append(child.agent_id)
            results.append("")

        flow.append_node(
            parent, SupervisingOutput(output=repl.drain(), waiting_on=child_ids)
        )
        # One path: submit each child to a task queue, then poll until they finish.
        # Each child self-drives (``run_agent``), so this works whether or not an
        # outer ``run_streaming`` loop is driving the queue. When a run exists we
        # submit to its queue so the stream observes child events and never
        # double-drives a child (the queue keeps one task per agent id); otherwise
        # a throwaway queue is enough to run the children and await them.
        run = flow.runs.get(parent.graph_id)
        tasks = run.tasks if run is not None else TaskQueue()
        for cid in child_ids:
            tasks.add(cid, lambda child=parent[cid]: flow.run_agent(child))
        while not all(parent[cid].finished for cid in child_ids):
            await tasks.changed()
        flow.append_node(parent, ResumeAction(resumed_from=child_ids))
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
