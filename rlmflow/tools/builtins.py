"""Host operations and per-REPL built-in tool factories."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import hashlib
import json
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NoReturn

from rlmflow.graph import calls
from rlmflow.graph.config import RECURSE, input_parts, validate_argument_name, visit_argument
from rlmflow.graph.nodes import RETURN, AgentRoot, Node
from rlmflow.repl import Repl, current_repl, record_metadata
from rlmflow.runtime.repl import (
    AGENT_REF_KEY,
    DoneSignal,
    NotYourAgent,
    encode_agent_references,
)
from rlmflow.tools.agents import AgentHandle, agent_handle, resolve_agent
from rlmflow.tools.declare import tool
from rlmflow.workspaces import COMMIT

if TYPE_CHECKING:
    from rlmflow.flow import Flow


def _saves(
    fn: Callable,
    *,
    key: Callable[..., str],
    receipt: Callable[..., Any] | None = None,
    restore: Callable[..., Any] | None = None,
) -> Callable:
    """A built-in saved under ``key(...)``, as ``receipt(result, ...)``, restored by ``restore``."""
    fn._tool_meta = dataclasses.replace(fn._tool_meta, key=key, receipt=receipt, restore=restore)
    return fn


LAUNCH_DESCRIPTION = """\
Start a run and return it immediately as an `Agent`. `agent=` names one of the \
agents listed below (default: {default}). JSON keyword arguments become the new run's \
REPL variables; wrap one as `Input(value, description="...")` to explain it. \
`await agent` returns its result.

Example — launch independent work together, then collect it:
```python
first = launch_agent("Produce deliverable A.")
second = launch_agent("Produce deliverable B.")
print(await asyncio.gather(first, second))
```"""

WORKSPACE_DESCRIPTION = """
Files: your directory is your own; a launched agent works in a copy of your files as they \
are at launch, and its edits don't reach yours. `workspace="shared"` makes it work in your \
directory instead; `workspace=other_agent` or `workspace=agent.commit` starts it from \
those files. `checkout(agent.commit)` makes your files that commit's."""


def _on_flow_loop(flow: Flow, g: Any, fn: Callable[[], Any]) -> Any:
    """Run a write to ``g`` on the event loop thread streaming it."""
    owner = flow.loop(g)
    try:
        current = asyncio.get_running_loop()
    except RuntimeError:
        current = None
    if owner is None or current is owner:
        return fn()

    async def run() -> Any:
        return fn()

    return asyncio.run_coroutine_threadsafe(run(), owner).result()


def _check_json(path: str, value: Any) -> Any:
    if isinstance(value, AgentHandle) or (isinstance(value, dict) and AGENT_REF_KEY in value):
        raise TypeError(f"argument {path!r} is an agent; await the agent first and pass its result")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        json.dumps(value, allow_nan=False)
        return value
    if isinstance(value, dict | list):
        return RECURSE
    raise TypeError(f"argument {path!r} must be JSON-compatible")


def check_args(args: dict[str, Any]) -> dict[str, Any]:
    """Launch arguments: named like Python variables, and strict JSON (``Input`` allowed)."""
    for name, value in args.items():
        validate_argument_name(name)
        wrapped = input_parts(value)
        visit_argument(wrapped[0] if wrapped is not None else value, _check_json, path=name)
    return args


def _plain(workspace: str) -> bool:
    """``"shared"`` or something shaped like a commit, rather than a run's name."""
    return workspace == "shared" or COMMIT.fullmatch(workspace) is not None


def launch(
    repl: Repl,
    task: str,
    *,
    agent: str | None = None,
    name: str | None = None,
    workspace: Any = None,
    **args: Any,
) -> AgentRoot:
    """Start a run launched by ``repl``'s running node."""
    if not isinstance(task, str):
        raise TypeError("task must be a string")
    if not repl.can_launch():
        raise ValueError(f"cannot start agents beyond max depth {repl.agent.max_depth}")
    # Keep the historical facade attribute as the monkeypatch point.
    from rlmflow import tools as tools_facade

    max_task = tools_facade.MAX_TASK
    if len(task) > max_task:
        raise ValueError(f"task exceeds {max_task} characters")
    if agent is not None and not isinstance(agent, str):
        raise TypeError("agent= takes the name of one of the Flow's agents")
    if workspace is not None and not (isinstance(workspace, str) and _plain(workspace)):
        workspace = resolve_agent(repl.graph, workspace)
    return repl.flow.launch(
        repl, task, agent=agent or child_agent(repl), name=name, workspace=workspace, **args
    )


def child_agent(repl: Repl) -> str:
    """The agent ``launch_agent`` starts when none is named."""
    return repl.agent.child_agent or repl.root.agent


def send(repl: Repl, target: object, text: str, **inputs: Any) -> Node:
    """Attach a message to another run, binding ``inputs`` there; it reopens a finished run."""
    if not isinstance(text, str):
        raise TypeError("message text must be a string")
    g = repl.graph
    run = resolve_agent(g, target)
    message = g.send(run, f"From {repl.root.path()}: {text}", by=repl.node, **inputs)
    repl.flow.land(message)
    return message


def _handle_for(g: Any, entry: dict[str, Any], *_args: Any, **_kwargs: Any) -> AgentHandle:
    """The handle a saved launch or send returns again."""
    agent_id = entry.get("agent", "")
    if agent_id in g:
        return agent_handle(g, g.agent_root(agent_id))
    return AgentHandle(agent_id=agent_id, name="(removed)", path="(removed)")


def _agent_saved(handle: AgentHandle, *_args: Any, **_kwargs: Any) -> dict[str, str]:
    return {"agent": handle.agent_id}


def _agent_id(g: Any, target: Any) -> Any:
    """The id a run reference names, for a call key; the reference itself if it names none."""
    try:
        return resolve_agent(g, target).id
    except (KeyError, LookupError, TypeError, ValueError):
        return encode_agent_references(target)


def launch_tool(repl: Repl):
    flow, g = repl.flow, repl.graph
    lock = threading.Lock()

    from rlmflow.agents.agent import launchable

    agents = launchable(repl)
    default = f"`{repl.agent.child_agent}`" if repl.agent.child_agent else "your own"
    description = LAUNCH_DESCRIPTION.format(default=default) + (f"\n{agents}" if agents else "")
    if getattr(flow.workspaces, "isolates_launches", False):
        description += "\n" + WORKSPACE_DESCRIPTION

    @tool(description, is_async=False)
    def launch_agent(
        task: str,
        *,
        agent: str | None = None,
        name: str | None = None,
        workspace: Any = None,
        **args: Any,
    ) -> AgentHandle:
        def start() -> AgentHandle:
            with lock:
                child = launch(
                    repl, task, agent=agent, name=name, workspace=workspace, **check_args(args)
                )
                return agent_handle(g, child)

        return _on_flow_loop(flow, g, start)

    def key(
        task: Any, *, agent: Any = None, name: Any = None, workspace: Any = None, **args: Any
    ) -> str:
        chosen = agent or child_agent(repl)
        resolved = flow.agents.get(chosen)
        parts = [
            calls.LAUNCH,
            task,
            chosen,
            name,
            check_args(args),
            resolved.summary() if resolved is not None else None,
        ]
        if workspace is not None:
            parts.append(workspace if isinstance(workspace, str) else _agent_id(g, workspace))
        return calls.key(*parts)

    return _saves(
        launch_agent, key=key, receipt=_agent_saved, restore=functools.partial(_handle_for, g)
    )


def await_tool(repl: Repl):
    """``await agent``: wait for a run this one launched, and save which one it was."""
    flow, g = repl.flow, repl.graph

    def returned(final: Node) -> Any:
        """The result of ``final``, with the ``return`` edge to the node that got it."""
        running = current_repl()
        node = running.node if running is not None else None
        if (
            node is not None
            and node.id in g
            and not g.edges(kind=RETURN, source=final.id, target=node.id)
        ):
            g.link(RETURN, final, node)
        return final.result

    async def result_of(agent_id: str, *, check: bool) -> Any:
        target = resolve_agent(g, agent_id)
        if check and repl.root not in target.lineage()[:-1]:
            name = target.name
            raise NotYourAgent(
                f"you can only await agents you launched; {name!r} is not one. "
                f"Message it with AGENTS[{name!r}].send(...) and end your turn "
                "to read its reply."
            )
        if target.status() != "done":
            if not flow.running(g):
                raise RuntimeError("awaiting an agent requires a running Flow")
            await flow.wait_for(g, lambda: target.id not in g or target.status() == "done")
            if target.id not in g:
                raise RuntimeError(f"agent {target.name!r} was removed")
        return returned(target.last_turn())

    def restore(entry: dict[str, Any], agent_id: str) -> Any:
        if "value" in entry:
            return entry["value"]
        final = g._nodes.get(entry.get("done", ""))
        if final is not None and final.terminal:
            return returned(final)
        return result_of(entry.get("agent") or agent_id, check=False)

    def receipt(_result: Any, agent_id: str) -> dict[str, str]:
        target = resolve_agent(g, agent_id)
        return {"agent": target.id, "done": target.last_turn().id}

    @tool("Wait for an agent you launched and return its result.")
    async def wait_agent(agent_id: str) -> Any:
        return await result_of(agent_id, check=True)

    def key(agent_id: Any) -> str:
        return calls.key(calls.AWAIT, _agent_id(g, agent_id))

    return _saves(wait_agent, key=key, receipt=receipt, restore=restore)


def _model_help(flow: Flow, default_model: str | None) -> str:
    names = ", ".join(f"`{key}`" for key in sorted(flow.llms))
    configured = f" (configured: {names})" if names else ""
    default = (
        f"Omitting `model` uses `{default_model}`." if default_model else "Pass `model` every call."
    )
    return f" `model=` takes a model name{configured}. {default}"


async def query(
    flow: Flow,
    prompt: str,
    *,
    model: str | None,
    schema: Any = None,
    **sampling: Any,
) -> Any:
    """One context-free model call; with ``schema``, a validated JSON value."""
    from rlmflow import schema as schemas

    if not isinstance(prompt, str):
        raise TypeError("llm_query(prompt) takes a str")
    normalized = schemas.json_schema_for(schema) if schema is not None else None
    text = (
        prompt
        if normalized is None
        else f"{prompt}\n\n{schemas.contract(normalized, source='query')}"
    )
    reply = await flow.complete([{"role": "user", "content": text}], model, **sampling)
    value = reply.text if normalized is None else schemas.parse(reply.text, normalized)
    if current_repl() is not None:
        record_metadata(
            "llm_query.calls",
            {
                "model": model,
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                "structured": normalized is not None,
                "usage": reply.usage.as_dict(),
            },
            append=True,
        )
    return value


def llm_query_tool(repl: Repl):
    flow = repl.flow
    default_model = repl.agent.model
    description = (
        "Make one model completion for extraction, summarization, or Q&A over the "
        "text you pass. The call sees no REPL, conversation, or tools, so use it for a "
        "one-shot transform, not multi-step owned work." + _model_help(flow, default_model)
    )

    @tool(description)
    async def llm_query(
        prompt: str,
        *,
        model: str | None = None,
        schema: Any = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        sampling = {
            key: value
            for key, value in {"temperature": temperature, "max_tokens": max_tokens}.items()
            if value is not None
        }
        return await query(flow, prompt, model=model or default_model, schema=schema, **sampling)

    def key(
        prompt: Any,
        *,
        model: Any = None,
        schema: Any = None,
        temperature: Any = None,
        max_tokens: Any = None,
    ) -> str:
        from rlmflow import schema as schemas

        normalized = schemas.json_schema_for(schema) if schema is not None else None
        return calls.key(
            "llm_query", prompt, model or default_model, normalized, temperature, max_tokens
        )

    return _saves(llm_query, key=key)


def submit_tool(repl: Repl):
    from rlmflow.schema import as_json_value, check

    output_schema = repl.agent.output_schema

    @tool(
        "Submit your final result as a string or strict JSON value. A valid result completes you.",
    )
    def submit(answer: object) -> NoReturn:
        value = check(answer, output_schema) if output_schema is not None else as_json_value(answer)
        raise DoneSignal(value)

    return submit


def send_tool(repl: Repl):
    flow, g = repl.flow, repl.graph

    @tool(
        "Message another agent and return that agent immediately. Keyword arguments "
        "become variables in its REPL. It reads the message at its next turn, which "
        "reopens it if it finished.",
        is_async=False,
    )
    def send_agent(target: Any, message: str, **inputs: Any) -> AgentHandle:
        def deliver() -> tuple[AgentHandle, str]:
            run = resolve_agent(g, target)
            if run.id == repl.root.id:
                raise ValueError("an agent cannot message itself")
            node = send(repl, run, message, **inputs)
            return agent_handle(g, run), node.id

        handle, message_id = _on_flow_loop(flow, g, deliver)
        sent[id(handle)] = message_id
        return handle

    sent: dict[int, str] = {}

    def key(target: Any, message: Any, **inputs: Any) -> str:
        return calls.key(calls.SEND, _agent_id(g, target), message, inputs)

    def receipt(handle: AgentHandle, *_args: Any, **_kwargs: Any) -> dict[str, str]:
        return {"agent": handle.agent_id, "message": sent.pop(id(handle), "")}

    return _saves(send_agent, key=key, receipt=receipt, restore=functools.partial(_handle_for, g))


def checkout_tool(repl: Repl):
    flow, g = repl.flow, repl.graph

    @tool(
        "Replace your files with another agent's current files, or a commit's.",
        is_async=False,
    )
    def checkout(source: Any) -> str:
        def resolve() -> str:
            if isinstance(source, str) and COMMIT.fullmatch(source):
                return source
            return flow.workspaces.commit(resolve_agent(g, source))

        commit = _on_flow_loop(flow, g, resolve)
        flow.workspaces.adopt(repl.root, commit)
        return commit

    return checkout
