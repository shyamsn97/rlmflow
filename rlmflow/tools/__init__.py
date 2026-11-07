"""REPL tools: the ``@tool`` decorator, its prompt docs, and the namespace each cell runs in.

The namespace is the same for every harness, since every cell runs in our REPL:
``submit``, ``launch_agent``, ``llm_query``, ``AGENTS`` (with ``await agent``
and ``agent.send``), ``Input``, plus the host's own tools.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import hashlib
import inspect
import json
import threading
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Protocol, runtime_checkable

from rlmflow.agent_tools import tools as agent_tools
from rlmflow.agent_tools.view import (
    AgentHandle,
    agent_handle,
    build_agent_directory,
    resolve_agent,
)
from rlmflow.graph.config import (
    RECURSE,
    Input,
    can_spawn,
    input_parts,
    validate_argument_name,
    visit_argument,
)
from rlmflow.graph.nodes import DONE, Node, Step, current_code, record_metadata
from rlmflow.runtime.repl import (
    AGENT_REF_KEY,
    AGENT_SEND_TOOL,
    AGENT_WAIT_TOOL,
    AGENTS_BINDING,
    DoneSignal,
    NotYourAgent,
    ToolMetadata,
    get_tool_metadata,
)

if TYPE_CHECKING:
    from rlmflow.flow import Flow


Where = Literal["host", "repl"]


def tool(
    description: str | None = None,
    *,
    name: str | None = None,
    where: Where = "host",
    is_async: bool | None = None,
    when: Callable[..., bool] | None = None,
) -> Callable:
    """Attach tool metadata to ``fn``.

    ``description`` defaults to the first paragraph of ``fn``'s docstring.
    ``where="host"`` (the default) runs the tool in this process, where its
    clients and connections live; agent code calls it over the REPL protocol.
    ``where="repl"`` copies it into the agent's REPL (and sandbox), for pure
    functions that should run next to the agent's files. ``is_async`` records
    whether agent code must ``await`` the tool; it defaults to ``async def``
    detection. ``when`` is ``(flow, node) -> bool``: when false, the tool is
    left out of the prompt and the REPL.

    A host tool's JSON result is saved on the cell that called it, so running
    the cell again (after an edit, or to rebuild its REPL) returns it instead of
    calling the tool twice.
    """
    if where not in ("host", "repl"):
        raise ValueError(f"tool where must be 'host' or 'repl', not {where!r}")

    def decorator(fn):
        resolved = inspect.iscoroutinefunction(fn) if is_async is None else is_async
        text = description if description is not None else _first_paragraph(fn.__doc__)
        fn._tool_meta = ToolMetadata(
            name=name or fn.__name__.removeprefix("tool_"),
            description=text.strip(),
            proxy=where == "host",
            is_async=resolved,
            when=when,
        )
        return fn

    return decorator


def _saves(fn: Callable, *, receipt: Callable[..., Any], restore: Callable[..., Any]) -> Callable:
    """A built-in tool that saves ``receipt(result, ...)`` and replays with ``restore(entry, ...)``."""
    fn._tool_meta = dataclasses.replace(fn._tool_meta, receipt=receipt, restore=restore)
    return fn


def _first_paragraph(doc: str | None) -> str:
    return inspect.cleandoc(doc).split("\n\n", 1)[0].replace("\n", " ") if doc else ""


def as_tool(fn: Any) -> Any:
    """``fn`` as a tool: a plain function gets metadata from its docstring and runs on the host."""
    if get_tool_metadata(fn) is not None or not inspect.isroutine(fn):
        return fn
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await fn(*args, **kwargs)

    else:

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

    return tool()(wrapper)


_JSON_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _json_type(prop: Mapping[str, Any]) -> str | None:
    kind = prop.get("type")
    if isinstance(kind, list):
        kind = next((item for item in kind if item != "null"), None)
    return kind if isinstance(kind, str) else None


def tool_from_schema(
    name: str,
    description: str,
    parameters: Mapping[str, Any] | None,
    call: Callable[[dict[str, Any]], Any],
    *,
    is_async: bool | None = None,
    where: Where = "host",
    when: Callable[..., bool] | None = None,
) -> Callable:
    """A tool whose arguments are a JSON Schema object, as MCP and function-calling APIs describe them.

    ``call(arguments)`` runs it with the arguments the agent passed, as a dict
    (only those it passed, so the callee applies its own defaults).
    """
    parameters = parameters or {}
    properties: Mapping[str, Any] = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    ordered = [
        *(p for p in properties if p in required),
        *(p for p in properties if p not in required),
    ]
    signature = inspect.Signature(
        [
            inspect.Parameter(
                prop,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=inspect.Parameter.empty
                if prop in required
                else properties[prop].get("default"),
                annotation=_JSON_TYPES.get(
                    _json_type(properties[prop]) or "", inspect.Parameter.empty
                ),
            )
            for prop in ordered
        ]
    )
    notes = [
        f"`{prop}`: {properties[prop]['description']}"
        for prop in ordered
        if properties[prop].get("description")
    ]
    text = description.strip() + (" Arguments: " + "; ".join(notes) + "." if notes else "")

    if inspect.iscoroutinefunction(call) if is_async is None else is_async:

        async def fn(*args: Any, **kwargs: Any) -> Any:
            return await call(dict(signature.bind(*args, **kwargs).arguments))

    else:

        def fn(*args: Any, **kwargs: Any) -> Any:
            return call(dict(signature.bind(*args, **kwargs).arguments))

    fn.__name__ = fn.__qualname__ = name
    fn.__signature__ = signature  # type: ignore[attr-defined]
    return tool(text, name=name, where=where, when=when)(fn)


@runtime_checkable
class Toolkit(Protocol):
    """Tools that need a live resource, such as a server connection.

    The Flow awaits ``open()`` on its own event loop when a run starts, adds the
    tools it returns, and awaits ``close()`` and removes them when the run ends.
    """

    async def open(self) -> Iterable[Any]: ...

    async def close(self) -> None: ...


def tool_allowed(fn: Any, flow: Any = None, node: Any = None) -> bool:
    meta = get_tool_metadata(fn)
    return meta is None or meta.when is None or bool(meta.when(flow, node))


def format_tool_line(fn: Callable) -> str:
    meta = get_tool_metadata(fn)
    if meta is None:
        return ""
    try:
        sig = str(inspect.signature(fn, eval_str=True))
    except (TypeError, ValueError, NameError):
        sig = "(...)"
    sig = sig.replace("rlmflow.agent_tools.view.", "").replace("AgentHandle", "Agent")
    prefix = "await " if meta.is_async else ""
    line = f"- `{prefix}{meta.name}{sig}`"
    return f"{line}: {meta.description}" if meta.description else line


def format_tool_docs(tools: Mapping[str, Any]) -> str:
    """One ``### Tools`` list for the callables an agent has bound."""
    lines = [line for value in tools.values() if (line := format_tool_line(value))]
    return "\n".join(["### Tools", *lines]) if lines else ""


def tool_name(fn: Any) -> str:
    meta = get_tool_metadata(fn)
    return meta.name if meta is not None else fn.__name__


RESERVED_TOOLS = frozenset(
    {
        "submit",
        "launch_agent",
        "llm_query",
        "Input",
        "NotYourAgent",
        "asyncio",
        "ENV",
        "AGENTS",
        AGENTS_BINDING,
        AGENT_WAIT_TOOL,
        AGENT_SEND_TOOL,
    }
)


def as_tool_items(tools: Any) -> list[Any]:
    """Normalize ``Flow(tools=...)`` to a flat list; one level of nesting is flattened."""
    if tools is None:
        return []
    if callable(tools) or isinstance(tools, Toolkit):
        return [tools]
    if isinstance(tools, Iterable) and not isinstance(tools, (str, bytes)):
        items: list[Any] = []
        for item in tools:
            if (
                not callable(item)
                and isinstance(item, Iterable)
                and not isinstance(item, (str, bytes))
            ):
                items.extend(item)
            else:
                items.append(item)
        return items
    raise TypeError(f"tools must be a tool or a sequence of tools, not {type(tools).__name__}")


# -- built-in tools -----------------------------------------------------------

LAUNCH_DESCRIPTION = """\
Start an agent and return it immediately as an `Agent`. `config=` names one of the \
configs listed below (default: `default`). JSON keyword arguments become the new agent's \
REPL variables; wrap one as `Input(value, description="...")` to explain it. \
`await agent` returns its result.

Example — launch independent work together, then collect it:
```repl
first = launch_agent("Produce deliverable A.")
second = launch_agent("Produce deliverable B.")
print(await asyncio.gather(first, second))
```"""


def _on_flow_loop(flow: Flow, fn: Callable[[], Any]) -> Any:
    """Run a graph write on the Flow's event loop thread."""
    owner = flow._loop
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
    if isinstance(value, dict):
        return RECURSE
    if isinstance(value, list):
        return RECURSE
    raise TypeError(f"argument {path!r} must be JSON-compatible")


def check_args(args: dict[str, Any]) -> dict[str, Any]:
    """Launch arguments: named like Python variables, and strict JSON (``Input`` allowed)."""
    for name, value in args.items():
        validate_argument_name(name)
        wrapped = input_parts(value)
        visit_argument(wrapped[0] if wrapped is not None else value, _check_json, path=name)
    return args


def _handle_for(g: Any, entry: dict[str, Any], *_args: Any, **_kwargs: Any) -> AgentHandle:
    """The handle a saved launch or send returns again."""
    agent_id = entry.get("agent", "")
    if agent_id in g:
        return agent_handle(g, g.agent(agent_id))
    return AgentHandle(agent_id=agent_id, name="(removed)", path="(removed)")


def _agent_saved(handle: AgentHandle, *_args: Any, **_kwargs: Any) -> dict[str, str]:
    return {"agent": handle.agent_id}


def launch_tool(flow: Flow, step: Step):
    g = step.graph
    lock = threading.Lock()

    configs = ", ".join(
        f"`{name}` (model `{config.model}`"
        + (f"; {config.description}" if config.description else "")
        + ")"
        for name, config in flow.configs.items()
    )
    description = LAUNCH_DESCRIPTION + (f"\nConfigs: {configs}" if configs else "")

    @tool(description, is_async=False)
    def launch_agent(
        task: str, *, config: str | None = None, name: str | None = None, **args: Any
    ) -> AgentHandle:
        def launch() -> AgentHandle:
            with lock:
                child = agent_tools.spawn(
                    flow, step, task, config=config, name=name, **check_args(args)
                )
                return agent_handle(g, child)

        return _on_flow_loop(flow, launch)

    return _saves(launch_agent, receipt=_agent_saved, restore=functools.partial(_handle_for, g))


def await_tool(flow: Flow, step: Step):
    """``await agent``: wait for an agent this one launched, and save which one it was."""
    g = step.graph

    def returned(done: Node) -> Any:
        """The result of ``done``, with the ``return`` edge to the cell that got it."""
        cell = current_code()
        if cell is not None and cell.id in g:
            g.link_return(done, cell)
        return done.data.get("result")

    async def result_of(agent_id: str, *, check: bool) -> Any:
        target = resolve_agent(g, agent_id)
        if check and step.agent not in g.lineage(target)[:-1]:
            raise NotYourAgent(
                f"you can only await agents you launched; {target.name!r} is not one. "
                f"Message it with AGENTS[{target.name!r}].send(...) and end your turn "
                "to read its reply."
            )
        if g.status(target) != "done":
            if flow.queue is None:
                raise RuntimeError("awaiting an agent requires a running Flow")
            await flow.queue.join(lambda: target.id not in g or g.status(target) == "done")
            if target.id not in g:
                raise RuntimeError(f"agent {target.name!r} was removed")
        return returned(g.history(target)[-1])

    def restore(entry: dict[str, Any], agent_id: str) -> Any:
        if "value" in entry:
            return entry["value"]
        done = g._nodes.get(entry.get("done", ""))
        if done is not None and done.type == DONE:
            return returned(done)
        return result_of(entry.get("agent") or agent_id, check=False)

    def receipt(_result: Any, agent_id: str) -> dict[str, str]:
        target = resolve_agent(g, agent_id)
        return {"agent": target.id, "done": g.history(target)[-1].id}

    @tool("Wait for an agent you launched and return its result.")
    async def wait_agent(agent_id: str) -> Any:
        return await result_of(agent_id, check=True)

    return _saves(wait_agent, receipt=receipt, restore=restore)


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
    reply, usage = await flow.complete([{"role": "user", "content": text}], model, **sampling)
    value = reply if normalized is None else schemas.parse(reply, normalized)
    code = current_code()
    if code is not None:
        record_metadata(
            "llm_query.calls",
            {
                "model": model,
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                "structured": normalized is not None,
                "usage": usage.as_dict(),
            },
            append=True,
            node=code,
        )
    return value


def llm_query_tool(flow: Flow, step: Step):
    default_model = step.agent.config.model
    description = (
        "Make one model completion for extraction, summarization, or Q&A over the "
        "text you pass. The call sees no REPL, history, or tools, so use it for a "
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

    return llm_query


def submit_tool(flow: Flow, step: Step):
    @tool(
        "Submit your final result as a string or strict JSON value. A valid result completes you.",
    )
    def submit(answer: object) -> NoReturn:
        raise DoneSignal(agent_tools.done(flow, step, answer))

    return submit


def send_tool(flow: Flow, step: Step):
    g = step.graph

    @tool(
        "Queue a message for another agent and return that agent immediately. The "
        "message reaches it at its next turn and reopens it if it finished.",
        is_async=False,
    )
    def send_agent(target: Any, message: str) -> AgentHandle:
        def send() -> AgentHandle:
            agent = resolve_agent(g, target)
            if agent.id == step.agent.id:
                raise ValueError("an agent cannot message itself")
            agent_tools.send(flow, step, agent, message)
            return agent_handle(g, agent)

        return _on_flow_loop(flow, send)

    return _saves(send_agent, receipt=_agent_saved, restore=functools.partial(_handle_for, g))


def repl_tools(flow: Flow, step: Step) -> dict[str, Any]:
    """The model-visible tools for one agent at one step, built-ins first."""
    tools: dict[str, Any] = {"submit": submit_tool(flow, step)}
    if can_spawn(step.graph, step.agent):
        tools["launch_agent"] = launch_tool(flow, step)
    tools["llm_query"] = llm_query_tool(flow, step)
    tools.update({name: fn for name, fn in flow.tools.items() if tool_allowed(fn, flow, step.node)})
    return tools


def build_namespace(flow: Flow, step: Step) -> dict[str, Any]:
    """Everything a cell's namespace binds: the tools plus ``AGENTS`` and ``Input``."""
    agents = build_agent_directory(step.graph, step.agent)
    return {
        **repl_tools(flow, step),
        **flow.namespace_for(step),
        "Input": Input,
        "NotYourAgent": NotYourAgent,
        "asyncio": asyncio,
        "AGENTS": agents,
        AGENTS_BINDING: agents,
        AGENT_WAIT_TOOL: await_tool(flow, step),
        AGENT_SEND_TOOL: send_tool(flow, step),
    }


__all__ = [
    "RESERVED_TOOLS",
    "Toolkit",
    "Where",
    "as_tool",
    "as_tool_items",
    "build_namespace",
    "check_args",
    "format_tool_docs",
    "launch_tool",
    "query",
    "repl_tools",
    "tool",
    "tool_allowed",
    "tool_from_schema",
    "tool_name",
]
