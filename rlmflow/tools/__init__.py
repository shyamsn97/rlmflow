"""REPL tools: the ``@tool`` decorator, its prompt docs, and the namespace each cell runs in.

The namespace is the same for every harness, since every cell runs in our REPL:
``submit``, ``launch_agent``, ``llm_query``, ``send_agent``, ``ask_agent``,
``AGENTS``, ``Input``, plus the host's own tools.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import threading
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NoReturn

from rlmflow.agent_tools import tools as agent_tools
from rlmflow.agent_tools.view import AgentHandle, agent_handle, build_agent_directory, resolve_agent
from rlmflow.engine.handoff import AWAITED_KEY
from rlmflow.graph.config import (
    RECURSE,
    Input,
    can_spawn,
    input_parts,
    validate_argument_name,
    visit_argument,
)
from rlmflow.graph.graph import STRUCTURAL, Edge
from rlmflow.graph.nodes import AGENT_REF_KEY, Node, Step, agent_ref_id, current_code
from rlmflow.runtime.repl import (
    AGENT_WAIT_TOOL,
    AGENTS_BINDING,
    DoneSignal,
    ToolMetadata,
    ToolRestore,
    get_tool_metadata,
)

if TYPE_CHECKING:
    from rlmflow.flow import Flow


def tool(
    description: str,
    *,
    name: str | None = None,
    proxy: bool = False,
    is_async: bool | None = None,
    when: Callable[..., bool] | None = None,
    restore: ToolRestore = None,
) -> Callable:
    """Attach tool metadata to ``fn``.

    ``is_async`` records whether agent code must ``await`` the tool; it defaults
    to ``async def`` detection. ``when`` is ``(flow, node) -> bool``: when false,
    the tool is left out of the prompt and the REPL. ``restore="result"`` saves
    the result of a successful call on the cell's ``code`` node and returns it
    when the REPL is rebuilt; a callable rebuilds a richer value from it.
    """
    if restore is not None and restore != "result" and not callable(restore):
        raise TypeError("tool restore must be None, 'result', or a callable")
    if restore is not None and not proxy:
        raise ValueError("tool restore requires proxy=True")

    def decorator(fn):
        resolved = inspect.iscoroutinefunction(fn) if is_async is None else is_async
        fn._tool_meta = ToolMetadata(
            name=name or fn.__name__.removeprefix("tool_"),
            description=description.strip(),
            proxy=proxy,
            is_async=resolved,
            when=when,
            restore=restore,
        )
        return fn

    return decorator


def tool_allowed(fn: Any, flow: Any = None, node: Any = None) -> bool:
    meta = get_tool_metadata(fn)
    return meta is None or meta.when is None or bool(meta.when(flow, node))


def format_tool_line(fn: Callable) -> str:
    meta = get_tool_metadata(fn)
    if meta is None:
        return ""
    try:
        sig = str(inspect.signature(fn, eval_str=True))
    except (TypeError, ValueError):
        sig = "(...)"
    sig = sig.replace("rlmflow.agent_tools.view.", "")
    prefix = "await " if meta.is_async else ""
    return f"- `{prefix}{meta.name}{sig}`: {meta.description}"


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
        "send_agent",
        "ask_agent",
        "connect_agent",
        "disconnect_agent",
        "llm_query",
        "Input",
        "asyncio",
        "ENV",
        "AGENTS",
        AGENTS_BINDING,
        AGENT_WAIT_TOOL,
    }
)


def as_tool_items(tools: Any) -> list[Any]:
    """Normalize ``Flow(tools=...)`` to a flat list; one level of nesting is flattened."""
    if tools is None:
        return []
    if callable(tools):
        return [tools]
    if isinstance(tools, Iterable) and not isinstance(tools, (str, bytes)):
        items: list[Any] = []
        for item in tools:
            if not callable(item) and isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                items.extend(item)
            else:
                items.append(item)
        return items
    raise TypeError(f"tools must be a tool or a sequence of tools, not {type(tools).__name__}")


# -- built-in tools -----------------------------------------------------------

LAUNCH_DESCRIPTION = """\
Start an agent and return its handle immediately. `config=` names one of the \
configs listed below (default: `default`). JSON keyword arguments, or other agents' \
handles, become the new agent's REPL variables; wrap one as `Input(value, \
description="...")` to explain it. `await handle` returns its result.

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


def _producer(g: Any, value: Any) -> Node | None:
    agent_id = value.agent_id if isinstance(value, AgentHandle) else agent_ref_id(value)
    if agent_id is None:
        return None
    if agent_id not in g or g.node(agent_id).type != "agent_start":
        raise ValueError(f"handle refers to unknown agent {agent_id!r}")
    return g.node(agent_id)


def _check_json(path: str, value: Any) -> Any:
    if isinstance(value, AgentHandle) or agent_ref_id(value) is not None:
        raise TypeError(f"argument {path!r} nests a handle; pass handles as top-level arguments")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        json.dumps(value, allow_nan=False)
        return value
    if isinstance(value, dict):
        if AGENT_REF_KEY in value:
            raise TypeError(f"argument {path!r} must contain an exact agent reference")
        return RECURSE
    if isinstance(value, list):
        return RECURSE
    raise TypeError(f"argument {path!r} must be JSON-compatible or an agent handle")


def split_args(g: Any, args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """JSON arguments, and top-level handles that become ``depends_on`` inputs."""
    plain: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    for name, value in args.items():
        validate_argument_name(name)
        wrapped = input_parts(value)
        raw, description = wrapped if wrapped is not None else (value, None)
        producer = _producer(g, raw)
        if producer is not None:
            inputs[name] = Input(producer, description=description) if description else producer
            continue
        visit_argument(raw, _check_json, path=name)
        plain[name] = value
    return plain, inputs


def launch_tool(flow: Flow, step: Step):
    g = step.graph
    lock = threading.Lock()

    def restore_launch(saved: object, task: str = "", **_ignored: Any) -> AgentHandle:
        agent_id = agent_ref_id(saved)
        if agent_id not in g:
            raise ValueError(f"recorded launch points to missing agent {agent_id!r}")
        return agent_handle(g, g.node(agent_id))

    configs = ", ".join(
        f"`{name}` (model `{config.model}`" + (f"; {config.description}" if config.description else "") + ")"
        for name, config in flow.configs.items()
    )
    description = LAUNCH_DESCRIPTION + (f"\nConfigs: {configs}" if configs else "")

    @tool(description, proxy=True, is_async=False, restore=restore_launch)
    def launch_agent(
        task: str, *, config: str | None = None, name: str | None = None, **args: Any
    ) -> AgentHandle:
        def launch() -> AgentHandle:
            with lock:
                plain, inputs = split_args(g, args)
                child = agent_tools.spawn(
                    flow, step, task, config=config, name=name, inputs=inputs or None, **plain
                )
                return agent_handle(g, child)

        return _on_flow_loop(flow, launch)

    return launch_agent


def _model_help(flow: Flow, default_model: str) -> str:
    names = ", ".join(f"`{key}`" for key in sorted(flow.llms))
    return f" `model=` takes a model name (configured: {names}). Omitting `model` uses `{default_model}`."


async def query(
    flow: Flow,
    prompt: str,
    *,
    model: str = "default",
    schema: Any = None,
    **sampling: Any,
) -> Any:
    """One context-free model call; with ``schema``, a validated JSON value."""
    from rlmflow import schema as schemas

    if not isinstance(prompt, str):
        raise TypeError("llm_query(prompt) takes a str")
    normalized = schemas.json_schema_for(schema) if schema is not None else None
    text = prompt if normalized is None else f"{prompt}\n\n{schemas.contract(normalized, source='query')}"
    reply, usage = await flow.complete([{"role": "user", "content": text}], model, **sampling)
    value = reply if normalized is None else schemas.parse(reply, normalized)
    code = current_code()
    if code is not None:
        code.annotate(
            "llm_query.calls",
            {
                "model": model,
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                "structured": normalized is not None,
                "usage": usage.as_dict(),
            },
            append=True,
        )
    return value


def llm_query_tool(flow: Flow, step: Step):
    default_model = step.agent.config.model
    description = (
        "Make one model completion for extraction, summarization, or Q&A over the "
        "text you pass. The call sees no REPL, history, or tools, so use it for a "
        "one-shot transform, not multi-step owned work." + _model_help(flow, default_model)
    )

    @tool(description, proxy=True, restore="result")
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
        "Submit your final result as a string or strict JSON value. A valid "
        "result completes you.",
        proxy=True,
    )
    def submit(answer: object) -> NoReturn:
        raise DoneSignal(agent_tools.done(flow, step, answer))

    return submit


def send_tool(flow: Flow, step: Step):
    @tool(
        "Queue a message for another agent without waiting. It reaches that agent "
        "at its next turn and reopens a finished agent. Returns the message id.",
        proxy=True,
        is_async=False,
        restore="result",
    )
    def send_agent(target: Any, message: str) -> str:
        return _on_flow_loop(flow, lambda: agent_tools.send(flow, step, target, message).id)

    return send_agent


def ask_tool(flow: Flow, step: Step):
    g = step.graph

    @tool(
        "Send a message to another agent and wait for the result of its next finished turn.",
        proxy=True,
        restore="result",
    )
    async def ask_agent(target: Any, message: str) -> Any:
        agent = resolve_agent(g, target)
        if agent.id == step.agent.id:
            raise ValueError("an agent cannot ask itself")
        if flow.queue is None:
            raise RuntimeError("ask_agent requires a running Flow")
        with waiting(flow, g, step.agent, agent):
            sent = agent_tools.send(flow, step, agent, message)

            def answered() -> bool:
                delivered = g.edges(kind="next", target=sent.id)
                frontier = g.frontier(agent)
                return bool(delivered) and frontier.type == "done" and frontier.seq > delivered[0].seq

            await flow.queue.join(answered)
        return g.result(agent)

    return ask_agent


def connect_tool(flow: Flow, step: Step):
    g = step.graph

    @tool(
        "Add an edge `kind` from you to another agent, with JSON `**data`; "
        "`reports_to` sends your results to it. Returns the edge id.",
        proxy=True,
        is_async=False,
        restore="result",
    )
    def connect_agent(target: Any, kind: str, **data: Any) -> str:
        if not isinstance(kind, str) or not kind or kind in STRUCTURAL:
            raise ValueError(f"edge kind must be a non-empty application kind, not {kind!r}")
        edge = Edge(kind=kind, source=step.agent.id, target=resolve_agent(g, target).id, data=data)
        return _on_flow_loop(flow, lambda: g.add_edge(edge).id)

    return connect_agent


def disconnect_tool(flow: Flow, step: Step):
    g = step.graph

    @tool("Remove an edge you added by id.", proxy=True, is_async=False, restore="result")
    def disconnect_agent(edge_id: str) -> None:
        _on_flow_loop(flow, lambda: g.remove_edge(edge_id))

    return disconnect_agent


@contextmanager
def waiting(flow: Flow, g: Any, waiter: Node, target: Node):
    """Register ``waiter -> target`` for the length of a wait, refusing cycles."""
    seen: set[str] = set()
    stack = [target.id]
    while stack:
        current = stack.pop()
        if current == waiter.id:
            raise RuntimeError(f"waiting on {target.name!r} would deadlock: it waits on this agent")
        if current in seen:
            continue
        seen.add(current)
        stack.extend(flow._waits.get(current, ()))
        stack.extend(producer.id for producer in g.producers(g.node(current)))
    targets = flow._waits.setdefault(waiter.id, [])
    targets.append(target.id)
    try:
        yield
    finally:
        targets.remove(target.id)
        if not targets:
            flow._waits.pop(waiter.id, None)


def wait_agent_tool(flow: Flow, step: Step):
    g = step.graph

    @tool("Wait for an existing agent and return its result.", proxy=True)
    async def wait_agent(agent_id: str) -> Any:
        target = resolve_agent(g, agent_id)
        if not g.terminal(target):
            if flow.queue is None:
                raise RuntimeError("waiting for an agent requires a running Flow")
            with waiting(flow, g, step.agent, target):
                await flow.queue.join(lambda: g.terminal(target))
        code = current_code()
        if code is not None and target.id not in code.data.setdefault(AWAITED_KEY, []):
            code.data[AWAITED_KEY].append(target.id)
        return g.result(target)

    return wait_agent


def repl_tools(flow: Flow, step: Step) -> dict[str, Any]:
    """The model-visible tools for one agent at one step, built-ins first."""
    tools: dict[str, Any] = {"submit": submit_tool(flow, step)}
    if can_spawn(step.graph, step.agent):
        tools["launch_agent"] = launch_tool(flow, step)
    tools["llm_query"] = llm_query_tool(flow, step)
    tools["send_agent"] = send_tool(flow, step)
    tools["ask_agent"] = ask_tool(flow, step)
    tools["connect_agent"] = connect_tool(flow, step)
    tools["disconnect_agent"] = disconnect_tool(flow, step)
    tools.update(
        {name: fn for name, fn in flow.tools.items() if tool_allowed(fn, flow, step.node)}
    )
    return tools


def build_namespace(flow: Flow, step: Step) -> dict[str, Any]:
    """Everything a cell's namespace binds: the tools plus ``AGENTS`` and ``Input``."""
    agents = build_agent_directory(step.graph, step.agent)
    return {
        **repl_tools(flow, step),
        **flow.namespace_for(step),
        "Input": Input,
        "asyncio": asyncio,
        "AGENTS": agents,
        AGENTS_BINDING: agents,
        AGENT_WAIT_TOOL: wait_agent_tool(flow, step),
    }


__all__ = [
    "RESERVED_TOOLS",
    "as_tool_items",
    "build_namespace",
    "format_tool_docs",
    "launch_tool",
    "query",
    "repl_tools",
    "split_args",
    "tool",
    "tool_allowed",
    "tool_name",
]
