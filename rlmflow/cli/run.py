"""``rlmflow run`` and ``rlmflow tui``: Fire classes that build a Flow and drive it.

Constructor flags pick the model, runtime, and tools. Methods take the query —
``rlmflow run tui "fix the tests"`` opens the dashboard, ``rlmflow run print
"fix the tests"`` streams the tree. ``rlmflow tui`` is the dashboard with no
``print`` verb, because that is the only thing it does.
"""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from typing import Any

from rlmflow.cli.options import CliError, RunOptions, resolve

# Anthropic for ``claude*``, OpenAI otherwise — the same rule ``client_for`` uses
# to pick a client, applied one step earlier so a missing key is one line of
# stderr instead of a stack trace from inside the SDK.
API_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


class RunCLI:
    """Run a coding agent over --workdir, checkpointing to <workdir>/graph.

    Flags pick the model and runtime; the verb picks the surface.

        rlmflow run tui "add a test for parse_args"
        rlmflow run print "fix the failing test"
        rlmflow run --model gpt-5 --workdir ./proj tui
    """

    def __init__(
        self,
        query: str | None = None,
        model: str | None = None,
        fast_model: str | None = None,
        reasoning_effort: str | None = None,
        workdir: str | None = None,
        docker_image: str | None = None,
        max_depth: int | None = None,
        max_iters: int | None = None,
        workers: int | None = None,
        tools: str | None = None,
        resume: str | None = None,
        agent: str | None = None,
    ) -> None:
        self.query = None if query is None else str(query)
        self.resume = resume
        self.agent = agent
        self.options, _ = resolve(
            {
                "model": model,
                "fast_model": fast_model,
                "reasoning_effort": reasoning_effort,
                "workdir": workdir,
                "docker_image": docker_image,
                "max_depth": max_depth,
                "max_iters": max_iters,
                "workers": workers,
                "tools": tools,
            }
        )

    def tui(self, query: str | None = None) -> None:
        """Open the Textual dashboard. Needs ``pip install "rlmflow[tui]"``.

        Args:
          query: What to do. Omit it to open the dashboard with nothing running.
        """
        run_agent(
            self.query if query is None else str(query),
            self.options,
            headless=False,
            resume=self.resume,
            agent=self.agent,
        )

    def print(self, query: str | None = None) -> None:
        """Skip the dashboard: stream the tree and print the final answer.

        Args:
          query: What to do. Required unless ``--resume`` already has a run.
        """
        run_agent(
            self.query if query is None else str(query),
            self.options,
            headless=True,
            resume=self.resume,
            agent=self.agent,
        )

    def __str__(self) -> str:
        """``rlmflow run`` with no verb opens the dashboard."""
        self.tui()
        return ""


class TuiCLI:
    """Open the coding agent in the Textual dashboard.

    Same flags as ``rlmflow run``, always interactive. Needs
    ``pip install "rlmflow[tui]"``.

        rlmflow tui
        rlmflow tui --query "add a test for parse_args"
        rlmflow tui --workdir ./proj --model gpt-5
    """

    def __init__(
        self,
        query: str | None = None,
        model: str | None = None,
        fast_model: str | None = None,
        reasoning_effort: str | None = None,
        workdir: str | None = None,
        docker_image: str | None = None,
        max_depth: int | None = None,
        max_iters: int | None = None,
        workers: int | None = None,
        tools: str | None = None,
        resume: str | None = None,
        agent: str | None = None,
    ) -> None:
        self.query = None if query is None else str(query)
        self.resume = resume
        self.agent = agent
        self.options, _ = resolve(
            {
                "model": model,
                "fast_model": fast_model,
                "reasoning_effort": reasoning_effort,
                "workdir": workdir,
                "docker_image": docker_image,
                "max_depth": max_depth,
                "max_iters": max_iters,
                "workers": workers,
                "tools": tools,
            }
        )

    def __str__(self) -> str:
        run_agent(
            self.query,
            self.options,
            headless=False,
            resume=self.resume,
            agent=self.agent,
        )
        return ""


def run_agent(
    query: str | None,
    options: RunOptions,
    *,
    headless: bool = False,
    resume: str | Path | None = None,
    agent: str | None = None,
    flow: Any = None,
) -> None:
    """Run the coding agent. Headless streams one query; otherwise the TUI opens."""
    workdir = Path(options.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    graph_dir = workdir / "graph"

    own_flow = flow is None and agent is None
    if flow is None:
        flow = build_flow(options, workdir=workdir, agent=agent)
    root = load_root(resume) if resume is not None else None

    # Naming a model we did not choose would be a lie when --agent built the flow.
    model = options.model if own_flow else (agent or "custom flow")
    print(f"rlmflow · {model} · {options.docker_image or 'local'} · {workdir}")
    if headless:
        _run_headless(flow, query, root, graph_dir)
    else:
        _run_tui(flow, query, root, graph_dir)
    print(f"run saved to {graph_dir}")


def build_flow(options: RunOptions, *, workdir: Path, agent: str | None = None) -> Any:
    """The coding agent by default; ``module:factory`` when the caller has their own."""
    from rlmflow import FILE_TOOLS, AgentConfig, DockerRuntime, Flow, LocalRuntime
    from rlmflow.llm import client_for

    if agent is not None:
        return load_factory(agent)

    require_api_key(options.model)
    runtime = (
        DockerRuntime(options.docker_image, working_directory=workdir)
        if options.docker_image
        else LocalRuntime(working_directory=workdir)
    )
    effort = options.reasoning_effort
    return Flow(
        client_for(options.model, reasoning_effort=effort),
        llm_clients={"fast": client_for(options.fast_model, reasoning_effort=effort)},
        runtime=runtime,
        tools=[FILE_TOOLS] if options.tools == "files" else None,
        workers=options.workers,
        root_config=AgentConfig(max_depth=options.max_depth, max_iters=options.max_iters),
    )


def load_factory(target: str) -> Any:
    """Import ``module:factory`` and call it; it must hand back a Flow."""
    from rlmflow import Flow

    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise CliError(f"--agent wants 'module:factory', not {target!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise CliError(f"cannot import {module_name!r}: {exc}") from exc
    factory = getattr(module, attr, None)
    if factory is None:
        raise CliError(f"{module_name!r} has no {attr!r}")
    if not callable(factory):
        raise CliError(f"{target} is not callable, so it cannot return a Flow")
    flow = factory()
    if not isinstance(flow, Flow):
        raise CliError(f"{target} returned {type(flow).__name__}, not a Flow")
    return flow


def load_root(source: str | Path) -> Any:
    from rlmflow import AgentStart, persistence

    path = Path(source)
    if not path.exists():
        raise CliError(f"path not found: {path}")
    try:
        root = persistence.load(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CliError(f"cannot read a run from {path}: {exc}") from exc
    if not isinstance(root, AgentStart):
        raise CliError(f"{path} holds a {type(root).__name__}, not a run to resume")
    return root


def require_api_key(model: str, environ: dict[str, str] | None = None) -> None:
    environ = os.environ if environ is None else environ
    provider = "anthropic" if model.startswith("claude") else "openai"
    key = API_KEYS[provider]
    if not environ.get(key):
        raise CliError(f"{key} is not set, and {model} needs it")


def _run_headless(flow: Any, query: str | None, root: Any, graph_dir: Path) -> None:
    """No dashboard: stream one query, printing the tree as it grows."""
    from rlmflow.consumers import ConsumerGroup, GraphCheckpointer, LiveTreeRenderer

    if root is None and not query:
        raise CliError("nothing to run: pass a query, or --resume a saved run")
    consumers = ConsumerGroup([LiveTreeRenderer(), GraphCheckpointer(graph_dir)])
    consumers.init(flow)
    if root is None:
        root = flow.start(query or "")
    elif query:
        _ask(root, query)

    async def stream() -> None:
        async for node in flow.run_streaming(root):
            consumers.handle(node)

    try:
        asyncio.run(stream())
    finally:
        consumers.close()
        flow.runtime.close_repls()
    print(f"\n{root.result() or '(no result)'}\n")


def _run_tui(flow: Any, query: str | None, root: Any, graph_dir: Path) -> None:
    """The dashboard drives its own turns; it only needs the flow and a place to save.

    A query here is the opening turn, not a different mode: the panels are up
    while it runs, and the prompt is there when it finishes.
    """
    from rlmflow.consumers import GraphCheckpointer
    from rlmflow.consumers.tui import FlowTUI

    ui = FlowTUI(root, sink=GraphCheckpointer(graph_dir))
    ui.init(flow)
    try:
        ui.run(query=query)
    finally:
        flow.runtime.close_repls()


def _ask(root: Any, query: str) -> None:
    """Queue another turn on a run that already exists."""
    from rlmflow.graph.nodes import UserQuery

    root.frontier.append(UserQuery(content=query))


__all__ = [
    "RunCLI",
    "TuiCLI",
    "build_flow",
    "load_factory",
    "load_root",
    "require_api_key",
    "run_agent",
]
