"""``rlmflow fork RUN NODE``: branch a saved run, then run the branch or drive it yourself.

Headless, the branch runs to the end with its own config (or ``--config``, a
config name on the project's Flow) and the run is saved. ``--interactive`` starts ``rlmflow serve`` on the branch
and prints the ``claude`` command that continues it: ``--resume <id>
--fork-session`` when NODE ended a Claude session turn, otherwise a fresh
session, which serve briefs on the REPL's cells and variables.
"""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from typing import Any

from rlmflow.cli import CliError, resolve_options

CONTINUE = "Continue the task."


def find_node(graph: Any, selector: str) -> Any:
    """A node by id or unique id prefix."""
    matches = [node for node in graph.nodes() if node.id == selector] or [
        node for node in graph.nodes() if node.id.startswith(selector)
    ]
    if len(matches) != 1:
        raise CliError(f"{'no' if not matches else 'more than one'} node matches {selector!r}")
    return matches[0]


def resume_command(graph: Any, fork: Any, plugin_dir: Path) -> str:
    """The ``claude`` invocation that continues ``fork`` in your own session."""
    from rlmflow.graph.nodes import Step
    from rlmflow.harnesses.sessions import Session

    session = Session.for_step(Step(graph, fork, graph.frontier(fork)))
    argv = ["claude", "--plugin-dir", str(plugin_dir)]
    if session.resume:
        argv += ["--resume", session.id, "--fork-session"]
    return shlex.join([*argv, CONTINUE])


def fork_run(
    run: str,
    node: str,
    *,
    config: str | None = None,
    interactive: bool = False,
    name: str | None = None,
    project: str = ".",
    out: str | None = None,
) -> None:
    from rlmflow import Graph
    from rlmflow.cli.serve import INTERACTIVE_CONFIG, ran_interactively

    run_dir = Path(run)
    if not run_dir.exists():
        raise CliError(f"path not found: {run_dir}")
    graph = Graph.load(run_dir)
    at = find_node(graph, node)
    owner = graph.node(at.agent_id)
    if interactive:
        config = INTERACTIVE_CONFIG
    elif config is None and ran_interactively(owner):
        raise CliError("this run's agent was interactive; pass --config, or --interactive")
    fork = graph.fork(at, name=name, config=config)
    out_dir = Path(out) if out is not None else run_dir
    graph.save(out_dir)
    print(f"forked {owner.name} at {at.type} {at.id} as {fork.name} ({fork.id})")

    if interactive:
        from rlmflow.agent_tools.cli import ToolError
        from rlmflow.agent_tools.hooks import PLUGIN_DIR, start_serve

        project_dir = Path(project).expanduser().resolve()
        try:
            start_serve(project_dir, ["--graph", str(out_dir.resolve()), "--root", fork.id])
        except ToolError as exc:
            raise CliError(str(exc)) from exc
        print(f"serving {out_dir} for {project_dir}; continue it with:\n")
        print(f"  cd {shlex.quote(str(project_dir))} && {resume_command(graph, fork, PLUGIN_DIR)}")
        return

    _run_headless(graph, fork, out_dir, project)


def _run_headless(graph: Any, fork: Any, out_dir: Path, project: str) -> None:
    from rlmflow.cli.run import build_flow
    from rlmflow.consumers import GraphCheckpointer

    project_dir = Path(project).expanduser().resolve()
    options, _ = resolve_options({"workdir": str(project_dir)}, cwd=project_dir)
    flow = build_flow(options, workdir=project_dir)
    checkpointer = GraphCheckpointer(out_dir)

    async def drive() -> None:
        try:
            async for step in flow.run_streaming(graph):
                checkpointer.handle(step, graph)
        finally:
            await flow.aclose()

    try:
        asyncio.run(drive())
    except KeyError as exc:
        raise CliError(f"{exc.args[0]}; set `flow` in rlmflow.toml to a Flow that has it") from exc
    graph.save(out_dir)
    print(f"\n{graph.result(fork) or '(no result)'}\n")
    print(f"run saved to {out_dir}")


__all__ = ["CONTINUE", "find_node", "fork_run", "resume_command"]
