"""``rlmflow fork RUN NODE``: copy a saved run cut after NODE, then run the copy or drive it yourself.

The copy is saved to its own directory (``--out``, by default beside RUN); RUN
is left as it is. Headless, the copy runs to the end with its runs' agents
(``--agent``, an agent name on the project's Flow, for NODE's run) and is
saved. ``--interactive`` starts ``rlmflow serve`` on the copy and prints the
``claude`` command that continues it: ``--resume <id> --fork-session`` when
NODE ended a Claude session step, otherwise a fresh session, which serve
sends the whole run.
"""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from typing import Any

from rlmflow.cli.app import CliError, resolve_options

CONTINUE = "Continue the task."


def find_node(graph: Any, selector: str) -> Any:
    """A node by id or unique id prefix."""
    matches = [node for node in graph.nodes() if node.id == selector] or [
        node for node in graph.nodes() if node.id.startswith(selector)
    ]
    if len(matches) != 1:
        raise CliError(f"{'no' if not matches else 'more than one'} node matches {selector!r}")
    return matches[0]


def resume_command(graph: Any, run: Any, plugin_dir: Path) -> str:
    """The ``claude`` invocation that continues ``run`` in your own session."""
    from rlmflow.agents.agent import last_session

    last = last_session(run.nodes()[-1])
    argv = ["claude", "--plugin-dir", str(plugin_dir)]
    if last is not None:
        argv += ["--resume", last.data["session"], "--fork-session"]
    return shlex.join([*argv, CONTINUE])


def fork_run(
    run: str,
    node: str,
    *,
    agent: str | None = None,
    interactive: bool = False,
    project: str = ".",
    out: str | None = None,
) -> None:
    from rlmflow import Graph
    from rlmflow.cli.serve import PERSON

    run_dir = Path(run)
    if not run_dir.exists():
        raise CliError(f"path not found: {run_dir}")
    graph = Graph.load(run_dir)
    at = find_node(graph, node)
    owner = at.root
    if interactive:
        agent = PERSON
    elif agent is None and owner.agent == PERSON:
        raise CliError("a person ran this run; pass --agent, or --interactive")
    branch = graph.fork(at)
    run_copy = branch.agent_root(owner.id)
    if agent is not None:
        run_copy.agent = agent
    tag = branch.id.removeprefix("graph_")[:8]
    out_dir = Path(out) if out is not None else run_dir.with_name(f"{run_dir.name}-fork-{tag}")
    branch.save(out_dir)
    print(f"forked {owner.name} after {at.id} into {out_dir}")

    if interactive:
        from rlmflow.agent_tools.cli import ToolError
        from rlmflow.agent_tools.hooks import PLUGIN_DIR, start_serve

        project_dir = Path(project).expanduser().resolve()
        try:
            start_serve(project_dir, ["--graph", str(out_dir.resolve()), "--root", run_copy.id])
        except ToolError as exc:
            raise CliError(str(exc)) from exc
        print(f"serving {out_dir} for {project_dir}; continue it with:\n")
        print(
            f"  cd {shlex.quote(str(project_dir))} && {resume_command(branch, run_copy, PLUGIN_DIR)}"
        )
        return

    _run_headless(branch, run_copy, out_dir, project)


def _run_headless(graph: Any, run: Any, out_dir: Path, project: str) -> None:
    from rlmflow import Checkpoint
    from rlmflow.cli.run import build_flow

    project_dir = Path(project).expanduser().resolve()
    options, _ = resolve_options({"workdir": str(project_dir)}, cwd=project_dir)
    flow = build_flow(options, workdir=project_dir)

    async def drive() -> None:
        try:
            await flow.arun(graph, hooks=[Checkpoint(out_dir)])
        finally:
            await flow.aclose()

    try:
        asyncio.run(drive())
    except KeyError as exc:
        raise CliError(f"{exc.args[0]}; set `flow` in rlmflow.toml to a Flow that has it") from exc
    graph.save(out_dir)
    print(f"\n{run.final_result() or '(no result)'}\n")
    print(f"run saved to {out_dir}")


__all__ = ["CONTINUE", "find_node", "fork_run", "resume_command"]
