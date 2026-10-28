"""``rlmflow`` command-line entry point.

Run an agent, or read one you already ran::

    rlmflow tui                                   the coding agent in the dashboard
    rlmflow tui --query "add a test"              same, first turn already going
    rlmflow run tui "add a test for parse_args"   same, query as a positional
    rlmflow run print "fix the failing test"      no dashboard, just the stream
    rlmflow run --workdir ./proj --model gpt-5 tui
    rlmflow view show runs/coding/graph           the tree, then the timeline
    rlmflow view show runs/coding/graph --step 3  just that step, with its content
    rlmflow render html runs/coding/graph r.html  steppable, single file
    rlmflow render gif runs/coding/graph run.gif  the run, animated
    rlmflow render browser runs/coding/graph      the Gradio viewer
    rlmflow config show                           resolved settings and their sources
    rlmflow version                               versions and installed extras

Inside an agent's session (the harness sets ``RLMFLOW_URL/_AGENT/_TOKEN``)::

    rlmflow repl 'print(len(haystack))'           a cell in the agent's REPL
    rlmflow spawn --config worker "search 000-099"
    rlmflow wait / send / done / agents

Your own Claude Code session as the root agent (see ``rlmflow plugin path``)::

    rlmflow serve --project .                     what `/rlmflow:run` starts for you
    rlmflow hook prompt|stop|end                  the plugin's hooks
    rlmflow fork RUN NODE --interactive           branch a run, continue it yourself

``python -m rlmflow …`` works the same.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

CONFIG_NAME = "rlmflow.toml"
USER_CONFIG = Path("~/.config/rlmflow/config.toml")
ENV_PREFIX = "RLMFLOW_"


class CliError(Exception):
    """A failure the user can act on. The entry point prints it and exits 1."""


@dataclass(frozen=True)
class RunOptions:
    """One resolved configuration for a ``run``."""

    model: str = "gpt-5"
    fast_model: str = "gpt-5-mini"
    reasoning_effort: str | None = None
    workdir: str = "."
    docker_image: str | None = None
    max_depth: int = 1
    max_iters: int = 30
    workers: int = 8
    tools: str = "files"
    #: A ``module:factory`` that returns your own Flow, used instead of the coding one.
    flow: str | None = None

    def __post_init__(self) -> None:
        if self.tools not in ("files", "none"):
            raise CliError(f"tools must be 'files' or 'none', not {self.tools!r}")


# Keyed by field name; the value is the default, which is also how ``_coerce``
# learns whether a string from a file or the environment needs to become an int.
DEFAULTS = {field.name: field.default for field in fields(RunOptions)}


def env_var(name: str) -> str:
    """``max_iters`` -> ``RLMFLOW_MAX_ITERS``."""
    return f"{ENV_PREFIX}{name.upper()}"


def resolve_options(
    overrides: dict[str, Any] | None = None,
    *,
    cwd: Path | None = None,
    home: Path | None = None,
    environ: dict[str, str] | None = None,
) -> tuple[RunOptions, dict[str, str]]:
    """Merge every source into one ``RunOptions``, plus a field -> source map.

    The map is what ``rlmflow config`` prints: with four places a model name can
    come from, "which one won" is the only question worth answering.
    """
    environ = os.environ if environ is None else environ
    cwd = Path.cwd() if cwd is None else cwd
    home = Path.home() if home is None else home

    values: dict[str, Any] = {}
    origins: dict[str, str] = dict.fromkeys(DEFAULTS, "default")

    layers = [
        ("user config", read_toml(user_config(home))),
        ("project config", read_toml(cwd / CONFIG_NAME)),
        ("env", _from_env(environ)),
        ("flag", {k: v for k, v in (overrides or {}).items() if v is not None}),
    ]
    for source, layer in layers:
        for name, value in layer.items():
            if name not in DEFAULTS:
                raise CliError(f"unknown {source} setting {name!r} (see `rlmflow config`)")
            values[name] = _coerce(name, value)
            origins[name] = source

    return RunOptions(**values), origins


def read_toml(path: Path) -> dict[str, Any]:
    """The ``[run]`` table of ``path``, or nothing if it isn't there."""
    import tomllib

    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise CliError(f"cannot read {path}: {exc}") from exc
    table = data.get("run", {})
    if not isinstance(table, dict):
        raise CliError(f"{path}: [run] must be a table")
    return table


def user_config(home: Path | None = None) -> Path:
    """The per-machine config file, or whatever ``RLMFLOW_CONFIG`` points at."""
    override = os.environ.get("RLMFLOW_CONFIG")
    if override:
        return Path(override).expanduser()
    home = Path.home() if home is None else home
    return Path(str(USER_CONFIG).replace("~", str(home), 1))


def _from_env(environ: dict[str, str]) -> dict[str, Any]:
    found = {}
    for name in DEFAULTS:
        value = environ.get(env_var(name))
        if value:
            found[name] = value
    return found


def _coerce(name: str, value: Any) -> Any:
    """Files and the environment hand over strings; the dataclass wants types."""
    if isinstance(DEFAULTS[name], int):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise CliError(f"{name} must be a whole number, not {value!r}") from exc
    return value if value is None else str(value)


STARTER = """# rlmflow defaults for this directory. Flags still win, then the
# RLMFLOW_* environment variables, then this file.
[run]
model = "{model}"
fast_model = "{fast_model}"
max_iters = {max_iters}
"""


class ConfigCLI:
    """The settings `rlmflow run` resolves, and the files they come from."""

    def show(self) -> None:
        """Print every setting, its value, and which source won it."""
        options, origins = resolve_options()
        width = max(len(name) for name in origins)
        for name, origin in origins.items():
            print(
                f"{name:<{width}}  {str(getattr(options, name)):<24}  {origin:<15}  {env_var(name)}"
            )

    def path(self) -> None:
        """Print the config files that are read, in the order they are read."""
        for label, path in (("user", user_config()), ("project", Path.cwd() / CONFIG_NAME)):
            print(f"{label:<8} {path}  {'found' if path.is_file() else 'absent'}")

    def init(self, out: str = CONFIG_NAME) -> None:
        """Write a starter config file holding the settings in effect right now.

        Args:
          out: Where to write it. Defaults to ./rlmflow.toml.
        """
        path = Path(out).expanduser()
        if path.exists():
            raise CliError(f"{path} already exists")
        options: RunOptions = resolve_options()[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            STARTER.format(
                model=options.model,
                fast_model=options.fast_model,
                max_iters=options.max_iters,
            )
        )
        print(f"wrote {path}")


class VersionCLI:
    """Print the rlmflow version, the Python running it, and the extras present."""

    def __str__(self) -> str:
        import importlib.metadata
        import importlib.util
        import platform

        try:
            installed = importlib.metadata.version("rlmflow")
        except importlib.metadata.PackageNotFoundError:  # a source checkout, uninstalled
            installed = "unknown"
        print(f"rlmflow {installed}")
        print(f"python {platform.python_version()} on {platform.system().lower()}")
        extras = ("openai", "anthropic", "textual", "rich", "gradio", "docker", "modal", "PIL")
        present = [name for name in extras if importlib.util.find_spec(name) is not None]
        print(f"extras: {', '.join(present) if present else 'none'}")
        return ""


class PluginCLI:
    """The Claude Code plugin that makes your session an rlmflow root agent."""

    def path(self) -> None:
        """Print the plugin directory, for ``claude --plugin-dir``."""
        from rlmflow.agent_tools.hooks import PLUGIN_DIR

        print(PLUGIN_DIR)


class RlmflowCLI:
    """Run agents, and read the graphs they leave behind.

    rlmflow tui                             the coding agent, in the dashboard
    rlmflow tui --query "fix the tests"     same, with the first turn underway
    rlmflow run tui "fix the tests"         the dashboard, query as a positional
    rlmflow run print "fix the tests"       no dashboard: stream it, print the answer
    rlmflow view show runs/coding/graph     the agent tree, then the timeline
    rlmflow render gif runs/coding/graph r.gif
    rlmflow config show                     resolved settings and where each came from
    rlmflow version                         versions, and which extras are installed
    """

    def __init__(self) -> None:
        from rlmflow.cli.run import RunCLI, TuiCLI
        from rlmflow.cli.view import RenderCLI, ViewCLI

        self.run = RunCLI
        self.tui = TuiCLI
        self.view = ViewCLI()
        self.render = RenderCLI()
        self.config = ConfigCLI()
        self.version = VersionCLI()
        self.plugin = PluginCLI()

    def serve(
        self,
        project: str = ".",
        graph: str | None = None,
        root: str | None = None,
        model: str | None = None,
        fast_model: str | None = None,
        docker_image: str | None = None,
        max_depth: int | None = None,
        workers: int | None = None,
        tools: str | None = None,
        flow: str | None = None,
    ) -> None:
        """Serve a run whose root agent is a Claude Code session; `/rlmflow:run` starts this.

        Args:
          project: The working directory; serve.json goes in its .rlmflow/.
          graph: A run directory to continue; default: a new one in .rlmflow/runs/.
          root: The agent the session drives; default: the run's interactive agent.
          flow: A ``module:factory`` that returns your Flow, instead of the coding one.
        """
        from rlmflow.cli.serve import serve_main

        overrides = {
            "model": model,
            "fast_model": fast_model,
            "docker_image": docker_image,
            "max_depth": max_depth,
            "workers": workers,
            "tools": tools,
            "flow": flow,
        }
        serve_main(project, graph_dir=graph, root=root, overrides=overrides)

    def fork(
        self,
        run: str,
        node: str,
        config: str | None = None,
        interactive: bool = False,
        name: str | None = None,
        project: str = ".",
        out: str | None = None,
    ) -> None:
        """Branch a saved run at NODE, then run the branch or continue it yourself.

        Args:
          run: The run directory.
          node: The node id (or unique prefix) to branch after.
          config: Run the branch with this config (a name on the project's Flow) instead of the original's.
          interactive: Continue the branch in your own Claude Code session.
          name: The branch's agent name.
          project: With --interactive, the working directory to serve.
          out: Where to save the result; default: RUN itself.
        """
        from rlmflow.cli.fork import fork_run

        fork_run(run, node, config=config, interactive=interactive, name=name, project=project, out=out)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``rlmflow`` script: 0 fine, 1 handled error, 2 bad usage."""
    from rlmflow.agent_tools import cli as agent_cli

    command = list(sys.argv[1:] if argv is None else argv)
    if command and command[0] in agent_cli.COMMANDS:
        return agent_cli.main(command)
    if command and command[0] == "hook":
        from rlmflow.agent_tools import hooks

        return hooks.main(command[1:])
    if command and command[0] == "mcp":
        from rlmflow.agent_tools import mcp

        return mcp.main()

    import fire
    from fire.core import FireExit

    try:
        fire.Fire(RlmflowCLI(), command=command, name="rlmflow")
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FireExit as exc:  # --help, or Fire could not match the arguments
        return int(exc.code)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return 0


__all__ = [
    "CONFIG_NAME",
    "CliError",
    "ConfigCLI",
    "PluginCLI",
    "RlmflowCLI",
    "RunOptions",
    "env_var",
    "main",
    "read_toml",
    "resolve_options",
    "user_config",
]
