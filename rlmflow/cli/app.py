"""The Fire component behind the ``rlmflow`` script, and the entry point itself.

The root is composition, nothing more: each command is a class that its own
module owns. Fire turns a class into a group, ``__init__`` into flags, methods
into verbs, and the docstring into help.
"""

from __future__ import annotations

import sys

from rlmflow.cli.config import ConfigCLI
from rlmflow.cli.options import CliError
from rlmflow.cli.run import RunCLI, TuiCLI
from rlmflow.cli.view import RenderCLI, ViewCLI


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
        self.run = RunCLI
        self.tui = TuiCLI
        self.view = ViewCLI()
        self.render = RenderCLI()
        self.config = ConfigCLI()
        self.version = VersionCLI()


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``rlmflow`` script: 0 fine, 1 handled error, 2 bad usage."""
    import fire
    from fire.core import FireExit

    command = list(sys.argv[1:] if argv is None else argv)
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


__all__ = ["CliError", "RlmflowCLI", "main"]
