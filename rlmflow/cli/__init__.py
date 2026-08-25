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

``python -m rlmflow …`` works the same.
"""

from __future__ import annotations

from rlmflow.cli.app import RlmflowCLI, main
from rlmflow.cli.options import CliError, RunOptions

__all__ = ["CliError", "RlmflowCLI", "RunOptions", "main"]
