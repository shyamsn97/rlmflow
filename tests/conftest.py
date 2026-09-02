"""Suite-wide pytest hooks."""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from helpers import TestRuntime


@pytest.fixture(autouse=True)
def use_in_process_runtime_by_default(monkeypatch):
    """Keep worker-process coverage explicit instead of taxing every flow test."""
    flow_module = importlib.import_module("rlmflow.flow")
    monkeypatch.setattr(flow_module, "LocalRuntime", TestRuntime)


def pytest_deselected(items: list[Any]) -> None:
    """Name what a filter dropped, and why, instead of leaving a bare count.

    ``-m "live and not slow"`` reporting only "5 deselected" says nothing about
    which behavior went unmeasured, and these checks are the ones worth knowing
    that about: a paid run that silently skipped the boids scenario looks exactly
    like one that covered it. One line, since the point is to be scannable.
    """
    if not items:
        return
    config = items[0].config
    if config.option.verbose < 0:  # -q: the caller asked for less, not more
        return
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    why = " ".join(
        part
        for part in (
            f'-m "{config.option.markexpr}"' if config.option.markexpr else "",
            f'-k "{config.option.keyword}"' if config.option.keyword else "",
        )
        if part
    )
    names = ", ".join(item.name for item in items)
    reporter.write_line(f"deselected by {why or 'filter'}: {names}", yellow=True)
