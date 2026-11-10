"""Live runs against the real ``claude`` CLI and model APIs; opt in with ``RLMFLOW_LIVE=1``."""

import os
import shutil

import pytest


def pytest_collection_modifyitems(config, items):
    reason = None
    if os.environ.get("RLMFLOW_LIVE") != "1":
        reason = "live runs are opt-in: set RLMFLOW_LIVE=1"
    elif shutil.which("claude") is None:
        reason = "live runs need the claude CLI on PATH"
    elif not os.environ.get("OPENAI_API_KEY"):
        reason = "live runs need OPENAI_API_KEY for the workers"
    for item in items:
        if "live" in item.keywords and reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))
