"""Suite-wide pytest hooks."""

from __future__ import annotations

import importlib

import pytest
from helpers import TestRuntime


@pytest.fixture(autouse=True)
def use_in_process_runtime_by_default(monkeypatch):
    """Keep worker-process coverage explicit instead of taxing every flow test."""
    flow_module = importlib.import_module("rlmflow.flow")
    monkeypatch.setattr(flow_module, "LocalRuntime", TestRuntime)
