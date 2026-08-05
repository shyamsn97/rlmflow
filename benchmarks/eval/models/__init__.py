"""Built-in benchmark models."""

from __future__ import annotations

from benchmarks.eval import MODELS, model
from benchmarks.eval.models import anthropic, fake, openai  # noqa: F401

__all__ = ["MODELS", "model"]
