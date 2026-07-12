"""Anthropic benchmark model."""

from __future__ import annotations

from benchmarks.eval import model
from benchmarks.eval.types import Model
from rflow.clients.llm import AnthropicClient


@model("anthropic")
class AnthropicModel(Model):
    provider = "anthropic"

    def __init__(self, name: str = "claude-4-sonnet", **kwargs) -> None:
        self.name = name
        self.client = AnthropicClient(model=name, **kwargs)
        self._usage = {"input_tokens": 0, "output_tokens": 0}

    def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        text, usage = self.client.completion(messages, **kwargs)
        self._usage = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
        return text

    def usage(self) -> dict[str, int]:
        return dict(self._usage)


__all__ = ["AnthropicModel"]
