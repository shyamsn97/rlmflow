"""OpenAI benchmark model."""

from __future__ import annotations

from benchmarks.eval import model
from benchmarks.eval.types import Model
from rflow.clients.llm import OpenAIClient


@model("openai")
class OpenAIModel(Model):
    provider = "openai"

    def __init__(self, name: str = "gpt-5-mini", **kwargs) -> None:
        self.name = name
        self.client = OpenAIClient(model=name, **kwargs)
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


__all__ = ["OpenAIModel"]
