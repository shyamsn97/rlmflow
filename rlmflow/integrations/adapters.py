"""Small compatibility adapters for examples built on ``rlmflow``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from rlmflow.flow import Flow
from rlmflow.graph import Graph, LLMUsage

try:
    import dspy as _dspy
except ModuleNotFoundError as exc:  # pragma: no cover - optional extra.
    _DSPY_IMPORT_ERROR: ModuleNotFoundError | None = exc
    _BaseLM = object
else:  # pragma: no cover - depends on optional extra.
    _DSPY_IMPORT_ERROR = None
    _BaseLM = _dspy.BaseLM


def messages_to_query(messages: list[dict[str, Any]]) -> str:
    """Project chat messages into a single agent query."""

    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content") or ""
        parts.append(f"{role}: {content if isinstance(content, str) else str(content)}")
    return "\n\n".join(parts)


class FlowLLM:
    """Expose a minimal :class:`Flow` through a sync ``chat(messages)`` API."""

    def __init__(self, flow: Flow) -> None:
        self.flow = flow
        self.last_graph: Graph | None = None
        self.last_usage = LLMUsage()

    def chat(self, messages: list[dict[str, Any]], *args: Any, **kwargs: Any) -> str:
        graph = Graph(query=messages_to_query(messages))
        result = self.flow.run(graph=graph)
        self.last_graph = graph
        self.last_usage = graph.usage()
        return result

    def completion(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        return self.chat([{"role": "user", "content": prompt}], *args, **kwargs)

    def close(self) -> None:
        self.flow.close_repls()


def _normalize_messages(
    prompt: str | None,
    messages: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    if messages is None:
        return [{"role": "user", "content": prompt or ""}]
    normalized: list[dict[str, str]] = []
    for message in messages:
        content = message.get("content") or ""
        normalized.append(
            {
                "role": str(message.get("role") or "user"),
                "content": content if isinstance(content, str) else str(content),
            }
        )
    return normalized


def _usage_dict(usage: LLMUsage | None) -> dict[str, int]:
    if usage is None:
        return {}
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
    }


def _chat_completion_response(*, model: str, text: str, usage: dict[str, int]) -> Any:
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=usage,
    )


class DSPyFlow(_BaseLM):
    """DSPy ``BaseLM`` adapter for any sync ``chat(messages)`` object."""

    def __init__(
        self,
        agent: Any,
        *,
        model: str = "rlmflow-minimal",
        model_type: str = "chat",
        **kwargs: Any,
    ) -> None:
        if _DSPY_IMPORT_ERROR is not None:  # pragma: no cover - optional extra.
            raise ModuleNotFoundError(
                "The DSPy integration requires the optional `dspy` dependency. "
                'Install it with `pip install -e ".[dspy]"`.'
            ) from _DSPY_IMPORT_ERROR

        super().__init__(model=model, model_type=model_type, **kwargs)
        self.agent = agent

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        request_messages = _normalize_messages(prompt, messages)
        text = self.agent.chat(request_messages, **{**self.kwargs, **kwargs})
        return _chat_completion_response(
            model=self.model,
            text=text,
            usage=_usage_dict(getattr(self.agent, "last_usage", None)),
        )

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.forward(prompt=prompt, messages=messages, **kwargs)


__all__ = ["DSPyFlow", "FlowLLM", "messages_to_query"]
