"""DSPy integration for using an RecursiveFlow agent as a DSPy language model.

Install with ``pip install rlmflow[dspy]``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from rflow.minimal.clients.llm import LLMClient, LLMUsage

try:
    import dspy
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra.
    if exc.name != "dspy":
        raise
    raise ModuleNotFoundError(
        "The DSPy integration requires the optional `dspy` dependency. "
        "Install it with `pip install rlmflow[dspy]`."
    ) from exc


def _normalize_messages(
    prompt: str | None,
    messages: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    if messages is None:
        return [{"role": "user", "content": prompt or ""}]

    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content") or ""
        normalized.append(
            {
                "role": role,
                "content": content if isinstance(content, str) else str(content),
            }
        )
    return normalized


def _usage_dict(usage: LLMUsage | None) -> dict[str, int]:
    if usage is None:
        return {}

    total_tokens = usage.input_tokens + usage.output_tokens
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": total_tokens,
    }


def _chat_completion_response(*, model: str, text: str, usage: dict[str, int]) -> Any:
    """Return the minimal OpenAI-style shape DSPy's BaseLM can process."""

    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
            )
        ],
        usage=usage,
    )


class RecursiveFlowLM(dspy.BaseLM):
    """Wrap an :class:`rflow.LLMClient` or :class:`rflow.RecursiveFlow` for DSPy.

    DSPy modules call this object like any other language model:

    ```python
    import dspy
    from rflow.integrations.dspy import RecursiveFlowLM

    dspy.configure(lm=RecursiveFlowLM(agent))
    ```
    """

    def __init__(
        self,
        agent: LLMClient,
        *,
        model: str = "rlmflow",
        model_type: str = "chat",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, model_type=model_type, **kwargs)
        self.agent = agent

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        request_messages = _normalize_messages(prompt, messages)
        request_kwargs = {**self.kwargs, **kwargs}
        text = self.agent.chat(request_messages, **request_kwargs)
        return _chat_completion_response(
            model=self.model,
            text=text,
            usage=_usage_dict(self.agent.last_usage),
        )

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.forward(prompt=prompt, messages=messages, **kwargs)


__all__ = ["RecursiveFlowLM"]
