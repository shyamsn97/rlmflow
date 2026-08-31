"""Prompt and control-loop conditions for delegation experiments."""

from __future__ import annotations

from enum import StrEnum

from rlmflow import AgentStart, Flow
from rlmflow.engine.steps import LLMRequestStep
from rlmflow.prompts import SystemPromptBuilder


class DelegationCondition(StrEnum):
    LOCAL = "local"
    CAPABILITY_ONLY = "capability_only"
    CURRENT_POLICY = "current_policy"

    @classmethod
    def parse(cls, value: str) -> DelegationCondition:
        try:
            return cls(value.replace("-", "_").lower())
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown delegation condition {value!r}; choose {choices}") from exc


class CapabilityOnlyStartStep(LLMRequestStep):
    """Start an agent directly without inserting the policy-bearing PlanQuery."""

    async def __call__(self, node):
        return await self.chat(node)


def system_prompt_for(condition: DelegationCondition | str) -> SystemPromptBuilder | None:
    condition = DelegationCondition.parse(condition) if isinstance(condition, str) else condition
    if condition is not DelegationCondition.CAPABILITY_ONLY:
        return None
    builder = SystemPromptBuilder()
    builder.sections.drop("strategy").drop("examples")
    return builder


def apply_condition(flow: Flow, condition: DelegationCondition | str) -> None:
    condition = DelegationCondition.parse(condition) if isinstance(condition, str) else condition
    if condition is DelegationCondition.CAPABILITY_ONLY:
        flow.update_step_fn(AgentStart, CapabilityOnlyStartStep)


__all__ = [
    "CapabilityOnlyStartStep",
    "DelegationCondition",
    "apply_condition",
    "system_prompt_for",
]
