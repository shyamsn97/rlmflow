"""The env boundary: one instance per rollout, shared by every agent in the tree.

The rollout is the episode. The root and every subagent act on the same env,
because they are all working on the same world — that is what delegation means
here. Concurrent siblings therefore serialize on one lock: two interleaved steps
against one world produce an episode no reward can be trusted on.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Protocol

from rlmflow.tools import tool


class Env(Protocol):
    """What RAO needs of an environment. OpenEnv satisfies it through an adapter."""

    async def reset(self) -> Any:
        """Start an episode and return the initial observation."""

    async def step(self, action: Any) -> Any:
        """Apply an action and return a result with observation, reward, and done."""

    async def close(self) -> None:
        """Release the container or process behind this env."""

    def action(self, **fields: Any) -> Any:
        """Build this env's typed action from keyword fields."""


@dataclass(frozen=True)
class EnvStep:
    """One env transition, attributed to the agent that caused it.

    ``agent_id`` is the load-bearing field: it is the only record of who did what
    to the shared world, and per-node local reward is derived from it.
    """

    agent_id: str
    action: dict[str, Any]
    observation: Any
    reward: float
    done: bool
    index: int


class EpisodeOver(RuntimeError):
    """Raised when an agent acts after the episode ended or the budget ran out.

    Surfaced to the agent as a normal tool error, which it can read and recover
    from by finishing.
    """


def plain(value: Any) -> Any:
    """Reduce an observation to plain data, so it survives the REPL boundary.

    A pydantic model or dataclass defined inside an env package cannot be
    unpickled in an agent's worker, which may not have that package installed.
    """
    for name in ("model_dump", "dict"):
        method = getattr(value, name, None)
        if callable(method):
            return method()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


@dataclass
class EnvSession:
    """One env bound to one rollout: serialized access, plus the step log."""

    env: Env
    max_steps: int | None = None
    steps: list[EnvStep] = field(default_factory=list)
    done: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def reset(self) -> Any:
        observation = await self.env.reset()
        self.steps.clear()
        self.done = False
        return plain(observation)

    async def act(self, agent_id: str, **fields: Any) -> EnvStep:
        """Take one step on behalf of ``agent_id``, one caller at a time."""
        async with self.lock:
            if self.done:
                raise EpisodeOver("the episode is over; finish with what you have")
            if self.max_steps is not None and len(self.steps) >= self.max_steps:
                self.done = True
                raise EpisodeOver(f"this rollout is capped at {self.max_steps} env steps")
            result = await self.env.step(self.env.action(**fields))
            step = EnvStep(
                agent_id=agent_id,
                action=dict(fields),
                observation=plain(getattr(result, "observation", result)),
                reward=float(getattr(result, "reward", 0.0) or 0.0),
                done=bool(getattr(result, "done", False)),
                index=len(self.steps),
            )
            self.steps.append(step)
            self.done = step.done
            return step

    def reward_for(self, agent_id: str) -> float:
        """Env reward earned by this agent's own actions.

        Exact and cheap, and only possible *because* the env is shared — a
        per-agent env could not attribute anything. It degrades on
        terminal-reward envs, where whichever agent lands the final step
        collects everything; those need a different local score.
        """
        return sum(step.reward for step in self.steps if step.agent_id == agent_id)

    @property
    def total_reward(self) -> float:
        return sum(step.reward for step in self.steps)

    def actors(self) -> set[str]:
        return {step.agent_id for step in self.steps}

    async def close(self) -> None:
        await self.env.close()


def env_tool(session: EnvSession, agent_id: str):
    """The one env tool, bound to a rollout's session and to one agent."""

    @tool(
        "Act on the shared environment; returns {observation, reward, done}. "
        "Every agent in this run acts on the same environment, so your actions "
        "and your subagents' actions change the same world.",
        proxy=True,
    )
    async def env_step(**fields: Any) -> dict[str, Any]:
        step = await session.act(agent_id, **fields)
        return {"observation": step.observation, "reward": step.reward, "done": step.done}

    return env_step


class OpenEnvAdapter:
    """Wrap a Hugging Face OpenEnv ``EnvClient`` as an :class:`Env`.

    Thin by design: the client is already async and already owns its container,
    so this only builds typed actions and hands back the step result.
    """

    def __init__(self, client: Any, action_cls: Any) -> None:
        self.client = client
        self.action_cls = action_cls

    def action(self, **fields: Any) -> Any:
        return self.action_cls(**fields)

    async def reset(self) -> Any:
        result = await self.client.reset()
        return getattr(result, "observation", result)

    async def step(self, action: Any) -> Any:
        return await self.client.step(action)

    async def close(self) -> None:
        for name in ("close", "aclose", "__aexit__"):
            method = getattr(self.client, name, None)
            if method is None:
                continue
            result = method(None, None, None) if name == "__aexit__" else method()
            if asyncio.iscoroutine(result):
                await result
            return


__all__ = [
    "Env",
    "EnvSession",
    "EnvStep",
    "EpisodeOver",
    "OpenEnvAdapter",
    "env_tool",
    "plain",
]
