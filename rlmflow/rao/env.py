"""The environment boundary: one env per rollout, shared by every agent in the tree.

The rollout is the episode. The root and every subagent act on the same env,
because they are all working on the same world — that is what delegation means
here. Concurrent siblings therefore serialize on one lock: two interleaved steps
against one world produce an episode no reward can be trusted on.

The protocol accepts the official Gymnasium ``reset``/``step`` tuple API and
OpenEnv's equivalent result objects. ``rlmflow`` imports neither package in its
core; real environments and lightweight test doubles satisfy the same methods.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Protocol

from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import Node, record_metadata
from rlmflow.tools import tool


class Env(Protocol):
    """The Gymnasium/OpenEnv methods a rollout uses.

    Methods may be synchronous (Gymnasium) or awaitable (OpenEnv);
    :func:`resolve` handles either.
    """

    def reset(self, **kwargs: Any) -> Any:
        """Start an episode; returns a ``StepResult`` carrying the observation."""

    def step(self, action: Any) -> Any:
        """Apply an action; returns a ``StepResult``."""

    def close(self) -> Any:
        """Disconnect, and stop the container if this client started one."""


class ForkableEnv(Env, Protocol):
    """An agent-local episode view over a world shared with its parent.

    Forking is synchronous because child creation happens inside the synchronous
    ``launch_agent`` boundary. The returned env may share mutable world state,
    but must own its task, termination state, and reward bookkeeping.
    """

    def fork(self, task: str, args: dict[str, Any]) -> Env:
        """Create one child episode without resetting the shared world."""


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

    Surfaced to the agent as an ordinary tool error, which it can read and
    recover from by finishing.
    """


async def resolve(value: Any) -> Any:
    """Await ``value`` if it is awaitable, else return it.

    OpenEnv hands back an object that is both awaitable and synchronously
    readable; a plain fake hands back a value or a coroutine. One helper covers
    all three without the caller caring which client it holds.
    """
    return await value if inspect.isawaitable(value) else value


def plain(value: Any) -> Any:
    """Reduce an observation to plain data, so it survives the REPL boundary.

    OpenEnv observations are pydantic models, and one defined inside an env
    package cannot be unpickled in an agent's worker, which may not have that
    package installed. ``GenericEnvClient`` already hands back dicts; a typed
    client does not.
    """
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def reward_of(result: Any) -> float:
    """The reward for one step, as OpenEnv reports it.

    ``StepResult.reward`` is ``Optional[float]`` and the base ``Observation``
    carries its own ``reward``, so a ``None`` on the result falls back to the
    observation before being read as zero.
    """
    reward = result[1] if isinstance(result, tuple) and len(result) in (4, 5) else None
    if reward is None:
        reward = getattr(result, "reward", None)
    if reward is None:
        reward = getattr(getattr(result, "observation", None), "reward", None)
    return float(reward or 0.0)


def done_of(result: Any) -> bool:
    if isinstance(result, tuple):
        if len(result) == 5:
            return bool(result[2] or result[3])
        if len(result) == 4:
            return bool(result[2])
    if getattr(result, "done", False):
        return True
    return bool(getattr(getattr(result, "observation", None), "done", False))


def observation_of(result: Any) -> Any:
    """Read observations from Gymnasium tuples or OpenEnv result objects."""
    if isinstance(result, tuple):
        if len(result) in (2, 4, 5):
            return result[0]
        raise ValueError("environment returned an unsupported tuple shape")
    return getattr(result, "observation", result)


@dataclass
class EnvSession:
    """One agent-local env view, serialized with every view of the same world.

    ``action_cls`` is the env's typed ``Action`` when the caller has one. Left
    unset, actions go out as plain dicts, which is what ``GenericEnvClient``
    accepts and what lets an agent act on an env whose package is not installed.

    Forked sessions share ``steps`` and ``lock`` but not ``done`` or their local
    step count. A child can therefore finish its assignment while its parent
    continues acting on the shared world.
    """

    env: Env
    action_cls: Any = None
    max_steps: int | None = None
    steps: list[EnvStep] = field(default_factory=list)
    done: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _local_steps: int = field(default=0, init=False, repr=False)

    def action(self, **fields: Any) -> Any:
        return dict(fields) if self.action_cls is None else self.action_cls(**fields)

    async def reset(self, **kwargs: Any) -> Any:
        result = await resolve(self.env.reset(**kwargs))
        self.steps.clear()
        self.done = False
        self._local_steps = 0
        return plain(observation_of(result))

    def fork(self, task: str, args: dict[str, Any]) -> tuple[EnvSession, Any]:
        """Fork a child episode and synchronously return its initial observation."""
        fork = getattr(self.env, "fork", None)
        if not callable(fork):
            # Compatibility for envs whose actions may be shared but whose
            # protocol has no independent child episode.
            return self, None
        child_env = fork(task, args)
        if inspect.isawaitable(child_env):
            raise TypeError("Env.fork() must be synchronous")
        child = EnvSession(
            child_env,
            action_cls=self.action_cls,
            max_steps=self.max_steps,
            steps=self.steps,
            lock=self.lock,
        )
        result = child_env.reset()
        if inspect.isawaitable(result):
            raise TypeError("a forked env's reset() must be synchronous")
        return child, plain(observation_of(result))

    async def act(self, agent_id: str, **fields: Any) -> EnvStep:
        """Take one step on behalf of ``agent_id``, one caller at a time."""
        async with self.lock:
            if self.done:
                raise EpisodeOver("the episode is over; finish with what you have")
            if self.max_steps is not None and self._local_steps >= self.max_steps:
                self.done = True
                raise EpisodeOver(f"this agent is capped at {self.max_steps} env steps")
            result = await resolve(self.env.step(self.action(**fields)))
            step = EnvStep(
                agent_id=agent_id,
                action=dict(fields),
                observation=plain(observation_of(result)),
                reward=reward_of(result),
                done=done_of(result),
                index=len(self.steps),
            )
            self.steps.append(step)
            self._local_steps += 1
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
        await resolve(self.env.close())


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
        record_metadata(
            "environment.steps",
            {
                "agent_id": step.agent_id,
                "action": step.action,
                "observation": step.observation,
                "reward": step.reward,
                "done": step.done,
                "index": step.index,
            },
            append=True,
        )
        return {"observation": step.observation, "reward": step.reward, "done": step.done}

    return env_step


def environment_rewards(graph: Graph, root: Node) -> dict[str, float]:
    """Sum graph-recorded environment rewards by acting agent in ``root``'s tree."""
    agents = graph.tree(root)
    rewards = {graph.path(agent): 0.0 for agent in agents}
    for node in (node for agent in agents for node in graph.nodes(agent=agent.id)):
        for receipt in node.metadata.get("environment.steps", []):
            if not isinstance(receipt, dict):
                continue
            actor = receipt.get("agent_id")
            if actor in rewards:
                rewards[actor] += float(receipt.get("reward", 0.0) or 0.0)
    return rewards


async def open_env(
    *,
    base_url: str | None = None,
    hub_id: str | None = None,
    docker_image: str | None = None,
    **kwargs: Any,
) -> Any:
    """Connect a real OpenEnv client, one of the three ways OpenEnv offers.

    A thin wrapper over ``GenericEnvClient``, which needs no env package
    installed — that is what makes one ``env_step(**fields)`` tool work against
    any env. Pass a typed client to :class:`EnvSession` directly, with its
    ``action_cls``, if you want typed actions instead.

    Async because ``from_env`` and ``from_docker_image`` return a handle that has
    to be awaited before a container is up; awaiting here means every path hands
    back a client that is ready to step.
    """
    try:
        from openenv import GenericEnvClient
    except ImportError as exc:  # pragma: no cover - exercised by optional deps
        raise ImportError(
            "RAO environments need the OpenEnv SDK. Install it with "
            "`pip install openenv` or `pip install rlmflow[openenv]`."
        ) from exc

    if base_url is not None:
        return GenericEnvClient(base_url=base_url, **kwargs)
    if hub_id is not None:
        return await GenericEnvClient.from_env(hub_id, **kwargs)
    if docker_image is not None:
        return await GenericEnvClient.from_docker_image(docker_image, **kwargs)
    raise ValueError("pass one of base_url, hub_id, or docker_image")


__all__ = [
    "Env",
    "EnvSession",
    "EnvStep",
    "EpisodeOver",
    "ForkableEnv",
    "done_of",
    "env_tool",
    "environment_rewards",
    "open_env",
    "observation_of",
    "plain",
    "resolve",
    "reward_of",
]
