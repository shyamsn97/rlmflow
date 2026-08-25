"""Driving recursive rollouts against one shared env, and recording what was sampled."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rlmflow.engine.execution import Transition
from rlmflow.flow import Flow
from rlmflow.graph.nodes import (
    AgentConfig,
    AgentStart,
    ExecAction,
    LLMOutput,
    Node,
)
from rlmflow.llm import PooledLLMClient
from rlmflow.rao.credit import (
    DELEGATION_BONUS,
    IncompleteTreeError,
    NodeScore,
    agent_id,
    agents,
    assign_advantages,
    assign_depth_weights,
    score_tree,
)
from rlmflow.rao.env import Env, EnvSession, env_tool

#: Key the initial observation is handed to the root agent under, via ``INPUTS``.
OBSERVATION_INPUT = "observation"


@dataclass(frozen=True)
class TaskSpec:
    """One episode's task. The root's goal; child goals are authored by parents."""

    id: str
    goal: str
    #: Whatever the env factory needs to open the right episode (seed, level, split).
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnSample:
    """One sampled assistant turn, as the sampler saw it.

    Recorded at generation time because it cannot be reconstructed afterwards:
    sampling logprobs are gone once the call returns, and a renderer's exact
    tokenization is not recoverable from the transcript text.
    """

    prompt_tokens: list[int]
    sampled_tokens: list[int]
    logprobs: list[float]
    stop_reason: str = ""

    @property
    def trainable(self) -> bool:
        return bool(self.sampled_tokens) and len(self.logprobs) == len(self.sampled_tokens)


@dataclass(frozen=True)
class Budget:
    """Rollout limits. Explosion is now a container cost, not just a token cost."""

    max_depth: int = 2
    max_iters: int = 12
    max_children_per_agent: int | None = 4
    max_total_agents: int | None = 16
    max_env_steps: int | None = 64
    max_budget: int | None = None
    max_wall_time: float | None = None

    def agent_config(self) -> AgentConfig:
        return AgentConfig(
            max_depth=self.max_depth,
            max_iters=self.max_iters,
            max_budget=self.max_budget,
        )


class RolloutFlow(Flow):
    """A ``Flow`` that shares one env across a tree, records tokens, and enforces budgets.

    Three narrow overrides and no runtime change: ``build_tools`` hands every
    agent the env, ``llm_for_step``/``step`` record what was sampled, and
    ``resolve_child`` refuses launches past the budget.
    """

    def __init__(self, llm: Any, *, budget: Budget | None = None, **kwargs: Any) -> None:
        budget = budget or Budget()
        kwargs.setdefault("root_config", budget.agent_config())
        super().__init__(llm, **kwargs)
        self.budget = budget
        self.sessions: dict[int, EnvSession] = {}
        #: ``LLMOutput.id -> TurnSample``, so transcript order comes for free.
        self.samples: dict[str, TurnSample] = {}
        self.refusals: dict[str, int] = {}
        self._sample_sinks: dict[str, list[TurnSample]] = {}

    # -- Env sharing ------------------------------------------------------

    def bind(self, root: AgentStart, session: EnvSession) -> None:
        """Give one rollout tree its env."""
        self.sessions[id(root)] = session

    def unbind(self, root: AgentStart) -> None:
        self.sessions.pop(id(root), None)

    def session_for(self, node: Node) -> EnvSession | None:
        root = node.root
        return self.sessions.get(id(root)) if root is not None else None

    def build_tools(self, node: Node) -> dict[str, Any]:
        namespace = super().build_tools(node)
        session = self.session_for(node)
        if session is not None:
            # Bound per node, which is what lets one Flow drive every rollout of a
            # task while each tree only ever touches its own env.
            agent = node.parent_agent
            path = agent.config.path if agent is not None else "root"
            namespace["env_step"] = env_tool(session, path)
        return namespace

    # -- Token capture ----------------------------------------------------

    def records(self, model: str) -> bool:
        """Whether this model's client hands back the tokens it sampled."""
        return bool(getattr(self._llm_clients.get(model), "records_tokens", False))

    def llm_for_step(self, node: Node) -> PooledLLMClient:
        agent = node.parent_agent
        model = agent.config.model
        if not self.records(model):
            return super().llm_for_step(node)

        sink: list[TurnSample] = []
        self._sample_sinks[agent.id] = sink
        return self._create_pooled_llm(
            model,
            key=agent.id,
            request_kwargs={"sample_sink": sink},
        )

    async def step(self, node: Node) -> Transition:
        transition = await super().step(node)
        created = transition.created
        agent = created.parent_agent
        sink = self._sample_sinks.pop(
            agent.id if agent is not None else "",
            [],
        )
        if isinstance(created, LLMOutput):
            if sink:
                self.samples[created.id] = sink[-1]
        return transition

    def turns(self, agent: AgentStart) -> list[TurnSample]:
        """This agent's sampled turns, in transcript order."""
        return [
            self.samples[node.id]
            for node in agent.transcript()
            if isinstance(node, LLMOutput) and node.id in self.samples
        ]

    # -- Budgets ----------------------------------------------------------

    def resolve_child(
        self,
        action: ExecAction,
        spec: dict[str, Any],
        call_id: int,
    ) -> AgentStart:
        limit = self.budget.max_children_per_agent
        parent = action.parent_agent
        # Only genuinely new launches are refused: ``super()`` hands back an
        # already-attached child for a repeated call, and refusing that would
        # break replay once a cap is full.
        if limit is not None and len(parent.sub_agents) >= limit:
            if not attached(action, spec, call_id):
                self.refuse("max_children_per_agent")
                raise ValueError(f"cannot launch beyond {limit} children per agent")
        total = self.budget.max_total_agents
        if total is not None and len(agents(action.root)) >= total:
            if not attached(action, spec, call_id):
                self.refuse("max_total_agents")
                raise ValueError(f"cannot launch beyond {total} agents in one rollout")
        return super().resolve_child(action, spec, call_id)

    def refuse(self, reason: str) -> None:
        self.refusals[reason] = self.refusals.get(reason, 0) + 1


def attached(action: ExecAction, spec: dict[str, Any], call_id: int) -> bool:
    """Whether this launch call already has its child, as ``Flow`` decides."""
    name = spec.get("name")
    return any(
        isinstance(child, AgentStart)
        and (
            child.config.launch_call_id == call_id
            or (name is not None and child.config.name == name)
        )
        for child in action.children
    )


@dataclass
class RolloutTree:
    """One rollout: a tree of agents, the env they shared, and their scores."""

    task: TaskSpec
    root: AgentStart
    index: int
    session: EnvSession
    scores: dict[str, NodeScore] = field(default_factory=dict)
    error: str | None = None

    @property
    def rollout_id(self) -> str:
        return f"{self.task.id}/g{self.index}"

    @property
    def root_id(self) -> str:
        return agent_id(self.root)

    @property
    def agents(self) -> list[AgentStart]:
        return agents(self.root)

    @property
    def complete(self) -> bool:
        """Every agent terminal — not just the root."""
        return all(agent.terminal for agent in self.agents)

    @property
    def usable(self) -> bool:
        return self.error is None and bool(self.scores)


class Collector:
    """Run ``rollouts_per_task`` recursive rollouts per task and score them.

    Tasks run one at a time, rollouts within a task run together, so the number
    of live envs is ``rollouts_per_task`` rather than that times the batch.
    """

    def __init__(
        self,
        flow: RolloutFlow,
        open_env: Callable[[TaskSpec], Awaitable[Env]],
        *,
        rollouts_per_task: int = 4,
        delegation_bonus: float = DELEGATION_BONUS,
    ) -> None:
        self.flow = flow
        self.open_env = open_env
        self.rollouts_per_task = rollouts_per_task
        self.delegation_bonus = delegation_bonus

    async def collect(self, tasks: Sequence[TaskSpec]) -> list[RolloutTree]:
        trees: list[RolloutTree] = []
        for task in tasks:
            group = await self.rollout(task)
            # Leave-one-out is per task: the baseline is the other rollouts of the
            # same task, so it cannot be pooled across the batch.
            assign_advantages([(tree.root_id, tree.scores) for tree in group if tree.usable])
            trees.extend(group)
        # Depth weighting is per batch, since it equalizes each depth's total pull.
        assign_depth_weights([tree.scores for tree in trees if tree.usable])
        return trees

    async def rollout(self, task: TaskSpec) -> list[RolloutTree]:
        trees = await self.open(task)
        try:
            await self.drive([tree.root for tree in trees])
        except TimeoutError:
            for tree in trees:
                tree.error = tree.error or "wall time"
        except Exception as exc:  # noqa: BLE001 - one bad rollout must not void the rest
            for tree in trees:
                if not tree.complete:
                    tree.error = f"{type(exc).__name__}: {exc}"
        for tree in trees:
            self.score(tree)
        await self.release(trees)
        return trees

    async def open(self, task: TaskSpec) -> list[RolloutTree]:
        trees: list[RolloutTree] = []
        for index in range(self.rollouts_per_task):
            session = EnvSession(
                await self.open_env(task),
                max_steps=self.flow.budget.max_env_steps,
            )
            observation = await session.reset()
            root = self.flow.start(
                task.goal,
                inputs={OBSERVATION_INPUT: json.dumps(observation, default=str)},
            )
            self.flow.bind(root, session)
            trees.append(RolloutTree(task=task, root=root, index=index, session=session))
        return trees

    async def drive(self, roots: list[AgentStart]) -> None:
        async def run() -> None:
            async for _node in self.flow.run_streaming(*roots):
                pass

        timeout = self.flow.budget.max_wall_time
        await (asyncio.wait_for(run(), timeout) if timeout else run())

    def score(self, tree: RolloutTree) -> None:
        if tree.error is not None:
            return
        local = {agent_id(agent): tree.session.reward_for(agent_id(agent)) for agent in tree.agents}
        try:
            tree.scores = score_tree(tree.root, local, delegation_bonus=self.delegation_bonus)
        except IncompleteTreeError:
            tree.error = "incomplete"

    async def release(self, trees: Sequence[RolloutTree]) -> None:
        for tree in trees:
            self.flow.unbind(tree.root)
            for agent in tree.agents:
                self.flow.runtime.close_repl(agent)
            await tree.session.close()


__all__ = [
    "OBSERVATION_INPUT",
    "Budget",
    "Collector",
    "RolloutFlow",
    "RolloutTree",
    "TaskSpec",
    "TurnSample",
]
