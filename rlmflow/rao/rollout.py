"""Driving recursive rollouts against one shared env, and recording what was sampled."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rlmflow.engine.scheduler import StepResult
from rlmflow.flow import Flow
from rlmflow.graph.config import AgentConfig
from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import LLM_OUTPUT, LLMUsage, Node, Step, current_step
from rlmflow.rao.credit import (
    DELEGATION_BONUS,
    IncompleteTreeError,
    NodeScore,
    RewardFn,
    agent_id,
    agents,
    assign_advantages,
    assign_depth_weights,
    score_tree,
)
from rlmflow.rao.env import Env, EnvSession, env_tool, environment_rewards

#: Direct root argument containing the initial environment observation.
OBSERVATION_INPUT = "observation"
#: The model name the policy being trained is registered under.
POLICY = "policy"


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
            model=POLICY,
            max_depth=self.max_depth,
            max_iters=self.max_iters,
            max_budget=self.max_budget,
        )


class RolloutFlow(Flow):
    """A ``Flow`` that shares a world across agent-local envs and records tokens.

    Narrow overrides and no runtime change: ``namespace_for`` hands every agent
    the env, ``complete``/``run_step`` record what was sampled, and ``spawn``
    refuses launches past the budget.
    """

    def __init__(self, policy: Any, *, budget: Budget | None = None, **kwargs: Any) -> None:
        super().__init__(llm_clients={POLICY: policy, **kwargs.pop("llm_clients", {})}, **kwargs)
        self.budget = budget or Budget()
        self.root_config = self.add_config(self.budget.agent_config())
        #: Agent id -> ``EnvSession``.
        self.sessions: dict[str, EnvSession] = {}
        #: ``llm_output`` node id -> ``TurnSample``, so transcript order comes for free.
        self.samples: dict[str, TurnSample] = {}
        self.refusals: dict[str, int] = {}
        self._sample_sinks: dict[str, list[TurnSample]] = {}
        self._session_lock = threading.RLock()

    # -- Env sharing ------------------------------------------------------

    def bind(self, agent: Node, session: EnvSession) -> None:
        """Give one agent its episode view of the rollout's shared world."""
        with self._session_lock:
            self.sessions[agent.id] = session

    def unbind(self, graph: Graph, root: Node) -> None:
        with self._session_lock:
            for agent in graph.tree(root):
                self.sessions.pop(agent.id, None)

    def ensure_session(self, graph: Graph, agent: Node) -> EnvSession | None:
        """Idempotently bind restored children that bypassed ``spawn``."""
        with self._session_lock:
            if (session := self.sessions.get(agent.id)) is not None:
                return session
            parent = graph.launcher(agent)
            if parent is None:
                return None
            parent_session = self.ensure_session(graph, parent)
            if parent_session is None:
                return None
            child_session, _observation = parent_session.fork(agent.task, dict(agent.args))
            self.sessions[agent.id] = child_session
            return child_session

    def namespace_for(self, step: Step) -> dict[str, Any]:
        session = self.ensure_session(step.graph, step.agent)
        if session is None:
            return {}
        # Bound per agent: forkable envs share world state while keeping
        # independent tasks, termination, rewards, and local budgets.
        return {"env_step": env_tool(session, step.graph.path(step.agent))}

    # -- Token capture ----------------------------------------------------

    def records(self, model: str) -> bool:
        """Whether this model's client hands back the tokens it sampled."""
        return bool(getattr(self.llms.get(model), "records_tokens", False))

    async def complete(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> tuple[str, LLMUsage]:
        step = current_step()
        if step is None or not self.records(model):
            return await super().complete(messages, model, **kwargs)
        sink: list[TurnSample] = []
        self._sample_sinks[step.agent.id] = sink
        return await super().complete(messages, model, sample_sink=sink, **kwargs)

    async def run_step(self, step: Step) -> StepResult:
        result = await super().run_step(step)
        sink = self._sample_sinks.pop(step.agent.id, [])
        if result.created is not None and result.created.type == LLM_OUTPUT and sink:
            self.samples[result.created.id] = sink[-1]
        return result

    def turns(self, graph: Graph, agent: Node) -> list[TurnSample]:
        """This agent's sampled turns, in transcript order."""
        return [
            self.samples[node.id]
            for node in graph.nodes(LLM_OUTPUT, agent=agent.id)
            if node.id in self.samples
        ]

    # -- Budgets ----------------------------------------------------------

    def spawn(self, step: Step, task: str, *, config: str = "default", **kwargs: Any) -> Node:
        g, parent = step.graph, step.agent
        limit = self.budget.max_children_per_agent
        if limit is not None and len(g.children(parent)) >= limit:
            self.refuse("max_children_per_agent")
            raise ValueError(f"cannot launch beyond {limit} children per agent")
        total = self.budget.max_total_agents
        if total is not None and len(g.tree(g.lineage(parent)[0])) >= total:
            self.refuse("max_total_agents")
            raise ValueError(f"cannot launch beyond {total} agents in one rollout")
        session = self.ensure_session(g, parent)
        child_session = None
        if session is not None:
            named = {k: v for k, v in kwargs.items() if k not in ("name", "inputs")}
            child_session, observation = session.fork(task, named)
            if observation is not None:
                kwargs.setdefault(OBSERVATION_INPUT, json.dumps(observation, default=str))
        child = super().spawn(step, task, config=config, **kwargs)
        if child_session is not None:
            self.bind(child, child_session)
        return child

    def refuse(self, reason: str) -> None:
        self.refusals[reason] = self.refusals.get(reason, 0) + 1


@dataclass
class RolloutTree:
    """One rollout: a top-level agent's tree, the env they shared, and their scores."""

    task: TaskSpec
    graph: Graph
    root: Node
    index: int
    session: EnvSession
    scores: dict[str, NodeScore] = field(default_factory=dict)
    error: str | None = None

    @property
    def rollout_id(self) -> str:
        return f"{self.task.id}/g{self.index}"

    @property
    def root_id(self) -> str:
        return agent_id(self.graph, self.root)

    @property
    def agents(self) -> list[Node]:
        return agents(self.graph, self.root)

    @property
    def complete(self) -> bool:
        """Every agent terminal — not just the root."""
        return all(self.graph.terminal(agent) for agent in self.agents)

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
        reward_fn: RewardFn | None = None,
        local_scorer: Callable[[RolloutTree], Mapping[str, float]] | None = None,
    ) -> None:
        if reward_fn is not None and local_scorer is not None:
            raise ValueError("pass reward_fn or local_scorer, not both")
        self.flow = flow
        self.open_env = open_env
        self.rollouts_per_task = rollouts_per_task
        self.delegation_bonus = delegation_bonus
        self.reward_fn = reward_fn
        self.local_scorer = local_scorer

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
            await self.drive(trees[0].graph if trees else Graph())
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
        """One graph per task, holding each rollout as its own top-level agent."""
        graph = Graph()
        trees: list[RolloutTree] = []
        for index in range(self.rollouts_per_task):
            session = EnvSession(
                await self.open_env(task),
                max_steps=self.flow.budget.max_env_steps,
            )
            observation = await session.reset()
            root = graph.create_agent(
                task.goal,
                **{OBSERVATION_INPUT: json.dumps(observation, default=str)},
            )
            root.annotate(
                "rollout.task",
                {"id": task.id, "index": index, "metadata": task.metadata},
            )
            self.flow.bind(root, session)
            trees.append(
                RolloutTree(task=task, graph=graph, root=root, index=index, session=session)
            )
        return trees

    async def drive(self, graph: Graph) -> None:
        async def run() -> None:
            async for _item in self.flow.run_streaming(graph):
                pass

        timeout = self.flow.budget.max_wall_time
        await (asyncio.wait_for(run(), timeout) if timeout else run())

    def score(self, tree: RolloutTree) -> None:
        if tree.error is not None:
            return
        local = (
            dict(self.reward_fn(tree.graph, tree.root))
            if self.reward_fn is not None
            else (
                dict(self.local_scorer(tree))
                if self.local_scorer is not None
                else environment_rewards(tree.graph, tree.root)
            )
        )
        try:
            tree.scores = score_tree(
                tree.graph, tree.root, local, delegation_bonus=self.delegation_bonus
            )
        except IncompleteTreeError:
            tree.error = "incomplete"

    async def release(self, trees: Sequence[RolloutTree]) -> None:
        for tree in trees:
            sessions = {
                id(session): session
                for agent in tree.agents
                if (session := self.flow.sessions.get(agent.id)) is not None
            }
            self.flow.unbind(tree.graph, tree.root)
            for agent in tree.agents:
                self.flow.runtime.close_repl(agent)
            await asyncio.gather(*(session.close() for session in sessions.values()))


__all__ = [
    "OBSERVATION_INPUT",
    "POLICY",
    "Budget",
    "Collector",
    "RolloutFlow",
    "RolloutTree",
    "TaskSpec",
    "TurnSample",
]
