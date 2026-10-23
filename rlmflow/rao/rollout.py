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
from rlmflow.graph import reads
from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import (
    AgentConfig,
    AgentStart,
    ExecAction,
    LLMOutput,
    Node,
    Step,
    current_step,
)
from rlmflow.llm import PooledLLMClient
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
            resume_on_child_done=False,
        )


class RolloutFlow(Flow):
    """A ``Flow`` that shares a world across agent-local envs and records tokens.

    Three narrow overrides and no runtime change: ``build_tools`` hands every
    agent the env, ``llm_for_step``/``_drive`` record what was sampled, and
    ``resolve_child`` refuses launches past the budget.
    """

    def __init__(self, llm: Any, *, budget: Budget | None = None, **kwargs: Any) -> None:
        budget = budget or Budget()
        kwargs.setdefault("root_config", budget.agent_config())
        super().__init__(llm, **kwargs)
        self.budget = budget
        #: ``AgentStart.id -> EnvSession``.
        self.sessions: dict[str, EnvSession] = {}
        #: ``LLMOutput.id -> TurnSample``, so transcript order comes for free.
        self.samples: dict[str, TurnSample] = {}
        self.refusals: dict[str, int] = {}
        self._sample_sinks: dict[str, list[TurnSample]] = {}
        self._session_lock = threading.RLock()

    # -- Env sharing ------------------------------------------------------

    def bind(self, agent: AgentStart, session: EnvSession) -> None:
        """Give one agent its episode view of the rollout's shared world."""
        with self._session_lock:
            self.sessions[agent.id] = session

    def unbind(self, graph: Graph, root: AgentStart) -> None:
        with self._session_lock:
            for agent in reads.tree(graph, root):
                self.sessions.pop(agent.id, None)

    def ensure_session(self, graph: Graph, agent: AgentStart) -> EnvSession | None:
        """Idempotently bind restored children that bypassed ``resolve_child``."""
        with self._session_lock:
            if (session := self.sessions.get(agent.id)) is not None:
                return session
            parent = reads.launcher(graph, agent)
            if parent is None:
                return None
            parent_session = self.ensure_session(graph, parent)
            if parent_session is None:
                return None
            child_session, _observation = parent_session.fork(agent.content, dict(agent.args))
            self.sessions[agent.id] = child_session
            return child_session

    def session_for(self, node: Node) -> EnvSession | None:
        step = current_step()
        return self.ensure_session(step.graph, step.agent) if step is not None else None

    def build_tools(self, node: Node) -> dict[str, Any]:
        namespace = super().build_tools(node)
        session = self.session_for(node)
        if session is not None:
            # Bound per agent: forkable envs share world state while keeping
            # independent tasks, termination, rewards, and local budgets.
            step = current_step()
            namespace["env_step"] = env_tool(session, reads.path(step.graph, step.agent))
        return namespace

    # -- Token capture ----------------------------------------------------

    def records(self, model: str) -> bool:
        """Whether this model's client hands back the tokens it sampled."""
        return bool(getattr(self._llm_clients.get(model), "records_tokens", False))

    def llm_for_step(self, node: Node) -> PooledLLMClient:
        agent = current_step().agent
        model = agent.config.model
        if not self.records(model):
            return super().llm_for_step(node)

        sink: list[TurnSample] = []
        self._sample_sinks[agent.id] = sink
        return self._create_pooled_llm(
            model,
            request_kwargs={"sample_sink": sink},
        )

    async def _drive(self, step: Step) -> StepResult:
        transition = await super()._drive(step)
        created = transition.created
        sink = self._sample_sinks.pop(step.agent.id, [])
        if isinstance(created, LLMOutput) and sink:
            self.samples[created.id] = sink[-1]
        return transition

    def turns(self, graph: Graph, agent: AgentStart) -> list[TurnSample]:
        """This agent's sampled turns, in transcript order."""
        return [
            self.samples[node.id]
            for node in graph.nodes(LLMOutput, agent=agent)
            if node.id in self.samples
        ]

    # -- Budgets ----------------------------------------------------------

    def resolve_child(
        self,
        action: ExecAction,
        spec: dict[str, Any],
        call_id: int,
    ) -> AgentStart:
        step = current_step()
        g, parent = step.graph, step.agent
        limit = self.budget.max_children_per_agent
        if limit is not None and len(reads.children(g, parent)) >= limit:
            self.refuse("max_children_per_agent")
            raise ValueError(f"cannot launch beyond {limit} children per agent")
        total = self.budget.max_total_agents
        if total is not None and len(reads.tree(g, reads.lineage(g, parent)[0])) >= total:
            self.refuse("max_total_agents")
            raise ValueError(f"cannot launch beyond {total} agents in one rollout")
        session = self.ensure_session(g, parent)
        child_session = None
        if session is not None:
            args = dict(spec.get("args") or {})
            child_session, observation = session.fork(spec.get("task", ""), args)
            if observation is not None:
                args.setdefault(OBSERVATION_INPUT, json.dumps(observation, default=str))
            spec = {**spec, "args": args}
        child = super().resolve_child(action, spec, call_id)
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
        return agent_id(self.graph, self.root)

    @property
    def agents(self) -> list[AgentStart]:
        return agents(self.graph, self.root)

    @property
    def complete(self) -> bool:
        """Every agent terminal — not just the root."""
        return all(reads.terminal(self.graph, agent) for agent in self.agents)

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
                config=self.flow.root_config,
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
    "Budget",
    "Collector",
    "RolloutFlow",
    "RolloutTree",
    "TaskSpec",
    "TurnSample",
]
