"""Shared environment contract and the rlmflow runtime binding."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from benchmarks.eval.environments.environment import Environment
from benchmarks.eval.environments.episode import EpisodeCore
from benchmarks.eval.environments.episode import (
    EpisodeEvent as EnvironmentEvent,
)
from benchmarks.eval.environments.episode import (
    EpisodeOwner as EnvironmentInstance,
)
from rlmflow import Flow
from rlmflow.graph.nodes import AgentStart, ExecAction, Node
from rlmflow.runtime.repl_client import current_rpc_call_id
from rlmflow.tools import tool


def candidate_protocol_message(*, is_root: bool, selected: bool) -> str:
    if selected:
        return "Selection recorded. The root may now finish."
    if is_root:
        return (
            "PROTOCOL REQUIRED BEFORE finish(): call submit_candidate(...), then "
            "select_candidate(candidate_id). Merely naming an ID in the final answer "
            "does not count."
        )
    return (
        "Before finish(), call submit_candidate(...) for this branch and return its "
        "candidate_id so the root can execute select_candidate(candidate_id)."
    )


class EnvironmentEpisode:
    """Bind graph agents to the runtime-neutral episode state machine."""

    def __init__(
        self,
        adapter: Environment,
        *,
        max_actions: int,
        episode_id: str = "",
        candidate_budget: int = 0,
        expected_observation_sha256: str | None = None,
    ) -> None:
        self.core = EpisodeCore(
            adapter,
            max_actions=max_actions,
            episode_id=episode_id,
            candidate_budget=candidate_budget,
            expected_observation_sha256=expected_observation_sha256,
        )
        self._receipts: dict[tuple[str, int, str], dict[str, Any] | Exception] = {}

    @property
    def adapter(self) -> Environment:
        return self.core.adapter

    @property
    def candidates(self):
        return self.core.candidates

    @property
    def max_actions(self) -> int:
        return self.core.max_actions

    @property
    def events(self) -> list[EnvironmentEvent]:
        return self.core.events

    @property
    def accepted_action_count(self) -> int:
        return self.core.accepted_action_count

    @property
    def actions_remaining(self) -> int:
        return self.core.actions_remaining

    @property
    def initial_observation(self) -> dict[str, Any]:
        return self.core.initial_observation

    @property
    def primary_instance(self) -> EnvironmentInstance:
        return self.core.root_owner

    @property
    def scored_instance(self) -> EnvironmentInstance:
        instance_id = self.core.scored_instance_id
        if instance_id == self.core.root_owner.instance_id:
            return self.core.root_owner
        selected = self.candidates.selected
        return EnvironmentInstance(
            id=selected.owner_id,
            path=selected.owner_path,
            instance_id=instance_id,
            parent_instance_id=None,
            depth=selected.owner_depth,
            primary=False,
            created_at=0.0,
            fork_action_count=len(selected.environment_prefix),
        )

    def bind_root(self, root: AgentStart) -> EnvironmentInstance:
        return self.core.start(root.id, path=root.config.path, depth=root.config.depth)

    def fork_child(
        self,
        action: ExecAction,
        child: AgentStart,
        launch_call_id: int,
    ) -> EnvironmentInstance:
        parent = action.parent_agent
        if parent is None:
            raise RuntimeError("cannot fork an environment from a detached launch")
        if parent.id not in self.core.owners:
            self.bind_root(parent)
        prefix = self.core.prefix_at(parent.id, action.id, launch_call_id)
        return self.core.fork(
            child.id,
            parent_owner_id=parent.id,
            path=child.config.path,
            depth=child.config.depth,
            exec_action_id=action.id,
            call_id=launch_call_id,
            prefix=prefix,
        )

    def perform(
        self,
        node: Node,
        operation: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(node, ExecAction):
            raise TypeError("environment tools require an ExecAction")
        if not operation:
            raise ValueError("environment operation must be non-empty")
        call_id = current_rpc_call_id()
        receipt_key = (node.id, call_id, operation)
        previous = self._receipts.get(receipt_key)
        if isinstance(previous, Exception):
            raise RuntimeError(str(previous))
        if isinstance(previous, dict):
            return dict(previous)
        agent = node.parent_agent
        owner = self._ensure_owner(agent)
        try:
            result = self.core.perform(
                owner.id,
                operation,
                payload,
                exec_action_id=node.id,
                call_id=call_id,
            )
            if self.candidates is not None:
                result["_candidate_protocol"] = candidate_protocol_message(
                    is_root=owner.primary,
                    selected=self.candidates.selected is not None,
                )
            self._receipts[receipt_key] = dict(result)
            return result
        except Exception as exc:
            self._receipts[receipt_key] = exc
            raise

    def metrics(self) -> dict[str, Any]:
        return self.core.metrics()

    def artifacts(self) -> dict[str, str]:
        return self.core.artifacts()

    def close(self) -> None:
        self.core.close()

    def candidate_tools_for(self, node: Node) -> dict[str, Any]:
        if self.candidates is None:
            return {}

        @tool("Evaluate and register the current branch as a named candidate.", proxy=True)
        def submit_candidate(
            label: str,
            proposal: Any = None,
            parent_candidate_id: str | None = None,
        ) -> dict[str, Any]:
            if not isinstance(node, ExecAction):
                raise TypeError("candidate tools require an ExecAction")
            owner = self._ensure_owner(node.parent_agent)
            record = self.core.submit_candidate(
                owner.id,
                label=label,
                proposal=proposal,
                parent_candidate_id=parent_candidate_id,
            )
            return {
                **record.to_dict(public=True),
                "_candidate_protocol": candidate_protocol_message(
                    is_root=owner.primary,
                    selected=False,
                ),
            }

        @tool("List model-visible summaries for all evaluated candidates.", proxy=True)
        def list_candidates() -> list[dict[str, Any]]:
            return self.candidates.public_records()

        @tool("Root only: choose the candidate used for native scoring.", proxy=True)
        def select_candidate(candidate_id: str) -> dict[str, Any]:
            if not isinstance(node, ExecAction):
                raise TypeError("candidate tools require an ExecAction")
            owner = self._ensure_owner(node.parent_agent)
            record = self.core.select_candidate(candidate_id, owner_id=owner.id)
            return {
                "selected_candidate_id": record.id,
                "label": record.label,
                "public_summary": dict(record.public_summary),
            }

        return {
            "submit_candidate": submit_candidate,
            "list_candidates": list_candidates,
            "select_candidate": select_candidate,
        }

    def _ensure_owner(self, agent: AgentStart | None) -> EnvironmentInstance:
        if agent is None:
            raise RuntimeError("environment tool belongs to no graph agent")
        existing = self.core.owners.get(agent.id)
        if existing is not None:
            return existing
        if agent is agent.root:
            return self.bind_root(agent)
        if not isinstance(agent.parent, ExecAction):
            raise RuntimeError("environment child has no launch action")
        call_id = agent.parent.launch_call_id_for(agent.id)
        if call_id is None:
            raise RuntimeError(f"agent {agent.config.path!r} has no launch receipt")
        return self.fork_child(agent.parent, agent, call_id)


class EnvironmentFlow(Flow):
    """Flow binding graph ownership to a shared environment episode."""

    def __init__(self, *args: Any, environment: EnvironmentEpisode, **kwargs: Any) -> None:
        self.environment = environment
        super().__init__(*args, **kwargs)

    def start(self, query: str = "", **overrides: Any) -> AgentStart:
        root = super().start(query, **overrides)
        self.environment.bind_root(root)
        root.args["environment_observation"] = json.dumps(
            self.environment.initial_observation,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return root

    def build_tools(self, node: Node) -> dict[str, Any]:
        from benchmarks.eval.environments.registry import bind_tools

        try:
            environment_tools = bind_tools(
                self.environment.adapter.kind,
                lambda operation, payload=None: self.environment.perform(
                    node, operation, dict(payload or {})
                ),
                proxy=True,
            )
        except ValueError:
            factory = getattr(self.environment.adapter, "tools_for", None)
            if factory is None:
                raise
            environment_tools = factory(self.environment, node)
        return {
            **super().build_tools(node),
            **environment_tools,
            **self.environment.candidate_tools_for(node),
        }

    def resolve_child(
        self,
        action: ExecAction,
        spec: dict[str, Any],
        call_id: int,
    ) -> AgentStart:
        child = super().resolve_child(action, spec, call_id)
        self.environment.fork_child(action, child, call_id)
        return child

    async def run_streaming(
        self,
        root: AgentStart | str,
        *roots: AgentStart | str,
        **kwargs: Any,
    ) -> AsyncIterator[Node]:
        agents = [
            self.start(value) if isinstance(value, str) else value for value in (root, *roots)
        ]
        for agent in agents:
            self.environment.bind_root(agent)
            agent.args["environment_observation"] = json.dumps(
                self.environment.initial_observation,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        async for node in super().run_streaming(*agents, **kwargs):
            yield node

    async def aclose(self) -> None:
        try:
            await super().aclose()
        finally:
            self.environment.close()

__all__ = [
    "EnvironmentEpisode",
    "EnvironmentEvent",
    "EnvironmentFlow",
    "EnvironmentInstance",
    "candidate_protocol_message",
]
