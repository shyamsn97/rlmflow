"""Base Flow contract.

This module is intentionally small and import-safe: it defines the public /
overridable engine surface without importing the concrete :mod:`rflow.flow`
implementation. Prompt sections, built-in tool factories, and extensions should
type against :class:`BaseFlow` instead of the concrete ``Flow`` class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from rflow.clients.llm import LLMClient, LLMUsage
from rflow.graph import ChildHandle, Graph, LLMOutput
from rflow.graph.actions import Action
from rflow.runtime.context import EngineContext
from rflow.runtime.runtime import ReplBackend


class BaseOutputParser(ABC):
    """Structured-output parser surface used by ``BaseFlow``."""

    @abstractmethod
    def __call__(self, content: str, schema: Any) -> Any:
        """Parse and validate ``content`` against ``schema``."""

    @abstractmethod
    def system_prompt_hint(self, schema: Any) -> str:
        """Render the schema hint shown in a system prompt."""


class BaseFlow(ABC):
    """Abstract base for Flow implementations and supported override hooks."""

    max_depth: int
    max_iters: int | None
    child_max_iters: int | None
    max_output_length: int
    max_messages: int | None
    show_vars: bool
    enable_structured_output: bool
    output_parser: BaseOutputParser
    repls: dict[str, ReplBackend]

    @property
    @abstractmethod
    def llm_clients(self) -> Mapping[str, LLMClient]:
        """Named LLM clients available for model routing."""

    # ── user-facing lifecycle ──────────────────────────────────────────

    @abstractmethod
    def run(
        self,
        query: str,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Any | None = None,
    ) -> str:
        """Start a run and step it to completion."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        """LLMClient-compatible chat entrypoint."""

    @abstractmethod
    def completion(
        self, messages: list[dict[str, str]], *args, **kwargs
    ) -> tuple[str, LLMUsage]:
        """LLMClient-compatible completion entrypoint."""

    @abstractmethod
    def start(
        self,
        query: str,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Any | None = None,
    ) -> Graph:
        """Begin a new run and return the root graph."""

    @abstractmethod
    def step(
        self,
        override_graph: Graph | None = None,
        query: str | None = None,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Any | None = None,
    ) -> Graph:
        """Advance a run by one engine tick and return the graph."""

    @abstractmethod
    def set_graph(self, graph: Graph) -> Graph:
        """Set the current run graph without advancing it."""

    @abstractmethod
    def terminate(self, agent_ids: Iterable[str] | None = None) -> Graph:
        """Request forced final answers for the named agents."""

    @abstractmethod
    def close(self) -> None:
        """Release runtime resources."""

    # ── prompt and message hooks ───────────────────────────────────────

    @abstractmethod
    def build_system_prompt(self, graph: Graph) -> str:
        """Render the system prompt for one agent."""

    @abstractmethod
    def build_messages(
        self, graph: Graph, *, force_final: bool = False
    ) -> list[dict[str, str]]:
        """Render LLM messages for one agent turn."""

    @abstractmethod
    def first_prompt(
        self,
        query: str,
        inputs: dict[str, str] | None = None,
        *,
        depth: int = 0,
    ) -> str:
        """Render the first user prompt for an agent."""

    @abstractmethod
    def format_exec_output(self, output: str) -> str:
        """Format REPL stdout for the next user turn."""

    @abstractmethod
    def no_code_block_message(self) -> str:
        """Error text when the LLM omits a repl block."""

    # ── tool / REPL hooks ──────────────────────────────────────────────

    @abstractmethod
    def build_tools(self, engine_context: EngineContext | None = None) -> dict:
        """Build the REPL tool namespace."""

    @abstractmethod
    def repl_for(self, agent: Graph) -> ReplBackend:
        """Return or create the REPL backend for an agent."""

    @abstractmethod
    def make_repl(self, agent: Graph) -> ReplBackend:
        """Create a new REPL backend for an agent."""

    @abstractmethod
    def seed_agent_context(self, repl: ReplBackend, agent: Graph) -> None:
        """Seed per-agent trusted context into a REPL backend."""

    @abstractmethod
    def spawn_child(
        self,
        parent_agent_id: str,
        name: str,
        query: str,
        inputs: dict[str, str] | None = None,
        model: str = "default",
        output_schema: Any | None = None,
        *,
        strict_name: bool = False,
    ) -> ChildHandle | str:
        """Spawn a recursive child agent."""

    @abstractmethod
    def llm_query_batched(
        self,
        prompts: list[str],
        *,
        model: str = "default",
        output_schema: Any | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list:
        """Run parallel one-shot LLM calls for an agent."""

    # ── execution hooks ────────────────────────────────────────────────

    @abstractmethod
    def validate_code(self, code: str) -> str | None:
        """Return an error string when a code block should not run."""

    @abstractmethod
    def truncate_output(self, output: str) -> str:
        """Truncate or transform REPL output."""

    @abstractmethod
    def record_observation(
        self,
        agent: Graph,
        repl: ReplBackend,
        payload: object,
    ) -> None:
        """Record a code observation or error on an agent."""

    @abstractmethod
    def record_usage(self, usage: LLMUsage) -> None:
        """Record LLM token usage."""

    @abstractmethod
    def llm_client_for(self, agent: Graph, *, model: str | None = None) -> LLMClient:
        """Select the LLM client for an agent/model."""

    # ── graph action hooks ─────────────────────────────────────────────

    @abstractmethod
    def act(self, action: Action) -> None:
        """Execute one graph action."""

    @abstractmethod
    def step_llm(self, agent: Graph, *, force_final: bool) -> None:
        """Execute a CallLLM action."""

    @abstractmethod
    def step_exec(self, agent: Graph, llm_output: LLMOutput) -> None:
        """Execute an Exec action."""


__all__ = ["BaseFlow", "BaseOutputParser"]
