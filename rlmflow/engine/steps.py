"""What one step does: the transition mechanism and the default transitions.

Each transition is one decorated function per edge, resolved first-match;
semantic transitions are chosen once from a bounded graph read-set.
``DEFAULT_TRANSITIONS`` order is the policy: application transitions first, then
these, and the first match wins.

```text
AgentStart|UserQuery|ErrorOutput|ExecOutput + out_of_room      → to_final           → FinalQuery
AgentStart|UserQuery|ErrorOutput|ExecOutput + needs_truncation → to_summary         → TruncationSummary
VerificationFailure                                            → to_repair          → RepairQuery
AgentStart|UserQuery|ErrorOutput|ExecOutput                    → complete           → LLMOutput | DoneOutput
LLMOutput + one non-empty repl block                           → to_action          → ExecAction
LLMOutput + non-empty, no repl fence                           → to_text_reply      → DoneOutput | VerificationFailure
LLMOutput + invalid repl protocol                              → to_missing_action  → ExecAction(raise MissingReplError)
ExecAction                                                     → run_repl           → output node
```

Protocol violations never leave this loop. An invalid reply becomes an
engine-written ``ExecAction`` that raises, and a final action that runs without
``submit`` becomes an ``ErrorOutput``; both recover like any failing code.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from rlmflow.graph import ops, reads
from rlmflow.graph.config import NATIVE_BACKEND
from rlmflow.graph.nodes import (
    COLD_REPL_NOTE,
    AgentStart,
    BackendRun,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    FinalQuery,
    LLMOutput,
    LLMUsage,
    Node,
    RepairQuery,
    ReplDead,
    TransitionDecision,
    TruncationSummary,
    TurnMode,
    UserQuery,
    VerificationFailure,
    current_step,
)
from rlmflow.llm import join_chunks
from rlmflow.runtime.repl import ReplRun, ReplStatus
from rlmflow.tools.builtins import node_for_candidate
from rlmflow.utils import find_repl_blocks, has_repl_fence, repl_code_block, truncate_output

if TYPE_CHECKING:
    from rlmflow.flow import Flow

Condition = Callable[[Node], "bool | Awaitable[bool]"] | str
SemanticChoice = tuple[tuple[str, str, str], ...]
Judge = Callable[[Node, SemanticChoice], Awaitable[TransitionDecision]]


class TransitionPolicyError(ValueError):
    """Invalid transition policy declared by the host application."""


@dataclass(frozen=True, slots=True)
class TransitionFunction:
    """A producer carrying its complete automatic-transition declaration."""

    fn: Callable[..., Any]
    sources: tuple[type[Node], ...] | None
    when: Condition | None = None
    node_only: bool = False

    @classmethod
    def create(
        cls,
        fn: Callable[..., Any],
        *,
        source: type[Node] | tuple[type[Node], ...] | str,
        when: Condition | None = None,
        node_only: bool = False,
    ) -> TransitionFunction:
        _validate_condition(when)
        return cls(fn, _normalize_sources(source), when, node_only)

    @property
    def __name__(self) -> str:
        return self.fn.__name__

    @property
    def id(self) -> str:
        source = (
            "all"
            if self.sources is None
            else ",".join(f"{item.__module__}.{item.__qualname__}" for item in self.sources)
        )
        condition = (
            self.when
            if isinstance(self.when, str) or self.when is None
            else f"{self.when.__module__}.{self.when.__qualname__}"
        )
        fingerprint = json.dumps(
            {
                "condition": condition,
                "source": source,
                "handler": f"{self.fn.__module__}.{self.fn.__qualname__}",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(fingerprint.encode()).hexdigest()

    def accepts(self, node: Node) -> bool:
        return self.sources is None or isinstance(node, self.sources)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    async def produce(self, flow: Any, node: Node) -> Node:
        created = self.fn(node) if self.node_only else self.fn(flow, node)
        if inspect.isawaitable(created):
            created = await created
        if not isinstance(created, Node):
            raise TypeError(
                f"transition handler {self.__name__} returned "
                f"{type(created).__name__}, expected Node"
            )
        return created


def transition(
    *,
    source: type[Node] | tuple[type[Node], ...] | str,
    when: Condition | None = None,
):
    """Wrap one built-in producer with its automatic-transition declaration."""

    def decorate(fn: Callable[..., Any]) -> TransitionFunction:
        return TransitionFunction.create(fn, source=source, when=when)

    return decorate


async def resolve_transition(
    transitions: Iterable[TransitionFunction],
    node: Node,
    judge: Judge,
) -> tuple[TransitionFunction, Node, Node]:
    """Return ``(function, anchor, source)`` for the selected transition.

    Callable conditions before the first semantic condition retain first-match
    priority. At the first string condition, all source-compatible string
    conditions before the next unconditional fallback become one structured
    choice. A selected semantic edge wins there; ``continue`` resumes deterministic
    first-match resolution after that choice. The one durable decision is reused
    on replay.
    """
    step = current_step()
    items = tuple(transitions)
    recorded: TransitionDecision | None = None
    source = node
    while isinstance(source, TransitionDecision):
        if recorded is not None:
            raise ValueError("expected one transition decision per source node")
        recorded = source
        source = reads.prev(step.graph, source)
    if recorded is not None and recorded.source_node_id != source.id:
        raise ValueError("transition decision chain has inconsistent source nodes")

    anchor = node
    semantic_resolved = False
    for index, transition_fn in enumerate(items):
        if not transition_fn.accepts(source):
            continue
        when = transition_fn.when
        if isinstance(when, str):
            if semantic_resolved:
                continue
            semantic = _semantic_choices(items[index:], source)
            if recorded is not None:
                decision = recorded
                policy = tuple((rule_id, condition) for rule_id, _label, condition in semantic)
                if tuple(decision.candidates.items()) != policy:
                    raise ValueError("recorded transition decision no longer matches policy")
            else:
                decision = await judge(source, semantic)
                ops.append(step.graph, step.agent, decision, after=anchor)
                anchor = decision
            semantic_resolved = True
            if decision.selected_rule_id is not None:
                selected = next(
                    (
                        candidate
                        for candidate in items
                        if candidate.id == decision.selected_rule_id
                        and candidate.accepts(source)
                        and isinstance(candidate.when, str)
                    ),
                    None,
                )
                if selected is None:
                    raise ValueError("semantic transition selected an unknown rule")
                return selected, anchor, source
            continue
        elif when is None:
            return transition_fn, anchor, source
        else:
            outcome = when(source)
            if bool(await outcome if inspect.isawaitable(outcome) else outcome):
                return transition_fn, anchor, source
    raise TypeError(f"cannot step {type(source).__name__}")


def _semantic_choices(
    transitions: tuple[TransitionFunction, ...],
    source: Node,
) -> SemanticChoice:
    """Semantic choices reachable before the next unconditional fallback."""
    choices: list[tuple[str, str, str]] = []
    for transition_fn in transitions:
        if not transition_fn.accepts(source):
            continue
        when = transition_fn.when
        if when is None:
            break
        if isinstance(when, str):
            choices.append((transition_fn.id, transition_fn.__name__, when))
    if not choices:
        raise ValueError("semantic transition choice requires at least one candidate")
    if len({rule_id for rule_id, _label, _condition in choices}) != len(choices):
        raise TransitionPolicyError("semantic transitions must have unique rule IDs")
    if len({label for _rule_id, label, _condition in choices}) != len(choices):
        raise TransitionPolicyError(
            "semantic transitions in one choice must have unique function names"
        )
    return tuple(choices)


def _validate_condition(when: Any) -> None:
    if isinstance(when, str) and not when.strip():
        raise TransitionPolicyError("semantic transition condition cannot be empty")
    if when is not None and not isinstance(when, str) and not callable(when):
        raise TransitionPolicyError("when must be a callable, string, or None")


def _normalize_sources(
    source: type[Node] | tuple[type[Node], ...] | str,
) -> tuple[type[Node], ...] | None:
    if source == "all":
        return None
    sources = source if isinstance(source, tuple) else (source,)
    if not sources or any(
        not isinstance(item, type) or not issubclass(item, Node) for item in sources
    ):
        raise TransitionPolicyError("source must be a Node class, tuple of Node classes, or 'all'")
    return sources


DEFAULT_READ_SET = 12

CONTINUE = "continue"


async def choose_transition(
    flow: Any,
    node: Node,
    candidates: tuple[tuple[str, str, str], ...],
) -> TransitionDecision:
    """Ask ``flow``'s structured engine to select one eligible edge or none."""
    if not candidates:
        raise ValueError("semantic transition choice requires at least one candidate")
    candidate_map = {rule_id: condition for rule_id, _label, condition in candidates}
    labels = {label: rule_id for rule_id, label, _condition in candidates}
    if len(candidate_map) != len(candidates):
        raise ValueError("semantic transition candidates must have unique rule IDs")
    if len(labels) != len(candidates):
        raise ValueError("semantic transition candidates must have unique labels")

    step = current_step()
    limit = step.agent.config.keep_n_messages or DEFAULT_READ_SET
    read_nodes = reads.history(step.graph, node)[-limit:]
    state = [{"id": item.id, "type": item.type, "content": item.content} for item in read_nodes]
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":"))
    schema: dict[str, Any] = {
        "type": "string",
        "enum": [*labels, CONTINUE],
    }
    options = "\n".join(f"- `{label}`: {condition}" for _rule_id, label, condition in candidates)
    instruction = (
        "Choose exactly one transition whose condition is satisfied by the graph "
        f"nodes below, or `{CONTINUE}` to resume deterministic transition "
        "resolution. Return only that transition label. When uncertain, choose "
        f"`{CONTINUE}`.\n\nTransitions:\n"
        f"{options}"
    )
    started_at = time.time()
    result = await flow.structured_query(
        {
            "read_set": state,
            "candidates": [
                {"transition": label, "condition": condition}
                for _rule_id, label, condition in candidates
            ],
        },
        output_schema=schema,
        instruction=instruction,
    )
    value = result.value
    if not isinstance(value, str):
        raise TypeError("semantic transition selector must return a string")
    selected = None if value == CONTINUE else labels[value]
    usage = result.metadata.get("usage", {})
    usage = usage if isinstance(usage, dict) else {}
    return TransitionDecision(
        candidates=candidate_map,
        selected_rule_id=selected,
        source_node_id=node.id,
        read_set_node_ids=[item.id for item in read_nodes],
        read_set_hash=hashlib.sha256(encoded.encode()).hexdigest(),
        model=str(result.metadata.get("model", "default")),
        prompt_hash=hashlib.sha256(instruction.encode()).hexdigest(),
        engine_metadata=result.metadata,
        usage=LLMUsage(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            reasoning_tokens=int(usage.get("reasoning_tokens", 0) or 0),
        ),
        started_at=started_at,
        finished_at=time.time(),
    )


MAX_ITERS_EXCEEDED = "[max_iters exceeded]"
MAX_BUDGET_EXCEEDED = "[max_budget exceeded]"
MISSING_REPL_ERROR = "MissingReplError"
MISSING_SUBMIT_ERROR = "MissingSubmitError"


def _agent() -> AgentStart:
    return current_step().agent


def _history() -> list[Node]:
    step = current_step()
    return reads.history(step.graph, step.node)


def llm_turns() -> int:
    """Model turns in the stepping agent's history."""
    return sum(isinstance(item, LLMOutput) for item in _history())


def turn_budget() -> tuple[int | None, int]:
    """The stepping agent's turn limit, and the turns used against it.

    An open ``ask_agent`` with ``max_iters=`` counts only the turns since it.
    """
    from rlmflow.graph.nodes import ASK_KEY

    history = _history()
    limit = _agent().config.max_iters
    start = 0
    for index in range(len(history) - 1, -1, -1):
        item = history[index]
        if isinstance(item, DoneOutput):
            break
        if isinstance(item, UserQuery) and "max_iters" in item.metadata.get(ASK_KEY, {}):
            limit, start = item.metadata[ASK_KEY]["max_iters"], index
            break
    return limit, sum(isinstance(item, LLMOutput) for item in history[start:])


def final_turn_spent() -> bool:
    """Whether the stepping agent already used its final turn without finishing."""
    for item in reversed(_history()[:-1]):
        if isinstance(item, DoneOutput):
            return False
        if isinstance(item, FinalQuery):
            return True
    return False


class AgentBackend(Protocol):
    """Run one agent outside rlmflow's model-and-REPL loop.

    Optional hooks: ``bind(flow)`` when the Flow is built, ``aclose()`` when it
    closes.
    """

    async def run(
        self,
        agent: AgentStart,
        action: BackendRun,
        *,
        resume: bool,
    ) -> Any: ...


def check_backends(backends: Mapping[str, AgentBackend] | None) -> dict[str, Any]:
    """Validate a ``{name: backend}`` table for ``Flow(backends=...)``."""
    checked = dict(backends or {})
    for name, backend in checked.items():
        if not isinstance(name, str) or not name or name == NATIVE_BACKEND:
            raise ValueError(
                f"backend name must be a non-empty string other than {NATIVE_BACKEND!r}"
            )
        if not callable(getattr(backend, "run", None)):
            raise TypeError(f"backend {name!r} must implement AgentBackend.run")
    return checked


def uses_external_backend(node: Node) -> bool:
    return _agent().config.backend != NATIVE_BACKEND


@transition(source=(AgentStart, UserQuery), when=uses_external_backend)
async def to_backend(flow: Flow, node: Node) -> Node:
    """Open one durable external-agent invocation."""
    agent = _agent()
    flow.backend_for(agent)
    limit = agent.config.max_iters
    attempts = sum(isinstance(item, BackendRun) for item in _history())
    if limit is not None and attempts >= limit:
        return DoneOutput(result=MAX_ITERS_EXCEEDED)
    return BackendRun(backend=agent.config.backend)


@transition(source=BackendRun)
async def run_backend(flow: Flow, node: BackendRun) -> Node:
    """Run or resume one host-configured external agent."""
    agent = _agent()
    if node.backend != agent.config.backend:
        raise RuntimeError(
            f"backend run requires backend {node.backend!r}, not {agent.config.backend!r}"
        )
    backend = flow.backend_for(agent)
    resume = node.started
    node.started = True
    candidate = await backend.run(agent, node, resume=resume)
    return node_for_candidate(flow, agent, candidate)


def at_final(agent: AgentStart) -> bool:
    limit, used = turn_budget()
    return limit is not None and used == limit - 1


def budget_nearly_spent(node: Node) -> bool:
    """Whether one more average turn would exhaust the token budget."""
    step = current_step()
    g, agent = step.graph, step.agent
    limit = agent.config.max_budget
    if limit is None:
        return False
    run = reads.tree(g, reads.lineage(g, agent)[0])
    turns = sum(isinstance(n, LLMOutput) for a in run for n in g.nodes(agent=a.id))
    if turns == 0:
        return False
    spent = reads.usage(g, run).total
    return spent + spent / turns >= limit


def out_of_room(node: Node) -> bool:
    return not isinstance(node, FinalQuery) and (at_final(_agent()) or budget_nearly_spent(node))


def needs_truncation(node: Node) -> bool:
    if isinstance(node, (FinalQuery, TruncationSummary)):
        return False
    step = current_step()
    keep = step.agent.config.keep_n_messages
    if keep is None:
        return False
    keep = max(keep, 1)
    return len(reads.project(step.graph, node, keep=keep + 1)) > keep


@transition(
    source=(AgentStart, UserQuery, ErrorOutput, ExecOutput),
    when=out_of_room,
)
async def to_final(flow: Flow, node: Node) -> Node:
    if final_turn_spent():
        limit, used = turn_budget()
        exhausted = limit is not None and used >= limit
        return DoneOutput(result=MAX_ITERS_EXCEEDED if exhausted else MAX_BUDGET_EXCEEDED)
    return FinalQuery()


@transition(
    source=(AgentStart, UserQuery, ErrorOutput, ExecOutput),
    when=needs_truncation,
)
async def to_summary(flow: Flow, node: Node) -> TruncationSummary:
    return TruncationSummary()


@transition(source=VerificationFailure)
async def to_repair(flow: Flow, node: VerificationFailure) -> RepairQuery:
    return RepairQuery(
        content=(
            "Your previous submitted candidate was rejected. Repair it and call "
            f"submit(...) again.\n\n{node.feedback}"
        )
    )


@transition(source=(AgentStart, UserQuery, ErrorOutput, ExecOutput))
async def complete(flow: Flow, node: Node) -> Node:
    agent = _agent()
    limit, used = turn_budget()
    if limit is not None and used >= limit:
        return DoneOutput(result=MAX_ITERS_EXCEEDED)
    messages = flow.build_messages(node)
    reply, usage = await join_chunks(flow.llm_for_step(node).stream(messages))
    model = agent.config.model
    return LLMOutput(
        content=reply,
        code=repl_code_block(reply),
        usage=usage,
        prompt_id=agent.record_prompt(messages[0]["content"]),
        metadata={
            "llm": {
                "model": model,
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "reasoning_tokens": usage.reasoning_tokens,
                },
            }
        },
    )


def has_repl_action(node: LLMOutput) -> bool:
    """Whether the model explicitly chose the executable REPL path."""
    return bool(node.code)


@transition(source=LLMOutput, when=has_repl_action)
async def to_action(flow: Flow, node: LLMOutput) -> ExecAction:
    source = reads.prev(current_step().graph, node)
    return ExecAction(
        code=node.code,
        turn_mode=source.turn_mode if source is not None else TurnMode.NONE,
    )


def protocol_error(reply: str) -> str:
    """Name what makes ``reply`` an invalid action."""
    blocks = find_repl_blocks(reply)
    if len(blocks) > 1:
        return f"found {len(blocks)} ```repl blocks; send exactly one per turn"
    if blocks:
        return "the ```repl block is empty"
    if has_repl_fence(reply):
        return "the ```repl block was never closed (the reply may have been cut off)"
    return "the reply is empty"


def raise_action(name: str, message: str) -> str:
    """REPL code that raises ``name: message`` without binding a new name."""
    return f"raise type({name!r}, (Exception,), {{}})({message!r})"


def missing_repl_message(reason: str) -> str:
    return (
        f"{reason}. Nothing ran. To act, reply with exactly one non-empty ```repl "
        "block. To finish, call submit(value) in that block, or reply in plain "
        "text with no ```repl block."
    )


def is_text_reply(node: LLMOutput) -> bool:
    """A non-empty reply that never opened a ``repl`` fence is a final answer."""
    return bool(node.content.strip()) and not has_repl_fence(node.content)


@transition(source=LLMOutput, when=is_text_reply)
async def to_text_reply(flow: Flow, node: LLMOutput) -> Node:
    return node_for_candidate(flow, _agent(), node.content.strip())


# An invalid reply becomes an engine-written action that raises, so recovery runs
# through the same ExecAction -> ErrorOutput loop as any failing code.
@transition(source=LLMOutput)
async def to_missing_action(flow: Flow, node: LLMOutput) -> ExecAction:
    source = reads.prev(current_step().graph, node)
    reason = protocol_error(node.content)
    return ExecAction(
        code=raise_action(MISSING_REPL_ERROR, missing_repl_message(reason)),
        turn_mode=source.turn_mode if source is not None else TurnMode.NONE,
        metadata={"protocol_error": reason},
    )


@transition(source=ExecAction)
async def run_repl(flow: Flow, node: ExecAction) -> Node:
    return node_for_run(node, await flow.wrapped_runtime.execute(node))


def node_for_run(
    node: ExecAction,
    run: ReplRun,
) -> Node:
    output = truncate_output(run.output, _agent().config.max_output_length)
    if run.status is ReplStatus.DONE:
        acceptance = run.acceptance or {}
        return DoneOutput(
            content=output,
            result=run.answer,
            schema_hash=acceptance.get("schema_hash"),
            verifier_hash=acceptance.get("verifier_hash"),
            verified=bool(acceptance.get("verified", False)),
        )
    if run.status is ReplStatus.REJECTED:
        failure = run.verification
        if failure is None:
            raise RuntimeError("rejected REPL run has no verification detail")
        return VerificationFailure(
            content=failure.feedback,
            candidate=failure.candidate,
            kind=failure.kind,
            feedback=failure.feedback,
            schema_hash=failure.schema_hash,
            verifier_hash=failure.verifier_hash,
            observations=output,
        )
    if run.status is ReplStatus.OK:
        if node.turn_mode is TurnMode.FINAL:
            missing = f"{MISSING_SUBMIT_ERROR}: the final turn must call submit(answer)"
            return ErrorOutput(
                content=f"{output}\n{missing}" if output else missing,
                metadata={"protocol_error": "final action did not call submit"},
            )
        return ExecOutput(content=output or "(no output)")
    if run.status is ReplStatus.ERROR:
        return ErrorOutput(content=output)
    if run.status is ReplStatus.DEAD:
        content = f"{output}\n{COLD_REPL_NOTE}" if output else COLD_REPL_NOTE
        return ReplDead(content=content)
    raise ValueError(f"unknown repl status {run.status!r}")


DEFAULT_TRANSITIONS = (
    to_backend,
    run_backend,
    to_final,
    to_summary,
    to_repair,
    complete,
    to_action,
    to_text_reply,
    to_missing_action,
    run_repl,
)


__all__ = [
    "AgentBackend",
    "DEFAULT_TRANSITIONS",
    "TransitionFunction",
    "TransitionPolicyError",
    "check_backends",
    "choose_transition",
    "llm_turns",
    "resolve_transition",
    "transition",
    "turn_budget",
]
