"""Rewind and branch helpers for the Shepherd example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from rlmflow import AgentStart, Flow, LLMOutput, Node, UserQuery, tool


class Plan(TypedDict):
    """How far to rewind, and which box takes which goal in what sequence.

    ``order`` is a list like ``["B2->G4", "B3->G3 from below"]``. Nothing parses
    it — it reaches the worker as written — so the planner picks the one thing
    that makes branches differ and the worker still chooses its own route.
    """

    rewind: int
    order: list[str]


def score(env: dict) -> float:
    if env.get("solved"):
        return 1000.0 - float(env.get("pushes", 0))
    return -1000.0 - float(env.get("dist", 0))


def landed_push(output: Node | None) -> bool:
    """True when a turn's stdout reports a push that actually moved a box.

    ``Sokoban.push`` announces success with a ``pushed ...`` line, but a worker
    may print its own reasoning around it, so scan the lines rather than
    requiring the report to come first.
    """
    if output is None:
        return False
    return any(
        line.lstrip().startswith("pushed ")
        for line in (output.content or "").splitlines()
    )


def push_turns(agent: AgentStart) -> list[LLMOutput]:
    """Successful worker pushes, oldest first."""
    pushes = []
    for node in agent.transcript():
        if not isinstance(node, LLMOutput) or "push(" not in (node.code or ""):
            continue
        action = node.next
        if landed_push(action.next if action is not None else None):
            pushes.append(node)
    return pushes


def rewind_point(agent: AgentStart, rewind: int) -> tuple[Node, int]:
    pushes = push_turns(agent)
    if not pushes:
        raise ValueError("worker has no pushes to rewind")
    depth = max(1, min(int(rewind), len(pushes)))
    cut = pushes[-depth].prev
    while isinstance(cut, UserQuery):
        cut = cut.prev
    return cut, depth


def snapshot(flow: Flow, agent: AgentStart, rewind: int = 0) -> dict:
    """The agent's board as plain data, read from its REPL env.

    The game itself lives wherever its REPL runs, which is a worker process under
    every runtime but the in-process one, so ``Sokoban._publish`` posts each
    reading the host needs to ``env`` and the host never touches the object.
    """
    env = flow.runtime.repl_for(agent).env
    return {
        "rewind": rewind,
        # Ruled, because ``board`` is the only description of the walls a reader
        # gets — there is no wall list to fall back on.
        "board": env.get("grid", ""),
        "status": env.get("status", ""),
        "player": env.get("player"),
        "boxes": env.get("boxes", {}),
        "goals": env.get("goals", {}),
        "legal_pushes": env.get("legal_pushes", []),
    }


async def replayed_fork(
    flow: Flow,
    worker: AgentStart,
    rewind: int,
) -> tuple[AgentStart, int]:
    cut, depth = rewind_point(worker, rewind)
    fork = cut.fork()
    await flow.replay(fork)
    return fork, depth


@dataclass
class Branch:
    flow: Flow
    graph: AgentStart
    index: int
    rewind: int
    order: list[str]

    @property
    def name(self) -> str:
        return f"branch{self.index}"

    @property
    def env(self) -> dict:
        return self.flow.runtime.repl_for(self.graph).env

    @property
    def result(self):
        return self.graph.result()

    @property
    def solved(self) -> bool:
        return bool(self.env.get("solved"))

    @property
    def dist(self) -> int:
        return int(self.env.get("dist", 0))

    @property
    def pushes(self) -> int:
        return int(self.env.get("pushes", 0))

    @property
    def score(self) -> float:
        return score(self.env)

    def board(self) -> str:
        return self.env.get("board", "")


def recovery_tools(
    flow: Flow,
    worker: AgentStart,
    plans: list[Plan] | None = None,
    *,
    expected_plans: int | None = None,
) -> dict:
    """Values injected into the shepherd's REPL."""
    jam = snapshot(flow, worker)
    max_rewind = len(push_turns(worker))

    # Both tools run on the host, not in the shepherd's REPL: ``preview`` drives a
    # graph replay and ``branch`` records into the host's ``plans`` list, and a copy
    # shipped into a worker process would replay nothing and record into the copy.
    @tool("Inspect the worker after rewinding pushes, without playing.", proxy=True)
    async def preview(rewind: int) -> dict:
        fork, depth = await replayed_fork(flow, worker, rewind)
        try:
            return snapshot(flow, fork, depth)
        finally:
            flow.runtime.close_repl(fork)

    @tool("Record {rewind, order} recovery plans, then stop.", proxy=True)
    def branch(specs: list[Plan]) -> str:
        """Take the plans as written. What makes a *good* spread of branches is
        the prompt's job to explain, not this tool's job to enforce; only the
        fields ``prepare_branch`` cannot run without are checked."""
        if plans is None:
            raise RuntimeError("branch recording is not enabled")
        if plans:
            raise ValueError("plans were already recorded")
        if expected_plans is not None and len(specs) != expected_plans:
            raise ValueError(f"expected {expected_plans} plans, got {len(specs)}")
        normalized: list[Plan] = []
        for spec in specs:
            rewind = int(spec["rewind"])
            if not 1 <= rewind <= max_rewind:
                raise ValueError(f"rewind must be between 1 and {max_rewind}")
            order = [str(step).strip() for step in spec["order"] if str(step).strip()]
            if not order:
                raise ValueError("order needs at least one B<n>->G<n> step")
            normalized.append({"rewind": rewind, "order": order})
        plans.extend(normalized)
        return f"recorded {len(plans)} plans; stop this block"

    tools = {
        "jam": jam,
        "max_rewind": max_rewind,
        "preview": preview,
    }
    if plans is not None:
        tools["branch"] = branch
    return tools


async def prepare_branch(
    flow: Flow,
    worker: AgentStart,
    plan: Plan,
    index: int,
    budget: int,
) -> Branch:
    fork, depth = await replayed_fork(flow, worker, plan["rewind"])
    state = snapshot(flow, fork, depth)
    order = plan["order"]
    order_text = ", then ".join(order)
    fork.config.max_iters = fork.llm_turns() + budget + 3
    fork.config.name = f"branch{index}"
    fork.config.inputs.pop("_worker_strategy", None)
    fork.config.inputs["_shepherd_order"] = order_text
    fork.frontier.append(
        UserQuery(
            content=(
                f"Rewound {depth} pushes. This is a fresh recovery attempt, so "
                "abandon the plan you were following.\n\n"
                f"{state['status']}\n\n"
                f"Lock the boxes in this order: {order_text}. The routes are "
                "yours to work out."
            )
        )
    )
    branch = Branch(flow, fork, index, depth, order)
    flow.runtime.close_repl(fork)
    return branch


__all__ = [
    "Branch",
    "landed_push",
    "prepare_branch",
    "recovery_tools",
    "rewind_point",
    "score",
    "snapshot",
]
