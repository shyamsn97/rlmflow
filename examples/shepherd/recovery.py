"""Rewind and branch helpers for the Shepherd example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from rlmflow import CODE, DONE, LLM_OUTPUT, MESSAGE, Flow, Graph, Node, tool


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
        line.lstrip().startswith("pushed ") for line in (output.get("output") or "").splitlines()
    )


def push_turns(g: Graph, agent: Node) -> list[Node]:
    """Successful worker pushes, oldest first.

    Judged by what the turn's output reports rather than what its code called: under
    ``--simple-moves`` a shove is spelled ``move(...)``, so reading the code would
    find no pushes at all and leave the shepherd nothing to rewind.
    """
    history = g.history(agent)
    pushes = []
    for index, node in enumerate(history):
        if node.type != LLM_OUTPUT:
            continue
        output = history[index + 1] if index + 1 < len(history) else None
        if output is not None and output.type == CODE and landed_push(output):
            pushes.append(node)
    return pushes


def rewind_point(g: Graph, agent: Node, rewind: int) -> tuple[Node, int]:
    pushes = push_turns(g, agent)
    if not pushes:
        raise ValueError("worker has no pushes to rewind")
    depth = max(1, min(int(rewind), len(pushes)))
    cut = g.prev(pushes[-depth])
    while cut.type == MESSAGE:
        cut = g.prev(cut)
    return cut, depth


def snapshot(flow: Flow, agent: Node, rewind: int = 0) -> dict:
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
    g: Graph,
    worker: Node,
    rewind: int,
    name: str,
    *,
    extra_turns: int | None = None,
    done: str | None = None,
) -> tuple[Node, int]:
    """Fork ``worker`` ``rewind`` pushes back and rebuild the fork's REPL there.

    ``extra_turns`` caps the fork at that many turns past the fork point.
    ``done`` finishes it at once, before its REPL is rebuilt, so a running flow
    never picks it up.
    """
    cut, depth = rewind_point(g, worker, rewind)
    overrides = {}
    if extra_turns is not None:
        turns = sum(node.type == LLM_OUTPUT for node in g.history(cut))
        overrides["max_iters"] = turns + extra_turns
    fork = g.fork(cut, name=name, **overrides)
    if done is not None:
        g.append(fork, Node(DONE, result=done))
    await flow.replay(g, fork)
    return fork, depth


@dataclass
class Branch:
    flow: Flow
    g: Graph
    agent: Node
    index: int
    rewind: int
    order: list[str]

    @property
    def name(self) -> str:
        return f"branch{self.index}"

    @property
    def env(self) -> dict:
        return self.flow.runtime.repl_for(self.agent).env

    @property
    def result(self):
        return self.g.result(self.agent)

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
    g: Graph,
    worker: Node,
    plans: list[Plan] | None = None,
    *,
    expected_plans: int | None = None,
) -> dict:
    """Values injected into the shepherd's REPL."""
    jam = snapshot(flow, worker)
    max_rewind = len(push_turns(g, worker))
    previews = 0

    # Both tools run on the host, not in the shepherd's REPL: ``preview`` drives a
    # graph replay and ``branch`` records into the host's ``plans`` list, and a copy
    # shipped into a worker process would replay nothing and record into the copy.
    # A preview is a fork like any other, so it stays in the graph; it is born
    # finished, which keeps the run from ever driving it.
    @tool("Inspect the worker after rewinding pushes, without playing.", proxy=True)
    async def preview(rewind: int) -> dict:
        nonlocal previews
        name = f"preview{previews}"
        previews += 1
        fork, depth = await replayed_fork(flow, g, worker, rewind, name, done=f"[{name}]")
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
    g: Graph,
    worker: Node,
    plan: Plan,
    index: int,
    budget: int,
    slack: int = 4,
) -> Branch:
    # ``budget`` is a push allowance, not a turn count, so the branch also needs turns
    # for the ones that land nothing: a refused action ends its turn, and under
    # ``--simple-moves`` the worker routes itself and misroutes more. Too little slack
    # and a branch dies at its turn ceiling mid-solve, which reads as giving up.
    fork, depth = await replayed_fork(
        flow, g, worker, plan["rewind"], f"branch{index}", extra_turns=budget + slack
    )
    state = snapshot(flow, fork, depth)
    order = plan["order"]
    order_text = ", then ".join(order)
    fork.args["_shepherd_order"] = order_text
    g.send(
        fork,
        f"Rewound {depth} pushes. This is a fresh recovery attempt, so "
        "abandon the plan you were following.\n\n"
        f"{state['status']}\n\n"
        f"Lock the boxes in this order: {order_text}. The routes are "
        "yours to work out.",
        sender="shepherd",
    )
    return Branch(flow, g, fork, index, depth, order)


__all__ = [
    "Branch",
    "landed_push",
    "prepare_branch",
    "recovery_tools",
    "rewind_point",
    "score",
    "snapshot",
]
