"""A shepherd rewinds a stuck Sokoban worker and runs recovery forks in parallel.

Run:
    python examples/shepherd/shepherd.py
    python examples/shepherd/shepherd.py --gradio
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from rlmflow import (
    AgentStart,
    DoneOutput,
    ExecOutput,
    Flow,
    LLMOutput,
    Node,
    PromptProfile,
    UserPromptBuilder,
)

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import add_flow_args, add_model_args, add_out_dir_arg, build_client  # noqa: E402
from recovery import Plan, prepare_branch, push_turns, recovery_tools  # noqa: E402
from sokoban import Sokoban  # noqa: E402
from view import (  # noqa: E402
    PanelViewer,
    export_run_traces,
    panel_status,
    side_by_side,
    trace_line,
)

BOARD = [
    "############",
    "#  .   .  ##",
    "### # # # ##",
    "#@$        #",
    "#   # # # .#",
    "#  $     $ #",
    "#  ## # #  #",
    "# .     .  #",
    "############",
]

CONSTRUCT = (
    f"game = Sokoban({BOARD!r}, env=ENV)\n"
    "def push(box, direction):\n"
    "    print(game.push(box, direction))\n"
)

WORKER_SYSTEM = """\
Play Sokoban one box-push at a time in a Python REPL.
Boxes are pushed, never pulled, and a box locks for good once it reaches a goal.
To push a box you stand on the far side of it, so a box against a wall only
slides along that wall, and a goal can only be entered from a side you can stand
behind.

Settle a box's last push before you start moving it: the square it sits on to
enter the goal, and the square you stand on to shove it from there. Both must be
floor. Then route the box to that square — reaching the goal from any other side
is not progress, and a box parked one square off a goal it cannot enter is lost
for the rest of the game.

Before each push, say which box you are moving, where it lands, and how it leaves
that square afterwards. If it cannot leave, push something else.
Then send exactly one ```repl``` block calling push("B<n>", "direction").
Call done("solved") when the board says SOLVED, done("stuck") when nothing is
left to push.
"""

WORKER_QUERY = "Push every box onto a target."

# The jam is a fixed script, so it is written rather than played. See ``play_jam``.
JAM_CODE = 'push("B1", "right")'
JAM_REPLY = (
    "B1 is right next to me and there is open floor to its right, so I'll keep "
    "shoving it that way and worry about the other boxes later.\n\n"
    f"```repl\n{JAM_CODE}\n```"
)

META_QUERY = """\
The Sokoban worker in `jam` walked itself into a bad irreversible plan. Rewind it
and restart it {n} different ways; the best branch is the one that counts.

- `INPUTS["jam"]`: the stuck state as JSON. `INPUTS["max_rewind"]`: how far back
  you may go.
- `await preview(k)` for 1 <= k <= int(INPUTS["max_rewind"]) shows the board k
  pushes ago: `board`, the maze as ASCII with row and column rulers and `#` for
  wall; `boxes` as {{"B1": [row, col]}}; `goals` as {{"G1": [row, col]}}; and
  `legal_pushes` such as `"B1 up"`. Print `board` and read the walls off it —
  there is no wall list, and no other key holds the maze.
- `branch(plans)` records exactly {n} plans and ends planning.

Preview the depths worth seeing, then call `branch(plans)` once with a literal
list of {n} dicts shaped like
{{"rewind": k, "order": ["B2->G4 entering from (r,c)", "B1->G1 entering from (r,c)"]}}.

A plan is two decisions and nothing else. `rewind` says how much of the jam the
branch inherits: shallow keeps finished work but leaves the bad plan in the
worker's visible history, which it tends to imitate; deep frees the board but
costs pushes to rebuild. `order` names which box takes which goal and in what
sequence — the worker works out the routes itself, so do not give it moves.

Give each step the square the box enters its goal from, as a coordinate. A box is
pushed from the far side, so entering the goal at (r,c) means the box stands on
one of its neighbours and the player on the next square out, and both have to be
floor in `board`. Some goals have three such approaches and some have one, and
you have to count that off the grid — a phrase like "from the open side" is not
an answer and tells the worker nothing. The worker is a smaller model: left to
itself it walks a box to the nearest face of its goal, and if that face is not an
entrance the box sits one square away, unpushable, for the rest of the game.

Order matters because a box locks the moment it lands on a goal, and a locked box
is a wall forever after. Put a box that would block a corridor last, and one that
has to cross the board first.

There are more goals than boxes, so some goals stay empty and no box is owed the
goal sharing its number. Two plans that pair the same boxes with the same goals
are one plan however you sequence them, so spend the {n} slots on different
pairings — including the goals an obvious plan would leave out.
Do not call done() or launch_subagents().
"""


def scoreboard(game, agent: AgentStart) -> str:
    """How the branch is doing and what it is being judged on.

    Without this a worker cannot tell a short solve from a long one, does not
    know a budget exists, and does not know it is being compared with siblings —
    so nothing discourages it from wandering.
    """
    placed = sum(1 for pos in game.boxes if pos in game.targets)
    left = max(0, agent.config.max_iters - agent.llm_turns())
    return (
        f"Standing: {game.pushes} pushes, {placed}/{len(game.boxes)} boxes locked, "
        f"{game.dist} to go, ~{left} turns left.\n"
        "Only the best of several parallel attempts is kept: solving scores 1000 "
        f"minus total pushes, failing about {-1000 - game.dist}. The shortest solve "
        "wins, so don't wander or repeat a push."
    )


def board_prompt(flow: Flow, agent: AgentStart) -> str | None:
    try:
        game = flow.runtime.get_var(agent, "game")
    except Exception:  # noqa: BLE001 - the shepherd has no board
        return None

    game.begin_turn()
    legal = game.legal_pushes()
    if game.solved:
        action = 'Call done("solved").'
    elif not legal:
        action = 'Call done("stuck").'
    else:
        action = "Make exactly one listed push."

    order = agent.config.inputs.get("_shepherd_order")
    guidance = (
        f"\n\n{scoreboard(game, agent)}"
        f"\n\nStanding order: lock {order}. Each box goes to the goal named for "
        "it, even when another goal is nearer, and the routes are yours."
        if order
        else ""
    )
    return f"{game.status()}\n\nLegal pushes: {', '.join(legal) or 'none'}\n{action}{guidance}"


async def play_jam(
    flow: Flow,
    worker: AgentStart,
    pushes: int,
) -> AsyncIterator[Node]:
    """Write the worker into its jam without an LLM, yielding each node.

    The broken plan is a script — shove B1 east until the wall — so playing it
    with a model costs tokens and a prompt that has to argue a worker out of its
    own judgement, and it can still wander off and jam somewhere else. What the
    shepherd rewinds is a transcript, and a written one is the same nodes the
    live loop would have produced: the turns carry the myopic reasoning too, so a
    shallow rewind really does inherit a visibly bad plan.
    """

    async def land(node: Node) -> Node:
        return (await flow.step(node)).created

    setup = worker.frontier.append(
        LLMOutput(content="construct the Sokoban game", code=CONSTRUCT)
    )
    yield setup
    action = await land(setup)
    yield action
    yield await land(action)

    env = flow.runtime.repl_for(worker).env
    for _ in range(pushes):
        turn = worker.frontier.append(LLMOutput(content=JAM_REPLY, code=JAM_CODE))
        yield turn
        action = await land(turn)
        yield action
        yield await land(action)
        if env.get("solved") or env.get("blocked"):
            return


def board(flow: Flow, agent: AgentStart) -> str:
    return flow.runtime.get_var(agent, "game").render(ids=True)


async def run_shepherd(
    root: Path,
    *,
    model: str,
    worker_model: str,
    max_depth: int,
    max_iters: int,
    n_branches: int,
    jam_pushes: int,
    branch_pushes: int,
    dashboard=None,
) -> None:
    flow = Flow(
        build_client(model),
        llm_clients={"worker": build_client(worker_model)},
        prompt_profiles={
            "worker": PromptProfile(
                system=WORKER_SYSTEM,
                user=UserPromptBuilder(board_prompt),
                description="myopic Sokoban worker",
            )
        },
    )
    flow.inject("Sokoban", Sokoban)
    worker = flow.start(
        WORKER_QUERY,
        model="worker",
        prompt_profile="worker",
        max_depth=0,
        max_iters=max_iters,
        keep_n_messages=15,
    )
    shepherd = flow.start(
        META_QUERY.format(n=n_branches),
        max_depth=max_depth,
        max_iters=max_iters,
    )

    panels = PanelViewer(
        title="shepherd — rewind ▸ parallel recovery",
        cols=min(n_branches, 4),
        status_of=lambda agent: panel_status(flow, agent),
        board_of=lambda agent: board(flow, agent),
        frames_of=(
            (lambda agent: flow.runtime.get_env_var(agent, "frames"))
            if dashboard is not None
            else None
        ),
        sink=(dashboard.set_panels if dashboard is not None else None),
        paint=False,
    )
    labels: dict[str, str] = {}

    def announce(agent: AgentStart, name: str) -> None:
        labels[agent.id] = name
        panels.label(agent, name)

    def show(node: Node) -> None:
        panels.handle(node)
        line = trace_line(node, labels)
        if line:
            print(line, file=sys.stderr, flush=True)

    try:
        announce(worker, "worker")
        if dashboard is not None:
            dashboard.set_status("worker playing the bad plan…")
        async for node in play_jam(flow, worker, jam_pushes):
            show(node)
            worker.save(root / "worker")

        if flow.runtime.get_env_var(worker, "solved"):
            worker.save(root / "best")
            if dashboard is not None:
                dashboard.finish("done — worker solved unaided")
            return

        plans: list[Plan] = []
        if not push_turns(worker):
            # Nothing to rewind to, so every depth the planner could name is out
            # of range. Say so here instead of letting it burn its whole budget
            # discovering that 1 <= k <= 0 has no solutions.
            message = "no worker pushes to rewind — cannot branch"
            print(message, file=sys.stderr, flush=True)
            if dashboard is not None:
                dashboard.finish(f"done — {message}")
            return
        tools = recovery_tools(
            flow,
            worker,
            plans,
            expected_plans=n_branches,
        )
        shepherd.config.inputs = {
            "jam": json.dumps(tools["jam"]),
            "max_rewind": str(tools["max_rewind"]),
        }
        repl = flow.runtime.repl_for(shepherd)
        repl.inject("preview", tools["preview"])
        repl.inject("branch", tools["branch"])
        announce(shepherd, "shepherd")
        if dashboard is not None:
            dashboard.set_status("shepherd planning recovery branches…")

        async for node in flow.run_streaming(
            shepherd,
            until=lambda node, _root: bool(plans) and node.parent_agent is shepherd,
        ):
            show(node)
            shepherd.save(root / "shepherd")
        if not plans:
            return

        branches = [
            await prepare_branch(flow, worker, plan, index, branch_pushes)
            for index, plan in enumerate(plans)
        ]
        for branch in branches:
            announce(branch.graph, branch.name)
        if dashboard is not None:
            dashboard.set_proposals(
                [
                    (branch.rewind, ", then ".join(branch.order))
                    for branch in branches
                ]
            )
            dashboard.set_status("running recovery branches in parallel…")

        anchor = shepherd.frontier
        for branch in branches:
            anchor.append_child(branch.graph, name=branch.name)

        async for node in flow.run_streaming(
            shepherd,
            until=lambda node, _root: (
                node.parent_agent is shepherd and isinstance(node, ExecOutput)
            ),
        ):
            show(node)
            shepherd.save(root / "shepherd")

        best = max(branches, key=lambda branch: branch.score)
        summary = (
            f"Picked {best.name}: solved={best.solved}, score={best.score}, pushes={best.pushes}"
        )
        shepherd.frontier.append(DoneOutput(content=summary, result=summary))
        shepherd.save(root / "shepherd")

        print(
            side_by_side(
                [
                    (
                        f"{branch.name} [{'solved' if branch.solved else f'dist={branch.dist}'}]",
                        branch.board(),
                    )
                    for branch in branches
                ]
            )
        )
        export_run_traces(
            root,
            [("worker", flow.runtime.get_var(worker, "game"))]
            + [(branch.name, flow.runtime.get_var(branch.graph, "game")) for branch in branches],
        )
        best.graph.save(root / "best")
        print(f"\n{summary}\n{best.board()}")
        if dashboard is not None:
            dashboard.set_picked(best.name)
            dashboard.finish(f"done — solved={best.solved}, {best.pushes} pushes")
    finally:
        panels.close()
        await flow.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Shepherd backtrack-and-branch")
    add_model_args(parser, default="gpt-5", fast_default="gpt-5-mini")
    add_flow_args(parser, max_depth=1, max_iters=32)
    add_out_dir_arg(parser, "shepherd", help="Save worker and branch graphs here.")
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--jam-pushes", type=int, default=8)
    # A branch that rewinds deep has to rebuild most of the board, and the longer
    # solutions run close to 30 pushes, so a smaller budget punishes a branch for
    # following its orders.
    parser.add_argument("--branch-pushes", type=int, default=32)
    parser.add_argument("--time-budget", type=float, default=900.0)
    parser.add_argument("--gradio", action="store_true")
    parser.add_argument("--gradio-port", type=int, default=7860)
    parser.add_argument("--gradio-share", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_dir).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    async def run(dashboard=None) -> None:
        await asyncio.wait_for(
            run_shepherd(
                root,
                model=args.model,
                worker_model=args.fast_model,
                max_depth=args.max_depth,
                max_iters=args.max_iters,
                n_branches=args.branches,
                jam_pushes=args.jam_pushes,
                branch_pushes=args.branch_pushes,
                dashboard=dashboard,
            ),
            timeout=args.time_budget,
        )

    if args.gradio:
        from gradio_view import launch

        launch(run, port=args.gradio_port, share=args.gradio_share)
    else:
        asyncio.run(run())


if __name__ == "__main__":
    main()
