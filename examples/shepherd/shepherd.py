"""A shepherd rewinds a stuck Sokoban worker, runs recovery forks in parallel, and picks one.

Run:
    python examples/shepherd/shepherd.py
    python examples/shepherd/shepherd.py --gradio
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path

import cloudpickle

from rlmflow import Agent, Checkpoint, Flow, Graph, Node
from rlmflow.agents import steps
from rlmflow.graph.nodes import final_node
from rlmflow.llm import client_for
from rlmflow.repl import Repl
from rlmflow.view import LiveTree

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

import sokoban  # noqa: E402
from common import (  # noqa: E402
    add_flow_args,
    add_model_args,
    add_out_dir_arg,
)
from recovery import Plan, prepare_branch, push_turns, recovery_tools  # noqa: E402
from sokoban import Sokoban  # noqa: E402
from view import (  # noqa: E402
    PanelViewer,
    export_run_traces,
    grid_of_blocks,
    lane_key,
    panel_status,
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
    "def goto(row, col=None):\n"
    "    print(game.goto(row, col))\n"
    "def push(direction):\n"
    "    print(game.push(direction))\n"
)

# ``--simple-moves``: the classic action, and nothing to route with.
SIMPLE_CONSTRUCT = (
    f"game = Sokoban({BOARD!r}, env=ENV)\ndef move(directions):\n    print(game.move(directions))\n"
)

# The geometry is the same game whichever action space the worker is given, so both
# system prompts share it rather than drifting apart.
SOKOBAN_RULES = """\
Boxes are pushed, never pulled, and a box locks for good once it reaches a goal.
Pushing means standing on the far side of the box, so a box against a wall only
slides along that wall, and a goal can only be entered from a side you can stand
behind.

Settle a box's last push before you start moving it: the square it sits on to
enter the goal, and the square you stand on to shove it from there. Both must be
floor. Then route the box to that square — reaching the goal from any other side
is not progress, and a box parked one square off a goal it cannot enter is lost
for the rest of the game."""

CLOSING = """\
Before you push, say which box you are moving, where it lands, and how it leaves
that square afterwards. If it cannot leave, push something else.

A refused action raises and ends the turn there: nothing moved, nothing after it in
your block ran, and the next board tells you where you actually are. That is a
misread square, not a lost position — so read the board, take your position from it,
and pick a push from the list it gives you. A refused turn is the one turn you must
not give up on.

The board is the only authority on how the game is going. Call submit("solved") only
when it printed SOLVED, and submit("stuck") only when its list of possible pushes says
none. While it lists even one push you have something to do, and a false claim
scores the same as failing, while another turn costs almost nothing."""

WORKER_SYSTEM = f"""\
Play Sokoban in a Python REPL. You are the man on the board and you have two
calls: `goto(row, col)` walks you to that square by the shortest way around the
boxes, and `push("up")` shoves the box straight ahead of you one square and steps
into the square it left. Walking never disturbs a box, so a push is the only way
you can change what is still solvable.

{SOKOBAN_RULES}

One push per turn, and walking is free, so a turn is `goto` the square you shove
from and then the shove. Make one `repl` call, with code like

goto(5, 10)
push("up")

The board lists every push that is possible right now and the square to stand on
for each, so take that square from the list rather than counting squares. `goto`
refuses only when walls or boxes seal the square off; that ends the turn without
moving you, so nothing you wrote after it runs.

{CLOSING}
"""

SIMPLE_WORKER_SYSTEM = f"""\
Play Sokoban in a Python REPL. You are the man on the board and you have one call:
`move(["down", "down", "right"])` walks that route a square at a time, shoving any box
you step into. So a step into a box is irreversible, and nothing routes you — working
the route out off the grid is your own job.

{SOKOBAN_RULES}

One push per turn and walking is free, so a turn is a route ending in the shove. The
board lists the pushes that are possible and the square to stand on for each, so pick
one from that list, count the route to that square, and finish the route by stepping
into the box. Make one `repl` call, with code like

move(["down", "right", "right", "up", "right"])

A wrong step ends the turn where it stood and the rest of the route does not run, so
a long route is a long guess: the report tells you which steps landed and where you
ended up, and the next board is the truth.

{CLOSING}
"""

WORKER_QUERY = "Push every box onto a target."

# The jam is a fixed script, so it is written rather than played. See ``play_jam``.
# B1 starts directly right of the man and the shove steps him in behind it again,
# so the myopic plan needs no walking at all in either action space.
JAM_CODE = 'push("right")'
SIMPLE_JAM_CODE = 'move(["right"])'
JAM_REASON = (
    "B1 is right next to me and there is open floor to its right, so I'll keep "
    "shoving it that way and worry about the other boxes later."
)

META_QUERY = """\
The Sokoban worker in `jam` walked itself into a bad irreversible plan. Rewind it
and restart it {n} different ways; the best branch is the one that counts.

- `jam`: the stuck state as JSON. `max_rewind`: how far back
  you may go.
- `await preview(k)` for 1 <= k <= int(max_rewind) shows the board k
  pushes ago: `board`, the maze as ASCII with row and column rulers and `#` for
  wall; `boxes` as {{"B1": [row, col]}}; `goals` as {{"G1": [row, col]}}; and
  `legal_pushes` such as `"B1 up from (4,3)"`. Print `board` and read the walls off it —
  there is no wall list, and no other key holds the maze.
- `branch(plans)` records exactly {n} plans and ends planning.

Preview the depths worth seeing, then call `branch(plans)` once with a literal
list of {n} dicts shaped like
{{"rewind": k, "order": ["B2->G4 entering from (r,c)", "B1->G1 entering from (r,c)"]}}.

A plan is two decisions and nothing else. `rewind` says how much of the jam the
branch inherits: shallow keeps finished work but leaves the bad plan in the
worker's visible transcript, which it tends to imitate; deep frees the board but
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
Do not call submit() or launch_agent() while planning: once the branches have
played, you get their results and pick the one to keep.
"""

PICK_QUERY = """\
All {n} branches have played. Each one's plan, outcome and final board:

{results}

Pick the branch to keep. A solved board beats any unsolved one, and among solves
fewer total pushes is better; a branch's pushes include the ones it kept from the
worker. Check the boards rather than trusting the outcome line alone. Then call
submit("Picked branchN: <one sentence on why>") with the branch's exact name.
"""


def scoreboard(state: dict, turn: int, limit: int | None) -> str:
    """How the branch is doing and what it is being judged on.

    Without this a worker cannot tell a short solve from a long one, does not
    know a budget exists, and does not know it is being compared with siblings —
    so nothing discourages it from wandering.
    """
    dist = state.get("dist", 0)
    remaining = f"~{max(0, limit - turn)} turns left" if limit is not None else "no turn cap"
    return (
        f"Standing: {state.get('pushes', 0)} pushes, "
        f"{state.get('placed', 0)}/{state.get('box_count', 0)} boxes locked, "
        f"{dist} to go, {remaining}.\n"
        "Only the best of several parallel attempts is kept: solving scores 1000 "
        f"minus total pushes, failing about {-1000 - dist}. The shortest solve "
        "wins, so don't wander or repeat a push."
    )


def board_prompt(repl: Repl, node: Node, *, simple: bool = False) -> str | None:
    """The step's board section; built inside the agent's step."""
    # The board lives in the agent's REPL, a worker process, so the host reads the
    # published state out of ENV rather than reaching for the game object.
    env = repl.env
    state = dict(env)
    if "status" not in state:
        return None

    # Stamp the turn the game's one-push guard keys on. It ships with the next
    # run, so the worker sees it before it executes this turn's block.
    turn = steps(node)
    env["turn"] = turn
    legal = state.get("legal_pushes", [])
    here = state.get("pushes_here", [])
    doomed = state.get("doomed", [])
    if state.get("solved"):
        action = 'Call submit("solved").'
    elif not legal:
        action = 'Call submit("stuck").'
    elif doomed:
        # Other boxes can still be shoved around for another twenty turns, so without
        # being told, a branch spends its whole budget on a board it has already lost.
        action = (
            f"This board can no longer be solved: {', '.join(doomed)} can never reach "
            'any goal from where it sits. Call submit("stuck") now — pushing the other '
            "boxes cannot undo it, and only a rewind can."
        )
    elif simple:
        walk = ", ".join(state.get("legal_moves", [])) or "nowhere"
        step = f"Stepping {' or '.join(here)} shoves a box. " if here else "No box is against you. "
        action = (
            f"You can step: {walk}. {step}Walk to the square listed for the push you "
            "want, then step into the box — one push this turn."
        )
    elif here:
        action = (
            f"You can push {' or '.join(here)} from where you stand. To shove a "
            "different box, goto its square from the list first, then push once."
        )
    else:
        action = (
            "No box is against you, so goto the square listed for the push you want "
            "and shove from there — one push this turn."
        )

    # Workers quit after a refusal, reading their own mistake as a dead board, so the
    # count of what is still legal goes next to the refusal every turn.
    if legal and not state.get("solved") and not doomed:
        action += (
            f' {len(legal)} pushes are still legal, listed above, so do not call submit("stuck").'
        )
    if state.get("blocked") and not state.get("solved"):
        action = (
            "Your last turn was refused and stopped there, so nothing moved and this "
            "board is current. " + action
        )

    order = node.root.data.get("shepherd_order")
    guidance = (
        f"\n\n{scoreboard(state, turn, repl.agent.max_iters)}"
        f"\n\nStanding order: lock {order}. Each box goes to the goal named for "
        "it, even when another goal is nearer, and the routes are yours."
        if order
        else ""
    )
    return (
        f"{state['status']}\n\n"
        f"Pushes possible now, with the square to stand on: {', '.join(legal) or 'none'}\n"
        f"{action}{guidance}"
    )


class Worker(Agent):
    """The REPL loop with the Sokoban rules as its whole system prompt, shown its board every step."""

    def __init__(self, *args, simple_moves: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.simple_moves = simple_moves

    def messages(self, node, repl, since=None):
        messages = super().messages(node, repl, since)
        board = board_prompt(repl, node, simple=self.simple_moves)
        return [*messages, {"role": "user", "content": board}] if board else messages


async def play_jam(
    flow: Flow,
    g: Graph,
    worker: Node,
    pushes: int,
    *,
    construct: str = CONSTRUCT,
    jam_code: str = JAM_CODE,
) -> AsyncIterator[Node]:
    """Write the worker into its jam without a model, yielding each turn.

    The broken plan is a script — shove B1 east until the wall — so playing it
    with a model costs tokens and a prompt that has to argue a worker out of its
    own judgement, and it can still wander off and jam somewhere else. What the
    shepherd rewinds is a transcript, and a written one is the same nodes the
    live loop would have produced: the turns carry the myopic reasoning too, so a
    shallow rewind really does inherit a visibly bad plan.
    """

    async def written(text: str, code: str) -> Node:
        # Its code runs the way the live loop runs a reply's ``repl`` call.
        g.append(worker, Node("assistant", text, code=code))
        [node] = await flow.step(g, worker)
        return node

    setup = await written("construct the Sokoban game", construct)
    yield setup
    if setup.error:
        raise RuntimeError(f"failed to construct the Sokoban worker: {setup.output}")

    for _ in range(pushes):
        yield await written(JAM_REASON, jam_code)
        # Read the env afresh each turn: a worker returns its published state as a
        # new mapping per run, so a reference taken once goes stale.
        env = flow.repl(worker).env
        if env.get("solved") or env.get("blocked"):
            return


def board(flow: Flow, agent: Node) -> str:
    return flow.repl(agent).env.get("board") or ""


def branch_results(branches, jam_pushes: int) -> str:
    """What the shepherd reads to pick: each branch's plan, outcome and final board."""
    blocks = []
    for branch in branches:
        env = branch.env
        outcome = "solved" if branch.solved else str(branch.result or "ran out of turns")
        blocks.append(
            f"{branch.name}: rewound {branch.rewind}, order {', then '.join(branch.order)}\n"
            f"  {outcome}; {branch.pushes} pushes in total "
            f"({jam_pushes - branch.rewind} kept from the worker); "
            f"{env.get('placed', 0)}/{env.get('box_count', 0)} boxes locked; "
            f"{branch.dist} to go\n"
            f"{branch.board()}"
        )
    return "\n\n".join(blocks)


async def stream_branch(branch, run_dir: Path) -> AsyncIterator[tuple[Node, Graph]]:
    """Run one branch's graph, checkpointing it, and yield each node."""
    checkpoint = Checkpoint(run_dir / "branches" / branch.name)
    async for item in branch.flow.stream(branch.g, hooks=[checkpoint]):
        yield item, branch.g


def branch_heading(branch) -> str:
    state = "SOLVED" if branch.solved else f"dist {branch.dist}"
    return f"{branch.name} [push {branch.pushes} · {state}]"


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
    effort: str | None = None,
    worker_effort: str | None = None,
    simple_moves: bool = False,
    time_budget: float | None = None,
    dashboard=None,
    trace: bool = False,
) -> None:
    # A soft deadline, checked between nodes. Cancelling the whole run at the budget
    # would throw away every branch that had already solved, so the budget stops the
    # fan-out instead and the run still scores, saves, and exports what it has.
    deadline = time.monotonic() + time_budget if time_budget else None

    def expired() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    # A branch turn is a small decision — route the man, shove one box — so the
    # worker thinks at low effort, where a turn costs seconds instead of a minute.
    # The shepherd reads a jammed board and picks N distinct recoveries twice in a
    # whole run, so it keeps the model's own default effort.
    # The action space the worker plays with, and everything that has to agree with
    # it: what its REPL defines, how its prompts read, and how the scripted jam is
    # spelled. Pushes are counted the same either way, so runs stay comparable.
    construct = SIMPLE_CONSTRUCT if simple_moves else CONSTRUCT
    jam_code = SIMPLE_JAM_CODE if simple_moves else JAM_CODE

    def make_flow() -> Flow:
        """The one Flow the run, the shepherd and every branch stream on."""
        worker = Worker(
            "worker",
            system=SIMPLE_WORKER_SYSTEM if simple_moves else WORKER_SYSTEM,
            max_depth=0,
            max_iters=max_iters,
            simple_moves=simple_moves,
        )
        made = Flow(
            agents={
                "worker": worker,
                "shepherd": Agent("shepherd", max_depth=max_depth, max_iters=max_iters),
            },
            llm_clients={
                "shepherd": client_for(model, reasoning_effort=effort),
                "worker": client_for(worker_model, reasoning_effort=worker_effort),
            },
        )
        made.add_tool(Sokoban, name="Sokoban")
        return made

    # Each REPL runs in a worker process that has no examples directory on its
    # path, so ``sokoban`` is not importable there and pickling the class by
    # reference would arrive as a missing import. Sending the module by value
    # ships the code itself; the worker publishes host-readable trace data in ENV.
    cloudpickle.register_pickle_by_value(sokoban)
    flow = make_flow()
    # ``g`` is the worker's run and nothing else; the shepherd works on it from its
    # own graph, ``meta``. Each recovery branch is a fork of ``g`` cut at its rewind
    # point; all of them stream on this one Flow at once.
    g, meta = Graph(), Graph()
    worker = g.start(WORKER_QUERY, agent="worker")

    # The terminal view is the same shape as the other examples': one live agent
    # tree, redrawn in place, whether or not a Gradio dashboard is also running.
    # ``--trace`` swaps it for the raw one-line-per-node log. The boards ride along
    # as the tree's footer, except under Gradio, where the browser draws them.
    tree = (
        LiveTree(
            title="shepherd — rewind ▸ parallel recovery",
            footer=(lambda: panels.render()) if dashboard is None else None,
        )
        if not trace
        else None
    )
    panels = PanelViewer(
        title="" if tree is not None else "shepherd — rewind ▸ parallel recovery",
        cols=min(n_branches, 4),
        status_of=lambda graph, agent: panel_status(flow, graph, agent),
        board_of=lambda graph, agent: board(flow, agent),
        frames_of=(
            (lambda graph, agent: flow.repl(agent).env.get("frames"))
            if dashboard is not None
            else None
        ),
        sink=(dashboard.set_panels if dashboard is not None else None),
        paint=False,
    )
    labels: dict[str, str] = {}

    if tree is not None:
        tree.track(g, label="run")

    def announce(graph: Graph, agent: Node, name: str) -> None:
        """Name a lane."""
        labels[lane_key(graph, agent)] = name
        panels.label(graph, agent, name)

    def show(node: Node, graph: Graph = g) -> None:
        # Panels first: the tree paints straight after, and its footer reads them.
        panels(node)
        if tree is not None:
            tree(node)
        line = trace_line(graph, node, labels) if tree is None or dashboard is not None else ""
        if not line:
            return
        if dashboard is not None:
            # The browser has a log pane; the terminal keeps the tree.
            dashboard.log(line)
        if tree is None:
            print(line, file=sys.stderr, flush=True)

    def close_view() -> None:
        """Stop the live view before anything prints under it."""
        if tree is not None:
            tree.close()
        panels.close()

    run_dir, meta_dir = root / "graph", root / "shepherd"
    try:
        announce(g, worker, "worker")
        if dashboard is not None:
            dashboard.set_status("worker playing the bad plan…")
        # Scripted turns, not a stream: no model, so save once they're written.
        async for item in play_jam(
            flow, g, worker, jam_pushes, construct=construct, jam_code=jam_code
        ):
            show(item)
        g.save(run_dir)

        if flow.repl(worker).env.get("solved"):
            g.save(run_dir)
            if dashboard is not None:
                dashboard.finish("done — worker solved unaided")
            return

        # The jammed worker is finished here. Its REPL stays open for its trace,
        # which is exported at the end; forks cut before this node.
        show(g.append(worker, final_node("[jammed]", len(worker.nodes()))))

        plans: list[Plan] = []
        if not push_turns(g, worker):
            # Nothing to rewind to, so every depth the planner could name is out
            # of range. Say so here instead of letting it burn its whole budget
            # discovering that 1 <= k <= 0 has no solutions.
            message = "no worker pushes to rewind — cannot branch"
            close_view()
            print(message, file=sys.stderr, flush=True)
            if dashboard is not None:
                dashboard.finish(f"done — {message}")
            return
        tools = recovery_tools(
            flow,
            g,
            worker,
            plans,
            expected_plans=n_branches,
        )
        shepherd = meta.start(
            META_QUERY.format(n=n_branches),
            agent="shepherd",
            jam=json.dumps(tools["jam"]),
            max_rewind=str(tools["max_rewind"]),
        )
        flow.repl(shepherd).set(preview=tools["preview"], branch=tools["branch"])
        if tree is not None:
            tree.track(meta, label="shepherd")
        announce(meta, shepherd, "shepherd")
        if dashboard is not None:
            dashboard.set_status("shepherd planning recovery branches…")

        def planned(node: Node) -> bool:
            return (bool(plans) and node.root_id == shepherd.id) or expired()

        stream = flow.stream(meta, until=planned, hooks=[Checkpoint(meta_dir)])
        async for item in stream:
            show(item, meta)
        if not plans:
            close_view()
            ran_out = " before the time budget" if expired() else ""
            message = f"the shepherd produced no plans{ran_out}"
            print(message, file=sys.stderr, flush=True)
            if dashboard is not None:
                dashboard.finish(f"done — {message}")
            return

        # Each plan becomes a fork of the run at its rewind point, with the orders
        # injected as a message to the worker's copy.
        branches = []
        for index, plan in enumerate(plans):
            branch = await prepare_branch(
                flow,
                g,
                worker,
                plan,
                index,
                branch_pushes,
                slack=12 if simple_moves else 4,
            )
            branches.append(branch)
            announce(branch.g, branch.agent, branch.name)
            if tree is not None:
                tree.track(branch.g, label=branch.name)
        if dashboard is not None:
            dashboard.set_proposals(
                [(branch.rewind, ", then ".join(branch.order)) for branch in branches]
            )
            dashboard.set_status("running recovery branches in parallel…")

        async def play(branch) -> None:
            async for item, graph in stream_branch(branch, root):
                show(item, graph)
                if expired():
                    break

        await asyncio.gather(*(play(branch) for branch in branches))

        # The shepherd reads how every branch did and picks the one to keep.
        if dashboard is not None:
            dashboard.set_status("shepherd picking a branch…")
        meta.send(
            shepherd,
            PICK_QUERY.format(n=len(branches), results=branch_results(branches, jam_pushes)),
            sender="host",
        )
        async for item in flow.stream(meta, hooks=[Checkpoint(meta_dir)]):
            show(item, meta)
        summary = str(shepherd.final_result() or "")
        found = re.match(r"Picked (\S+?):", summary)
        best = next((branch for branch in branches if found and branch.name == found[1]), None)
        meta.save(meta_dir)
        if best is None:
            close_view()
            raise RuntimeError(f"the shepherd did not pick a branch by name: {summary!r}")
        g.save(run_dir)
        for branch in branches:
            branch.g.save(root / "branches" / branch.name)

        final_panels = [(branch_heading(branch), branch.board()) for branch in branches]
        # The live view repaints in place, so it has to stop before the report is
        # printed underneath it — otherwise the last repaint wipes the report.
        close_view()
        print(grid_of_blocks(final_panels, cols=min(len(final_panels), 4)))
        export_run_traces(
            root,
            [("worker", flow.repl(worker).env.get("step_frames"))]
            + [(branch.name, branch.env.get("step_frames")) for branch in branches],
        )
        print(f"\n{summary}\n{best.board()}")
        if dashboard is not None:
            # The live viewer is already frozen, so its best-effort status callback
            # can no longer overwrite SOLVED with a bare branch name.
            dashboard.set_panels(final_panels)
            dashboard.set_picked(best.name)
            dashboard.finish(f"done — solved={best.solved}, {best.pushes} pushes")
    finally:
        close_view()
        await flow.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Shepherd backtrack-and-branch")
    add_model_args(parser, default="gpt-6-sol", fast_default="gpt-6-sol")
    add_flow_args(parser, max_depth=1, max_iters=32)
    add_out_dir_arg(parser, "shepherd", help="Save the run graph and traces here.")
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--jam-pushes", type=int, default=8)
    # A branch that rewinds deep has to rebuild most of the board, and the longer
    # solutions run close to 30 pushes, so a smaller budget punishes a branch for
    # following its orders.
    parser.add_argument("--branch-pushes", type=int, default=32)
    parser.add_argument(
        "--simple-moves",
        action="store_true",
        help="Give the worker only move(direction), Sokoban's classic single action, "
        "instead of goto(row, col) + push(direction).",
    )
    parser.add_argument(
        "--effort",
        default="low",
        help="Reasoning effort for the shepherd.",
    )
    parser.add_argument(
        "--worker-effort",
        default="low",
        help="Reasoning effort for the worker and branches. Raise it if branches play badly.",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=1800.0,
        help="Stop launching further branch turns after this many seconds, then score.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print one line per node instead of the live agent tree.",
    )
    parser.add_argument("--gradio", action="store_true")
    parser.add_argument("--gradio-port", type=int, default=7860)
    parser.add_argument("--gradio-share", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_dir).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    async def run(dashboard=None) -> None:
        # ``run_shepherd`` honours the budget itself and still reports, so this only
        # fires if something stopped producing nodes altogether — a hung model call,
        # say — and the grace is what separates the two.
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
                effort=args.effort,
                worker_effort=args.worker_effort,
                simple_moves=args.simple_moves,
                time_budget=args.time_budget,
                dashboard=dashboard,
                trace=args.trace,
            ),
            timeout=args.time_budget + 300.0,
        )

    if args.gradio:
        from gradio_view import launch

        launch(run, port=args.gradio_port, share=args.gradio_share)
    else:
        asyncio.run(run())


if __name__ == "__main__":
    main()
