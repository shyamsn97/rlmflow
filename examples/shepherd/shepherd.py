"""Shepherd's flagship move: rewind a stuck agent, then branch a lot in parallel.

A single *worker* agent plays a tiny Sokoban puzzle one irreversible push at a
time (small/fast models are plenty — it is just pushing). Under that myopic
one-push-at-a-time contract it shoves a box into a frozen spot — a mistake no
amount of *forward* play can undo. That is the whole point: the capability the
worker lacks is *backtracking*, which Shepherd's primitives (fork/rewind over the
worker's own trajectory) supply. The board has TWO boxes and FOUR targets, so a
recovery has real choices (which targets to fill, which side of the central wall
to route around) — that is where the branch *diversity* comes from. The *strategy*
for how to recover comes from a second rlm — a **meta-agent** ("shepherd") that
reasons over the broken board and
returns structured ``Proposals``. It is a single parallel fan-out — no rounds:

  1. **Play forward.** Drive the worker with ``run_streaming(until=stuck_or_solved)``
     — a *state predicate* that halts the instant the environment publishes
     ``ENV["deadlock"]`` or ``ENV["solved"]``, not at every push. While it plays,
     collect the ``LLMOutput`` decision nodes we could rewind to.
  2. **Meta-agent proposes.** On a dead end, the shepherd rlm is shown the board
     and the worker's recent pushes and returns ``N`` ``Proposal``s (``N`` is a
     CLI arg it reads from its prompt), each with its own ``rewind`` depth and a
     one-line ``strategy`` — genuinely different recovery plans.
  3. **Rewind + fan out in parallel.** ``flow.fork`` the worker's trajectory at
     each proposal's chosen decision (dropping it so the model re-decides), inject
     the strategy, and play ALL branches concurrently. Each fork replays the good
     prefix byte-identically, so the branches are honest independent rollouts.
  4. **Pick the best.** Score every branch with the environment (solved?
     deadlocked? distance-to-target), keep the winner, discard the rest. Done.

The division of labor is strict: the shepherd rlm supplies the *policy*
(proposals), the env supplies the *verifier* (score), and Python supplies the
*mechanism* (fork/discard/pick) — it never solves the puzzle. Note how the game
enters the REPL: the
``Sokoban`` **class** (pure logic) is injected as a tool via ``Flow(tools=[...])``,
so every REPL — including forks — has it. The mutable game *instance* is built by
one tiny replayable action (``game = Sokoban(INPUTS["board"], env=ENV)``); that
way fork/revert reconstruct each branch's own game at the right state by replay,
instead of aliasing one shared object. The worker's LLM turns are the moves.

Board glyphs:  ``#`` wall · (space) floor · ``@`` player · ``$`` box ·
``.`` target · ``*`` box on target · ``+`` player on target

Run:
    python examples/shepherd/shepherd.py                 # live: needs an API key
    python examples/shepherd/shepherd.py --self-test     # offline env check, no LLM
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from rflow import (
    AppendNode,
    ExecAction,
    ExecOutput,
    Flow,
    Graph,
    GraphCheckpointer,
    LLMOutput,
    StreamConsumer,
)

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import add_flow_args, add_model_args, add_out_dir_arg, build_client  # noqa: E402

# A trap board: TWO boxes, FOUR targets, one central wall "pillar". Diversity is
# the point — with more targets than boxes, a recovery has real CHOICES: which
# two of the four targets to fill, which box serves which, and (thanks to the
# pillar at (3,4)) which side to route around. All six target-pairs are solvable,
# so the shepherd has genuinely different routes to describe, and the branches'
# final boards look visibly different.
#
# The player starts directly LEFT of the near box, so the myopic "keep pushing
# right" baseline shoves that box straight into the right wall — with no target
# in that column it is frozen forever (DEADLOCK after 4 pushes). The shepherd
# then rewinds and fans out over the distinct ways to place both boxes (BFS
# optimum = 5 pushes, e.g. RRDDR).
BOARD = [
    "########",
    "#   .  #",
    "#@$ .. #",
    "#   #  #",
    "#   $. #",
    "#      #",
    "########",
]


class Sokoban:
    """The whole game as a plain Python object — injected into the REPL as a tool.

    It is pure logic: apply a push, tell whether the board is solved or a box is
    frozen (a cheap verifier signal, never a solver), and render it. State lives
    on the instance; ``push`` publishes the outcome to the ``env`` mapping it was
    given (the REPL's ``ENV`` channel) so the host meta-agent can read it.
    """

    MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

    def __init__(self, board: list[str], env: dict | None = None) -> None:
        self.board = board
        self.walls: set[tuple[int, int]] = set()
        self.targets: set[tuple[int, int]] = set()
        self.boxes: set[tuple[int, int]] = set()
        self.player: tuple[int, int] | None = None
        for r, row in enumerate(board):
            for c, ch in enumerate(row):
                p = (r, c)
                if ch == "#":
                    self.walls.add(p)
                elif ch in ".*+":
                    self.targets.add(p)
                if ch in "$*":
                    self.boxes.add(p)
                if ch in "@+":
                    self.player = p
        self.env = env if env is not None else {}
        self.pushes = 0
        self._publish()

    @property
    def solved(self) -> bool:
        # More targets than boxes: solved when every box sits on some target.
        return len(self.boxes) > 0 and self.boxes <= self.targets

    @property
    def deadlock(self) -> bool:
        return (not self.solved) and self._stuck()

    @property
    def dist(self) -> int:
        return sum(
            min(abs(b[0] - t[0]) + abs(b[1] - t[1]) for t in self.targets)
            for b in self.boxes
            if b not in self.targets
        )

    def _stuck(self) -> bool:
        for r, c in self.boxes:
            if (r, c) in self.targets:
                continue
            row_wall = (r - 1, c) in self.walls or (r + 1, c) in self.walls
            col_wall = (r, c - 1) in self.walls or (r, c + 1) in self.walls
            if row_wall and col_wall:
                return True
            if row_wall and not any(t[0] == r for t in self.targets):
                return True
            if col_wall and not any(t[1] == c for t in self.targets):
                return True
        return False

    def _publish(self) -> None:
        self.env["solved"] = self.solved
        self.env["deadlock"] = self.deadlock
        self.env["dist"] = self.dist
        self.env["pushes"] = self.pushes
        # Box positions (for the host to signature branch diversity), and which
        # of them landed on targets.
        self.env["boxes"] = ";".join(f"{r},{c}" for r, c in sorted(self.boxes))
        self.env["filled"] = ";".join(
            f"{r},{c}" for r, c in sorted(self.boxes & self.targets)
        )

    def push(self, move: str) -> str:
        """Move the player one cell (U/D/L/R), pushing a box if one is ahead."""
        dr, dc = self.MOVES[move]
        pr, pc = self.player
        step = (pr + dr, pc + dc)
        if step in self.walls:
            return self.render() + f"\nILLEGAL move {move}; nothing moved."
        if step in self.boxes:
            beyond = (step[0] + dr, step[1] + dc)
            if beyond in self.walls or beyond in self.boxes:
                return self.render() + f"\nILLEGAL move {move}; nothing moved."
            self.boxes.discard(step)
            self.boxes.add(beyond)
        self.player = step
        self.pushes += 1
        self._publish()
        if self.solved:
            tail = "SOLVED"
        elif self.deadlock:
            tail = "DEADLOCK: a box is frozen and can never reach the target."
        else:
            tail = f"box-distance-to-target={self.dist}"
        return self.render() + "\n" + tail

    def render(self) -> str:
        rows = []
        for r, row in enumerate(self.board):
            line = []
            for c in range(len(row)):
                p = (r, c)
                if p in self.walls:
                    line.append("#")
                elif p == self.player:
                    line.append("+" if p in self.targets else "@")
                elif p in self.boxes:
                    line.append("*" if p in self.targets else "$")
                elif p in self.targets:
                    line.append(".")
                else:
                    line.append(" ")
            rows.append("".join(line))
        return "\n".join(rows)


# The single replayable action that constructs per-REPL game state from the
# injected class. Fork/revert re-run this (plus the worker's push()es) to rebuild
# a branch's own game — no shared mutable object, so reversibility is exact. The
# board is baked in as a literal (NOT routed through INPUTS): that keeps the game
# self-contained AND avoids the framework's "inspect INPUTS first" turn, which a
# weak worker loops on. ``push`` is wrapped to PRINT its result so the worker can
# actually see the board after every move (a bare ``push(...)`` returns a string
# that the exec REPL would otherwise discard, leaving the worker blind).
CONSTRUCT = (
    f"game = Sokoban({BOARD!r}, env=ENV)\n"
    "def push(m):\n"
    "    print(game.push(m))\n"
    "print(game.render())"
)

WORKER_QUERY = (
    "Solve this Sokoban puzzle by pushing every box ($) onto a target (.). The "
    "game is already set up in your REPL; the starting board is shown above.\n\n"
    'Call push("U"|"D"|"L"|"R"): it moves you one cell (U=up, D=down, L=left, '
    "R=right), pushes a box if one is directly ahead and the cell beyond is free, "
    "and PRINTS the resulting board plus the outcome.\n\n"
    "Strategy: a box is right in front of you and there are targets over to the "
    "right, so the obvious plan is to just keep pushing that box Right to close "
    "the distance — push Right, and keep pushing Right, until you can't anymore. "
    "Stay simple and greedy; don't overthink detours.\n\n"
    "Rules for this task:\n"
    "- Make EXACTLY ONE push(...) call per turn inside a ```repl block, then "
    "stop and read the printed board before choosing your next move.\n"
    '- When the board reports SOLVED, call done("solved").'
)

# The meta-agent ("shepherd") is *another rlm* — its own graph. When the worker
# dead-ends, it reasons over the broken board and returns structured Proposals:
# N distinct recovery branches, each with its own rewind depth. Python never
# writes the strategies; it only forks/scores/picks what the shepherd says.
class Proposal(BaseModel):
    rewind: int = Field(
        ge=1,
        description="how many of the worker's last decisions to undo (1 = redo only the most recent)",
    )
    strategy: str = Field(
        description="a short recovery plan for this branch, followed one push at a time"
    )


class Proposals(BaseModel):
    proposals: list[Proposal] = Field(
        description="distinct branches to explore, one fork each"
    )


META_QUERY = (
    "You are the shepherd of a myopic Sokoban worker that plays one push at a "
    "time and cannot backtrack. The board has TWO boxes and FOUR targets, so a "
    "solution only needs both boxes on targets — which means there are many "
    "genuinely different ways to win: which two targets to fill, which box serves "
    "which, and which way to route around the central wall.\n\n"
    "You will be shown the board where the worker got stuck and its recent "
    "pushes. Propose exactly {n} genuinely DIFFERENT recovery branches to explore "
    "in parallel — make the routes actually diverge (different target choices and "
    "different paths), not minor variations. For each branch set `rewind` = how "
    "many of the worker's last decisions to undo, and `strategy` = a short "
    "instruction, in your own words, describing that branch's route (which box "
    "goes to which target and roughly how), followed one push at a time. Do NOT "
    "solve the puzzle yourself and do not use the REPL — just submit your "
    'proposals via done({{"proposals": [...]}}).'
)

# A hand-verified winning line (BFS optimum: push the near box right onto a
# target, then the far box down-right onto another). Used ONLY to keep the
# offline self-test deterministic; the live run never uses it — the worker plays.
WIN_MOVES = "RRDDR"


def score(env: dict) -> float:
    """Verifier score for a branch's final state (never a solver).

    Shortest-path scoring: a solve is always positive (``1000 - pushes``) so it
    beats any failed branch, and among solvers fewer pushes ranks higher (an
    8-push solve scores 992, a 12-push one 988). Failed branches (deadlocked or
    still unsolved at the cap) share a hard ``-1000`` floor.
    """
    if env.get("solved"):
        return 1000.0 - float(env.get("pushes", 0))
    return -1000.0


async def seed_turn(flow: Flow, graph: Graph, code: str) -> None:
    """Record a scripted (llm_output, exec_action) pair and run it, so the game
    construction is a replayable ExecAction that every fork reconstructs."""
    graph.commit(LLMOutput(content="construct the Sokoban game", code=code))
    graph.commit(ExecAction(code=code))
    await flow.exec_turn(graph, code)


async def render(flow: Flow, graph: Graph) -> str:
    return (await flow.repl_for(graph).run("print(game.render())")).strip()


def signature(env_boxes: str) -> frozenset:
    """A branch's solution signature = the set of filled target cells, so we can
    count how many branches found *genuinely distinct* placements (not just
    distinct move sequences that land the same way)."""
    return frozenset(tuple(map(int, p.split(","))) for p in env_boxes.split(";") if p)


def side_by_side(labeled: list[tuple[str, str]], *, gap: str = "    ") -> str:
    """Lay several headed text blocks next to each other in one row. ``labeled``
    is a list of ``(heading, block)`` pairs; blocks may be multi-line."""
    if not labeled:
        return ""
    cols = [[head, *block.splitlines()] for head, block in labeled]
    height = max(len(col) for col in cols)
    widths = [max((len(line) for line in col), default=0) for col in cols]
    rows = []
    for r in range(height):
        cells = [
            (col[r] if r < len(col) else "").ljust(w) for col, w in zip(cols, widths)
        ]
        rows.append(gap.join(cells).rstrip())
    return "\n".join(rows)


def grid_of_blocks(labeled: list[tuple[str, str]], *, cols: int = 4) -> str:
    """``side_by_side`` wrapped into rows of at most ``cols`` blocks, so a large
    fan-out (e.g. 8 branches) renders as a grid instead of one very wide line."""
    cols = max(1, cols)
    rows = [
        side_by_side(labeled[i : i + cols]) for i in range(0, len(labeled), cols)
    ]
    return "\n\n".join(row for row in rows if row)


def _looks_like_grid(text: str) -> bool:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[0] == "#"
    return False


class PanelViewer(StreamConsumer):
    """Live side-by-side boards, one panel per lane, updated push-by-push.

    One panel per graph (keyed by ``graph_id``, first-seen order). Each panel
    shows that lane's latest ``ExecOutput`` block — which for this example is the
    rendered Sokoban board — so the worker and every recovery branch animate as
    their streams advance. Safe to share across concurrently-gathered streams:
    event handling runs on one asyncio loop, so updates never interleave.
    """

    def __init__(
        self,
        *,
        title: str = "",
        cols: int = 4,
        every_s: float = 0.1,
        status_of: Callable[[Graph], str] | None = None,
    ) -> None:
        self.title = title
        self.cols = cols
        self.every_s = every_s
        self.status_of = status_of
        self._order: list[str] = []
        self._labels: dict[str, str] = {}
        self._blocks: dict[str, str] = {}
        self._graphs: dict[str, Graph] = {}
        self._last_paint = 0.0
        self._closed = False

    def label(self, graph_id: str, label: str) -> None:
        if graph_id not in self._order:
            self._order.append(graph_id)
        self._labels[graph_id] = label

    def handle(self, event, graph: Graph | None) -> None:
        if graph is None or self._closed:
            return
        key = graph.graph_id
        if key not in self._order:
            self._order.append(key)
            self._labels.setdefault(key, graph.agent_id)
        self._graphs[key] = graph
        if isinstance(event, AppendNode) and isinstance(event.node, ExecOutput):
            text = (event.node.content or "").rstrip()
            if _looks_like_grid(text) or key not in self._blocks:
                self._blocks[key] = text
        now = time.monotonic()
        if self.every_s and now - self._last_paint < self.every_s:
            return
        self._paint()

    def _panels(self) -> list[tuple[str, str]]:
        panels = []
        for key in self._order:
            heading = self._labels.get(key, key)
            if self.status_of is not None and key in self._graphs:
                try:
                    status = self.status_of(self._graphs[key])
                except Exception:  # noqa: BLE001 - a viewer must never crash a run
                    status = ""
                if status:
                    heading = f"{heading}  [{status}]"
            panels.append((heading, self._blocks.get(key, "")))
        return panels

    def _paint(self) -> None:
        self._last_paint = time.monotonic()
        panels = self._panels()
        if not panels:
            return
        header = f"{self.title}\n\n" if self.title else ""
        if sys.stdout.isatty():
            print("\033[2J\033[H" + header + grid_of_blocks(panels, cols=self.cols), flush=True)
        else:  # Non-TTY (CI, piped): compact status lines, no screen repaint.
            print("\n".join([self.title, *(h for h, _ in panels)]).strip(), flush=True)

    def close(self) -> None:
        """Paint the final frame once; later events/closes are no-ops so the end
        state (and the report printed after it) is not wiped by a re-clear."""
        if self._closed:
            return
        self._last_paint = 0.0
        self._paint()
        self._closed = True


async def stream_saving(flow: Flow, graph: Graph, *, until, path: Path, extra=None):
    """Drive ``run_streaming`` through a per-event ``GraphCheckpointer`` so the
    graph snapshot is written to ``path`` after EVERY committed node (and on a
    final flush), then re-yield each event for the caller's own logic. This is
    why a run that hangs mid-turn still leaves a full, up-to-date trace on disk.

    ``extra`` is an optional additional ``StreamConsumer`` (e.g. a shared
    ``PanelViewer``) fed the same events — safe to share across concurrently
    gathered streams because event handling runs on one asyncio loop.
    """
    checkpointer = GraphCheckpointer(path)
    try:
        async for event in flow.run_streaming(graph=graph, until=until):
            checkpointer.handle(event, graph)
            if extra is not None:
                extra.handle(event, graph)
            yield event
    finally:
        checkpointer.close()


def panel_status(flow: Flow, graph: Graph) -> str:
    """Live one-line status for a board panel, read from the graph's REPL env."""
    env = flow.repl_for(graph).env
    if env.get("solved"):
        return "SOLVED"
    if env.get("deadlock"):
        return "DEADLOCK"
    pushes, dist = env.get("pushes"), env.get("dist")
    return f"push {pushes} · dist {dist}" if pushes is not None else ""


async def ask_shepherd(
    flow: Flow, shepherd: Graph, trunk: Graph, fork_points: list, n: int, path: Path
) -> list[Proposal]:
    """Consult the meta-agent rlm: inject the failed board + the worker's recent
    decisions onto its own graph, run one structured-output turn, and return the
    parsed proposals. The shepherd graph is synced to ``path`` every step."""
    env = flow.repl_for(trunk).env
    reason = (
        "a box is DEADLOCKED (frozen, can never reach the target)"
        if env.get("deadlock")
        else f"it stalled after {env.get('pushes')} pushes"
    )
    k = len(fork_points)
    moves = "\n".join(
        f"  {i + 1}. {(node.code or '').strip()}" for i, node in enumerate(fork_points)
    )
    shepherd.inject(
        f"The worker stopped: {reason}.\n\nCurrent board:\n{await render(flow, trunk)}\n\n"
        f"It made {k} push decision(s) (oldest first):\n{moves or '  (none)'}\n\n"
        f"Propose {n} genuinely different recovery branches now. Each branch's "
        f"`rewind` must be between 1 and {k} (how many of these {k} pushes to undo)."
    )
    async for _event in stream_saving(flow, shepherd, until="done", path=path):
        pass
    return Proposals(**json.loads(shepherd.result())).proposals[:n]


def _halt(flow: Flow, *, push_cap: int | None):
    """Build an ``until`` predicate: stop when the board is solved or dead, and
    (for the worker) after ``push_cap`` moves so the meta-agent can intervene
    before a myopic worker wanders forever."""

    def predicate(_event, graph: Graph) -> bool:
        env = flow.repl_for(graph).env
        if env.get("solved") or env.get("deadlock"):
            return True
        return push_cap is not None and env.get("pushes", 0) >= push_cap

    return predicate


async def branch_and_pick(
    flow: Flow,
    trunk: Graph,
    shepherd: Graph,
    *,
    root: Path,
    n_branches: int,
    worker_pushes: int,
    branch_pushes: int,
    viewer: PanelViewer,
) -> tuple[Graph, bool]:
    """Play the weak worker to a dead end, let the meta-agent rlm propose ``n``
    diverse recovery branches, fork them, play them ALL in parallel, pick the best.

    Every stream (worker, shepherd, each branch) is driven through
    ``stream_saving`` so the graphs are checkpointed to ``root`` after every step.
    """
    worker_halt = _halt(flow, push_cap=worker_pushes)
    branch_halt = _halt(flow, push_cap=branch_pushes)

    # 1. Play the worker forward until it dead-ends (or solves), collecting its
    #    actual PUSH turns as candidate rewind points (skip any non-move turns so
    #    rewind depths map to real decisions). Synced after every node.
    fork_points: list = []
    viewer.label(trunk.graph_id, "worker")
    async for event in stream_saving(
        flow, trunk, until=worker_halt, path=root / "worker", extra=viewer
    ):
        node = event.node
        if isinstance(event, AppendNode) and isinstance(node, LLMOutput) and "push(" in (node.code or ""):
            fork_points.append(node)

    env = flow.repl_for(trunk).env
    reason = (
        "solved" if env.get("solved")
        else "DEADLOCK" if env.get("deadlock")
        else f"stalled after {env.get('pushes')} pushes"
    )
    print(f"\n[worker] {reason}; board now:")
    print(await render(flow, trunk))
    if env.get("solved"):
        return trunk, True
    if not fork_points:
        return trunk, False

    # 2. Ask the META-AGENT rlm for N distinct proposals, each with its own rewind
    #    depth. Python never writes the strategies.
    proposals = await ask_shepherd(
        flow, shepherd, trunk, fork_points, n_branches, path=root / "shepherd"
    )
    print(f"[shepherd] proposed {len(proposals)} branches:")
    for i, p in enumerate(proposals):
        print(f"  proposal {i} (rewind={p.rewind}): {p.strategy}")

    # 3. Fork one branch per proposal at ITS rewind point (forks are sequential —
    #    they mutate flow state — but cheap), then play them ALL concurrently.
    forked: list[tuple[int, int, Graph]] = []
    for i, p in enumerate(proposals):
        rewind = max(1, min(p.rewind, len(fork_points)))
        # Forking trunk inherits its fast "worker" model, so each branch just
        # executes the shepherd's explicit plan one push at a time — no model
        # switch needed, and N parallel rollouts stay within the time budget.
        branch = await flow.fork(trunk, from_node_id=fork_points[-rewind].id)
        branch.inject(p.strategy)
        forked.append((i, rewind, branch))
        viewer.label(branch.graph_id, f"branch{i} (rewind={rewind})")

    async def play(i: int, branch: Graph) -> None:
        async for _event in stream_saving(
            flow, branch, until=branch_halt, path=root / f"branch{i}", extra=viewer
        ):
            pass

    await asyncio.gather(*(play(i, branch) for i, _rewind, branch in forked))
    viewer.close()  # paint the final live grid once; the report prints below it

    # 4. Score every branch with the env and pick the best (a solver wins outright).
    scored: list[tuple[int, Graph, float, str]] = []
    boards: list[tuple[str, str]] = []
    solved_sigs: set = set()
    for i, rewind, branch in forked:
        benv = flow.repl_for(branch).env
        status = (
            "solved" if benv.get("solved")
            else "deadlock" if benv.get("deadlock")
            else f"dist={benv.get('dist')}"
        )
        scored.append((i, branch, score(benv), status))
        boards.append((f"branch {i} [{status}]", await render(flow, branch)))
        if benv.get("solved"):
            solved_sigs.add(signature(benv.get("filled", "")))
        print(f"  branch {i} (rewind={rewind}): {status} (score={score(benv)})")

    # Diversity report: show every branch's final board side by side (distinct
    # placements are obvious at a glance) and count how many DISTINCT solved
    # target-fillings the fan-out actually produced.
    n_solved = sum(1 for _, _, s, _ in scored if s > 0)
    print(
        f"\n[diversity] {n_solved}/{len(scored)} branches solved via "
        f"{len(solved_sigs)} distinct target placements:"
    )
    print(side_by_side(boards))

    best_i, best, _best_score, _ = max(scored, key=lambda b: b[2])
    flow.discard(*[b for _, b, _, _ in scored if b is not best])
    print(f"\n[pick] best is branch {best_i}")
    best.save(root / "best")
    return best, bool(flow.repl_for(best).env.get("solved"))


async def run_shepherd(
    root: Path,
    *,
    model: str,
    worker_model: str,
    max_depth: int,
    max_iters: int,
    n_branches: int,
    worker_pushes: int,
    branch_pushes: int,
    request_timeout: float,
) -> None:
    # Inject the game class as a tool: every REPL (and every fork) gets it.
    # The worker plays with a small/fast model (myopic, blunders); the shepherd
    # meta-agent reasons with the stronger default model, and recovery branches
    # then execute its explicit plan on the fast model. A per-request timeout
    # keeps a single stuck model call from hanging the whole run.
    flow = Flow(
        build_client(model),
        llm_clients={"worker": build_client(worker_model)},
        tools=[Sokoban],
        max_depth=max_depth,
        max_iters=max_iters,
        llm_request_timeout=request_timeout,
    )
    trunk = Graph(query=WORKER_QUERY, model="worker")
    # The meta-agent is its own rlm (default/strong model), consulted once with
    # structured output when the worker dead-ends.
    shepherd = Graph(query=META_QUERY.format(n=n_branches), output_schema=Proposals)

    # One shared live view: the worker board, then every recovery branch side by
    # side, each updating push-by-push as the streams advance.
    viewer = PanelViewer(
        title="shepherd — worker deadlock ▸ parallel recovery branches",
        cols=min(n_branches, 4),
        status_of=lambda g: panel_status(flow, g),
    )

    try:
        await seed_turn(flow, trunk, CONSTRUCT)
        trunk.save(root / "worker")
        print(f"=== board === (worker={worker_model}, recovery/meta={model})")
        print(await render(flow, trunk))

        best, solved = await branch_and_pick(
            flow,
            trunk,
            shepherd,
            root=root,
            n_branches=n_branches,
            worker_pushes=worker_pushes,
            branch_pushes=branch_pushes,
            viewer=viewer,
        )

        print("\n=== final board ===")
        print(await render(flow, best))
        print(f"[done] solved={solved} pushes={flow.repl_for(best).env.get('pushes')}")
    finally:
        viewer.close()  # safety net; no-op if already closed after the branches
        flow.close_repls()


def self_test() -> None:
    """Offline check of the honest core: drive the injected ``Sokoban`` class
    directly (no LLM, no graph) and assert the trap deadlocks and the verified
    line solves."""
    trap = Sokoban(BOARD)
    print(trap.render())
    for _ in range(4):
        trap.push("R")
    assert trap.deadlock, "marching the box right should freeze it against the wall"
    print("trap confirmed: 'keep right' deadlocks against the wall")

    game = Sokoban(BOARD)
    for move in WIN_MOVES:
        game.push(move)
    assert game.solved and not game.deadlock, "verified line must solve the board"
    print(f"verified line solves in {len(WIN_MOVES)} moves")
    print(game.render())
    print("self-test OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shepherd backtrack-and-branch (Sokoban)")
    add_model_args(parser, default="gpt-5-mini", fast_default="gpt-5-mini")
    parser.add_argument(
        "--worker-model",
        default=None,
        help="Model for the myopic worker (defaults to --fast-model).",
    )
    add_flow_args(parser, max_depth=1, max_iters=32)
    add_out_dir_arg(parser, "shepherd", help="Save the worker/branch graphs here.")
    parser.add_argument(
        "--branches",
        type=int,
        default=8,
        help="how many diverse recovery branches the meta-agent proposes and we "
        "fork in parallel (each is a full push-by-push rollout, so this many LLM "
        "trajectories run concurrently)",
    )
    parser.add_argument(
        "--worker-pushes",
        type=int,
        default=6,
        help="moves the worker gets before the meta-agent intervenes (the myopic "
        "'keep right' baseline deadlocks against the wall after 4)",
    )
    parser.add_argument(
        "--branch-pushes",
        type=int,
        default=9,
        help="move cap per recovery branch, so none runs away (the optimal solve "
        "is 5 pushes; detours around the pillar need a few more)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=90.0,
        help="per-LLM-request timeout (seconds); guards against a stuck call",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=900.0,
        help="overall wall-clock budget (seconds) before the run is aborted",
    )
    parser.add_argument("--self-test", action="store_true", help="Offline env check.")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    root = Path(args.out_dir).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    async def _driver() -> None:
        await asyncio.wait_for(
            run_shepherd(
                root,
                model=args.model,
                worker_model=args.worker_model or args.fast_model,
                max_depth=args.max_depth,
                max_iters=args.max_iters,
                n_branches=args.branches,
                worker_pushes=args.worker_pushes,
                branch_pushes=args.branch_pushes,
                request_timeout=args.request_timeout,
            ),
            timeout=args.time_budget,
        )

    try:
        asyncio.run(_driver())
    except (asyncio.TimeoutError, TimeoutError):
        print(f"[timeout] aborted after exceeding the {args.time_budget:.0f}s budget")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
