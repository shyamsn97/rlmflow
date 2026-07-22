"""Shepherd: a meta-agent reverts & branches a stuck worker *from its own REPL*.

A weak *worker* plays a tiny Sokoban puzzle one irreversible push at a time and,
told to greedily "keep pushing right", jams a box against a wall — a wrong turn
no forward play can undo (a box can be pushed, never pulled). The capability it
lacks is *backtracking*, supplied not by Python glue but by a second rlm — the
**shepherd** — that does the recovery itself, in its REPL, via injected
primitives (never advertised as tools):

  - ``worker``         — read-only view of the jam (``.status/.board/.moves``).
  - ``preview(k)``     — post-rewind board/coords (no play); plan against these.
  - ``branch(specs)``  — each spec ``{"rewind": k, "strategy": s}`` forks the
                         worker, undoes its last ``k`` pushes, injects ``s`` plus
                         the post-rewind board, and runs the worker agent; specs
                         run in parallel and return ``Branch`` handles
                         (``.solved/.dist/.score/.board()``).
  - ``pick(branch)``   — commit the winner.

Everything else (which box→goal, how many branches, scoring) is the shepherd
writing Python. ``await branch(...)`` works because ``LocalRepl.run`` allows
top-level await — the same nested-driving pattern as ``launch_subagents``, over
independent forked graphs (see ``docs/internal/shepherd_rlm_redesign.md``).
Python only does setup (build the stuck worker) + teardown (report, keep winner);
the env is the verifier (``score``), never a solver. ``Sokoban`` is
``flow.inject``-ed into every REPL (incl. forks); the game instance is built by
one replayable action so fork/revert reconstruct each branch's game by replay.

Board glyphs:  ``#`` wall · (space) floor · ``@`` player · ``$`` box ·
``.`` target · ``*`` box on target · ``+`` player on target

Run:
    python examples/shepherd/shepherd.py                 # live agent tree; needs an API key
    python examples/shepherd/shepherd.py --gradio        # live board dashboard instead
    python examples/shepherd/shepherd.py --self-test     # offline env check, no LLM
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

from rflow import (
    ConsumerGroup,
    ExecAction,
    ExecOutput,
    Flow,
    Graph,
    GraphCheckpointer,
    LLMOutput,
    LiveGraphTree,
    PromptProfile,
    inject_tools,
    register_halt,
    tool,
)

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import add_flow_args, add_model_args, add_out_dir_arg, build_client  # noqa: E402
from sokoban import Sokoban  # noqa: E402
from view import PanelViewer, panel_status, side_by_side  # noqa: E402

# Trap board: 3 boxes / 4 goals, with interior walls for routing choices (which
# side to approach, which goal to fill) without forcing deadlocks for correct play.
# The ``##`` at (2,9) turns the far-right cell into a NE corner: greedily pushing
# box B1 right freezes it there after 6 pushes, and that jammed state is UNSOLVABLE,
# so rewind is required. The surplus goal (4 goals, 3 boxes) lets branches fill
# different goals, so recoveries diverge into distinct box->goal assignments.
# Verified offline: solvable from the start, and B1-right deadlocks (B1 frozen).
BOARD = [
    "###########",
    "#  .   .  #",
    "#  ##    ##",
    "# @$      #",
    "#     ##  #",
    "#  $  .   #",
    "#      $  #",
    "#  .   .  #",
    "###########",
]


# The single replayable action that builds per-REPL game state; fork/revert re-run
# it (plus the worker's pushes) to rebuild each branch's own game, so reversibility
# is exact. The board is a literal (not INPUTS) to skip the "inspect INPUTS" turn a
# weak worker loops on. ``push`` prints a play-by-play (the honest walk-and-shove);
# the full board is harness-fed each turn by ``board_prompt`` (see PromptProfile).
CONSTRUCT = (
    f"game = Sokoban({BOARD!r}, env=ENV)\n"
    "def push(box, direction):\n"
    "    print(game.push(box, direction))\n"
    "def legal_pushes():\n"
    "    return game.legal_pushes()\n"
)

# The one place the push mechanic is explained; every prompt below references it.
PUSH_RULES = (
    "push(box, direction) shoves ONE box one cell — box is a B<n> id (e.g. "
    '"B1") and direction is "up", "down", "left", or "right". You act at the box '
    "level: the game walks the player around to the right side of the box and "
    "shoves it for you (a real cell-by-cell walk, no teleport), so you never count "
    "steps or fumble the standing side — just say which box goes which way. A push "
    "is illegal (nothing moves) if the cell the box would enter is blocked, or the "
    "player can't reach the far side to push from. Boxes push, never pull: once a "
    "box sits flat against a wall it is stuck there, so never push a box against a "
    "wall unless its goal is on that wall. The legal pushes are listed for you each "
    "turn — pick one of those."
)

# The worker is a myopic Sokoban player, not an orchestrator: it gets this lean
# system prompt (a "worker" profile) instead of the full RLM prompt. It is still a
# REPL agent (one repl block/turn + done()), just without delegation/fan-out/INPUTS
# scaffolding. See docs/internal/shepherd_worker_prompt.md.
WORKER_SYSTEM = (
    "You are playing a Sokoban puzzle in a Python REPL, one push at a time.\n\n"
    "Protocol:\n"
    "- Each turn, write exactly ONE ```repl``` block, then stop. The updated board "
    "and the legal pushes are shown to you automatically — never call status() or "
    "legal_pushes().\n"
    "- The game is already built; act only through push(box, direction). "
    "push() prints the play-by-play of what it did (or why the push was illegal).\n"
    '- Call done("solved") ONLY when the board reports SOLVED.\n'
    '- Call done("stuck") ONLY when Legal pushes now: none. An illegal / rejected '
    "push is NOT stuck — pick a different legal push next turn. Never call "
    'done("blocked") / done("BLOCKED").\n\n'
    f"{PUSH_RULES}"
)

# The neutral base query (just the goal, no protocol — that lives in WORKER_SYSTEM,
# no strategy — that is injected), shared by the base worker and every fork.
WORKER_QUERY = (
    "Solve this Sokoban puzzle by pushing every box ($) onto a target (.). The "
    "game is already set up in your REPL; the current board is shown to you fresh "
    "at the end of every turn — act from that, no need to call status()."
)


def board_prompt(flow: Flow, graph: Graph) -> str | None:
    """Per-turn observation + legal pushes + action cue, committed as a UserQuery.

    This is the per-turn hook that arms the one-push-per-turn guard: the user
    builder runs exactly once per live LLM turn (never during fork replay, which
    only re-execs ``ExecAction``s), so ``begin_turn`` here resets the guard for
    whoever is driving — the base worker or a self-driven recovery child alike,
    with no host loop in the middle.
    """
    try:
        game = flow.get_var(graph, "game")
    except Exception:
        return None
    game.begin_turn()  # arm the one-push guard for this turn (replay never runs this)
    legal = ", ".join(game.legal_pushes()) or "none"
    if game.solved:
        cue = 'Board is SOLVED — call done("solved") now.'
    elif legal == "none":
        cue = 'No legal pushes left — call done("stuck").'
    else:
        cue = (
            "Make your next push now: pick ONE of the legal pushes above and call "
            "push(box, direction) in a single ```repl``` block. Do NOT call done() "
            "while legal pushes remain (an illegal attempt is not the end)."
        )
    return (
        "Current board:\n"
        f"{game.status()}\n\n"
        f"Legal pushes now: {legal}\n\n"
        f"{cue} "
        "Do not call status() or legal_pushes() — this view is already current."
    )

# The deliberate blunder-inducer, injected onto the worker only (kept out of
# WORKER_QUERY so forks don't inherit it): greedily march box B1 right.
MYOPIC_STRATEGY = (
    "Strategy: box B1 is right in front of you, so the obvious greedy plan is to "
    'just keep pushing it right — push("B1", "right") EVERY turn, over and over. '
    "Only ever push B1, never another box; stay simple and greedy and don't "
    "overthink detours."
)

# Injected onto each fork AFTER the rewind: overrides the failed "keep right"
# instinct and grounds the worker on the post-rewind board (jam coords are stale).
BRANCH_GUIDANCE = (
    'Your earlier "keep pushing right" plan FAILED — it shoved box B1 into the '
    "wall. That last mistake was undone ({k} push decision(s) rewound). Ignore "
    "the old plan and follow the recovery plan below, working its boxes in order: "
    "finish placing ONE box fully on its goal before starting the next. The plan "
    "tells you WHICH box goes WHERE and in what ORDER — call push(box, direction) "
    "to move each box (the game walks the player around for you). Each turn the "
    "legal pushes are shown to you — pick the single push that makes progress on "
    "the current box, and keep going one push at a time "
    'until the board reports SOLVED, then call done("solved"). If a push is '
    "rejected as illegal, try another legal push — do not call done().\n\n"
    "Your starting state AFTER the rewind (authoritative — use THESE coordinates; "
    "any jammed-board coords in the plan are stale):\n"
    "{status}\n\n"
    "Player: {player}\n"
    "Boxes: {boxes}\n"
    "Goals: {goals}\n\n"
    "Recovery plan for this branch:\n{strategy}"
)

# The shepherd's query: a plain REPL agent (no output schema) whose primitives
# (worker/branch/pick) are injected; this prompt just describes them and the job.
META_QUERY = """\
You are the shepherd of a myopic Sokoban worker. It jammed a box against a wall \
and cannot backtrack. Recover it from your REPL by reverting and branching.

Rules: {rules}

Injected (call directly):
- `worker.status()` / `worker.board()` — text to eyeball (not data).
- `worker.player` / `worker.boxes` / `worker.goals` — jammed coords; STALE for \
plans — use `preview` coords instead.
- `worker.moves()` — push list; `len(...)` is max rewind.
- `await preview(k)` → dict with keys `rewind`, `board`, `status`, `player`, \
`boxes`, `goals` (`rewind` is the clamped depth). Preview each k before planning.
- `await branch([{{"rewind": k, "strategy": "..."}}, ...])` — fork, undo k \
pushes, inject strategy, run in parallel. Each handle: `.solved`, `.dist`, \
`.score`, `.board()`.
- `pick(branch)` — commit the winner.

Job: inspect jam → preview → ~{n} diverse branches (cap {cap} pushes each) → \
compare `.score` → `pick` best → `done(...)`.

Rewind depth: the worker's last attempt is often an ILLEGAL push that moved \
nothing. `rewind=1` alone usually leaves the box still jammed — preview and \
rewind enough successful pushes to free it (often 2–4). Diversity is in \
`strategy` (box→goal assignment / order / approach), not only in rewind. More \
goals than boxes — vary which goals you fill.

Each `strategy` is a PLAN against `preview(k)` coords — which box → which goal, \
in what order, rough route/side. NOT a push list (those go wrong). Example \
shape (replace coords from preview):
  "Unstack B1 first.
   1) B1 -> goal (1,4): clear aside, then bring up.
   2) B2 -> goal (5,3): push down once path is free.
   3) B3 -> goal (1,2): left along row 1."
"""

def score(env: dict) -> float:
    """Verifier score (never a solver): any solve is positive and beats any
    unsolved branch (fewer pushes ranks higher); unsolved branches rank by how
    close they got, so we still keep the most-progressed one if none solves."""
    if env.get("solved"):
        return 1000.0 - float(env.get("pushes", 0))
    return -1000.0 - float(env.get("dist", 0))


async def seed_turn(flow: Flow, graph: Graph, code: str) -> None:
    """Record a scripted (llm_output, exec_action) pair and run it, so game
    construction is a replayable ExecAction every fork reconstructs."""
    graph.commit(LLMOutput(content="construct the Sokoban game", code=code))
    graph.commit(ExecAction(code=code))
    await flow.exec_turn(graph, code)


async def render(flow: Flow, graph: Graph) -> str:
    return (await flow.repl_for(graph).run("print(game.render())")).strip()


def export_run_traces(root: Path, named_games: list) -> None:
    """Persist each game's play-by-play to ``root/traces/``: a JSON frame trace
    (``<name>.json``, one ``{label, board}`` per honest sub-step) always, plus an
    animated board GIF (``<name>.gif``) when Pillow + the sprite tiles are present.
    The JSON is the raw record (replayable/inspectable offline); the GIF is just a
    rendering of it, so the run never hard-depends on Pillow."""
    import json

    import sprites

    out = root / "traces"
    out.mkdir(parents=True, exist_ok=True)
    have_tiles = sprites.available()
    if not have_tiles:
        print("[viz] Pillow/tiles unavailable — writing trace JSON only (no GIFs)")
    for name, game in named_games:
        steps = list(getattr(game, "step_frames", []))
        trace = [{"label": label, "board": board} for label, board in steps]
        (out / f"{name}.json").write_text(json.dumps(trace, indent=2))
        renders = [board for _label, board in steps]
        wrote_gif = have_tiles and sprites.save_gif(renders, out / f"{name}.gif")
        suffix = " + .gif" if wrote_gif else ""
        print(f"[viz] {name}: {len(trace)} steps -> {out}/{name}.json{suffix}")


def signature(assignment: str) -> tuple:
    """A branch's solution signature = the box->goal ASSIGNMENT (each box's final
    cell, in stable id order), to count how many distinct solutions were found."""
    return tuple(tuple(map(int, p.split(","))) for p in assignment.split(";") if p)


def is_push(node: LLMOutput) -> bool:
    """A worker decision turn whose code calls ``push(...)``. Excludes CONSTRUCT
    (defines ``push`` / builds the game) and ``done(...)`` turns."""
    code = node.code or ""
    return "push(" in code and "Sokoban(" not in code


def is_successful_push(graph: Graph, node: LLMOutput) -> bool:
    """True when ``node`` is a push turn whose next ``ExecOutput`` actually shoved
    a box.

    Rewind anchors on box-*pushes* (the irreversible progress): an illegal push
    that moved nothing isn't a rewind step — undoing it wouldn't free the jam. A
    successful push prints a play-by-play headed ``pushed <box> <dir> (...)``; an
    illegal one reads ``ILLEGAL``/``NO``/``UNKNOWN``.
    """
    if not is_push(node):
        return False
    seen = False
    for n in graph.nodes:
        if n.id == node.id:
            seen = True
            continue
        if not seen:
            continue
        if isinstance(n, ExecOutput):
            return (n.content or "").lstrip().startswith("pushed")
        if isinstance(n, LLMOutput):
            return False
    return False


def push_turns(graph: Graph) -> list[LLMOutput]:
    """Successful push decisions, oldest first — the rewind candidates."""
    return [
        n
        for n in graph.nodes
        if isinstance(n, LLMOutput) and is_successful_push(graph, n)
    ]


class Branch:
    """A handle to one recovery rollout, returned by ``branch(...)``. Wraps the
    forked graph and reads the env's verifier signals so the shepherd can compare
    rollouts (``.solved`` / ``.dist`` / ``.score`` / ``.board()``)."""

    def __init__(
        self, flow: Flow, graph: Graph, *, index: int, rewind: int, strategy: str
    ) -> None:
        self._flow = flow
        self.graph = graph
        self.index = index
        self.rewind = rewind
        self.strategy = strategy

    @property
    def _env(self) -> dict:
        return self._flow.repl_for(self.graph).env

    @property
    def solved(self) -> bool:
        return bool(self._env.get("solved"))

    @property
    def dist(self) -> int:
        return int(self._env.get("dist", 0))

    @property
    def pushes(self) -> int:
        return int(self._env.get("pushes", 0))

    @property
    def score(self) -> float:
        return score(self._env)

    def board(self) -> str:
        # LocalRuntime: get_var returns the live game object -> render directly.
        return self._flow.get_var(self.graph, "game").render()

    def __repr__(self) -> str:
        state = "solved" if self.solved else f"dist={self.dist}"
        return f"<branch{self.index} rewind={self.rewind} {state} score={self.score:.0f}>"


class WorkerView:
    """Read-only view of the jammed worker injected as ``worker``. No control
    methods — the worker already ran; the shepherd only reverts/branches it."""

    def __init__(self, flow: Flow, graph: Graph) -> None:
        self._flow = flow
        self.graph = graph

    def _game(self):
        return self._flow.get_var(self.graph, "game")

    def board(self) -> str:
        return self._game().render()

    def status(self) -> str:
        return self._game().status()

    @property
    def player(self) -> tuple[int, int]:
        """The worker's cell as ``(row, col)`` — a plain tuple, not a string."""
        return self._game().player

    @property
    def boxes(self) -> list[tuple[int, int]]:
        """Box cells as a sorted list of ``(row, col)`` tuples."""
        return sorted(self._game().boxes)

    @property
    def goals(self) -> list[tuple[int, int]]:
        """Goal cells as a sorted list of ``(row, col)`` tuples."""
        return sorted(self._game().targets)

    def moves(self) -> list[str]:
        """The worker's push decisions, oldest first — the rewind candidates
        (``rewind=1`` redoes only the last one)."""
        return [(n.code or "").strip() for n in push_turns(self.graph)]


def make_shepherd_tools(
    flow: Flow,
    worker: Graph,
    shepherd: Graph,
):
    """Build the shepherd's injected REPL surface. Returns ``(namespace, branches,
    picked)``: the namespace is injected into the shepherd's REPL, and the host
    reads ``branches``/``picked`` after the shepherd finishes.

    Recovery branches are **children of the shepherd**: ``branch`` rewinds the
    worker into forks, then hands them to ``flow.launch_subgraphs`` — the warm,
    graph-first counterpart to ``launch_subagents``. The forks are adopted under
    the shepherd (one
    ``graph_id``, nested ``agent_id``s) and self-drive on the shepherd's own task
    queue, so their events flow through the shepherd's run — one save root, one live
    tree, no off-tree side-runs. Each child stops itself (``done`` on solved/stuck),
    with the depth>0 ``child_max_iters`` as the hard push cap (~``branch_pushes``).
    """
    branches: list[Branch] = []
    picked: dict[str, Branch] = {}
    view = WorkerView(flow, worker)

    def clamp_rewind(rewind: int) -> int:
        points = push_turns(worker)
        if not points:
            raise ValueError("worker has no push decisions to rewind")
        return max(1, min(int(rewind), len(points)))

    def snapshot(fork: Graph, k: int) -> dict:
        """Post-rewind board/coords from a live forked REPL."""
        game = flow.get_var(fork, "game")
        return {
            "rewind": k,
            "board": game.render(),
            "status": game.status(),
            "player": game.player,
            "boxes": sorted(game.boxes),
            "goals": sorted(game.targets),
        }

    async def open_rewind(rewind: int) -> tuple[Graph, int, dict]:
        """Fork the worker at ``rewind`` push decisions and return ``(fork, k, snapshot)``."""
        k = clamp_rewind(rewind)
        # Count PUSH turns, not raw decision turns: the trajectory also holds the
        # scripted construct turn and the final done("stuck") turn, so a plain
        # rewind(n=k) would undo those instead of real moves and leave the jam
        # in place. ``where=is_successful_push`` anchors on the k-th push from the end.
        fork = await flow.rewind(
            worker, n=k, where=lambda node: is_successful_push(worker, node)
        )
        return fork, k, snapshot(fork, k)

    @tool(
        "Rewind the worker by `rewind` push decisions WITHOUT playing; returns a "
        "dict of the post-rewind board/status/player/boxes/goals. Write strategy "
        "plans against THESE coords (worker.boxes is the stale jammed state)."
    )
    async def preview(rewind: int) -> dict:
        """Show the board after undoing ``rewind`` pushes — without playing.
        Use this to write strategy plans against the coords the fork will
        actually start from (``worker.boxes`` is the jammed state)."""
        fork, _k, snap = await open_rewind(rewind)
        flow.discard(fork)
        return snap

    async def prepare_branch(
        rewind: int, strategy: str
    ) -> tuple[Graph, str, str, Branch]:
        """Rewind the worker into a fork and build its recovery guidance, child
        name, and ``Branch`` handle. The guidance is the fork's next user turn;
        the fork carries the worker's prompt_profile/model, so nothing else is
        needed. Returns ``(fork, guidance, name, branch)`` for ``launch_subgraphs``."""
        fork, k, snap = await open_rewind(rewind)
        guidance = BRANCH_GUIDANCE.format(
            k=k,
            status=snap["status"],
            player=snap["player"],
            boxes=snap["boxes"],
            goals=snap["goals"],
            strategy=strategy,
        )
        idx = len(branches)
        b = Branch(flow, fork, index=idx, rewind=k, strategy=strategy)
        branches.append(b)
        return fork, guidance, f"b{idx}", b

    @tool(
        "Fork the worker, undo the last `rewind` push decisions per spec, inject "
        "each recovery strategy + its post-rewind board, and run the forks as "
        "child agents IN PARALLEL. specs=[{'rewind': int, 'strategy': str}]; "
        "returns one Branch handle per spec (with .solved / .dist / .score / .board())."
    )
    async def branch(specs: list[dict]) -> list[Branch]:
        """Revert-and-branch primitive. Each spec ``{"rewind": int, "strategy":
        str}`` rewinds the worker's last ``rewind`` push-decisions into a fork and
        hands it the recovery plan. The forks are launched as the shepherd's own
        children (via ``launch_subgraphs``, the warm counterpart to
        ``launch_subagents``) and run in parallel on the shepherd's queue; returns
        the ``Branch`` handles in order."""
        prepared = [await prepare_branch(**s) for s in specs]
        await flow.launch_subgraphs(
            shepherd,
            [fork for fork, _g, _n, _b in prepared],
            queries=[guidance for _f, guidance, _n, _b in prepared],
            names=[name for _f, _g, name, _b in prepared],
        )
        return [b for _f, _g, _n, b in prepared]

    @tool("Commit the winning Branch as the recovery; the host keeps it, discards the rest.")
    def pick(b: Branch) -> str:
        """Commit the winning branch; the host keeps it and discards the rest."""
        picked["branch"] = b
        return f"picked branch{b.index} ({'solved' if b.solved else f'dist={b.dist}'})"

    namespace = {"worker": view, "preview": preview, "branch": branch, "pick": pick}
    return namespace, branches, picked


def register_halts(flow: Flow, *, jam_pushes: int) -> None:
    """Name the host-driven ``worker_stop`` boundary once, up front, so the worker's
    ``run_streaming`` selects it by name (see ``rflow.tools.graph_ops``). It stops
    the instant the worker solves, jams, or hits its push budget — the cue to hand
    off to the shepherd. (Recovery branches self-drive via ``launch_subgraphs``, so
    they aren't host-driven and can't use a named ``until``; their push budget is
    enforced by ``Flow(child_max_iters=...)`` instead — see ``run_shepherd``.)"""

    def env(graph: Graph) -> dict:
        return flow.repl_for(graph).env

    register_halt(
        flow,
        "worker_stop",
        lambda _e, g: bool(
            env(g).get("solved")
            or env(g).get("blocked")
            or env(g).get("pushes", 0) >= jam_pushes
        ),
    )


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
    request_timeout: float,
    dashboard=None,
) -> None:
    # Worker plays on the fast model (myopic); the shepherd reasons on the strong
    # default model and drives recovery from its REPL. Timeout guards stuck calls.
    #
    # The recovery branches self-drive (``launch_subgraphs``), so no host ``until``
    # can cap them — their push budget is the depth>0 ``child_max_iters``, checked
    # in the agent loop. Each child's LLM turns are ~ CONSTRUCT(1) + total pushes +
    # done(1), so ``branch_pushes + 3`` caps a branch at about ``branch_pushes``
    # pushes (illegal-push retries also burn turns, which is fine — they're wasted).
    # The shepherd itself is the root, so it keeps the roomier ``max_iters``.
    flow = Flow(
        build_client(model),
        llm_clients={"worker": build_client(worker_model)},
        # The worker is a myopic player, not an orchestrator: a lean profile (REPL
        # protocol + PUSH_RULES only) plus harness-fed board each turn. The
        # shepherd stays on the full default prompt. prompt_router="graph" honors
        # Graph.prompt_profile without advertising "worker" to the shepherd LLM.
        prompts={
            "worker": PromptProfile(
                system=WORKER_SYSTEM,
                user=board_prompt,
                description="myopic Sokoban worker",
            )
        },
        prompt_router="graph",
        max_depth=max_depth,
        max_iters=max_iters,
        child_max_iters=branch_pushes + 3,
        llm_request_timeout=request_timeout,
    )
    # Ambient dependency (not a tool): in every REPL incl. forks, so replaying
    # CONSTRUCT works on every branch; unadvertised (no @tool metadata).
    flow.inject("Sokoban", Sokoban)
    # Named stop conditions the worker run and each branch rollout select by name.
    register_halts(flow, jam_pushes=jam_pushes)

    # prompt_profile="worker" stamps this graph (and its forks, which deep-copy
    # it) onto the lean profile above; prompt_router="graph" reads that stamp.
    worker = Graph(query=WORKER_QUERY, model="worker", prompt_profile="worker")
    shepherd = Graph(
        query=META_QUERY.format(rules=PUSH_RULES, n=n_branches, cap=branch_pushes)
    )

    # Boards (terminal / Gradio sink) + live agent tree — one ConsumerGroup fans
    # handle/close; label() stays on the individual consumers (not a core API).
    title = "shepherd — worker hits a wall ▸ parallel recovery branches"
    panels = PanelViewer(
        title=title,
        cols=min(n_branches, 4),
        status_of=lambda g: panel_status(flow, g),
        board_of=lambda g: flow.get_var(g, "game").render(),
        # The game publishes this turn's sub-step renders to env["frames"]; the grid
        # reads them off the metadata channel (rather than reaching into the object)
        # and animates the honest walk-around. Lanes with no game (the shepherd)
        # raise KeyError, which the viewer treats as "no frames".
        frames_of=lambda g: flow.get_env_var(g, "frames"),
        sink=(dashboard.set_panels if dashboard is not None else None),
    )
    live = LiveGraphTree(title=title, every_s=0.1)
    viewers = ConsumerGroup([panels, live])
    if dashboard is not None:
        dashboard.set_status("worker playing forward (myopic 'keep right')…")

    def label_viewers(graph_id: str, label: str) -> None:
        for consumer in viewers.consumers:
            fn = getattr(consumer, "label", None)
            if callable(fn):
                fn(graph_id, label)

    try:
        # 1. Setup: build and play the stuck worker forward (the problem
        #    statement). MYOPIC_STRATEGY is on the worker only, so forks re-decide.
        await seed_turn(flow, worker, CONSTRUCT)
        worker.inject(MYOPIC_STRATEGY)
        worker.save(root / "worker")
        print(f"=== board === (worker={worker_model}, shepherd={model})")
        print(await render(flow, worker))
        label_viewers(worker.graph_id, "worker")
        # The one-push guard is armed per turn by ``board_prompt`` (the worker's
        # user builder), so no host-loop ``begin_turn`` is needed here.
        saver = GraphCheckpointer(root / "worker")
        try:
            async for event in flow.run_streaming(
                graph=worker, until="worker_stop"
            ):
                saver.handle(event, worker)
                viewers.handle(event, worker)
        finally:
            saver.close()

        wenv = flow.repl_for(worker).env
        reason = (
            "solved" if wenv.get("solved")
            else "hit a wall" if wenv.get("blocked")
            else f"stopped after {wenv.get('moves')} moves ({wenv.get('pushes')} pushes)"
        )
        live.close()  # stop Rich Live before the handoff print
        print(f"\n[worker] {reason}")
        if wenv.get("solved"):
            print("[worker] solved unaided; no recovery needed")
            worker.save(root / "best")
            if dashboard is not None:
                dashboard.finish("done — worker solved unaided")
            return

        # 2. Hand control to the shepherd: a plain REPL agent that inspects,
        #    branches, scores, and picks entirely in its own REPL.
        if dashboard is not None:
            dashboard.set_status(f"worker {reason} — shepherd recovering in its REPL…")
        live = LiveGraphTree(title=title, every_s=0.1)
        viewers = ConsumerGroup([panels, live])
        label_viewers(worker.graph_id, "worker")
        label_viewers(shepherd.graph_id, "shepherd")
        ns, branches, picked = make_shepherd_tools(flow, worker, shepherd)
        # Scope the primitives to the shepherd's REPL only, so they never leak
        # onto the worker or its forks.
        inject_tools(flow, ns, only=shepherd)
        saver = GraphCheckpointer(root / "shepherd")
        try:
            async for event in flow.run_streaming(graph=shepherd, until="done"):
                saver.handle(event, shepherd)
                viewers.handle(event, shepherd)
        finally:
            saver.close()
        viewers.close()

        # 3. Teardown: commit the winner (the RLM's pick, else best-scored),
        #    report diversity, keep the winner, discard the rest.
        if not branches:
            print("[shepherd] created no branches; nothing to pick")
            if dashboard is not None:
                dashboard.finish("done — shepherd made no branches")
            return

        best = picked.get("branch")
        if best is None:
            best = max(branches, key=lambda b: b.score)
            print(f"[pick] shepherd did not call pick(); defaulting to branch{best.index}")

        board_cols = [
            (f"branch{b.index} [{'solved' if b.solved else f'dist={b.dist}'}]", b.board())
            for b in branches
        ]
        solved_sigs = {
            signature(flow.repl_for(b.graph).env.get("assignment", ""))
            for b in branches
            if b.solved
        }
        n_solved = sum(1 for b in branches if b.solved)
        print(
            f"\n[diversity] {n_solved}/{len(branches)} branches solved via "
            f"{len(solved_sigs)} distinct box→goal assignments:"
        )
        print(side_by_side(board_cols))

        # Persist play-by-play traces (+ GIFs) for the worker and every branch while
        # their REPLs are still open — closing a loser's REPL drops its live game.
        export_run_traces(
            root,
            [("worker", flow.get_var(worker, "game"))]
            + [(f"branch{b.index}", flow.get_var(b.graph, "game")) for b in branches],
        )

        # Branches are children under the shepherd's single graph_id, so we can't
        # discard by graph_id (that would close the shepherd + siblings). Close each
        # loser's REPL by its own (graph_id, agent_id) key instead.
        for b in branches:
            if b is not best:
                flow.close_repl(b.graph)
        best.graph.save(root / "best")
        print(f"\n[pick] best is branch{best.index}")
        print("\n=== final board ===")
        print(best.board())
        print(f"[done] solved={best.solved} pushes={best.pushes}")
        if dashboard is not None:
            dashboard.set_picked(f"branch{best.index}")
            dashboard.finish(f"done — solved={best.solved}, {best.pushes} pushes")
    finally:
        viewers.close()
        flow.close_repls()


def self_test() -> None:
    """Offline check (no LLM): greedily pushing B1 right walks the player around
    and shoves B1 across the board (a real cell-by-cell chain, no teleport) until
    it jams against the wall, a legal push still remains from the jam, and the
    one-push-per-turn guard rejects a second push in the turn."""
    trap = Sokoban(BOARD)
    print(trap.render())
    for _ in range(12):  # guard is inactive until begin_turn(), so this loop is fine
        trap.push("B1", "right")
        if trap.blocked:
            break
    assert trap.pushes >= 3, "pushing B1 right should shove it several cells first"
    assert trap.blocked, "B1 should jam against the far wall"
    assert trap.legal_pushes(), "other pushes remain from the jam (recovery is possible)"
    print(
        f"trap confirmed: 'keep pushing B1 right' jams it after {trap.pushes} pushes "
        f"({trap.moves} honest sub-moves)"
    )

    guard = Sokoban(BOARD)
    guard.begin_turn()
    guard.push("B1", "right")
    try:
        guard.push("B1", "right")
    except RuntimeError:
        print("guard confirmed: a second push in one turn is rejected")
    else:
        raise AssertionError("one-push-per-turn guard did not fire")
    print("self-test OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shepherd backtrack-and-branch (Sokoban)")
    add_model_args(parser, default="gpt-5", fast_default="gpt-5-mini")
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
        help="advisory branch count suggested to the shepherd (it decides)",
    )
    parser.add_argument(
        "--jam-pushes",
        type=int,
        default=6,
        help="pushes the worker plays forward before handing off (this board deadlocks "
        "B1 in exactly 6 right-pushes, so 6 stops the worker right on the clean jam — "
        "a larger cap lets it wander onto other boxes past the jam)",
    )
    parser.add_argument(
        "--branch-pushes",
        type=int,
        default=20,
        help="push cap per recovery branch (a clean recovery needs well under this)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="per-LLM-request timeout (seconds); guards against a stuck call. "
        "Kept generous because the shepherd's first turn is a big reasoning-model prompt",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=900.0,
        help="overall wall-clock budget (seconds) before the run is aborted",
    )
    parser.add_argument("--self-test", action="store_true", help="Offline env check.")
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Open a live Gradio board dashboard instead of the terminal agent tree.",
    )
    parser.add_argument("--gradio-port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument(
        "--gradio-share",
        action="store_true",
        help="Expose the Gradio viewer via a public share link.",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    root = Path(args.out_dir).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    async def _driver(dashboard=None) -> None:
        await asyncio.wait_for(
            run_shepherd(
                root,
                model=args.model,
                worker_model=args.worker_model or args.fast_model,
                max_depth=args.max_depth,
                max_iters=args.max_iters,
                n_branches=args.branches,
                jam_pushes=args.jam_pushes,
                branch_pushes=args.branch_pushes,
                request_timeout=args.request_timeout,
                dashboard=dashboard,
            ),
            timeout=args.time_budget,
        )

    if args.gradio:
        # The Gradio server owns the main thread; the run drives on a worker
        # thread and streams its progress into the dashboard it is handed.
        from gradio_view import launch

        launch(
            _driver,
            port=args.gradio_port,
            share=args.gradio_share,
        )
        return

    try:
        asyncio.run(_driver())
    except (asyncio.TimeoutError, TimeoutError):
        print(f"[timeout] aborted after exceeding the {args.time_budget:.0f}s budget")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
