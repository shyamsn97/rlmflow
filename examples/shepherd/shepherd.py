"""Shepherd: a meta-agent reverts & branches a stuck worker *from its own REPL*.

A weak *worker* plays a tiny Sokoban puzzle one irreversible push at a time and,
told to greedily "keep pushing right", jams a box against a wall — a wrong turn
no forward play can undo (a box can be pushed, never pulled). The capability it
lacks is *backtracking*, supplied not by Python glue but by a second rlm — the
**shepherd** — that does the recovery itself, in its REPL, via primitives
injected into that one REPL (never registered as flow-wide tools, so the worker
and its forks never see them):

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
top-level await: the recovery rollouts are independent forked graphs, driven by
``parallel_stream`` from inside the shepherd's own turn.

A rewind is ``Node.fork()`` at the turn before the k-th push from the end; the
fork's REPL is rebuilt by ``Flow.replay``, which re-execs the recorded
``ExecAction``s and so replays exactly the pushes that were kept.
Python only does setup (build the stuck worker) + teardown (report, keep winner);
the env is the verifier (``score``), never a solver. ``Sokoban`` is
``flow.inject``-ed into every REPL (incl. forks); the game instance is built by
one replayable action so fork/replay reconstruct each branch's game.

Board glyphs:  ``#`` wall · (space) floor · ``@`` player · ``$`` box ·
``.`` target · ``*`` box on target · ``+`` player on target

Terminal visualization now lives in ``rlmflow.consumers``. This demo prints one
short line per streamed node (on stderr — a running REPL captures stdout) plus a
board grid at each milestone, and ``--gradio`` animates the boards push-by-push.

Run:
    python examples/shepherd/shepherd.py                 # node trace + boards; needs an API key
    python examples/shepherd/shepherd.py --gradio        # live board dashboard instead
    python examples/shepherd/shepherd.py --self-test     # offline env check, no LLM
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

from rlmflow import (
    AgentStart,
    DoneOutput,
    ExecAction,
    ExecOutput,
    Flow,
    LLMOutput,
    Node,
    PromptProfile,
    UserPromptBuilder,
    UserQuery,
    parallel_stream,
    start,
    tool,
)

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import (  # noqa: E402, RUF100
    add_flow_args,
    add_model_args,
    add_out_dir_arg,
    build_client,
)
from sokoban import Sokoban  # noqa: E402, RUF100
from view import PanelViewer, panel_status, side_by_side  # noqa: E402, RUF100

# Trap board: 3 boxes / 4 goals. The player starts hard against the left wall with
# B1 just to its right, so the myopic "always push right" worker shoves B1 straight
# down the top corridor into the NE corner (``##`` cap at row2/c10) after 8 pushes;
# that jammed state is UNSOLVABLE, so rewind is required. A wall band with gaps sits
# above the corridor (row2) so B1 can only be redirected off the death line at a few
# columns -- the valid rewind points (depths 1/5/7). The surplus goal (4 goals, 3
# boxes) plus scattered, non-aligned goal placement lets branches fill different
# goals, so recoveries diverge into distinct box->goal assignments.
# Verified offline: solvable from the start, and B1-right deadlocks (B1 frozen).
BOARD = [
    "############",
    "#  .    . ##",
    "### # # # ##",
    "#@$        #",
    "#   # ### .#",
    "#  $     $ #",
    "#  ## # #  #",
    "# .        #",
    "############",
]


# The single replayable action that builds per-REPL game state; a fork's replay
# re-runs it (plus the pushes that survived the rewind) to rebuild that branch's
# own game, so reversibility is exact. The board is a literal (not INPUTS) to skip
# the "inspect INPUTS" turn a weak worker loops on. ``push`` prints a play-by-play
# (the honest walk-and-shove); the full board is harness-fed each turn by
# ``board_prompt`` (see PromptProfile).
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
# scaffolding.
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


def board_prompt(flow: Flow, agent: AgentStart) -> str | None:
    """Per-turn observation + legal pushes + action cue, committed as a UserQuery.

    This is the per-turn hook that arms the one-push-per-turn guard: the user
    builder runs exactly once per live LLM turn (never during a fork's replay,
    which only re-execs ``ExecAction``s), so ``begin_turn`` here resets the guard
    for whoever is driving — the base worker or a recovery branch alike, with no
    host loop in the middle.
    """
    try:
        game = flow.runtime.get_var(agent, "game")
    except Exception:  # noqa: BLE001 - missing runtime state means no live board
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

Make the branches genuinely different — this is a search, not {n} copies of one \
plan. Vary BOTH axes:
- Rewind depth: don't send every branch at the same `k`. Not every depth frees \
the box — some just leave it jammed against the same wall — so preview a range of \
depths, keep the ones whose board actually opens an escape, and spread branches \
across at least 2–3 of those distinct rewinds so they start from genuinely \
different boards.
- Strategy: write each plan by hand for that branch's preview board — a distinct \
box→goal assignment, order, and approach. There are more goals than boxes, so \
different branches should fill different goals. Do NOT mass-produce strategies \
from one formula/loop; a set of near-identical Manhattan plans is a wasted search.

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


async def seed_turn(flow: Flow, agent: AgentStart, code: str) -> None:
    """Record a scripted (llm_output, exec_action) pair and run it, so game
    construction is a replayable ExecAction every fork reconstructs."""
    llm_output = agent.frontier.append(LLMOutput(content="construct the Sokoban game", code=code))
    await flow.step(llm_output.append(ExecAction(code=code)))


def board(flow: Flow, agent: AgentStart) -> str:
    """The agent's live board. LocalRuntime hands back the real game object."""
    return flow.runtime.get_var(agent, "game").render()


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
        trace = [{"label": label, "board": board_text} for label, board_text in steps]
        (out / f"{name}.json").write_text(json.dumps(trace, indent=2))
        renders = [board_text for _label, board_text in steps]
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


def is_successful_push(node: LLMOutput) -> bool:
    """True when ``node`` is a push turn whose exec actually shoved a box.

    Rewind anchors on box-*pushes* (the irreversible progress): an illegal push
    that moved nothing isn't a rewind step — undoing it wouldn't free the jam. A
    successful push prints a play-by-play headed ``pushed <box> <dir> (...)``; an
    illegal one reads ``ILLEGAL``/``NO``/``UNKNOWN``. A turn's exec output is two
    hops down its own agent: the ``ExecAction`` the engine derived, then its result.
    """
    if not is_push(node):
        return False
    action = node.next
    result = action.next if action is not None else None
    return isinstance(result, ExecOutput) and (result.content or "").lstrip().startswith("pushed")


def push_turns(agent: AgentStart) -> list[LLMOutput]:
    """Successful push decisions, oldest first — the rewind candidates."""
    return [n for n in agent.transcript() if isinstance(n, LLMOutput) and is_successful_push(n)]


def rewind_point(agent: AgentStart, rewind: int) -> tuple[Node, int]:
    """The node to fork at to undo ``rewind`` push decisions, and the clamped depth.

    Counts PUSH turns, not raw turns: the transcript also holds the scripted
    construct turn, the injected strategy, and any illegal-push turns, so cutting
    a fixed number of nodes off the end would undo those instead of real moves and
    leave the jam in place. The cut lands just before the k-th push from the end,
    skipping back over that turn's harness-fed board (its ``UserQuery``s), which
    the fork will be shown afresh anyway.
    """
    pushes = push_turns(agent)
    if not pushes:
        raise ValueError("worker has no push decisions to rewind")
    k = max(1, min(int(rewind), len(pushes)))
    cut = pushes[-k].prev
    while isinstance(cut, UserQuery):
        cut = cut.prev
    return cut, k


def trace_line(node: Node, lanes: dict[str, str]) -> str:
    """One short line for a streamed node — what the live agent tree used to show.

    ``lanes`` maps a root agent's id to the host's name for it ("worker",
    "shepherd", "branch3"), since every graph here is its own root and so is
    called ``root`` by the engine.
    """
    if isinstance(node, ExecAction):
        return ""  # the code already went by on the LLM turn that wrote it
    agent = node.parent_agent
    lane = lanes.get(agent.root.id, agent.config.name)
    if isinstance(node, LLMOutput):
        detail = node.code
    elif isinstance(node, DoneOutput):
        detail = f"done({node.result!r})"
    else:
        detail = node.content
    head = next((line for line in (detail or "").splitlines() if line.strip()), "")
    return f"[{lane:>9}] {node.type:<13} {head.strip()[:88]}"


class Branch:
    """A handle to one recovery rollout, returned by ``branch(...)``. Wraps the
    forked graph and reads the env's verifier signals so the shepherd can compare
    rollouts (``.solved`` / ``.dist`` / ``.score`` / ``.board()``)."""

    def __init__(
        self, flow: Flow, graph: AgentStart, *, index: int, rewind: int, strategy: str
    ) -> None:
        self._flow = flow
        self.graph = graph
        self.index = index
        self.rewind = rewind
        self.strategy = strategy

    @property
    def _env(self) -> dict:
        return self._flow.runtime.repl_for(self.graph).env

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
        return board(self._flow, self.graph)

    def __repr__(self) -> str:
        state = "solved" if self.solved else f"dist={self.dist}"
        return f"<branch{self.index} rewind={self.rewind} {state} score={self.score:.0f}>"


class WorkerView:
    """Read-only view of the jammed worker injected as ``worker``. No control
    methods — the worker already ran; the shepherd only reverts/branches it."""

    def __init__(self, flow: Flow, graph: AgentStart) -> None:
        self._flow = flow
        self.graph = graph

    def _game(self):
        return self._flow.runtime.get_var(self.graph, "game")

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


def make_shepherd_primitives(
    flow: Flow,
    worker: AgentStart,
    *,
    budget: int,
    run_branches,
):
    """Build the shepherd's injected REPL surface. Returns ``(namespace, branches,
    picked)``: the namespace goes into the shepherd's REPL only, and the host reads
    ``branches``/``picked`` after the shepherd finishes.

    A recovery branch is a **fork of the worker's own graph**: ``Node.fork()`` at
    the turn before the k-th push from the end, whose REPL ``Flow.replay`` rebuilds
    by re-executing the ``ExecAction``s that survived the cut. Each fork is a root
    in its own right, so ``run_branches`` drives them all together on this same
    Flow (``parallel_stream``) and each gets its own ``until`` boundary — the push
    cap that used to be a ``child_max_iters`` — with ``max_iters`` left as a
    backstop for turns burned on illegal pushes.
    """
    branches: list[Branch] = []
    picked: dict[str, Branch] = {}
    view = WorkerView(flow, worker)

    def snapshot(fork: AgentStart, k: int) -> dict:
        """Post-rewind board/coords from a replayed forked REPL."""
        game = flow.runtime.get_var(fork, "game")
        return {
            "rewind": k,
            "board": game.render(),
            "status": game.status(),
            "player": game.player,
            "boxes": sorted(game.boxes),
            "goals": sorted(game.targets),
        }

    async def open_rewind(rewind: int) -> tuple[AgentStart, int]:
        """Fork the worker at ``rewind`` push decisions and warm the fork's REPL."""
        cut, k = rewind_point(worker, rewind)
        fork = cut.fork()
        fork.config.max_iters = budget + 3
        # A fork is a graph this Flow has not run, so its namespace does not exist
        # yet; replaying it here (rather than leaving it to the first step) is what
        # makes the post-rewind board readable before anyone plays on it.
        await flow.replay(fork)
        return fork, k

    @tool(
        "Rewind the worker by `rewind` push decisions WITHOUT playing; returns a "
        "dict of the post-rewind board/status/player/boxes/goals. Write strategy "
        "plans against THESE coords (worker.boxes is the stale jammed state)."
    )
    async def preview(rewind: int) -> dict:
        """Show the board after undoing ``rewind`` pushes — without playing.
        Use this to write strategy plans against the coords the fork will
        actually start from (``worker.boxes`` is the jammed state)."""
        fork, k = await open_rewind(rewind)
        snap = snapshot(fork, k)
        flow.runtime.close_repl(fork)  # a preview is thrown away; only its board matters
        return snap

    async def prepare_branch(rewind: int, strategy: str) -> tuple[str, Branch]:
        """Rewind the worker into a fork and hand it its recovery guidance as the
        fork's next user turn. The fork carries the worker's prompt profile and
        model, so nothing else is needed. Returns ``(name, branch)``."""
        fork, k = await open_rewind(rewind)
        snap = snapshot(fork, k)
        fork.frontier.append(
            UserQuery(
                content=BRANCH_GUIDANCE.format(
                    k=k,
                    status=snap["status"],
                    player=snap["player"],
                    boxes=snap["boxes"],
                    goals=snap["goals"],
                    strategy=strategy,
                )
            )
        )
        index = len(branches)
        handle = Branch(flow, fork, index=index, rewind=k, strategy=strategy)
        branches.append(handle)
        return f"branch{index}", handle

    @tool(
        "Fork the worker, undo the last `rewind` push decisions per spec, inject "
        "each recovery strategy + its post-rewind board, and play the forks OUT IN "
        "PARALLEL. specs=[{'rewind': int, 'strategy': str}]; returns one Branch "
        "handle per spec (with .solved / .dist / .score / .board())."
    )
    async def branch(specs: list[dict]) -> list[Branch]:
        """Revert-and-branch primitive. Each spec ``{"rewind": int, "strategy":
        str}`` rewinds the worker's last ``rewind`` push-decisions into a fork and
        hands it the recovery plan. The forks run together on the shepherd's own
        Flow and stream into the same viewer; returns the handles in order."""
        prepared = [await prepare_branch(**spec) for spec in specs]
        await run_branches([(name, handle.graph) for name, handle in prepared])
        return [handle for _name, handle in prepared]

    @tool("Commit the winning Branch as the recovery; the host keeps it, discards the rest.")
    def pick(b: Branch) -> str:
        """Commit the winning branch; the host keeps it and discards the rest."""
        picked["branch"] = b
        return f"picked branch{b.index} ({'solved' if b.solved else f'dist={b.dist}'})"

    namespace = {"worker": view, "preview": preview, "branch": branch, "pick": pick}
    return namespace, branches, picked


def worker_halt(flow: Flow, *, jam_pushes: int):
    """Stop the worker the instant it solves, hits a wall, or spends its push
    budget — the cue to hand off to the shepherd. A boundary is a plain
    ``(node, root) -> bool`` handed to ``run_streaming(until=...)``."""

    def halt(_node: Node, root: AgentStart) -> bool:
        env = flow.runtime.repl_for(root).env
        return bool(env.get("solved") or env.get("blocked") or env.get("pushes", 0) >= jam_pushes)

    return halt


def branch_halt(flow: Flow, *, budget: int):
    """Stop a recovery branch on a solve or its push cap. Unlike the worker's
    boundary this ignores ``blocked``: a rejected push is not the end of a
    recovery, and the guidance tells the branch to try another legal one."""

    def halt(_node: Node, root: AgentStart) -> bool:
        env = flow.runtime.repl_for(root).env
        return bool(env.get("solved") or env.get("pushes", 0) >= budget)

    return halt


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
    flow = Flow(
        build_client(model),
        llm_clients={"worker": build_client(worker_model)},
        # The worker is a myopic player, not an orchestrator: a lean profile (REPL
        # protocol + PUSH_RULES only) plus a harness-fed board each turn. The
        # shepherd stays on the full default prompt. A graph picks its profile by
        # name, and a fork deep-copies that choice along with the rest of its config.
        prompt_profiles={
            "worker": PromptProfile(
                system=WORKER_SYSTEM,
                # A profile's user side has to be a builder: only ``Flow(user_prompt=)``
                # wraps a bare ``(flow, agent) -> str | None`` for you.
                user=UserPromptBuilder(board_prompt),
                description="myopic Sokoban worker",
            )
        },
        llm_request_timeout=request_timeout,
    )
    # Ambient dependency (not a tool): in every REPL incl. forks, so replaying
    # CONSTRUCT works on every branch; unadvertised (no @tool metadata).
    flow.inject("Sokoban", Sokoban)

    # prompt_profile="worker" stamps this graph (and its forks, which deep-copy
    # its config) onto the lean profile above. Iteration caps are per-agent: the
    # worker's is a backstop behind the ``until`` boundary below, and each fork
    # gets its own in ``open_rewind``.
    worker = start(
        query=WORKER_QUERY,
        model="worker",
        prompt_profile="worker",
        max_depth=0,
        max_iters=max_iters,
    )
    shepherd = start(
        query=META_QUERY.format(rules=PUSH_RULES, n=n_branches, cap=branch_pushes),
        max_depth=max_depth,
        max_iters=max_iters,
    )

    # Board panels: the Gradio dashboard's live sink, and the grid the host prints
    # at each milestone. ``paint=False`` because the terminal is a scrolling node
    # trace here, and a viewer that clears the screen would wipe it every push.
    title = "shepherd — worker hits a wall ▸ parallel recovery branches"
    panels = PanelViewer(
        title=title,
        cols=min(n_branches, 4),
        status_of=lambda agent: panel_status(flow, agent),
        board_of=lambda agent: board(flow, agent),
        # The game publishes this turn's sub-step renders to env["frames"]; the grid
        # reads them off the metadata channel (rather than reaching into the object)
        # and animates the honest walk-around, one repaint per frame. Only worth its
        # sleeps when something is actually watching them go by.
        frames_of=(
            (lambda agent: flow.runtime.get_env_var(agent, "frames"))
            if dashboard is not None
            else None
        ),
        sink=(dashboard.set_panels if dashboard is not None else None),
        paint=False,
    )
    lanes: dict[str, str] = {}

    def announce(agent: AgentStart, name: str) -> None:
        """Name a graph for both the trace and its board panel."""
        lanes[agent.id] = name
        panels.label(agent, name)

    def show(node: Node) -> None:
        panels.handle(node)
        line = trace_line(node, lanes)
        if line:
            # stderr, because the branches stream while the shepherd's own REPL
            # turn is still running and a REPL captures stdout: a trace printed
            # there would come back as that turn's output, in the next prompt.
            print(line, file=sys.stderr, flush=True)

    async def run_branches(named: list[tuple[str, AgentStart]]) -> None:
        """Play every recovery fork out at once on this Flow, into one viewer."""
        for name, fork in named:
            announce(fork, name)
        halt = branch_halt(flow, budget=branch_pushes)
        async for node in parallel_stream(flow, *(fork for _n, fork in named), until=halt):
            show(node)

    if dashboard is not None:
        dashboard.set_status("worker playing forward (myopic 'keep right')…")

    try:
        # 1. Setup: build and play the stuck worker forward (the problem
        #    statement). MYOPIC_STRATEGY is on the worker only, so forks re-decide.
        announce(worker, "worker")
        await seed_turn(flow, worker, CONSTRUCT)
        worker.frontier.append(UserQuery(content=MYOPIC_STRATEGY))
        print(f"=== board === (worker={worker_model}, shepherd={model})")
        print(board(flow, worker))
        # The one-push guard is armed per turn by ``board_prompt`` (the worker's
        # user builder), so no host-loop ``begin_turn`` is needed here. Saving
        # inside the loop is the checkpoint: a run directory per landed node.
        async for node in flow.run_streaming(
            worker, until=worker_halt(flow, jam_pushes=jam_pushes)
        ):
            show(node)
            worker.save(root / "worker")

        wenv = flow.runtime.repl_for(worker).env
        reason = (
            "solved"
            if wenv.get("solved")
            else "hit a wall"
            if wenv.get("blocked")
            else f"stopped after {wenv.get('moves')} moves ({wenv.get('pushes')} pushes)"
        )
        print(f"\n[worker] {reason}")
        print(panels.render())
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
        announce(shepherd, "shepherd")
        namespace, branches, picked = make_shepherd_primitives(
            flow, worker, budget=branch_pushes, run_branches=run_branches
        )
        # Straight into the shepherd's own REPL rather than through flow.inject,
        # which would put the primitives in every namespace — including the forks
        # the shepherd is about to launch.
        repl = flow.runtime.repl_for(shepherd)
        for name, value in namespace.items():
            repl.inject(name, value)
        async for node in flow.run_streaming(shepherd, until="done"):
            show(node)
            shepherd.save(root / "shepherd")

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
            signature(flow.runtime.repl_for(b.graph).env.get("assignment", ""))
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
            [("worker", flow.runtime.get_var(worker, "game"))]
            + [(f"branch{b.index}", flow.runtime.get_var(b.graph, "game")) for b in branches],
        )

        for b in branches:
            if b is not best:
                flow.runtime.close_repl(b.graph)
        best.graph.save(root / "best")
        print(f"\n[pick] best is branch{best.index}")
        print("\n=== final board ===")
        print(best.board())
        print(f"[done] solved={best.solved} pushes={best.pushes}")
        if dashboard is not None:
            dashboard.set_picked(f"branch{best.index}")
            dashboard.finish(f"done — solved={best.solved}, {best.pushes} pushes")
    finally:
        panels.close()
        await flow.aclose()


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
    # max_depth=0: neither agent delegates. The shepherd recovers by forking the
    # worker's graph from its own REPL, not by spawning subagents.
    add_flow_args(parser, max_depth=0, max_iters=32)
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
        default=8,
        help="pushes the worker plays forward before handing off (this board deadlocks "
        "B1 in exactly 8 right-pushes, so 8 stops the worker right on the clean jam — "
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
        help="Open a live Gradio board dashboard instead of the terminal node trace.",
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
    except TimeoutError:
        print(f"[timeout] aborted after exceeding the {args.time_budget:.0f}s budget")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
