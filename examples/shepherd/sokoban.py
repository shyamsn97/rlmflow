"""The Sokoban game env for the shepherd demo — pure logic, no LLM or graph.

Injected into every REPL as ``Sokoban``. The worker acts at the *strategic* layer:
``push(box_id, direction)`` shoves one box one cell using ordinary Sokoban
movement underneath — it walks the player cell-by-cell to the far side of the box
(BFS over floor, boxes are obstacles) and then steps in to shove it, with no
teleport. This demo adds one simplifying rule: a box locks once it reaches a goal.
Every sub-step is played out and recorded (``step_frames``) so exports can animate
the man walking. It still commits exactly one push per turn, so rewind granularity
is one decision per turn, rewind anchors on the box-*pushes* (the irreversible
progress), and the box->goal ASSIGNMENT signature is unchanged. ``legal_pushes()``
reports which pushes are physically possible right now — landing cell free and the
far side reachable (state derivation, never a solver or a plan).

Board glyphs:  ``#`` wall · (space) floor · ``@`` player · ``$`` box ·
``.`` target · ``*`` box on target · ``+`` player on target
"""

from __future__ import annotations

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

# Glyphs for ``render(ids=True)``: box ``Bn`` draws as the nth character, so a
# viewer can tell the boxes apart. ``sprites`` maps these to per-box colours.
BOX_GLYPHS = "123456789"
BOX_ON_GOAL_GLYPHS = "ABCDEFGHI"


def _norm_dir(d: str) -> tuple[int, int] | None:
    key = str(d).strip().lower()
    if key in DIRS:
        return DIRS[key]
    aliases = {"u": "up", "d": "down", "l": "left", "r": "right"}
    return DIRS.get(aliases.get(key, ""))


class Sokoban:
    """The whole game as a plain Python object, injected into every REPL. Pure
    logic: apply a box-level push (expanded into honest cell-by-cell moves), report
    solved/blocked/dist (cheap verifier signals, never a solver), render. ``push``
    publishes its outcome to the ``env`` mapping (the REPL's ``ENV`` channel) so the
    host can read state, and records one board frame per sub-step in ``step_frames``
    so the run can animate the play-by-play."""

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
        # Stable box identities (by start position) so the host can tell which box
        # ended on which goal — the distinct box->goal ASSIGNMENT is the diversity.
        # ``B1``..``Bn`` index into this list; entries update as boxes move.
        self._pos_by_id = sorted(self.boxes)
        self.env = env if env is not None else {}
        # ``moves`` is the honest sub-step count (walks + shoves); ``pushes`` is the
        # strategic decision count (one per ``push`` call). Turns == pushes.
        self.moves = 0
        self.pushes = 0
        # Full per-sub-step trace [(label, render), ...] for the whole game — saved
        # to disk and replayed by the GIF export to animate the man walking.
        self.step_frames: list[tuple[str, str]] = []
        # Just the *current* turn's sub-step renders, reset at the start of every
        # push/move and published to ``env["frames"]`` so the live grid can read it
        # back (``flow.runtime.get_env_var``) and animate this turn's walk-and-shove.
        self.turn_frames: list[str] = []
        # Set when a requested push can't be applied: the cue that the myopic
        # worker has run into a wall and it is time to rewind.
        self.blocked = False
        # One-push-per-turn guard, keyed on the turn marker the host writes into
        # ``env`` before each live turn. Reading the marker instead of being armed
        # by a host method call keeps the guard working when the game lives in a
        # worker process, where the host cannot touch this object. It also stays
        # inactive during deterministic fork REPLAY, which re-runs each push
        # back-to-back with no host turn boundary and so writes no marker.
        self._turn = None
        # Filled on first use by ``dead_cells``; depends only on walls and goals.
        self._dead: set[tuple[int, int]] | None = None
        self._publish()

    # --- ids -------------------------------------------------------------
    def box_items(self) -> list[tuple[str, tuple[int, int]]]:
        """``[("B1", (r,c)), ...]`` in stable id order (current positions)."""
        return [(f"B{i + 1}", pos) for i, pos in enumerate(self._pos_by_id)]

    def goal_items(self) -> list[tuple[str, tuple[int, int]]]:
        """``[("G1", (r,c)), ...]`` in stable id order.

        Same ordering the planner sees, so a strategy naming ``G3`` and a worker
        reading the board mean the same square.
        """
        return [(f"G{i + 1}", pos) for i, pos in enumerate(sorted(self.targets))]

    def _goal_at(self, cell: tuple[int, int]) -> str:
        for gid, pos in self.goal_items():
            if pos == cell:
                return gid
        return "?"

    def _box_pos(self, box) -> tuple[int, int] | None:
        if isinstance(box, str) and box[:1] in "Bb" and box[1:].isdigit():
            i = int(box[1:]) - 1
            if 0 <= i < len(self._pos_by_id):
                return self._pos_by_id[i]
            return None
        pos = tuple(box) if isinstance(box, (tuple, list)) else box
        return pos if pos in self.boxes else None

    # --- verifier signals (never a solver) -------------------------------
    @property
    def solved(self) -> bool:
        # More targets than boxes: solved when every box sits on some target.
        return len(self.boxes) > 0 and self.boxes <= self.targets

    @property
    def dist(self) -> int:
        return sum(
            min(abs(b[0] - t[0]) + abs(b[1] - t[1]) for t in self.targets)
            for b in self.boxes
            if b not in self.targets
        )

    def legal_moves(self) -> list[str]:
        """Directions the player can move right now: a plain walk onto floor, or a
        push when a box is directly ahead with an empty cell behind it. State
        derivation, never a solver or a plan."""
        r, c = self.player
        out = []
        for name, (dr, dc) in DIRS.items():
            ahead = (r + dr, c + dc)
            if ahead in self.walls:
                continue
            if ahead in self.boxes:
                if ahead in self.targets:
                    continue
                beyond = (r + 2 * dr, c + 2 * dc)
                if beyond in self.walls or beyond in self.boxes:
                    continue
            out.append(name)
        return out

    def _path(self, start, goal) -> list[str] | None:
        """Shortest walk (list of direction names) for the *player* from ``start``
        to ``goal`` over empty floor — walls and boxes are obstacles. Returns
        ``[]`` if already there, ``None`` if unreachable. Just walks the man
        around; never a Sokoban solver (boxes never move here)."""
        if start == goal:
            return []
        from collections import deque

        seen = {start}
        queue = deque([(start, [])])
        while queue:
            cell, path = queue.popleft()
            for name, (dr, dc) in DIRS.items():
                nxt = (cell[0] + dr, cell[1] + dc)
                if nxt in seen or nxt in self.walls or nxt in self.boxes:
                    continue
                if nxt == goal:
                    return path + [name]
                seen.add(nxt)
                queue.append((nxt, path + [name]))
        return None

    def dead_cells(self) -> set[tuple[int, int]]:
        """Squares a box can never leave alive.

        Derived from the walls alone: pull each goal backwards as far as a box
        could have come from, and whatever is never reached is a square from
        which no goal is reachable. Board geometry never changes, so this is
        computed once. It is a property of the maze, not a solver — it says
        nothing about whether *this* position is winnable.

        For host-side analysis only, e.g. telling why a branch failed. Never put
        it in a prompt: working out which pushes are safe is the worker's job,
        and handing over the answer is what the task is asking for.
        """
        if self._dead is None:
            from collections import deque

            floor = {
                (r, c)
                for r, row in enumerate(self.board)
                for c in range(len(row))
                if (r, c) not in self.walls
            }
            live = set(self.targets)
            queue = deque(self.targets)
            while queue:
                r, c = queue.popleft()
                for dr, dc in DIRS.values():
                    back = (r - dr, c - dc)
                    stand = (r - 2 * dr, c - 2 * dc)
                    if back in floor and stand in floor and back not in live:
                        live.add(back)
                        queue.append(back)
            self._dead = floor - live
        return self._dead

    def legal_pushes(self) -> list[str]:
        """``["B1 right", "B2 up", ...]`` — pushes physically possible right now:
        the box's landing cell is free and the player can walk to the far side.
        State derivation, never a solver or a plan."""
        out = []
        for bid, pos in self.box_items():
            if pos in self.targets:
                continue
            for name, (dr, dc) in DIRS.items():
                dest = (pos[0] + dr, pos[1] + dc)
                stand = (pos[0] - dr, pos[1] - dc)
                if dest in self.walls or dest in self.boxes:
                    continue
                if stand in self.walls or stand in self.boxes:
                    continue
                if self._path(self.player, stand) is not None:
                    out.append(f"{bid} {name}")
        return out

    # --- observation -----------------------------------------------------
    def _id_at(self, cell: tuple[int, int]) -> str:
        for bid, pos in self.box_items():
            if pos == cell:
                return bid
        return "?"

    def _describe(self, cell: tuple[int, int]) -> str:
        """What sits in ``cell``, in the worker's vocabulary."""
        if cell in self.walls:
            return "wall"
        if cell == self.player:
            return "player"
        if cell in self.boxes:
            bid = self._id_at(cell)
            if cell in self.targets:
                return f"box-on-target {bid} on {self._goal_at(cell)}"
            return f"box {bid}"
        if cell in self.targets:
            return f"target {self._goal_at(cell)}"
        return "empty"

    def _neighbors(self, cell: tuple[int, int]) -> str:
        """``up=... down=... left=... right=...`` for the 4 cells around ``cell``."""
        return "  ".join(
            f"{name}={self._describe((cell[0] + dr, cell[1] + dc))}"
            for name, (dr, dc) in DIRS.items()
        )

    def grid(self) -> str:
        """The board with row and column numbers down the side and across the top.

        Adds nothing the board does not already say, but the wall that decides a
        push is often two rows up and ten columns across from the piece being
        moved, and counting that far along an unlabelled row is where a reader
        slips. Viewers keep the bare ``render`` — this is the model's copy.
        """
        rows = self.render().splitlines()
        width = max(len(row) for row in rows)
        pad = len(str(len(rows) - 1))
        head = " " * (pad + 1)
        tens = "".join(str(c // 10) if c >= 10 else " " for c in range(width))
        units = "".join(str(c % 10) for c in range(width))
        body = [f"{r:>{pad}} {row}" for r, row in enumerate(rows)]
        return "\n".join([f"{head}{tens}", f"{head}{units}", *body])

    def status(self) -> str:
        """Board observation: grid, coords, per-direction adjacency.
        Legal pushes are harness-injected separately (see ``board_prompt``) —
        not something the agent should call for."""
        goals = ", ".join(f"{gid} ({r},{c})" for gid, (r, c) in self.goal_items())
        around = [f"  Player ({self.player[0]},{self.player[1]}): {self._neighbors(self.player)}"]
        around += [f"  {bid} ({p[0]},{p[1]}): {self._neighbors(p)}" for bid, p in self.box_items()]
        return (
            f"Current Grid:\n{self.grid()}\n\n"
            f"Player: ({self.player[0]},{self.player[1]})\n"
            f"Goals: {goals}\n"
            f"Adjacency (what's up/down/left/right of each piece):\n" + "\n".join(around)
        )

    def _publish(self) -> None:
        self.env["solved"] = self.solved
        self.env["blocked"] = self.blocked
        self.env["dist"] = self.dist
        self.env["moves"] = self.moves
        self.env["pushes"] = self.pushes
        # Everything the host reads about the board goes through ``env`` as plain
        # data. The game object itself stays where it was built, which is a worker
        # process under every runtime but the in-process one, so the host cannot
        # call methods on it.
        self.env["status"] = self.status()
        self.env["board"] = self.render(ids=True)
        self.env["grid"] = self.grid()
        self.env["legal_pushes"] = self.legal_pushes()
        self.env["player"] = self.player
        self.env["boxes"] = dict(self.box_items())
        self.env["goals"] = dict(self.goal_items())
        self.env["box_count"] = len(self.boxes)
        self.env["placed"] = sum(1 for pos in self.boxes if pos in self.targets)
        # This turn's per-sub-step board renders (walk frames + the shove) for the
        # live grid to animate; empty on a turn that moved nothing (a rejected push).
        self.env["frames"] = list(self.turn_frames)
        # Box positions in stable id order = the branch's box->goal ASSIGNMENT,
        # the signature used to count distinct solutions across the fan-out.
        self.env["assignment"] = ";".join(f"{r},{c}" for r, c in self._pos_by_id)

    # --- action ----------------------------------------------------------
    def _claim_turn(self, action: str) -> None:
        """Spend this turn's single action, or refuse a second one.

        The host stamps ``env["turn"]`` before each live turn. An unstamped run is
        replay, which is expected to apply its pushes back-to-back.
        """
        turn = self.env.get("turn")
        if turn is None:
            return
        if turn == self._turn:
            raise RuntimeError(f"one {action} per turn — read the board, then act once next turn")
        self._turn = turn

    def _reject(self, msg: str) -> str:
        self.blocked = True
        self._publish()
        return msg

    def _record(self, label: str) -> None:
        """Snapshot the board after a sub-step: append to the full trace (disk +
        GIF) and this turn's frames (the live per-turn ``env["frames"]``). These
        are viewer-only, so they use the per-box identity glyphs."""
        render = self.render(ids=True)
        self.step_frames.append((label, render))
        self.turn_frames.append(render)

    def _apply(self, direction) -> str | None:
        """Raw one-cell step, the honest primitive ``push`` walks the man with: walk
        onto floor, or shove a box that has an empty cell behind it. No per-turn
        guard and no planning — mutates state, bumps counters, records a frame.
        Returns a one-line description, or ``None`` when nothing can move."""
        delta = _norm_dir(direction)
        if delta is None:
            return None
        dr, dc = delta
        r, c = self.player
        ahead = (r + dr, c + dc)
        if ahead in self.walls:
            return None
        if ahead in self.boxes:
            if ahead in self.targets:
                return None
            beyond = (r + 2 * dr, c + 2 * dc)
            if beyond in self.walls or beyond in self.boxes:
                return None
            bid = self._id_at(ahead)
            self.boxes.discard(ahead)
            self.boxes.add(beyond)
            self._pos_by_id[self._pos_by_id.index(ahead)] = beyond
            self.player = ahead
            self.moves += 1
            self.pushes += 1
            self._publish()
            self._record(f"push {direction} {bid}")
            return f"push {direction}: {bid} {ahead}->{beyond}"
        self.player = ahead
        self.moves += 1
        self._publish()
        self._record(f"walk {direction}")
        return f"walk {direction}: {(r, c)}->{ahead}"

    def move(self, direction) -> str:
        """Low-level one-cell step the strategic ``push`` is built on: walk or shove
        exactly one cell, reporting why if blocked. Subject to the one-action guard
        during live play. Workers use ``push``; this exists for tests/primitives."""
        self._claim_turn("move")
        self.turn_frames = []
        delta = _norm_dir(direction)
        if delta is None:
            return self._reject(f"UNKNOWN direction {direction!r} (use up/down/left/right).")
        desc = self._apply(direction)
        if desc is None:
            return self._reject(f"ILLEGAL: can't move {direction} — nothing moved.")
        self.blocked = False
        self._publish()
        return desc + ("  SOLVED!" if self.solved else "")

    def push(self, box, direction) -> str:
        """Shove one box one cell ``direction`` — the worker's single strategic
        action. Plays real Sokoban underneath: BFS-walks the player cell-by-cell to
        the far side of the box (no teleport) then steps in to shove it, printing
        every sub-step. Commits exactly one push per turn. Illegal pushes report
        why and change nothing. ``direction`` is up/down/left/right; ``box`` is a
        ``B<n>`` id or an ``(r, c)`` coordinate."""
        self._claim_turn("push")
        self.turn_frames = []
        delta = _norm_dir(direction)
        if delta is None:
            return self._reject(f"UNKNOWN direction {direction!r} (use up/down/left/right).")
        pos = self._box_pos(box)
        if pos is None:
            return self._reject(f"NO box {box!r} on the board.")
        bid = self._id_at(pos)
        if pos in self.targets:
            return self._reject(f"LOCKED: box {bid} is already on a goal.")
        dr, dc = delta
        dest = (pos[0] + dr, pos[1] + dc)
        stand = (pos[0] - dr, pos[1] - dc)
        if dest in self.walls or dest in self.boxes:
            return self._reject(f"ILLEGAL: box {bid} can't go {direction} — blocked ahead.")
        if stand in self.walls or stand in self.boxes:
            return self._reject(
                f"ILLEGAL: no room to stand behind box {bid} to push it {direction}."
            )
        route = self._path(self.player, stand)
        if route is None:
            return self._reject(
                f"ILLEGAL: can't reach the far side of box {bid} to push it {direction}."
            )
        # Walk the man to the pushing side, then step into the box — every sub-step
        # is a real move that mutates state and records a frame.
        steps = [f"  {self._apply(d)}" for d in route]
        steps.append(f"  {self._apply(direction)}")
        self.blocked = False
        self._publish()
        out = "\n".join([f"pushed {bid} {direction} ({len(steps)} moves):", *steps])
        if self.solved:
            out += "\n  SOLVED!"
        return out

    def render(self, ids: bool = False) -> str:
        """The board as ASCII.

        ``ids=True`` draws each box as its own number (``1``..``9`` off a goal,
        ``A``..``I`` on one) instead of the shared ``$``/``*``, so viewers can
        follow an individual box and colour it. The model-facing board keeps the
        standard glyphs, so this never changes what a worker reads.
        """
        numbered = (
            {pos: index for index, pos in enumerate(self._pos_by_id) if index < len(BOX_GLYPHS)}
            if ids
            else {}
        )
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
                    index = numbered.get(p)
                    if index is None:
                        line.append("*" if p in self.targets else "$")
                    elif p in self.targets:
                        line.append(BOX_ON_GOAL_GLYPHS[index])
                    else:
                        line.append(BOX_GLYPHS[index])
                elif p in self.targets:
                    line.append(".")
                else:
                    line.append(" ")
            rows.append("".join(line))
        return "\n".join(rows)


__all__ = ["BOX_GLYPHS", "BOX_ON_GOAL_GLYPHS", "Sokoban"]
