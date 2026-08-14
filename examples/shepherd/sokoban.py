"""The Sokoban game env for the shepherd demo — pure logic, no LLM or graph.

Injected into every REPL as ``Sokoban``. The worker plays the man directly, and the
example picks one of two action spaces for it (see ``--simple-moves``):

* ``goto(row, col)`` walks him to a square around the boxes, and
  ``push(direction)`` shoves whatever box is straight ahead. Splitting them keeps
  every irreversible step deliberate — walking can never nudge a box — while leaving
  the worker the part that is actually a decision: which box, which way, what order.
* ``move(directions)`` alone, Sokoban's classic action taking a whole route: step a
  cell at a time, shoving a box if one is ahead. Harder, because routing is the
  worker's problem again and a miscounted step shoves something it never meant to
  touch.

Both spend one shove per turn, so turns, scores and rewind depths compare directly.
Any refused action raises :class:`IllegalAction` and ends the turn without changing
the board, because every line a block runs after a refusal was written for a position
the man never reached. This demo adds one simplifying rule: a box locks once it
reaches a goal.

Walking is free within a turn and a turn commits exactly one *shove*, so a turn is
one irreversible decision: rewind granularity stays one push, and the box->goal
ASSIGNMENT signature is unchanged. Every sub-step is recorded (``step_frames``), one
frame per cell walked, so exports can animate the man rather than teleport him.

``pushes_here()`` gives the directions the player can shove from where it stands,
``legal_moves()`` the directions it can step at all, and ``legal_pushes()`` the box
shoves physically possible anywhere on the board, each quoting the square to stand
on — landing cell free and the far side reachable (state derivation, never a solver
or a plan). ``doomed()`` names boxes that can no longer reach any goal, so a lost
board can be called lost instead of played out.

Board glyphs:  ``#`` wall · (space) floor · ``@`` player · ``$`` box ·
``.`` target · ``*`` box on target · ``+`` player on target
"""

from __future__ import annotations

from typing import NoReturn

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

# Glyphs for ``render(ids=True)``: box ``Bn`` draws as the nth character, so a
# viewer can tell the boxes apart. ``sprites`` maps these to per-box colours.
BOX_GLYPHS = "123456789"
BOX_ON_GOAL_GLYPHS = "ABCDEFGHI"


class IllegalAction(RuntimeError):
    """An action the board refused. Ends the turn without changing anything."""


def _norm_dir(d: str) -> tuple[int, int] | None:
    key = str(d).strip().lower()
    if key in DIRS:
        return DIRS[key]
    aliases = {"u": "up", "d": "down", "l": "left", "r": "right"}
    return DIRS.get(aliases.get(key, ""))


class Sokoban:
    """The whole game as a plain Python object, injected into every REPL. Pure
    logic: walk or shove one cell, report solved/blocked/dist (cheap verifier
    signals, never a solver), render. Every action publishes its outcome to the
    ``env`` mapping (the REPL's ``ENV`` channel) so the host can read state, and
    records one board frame in ``step_frames`` so the run can animate the
    play-by-play."""

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
        # ``moves`` counts every cell the man travels (walks + shoves); ``pushes``
        # counts the irreversible ones. One shove per turn, so turns == pushes.
        self.moves = 0
        self.pushes = 0
        # Full per-sub-step trace [(label, render), ...] for the whole game — saved
        # to disk and replayed by the GIF export to animate the man walking.
        self.step_frames: list[tuple[str, str]] = []
        # Just the *current* turn's sub-step renders, published to ``env["frames"]``
        # so the live grid can read it back (``flow.runtime.get_env_var``) and
        # animate the whole walk-and-shove. A turn may take many actions, so these
        # reset when the turn marker changes rather than on every call.
        self.turn_frames: list[str] = []
        self._frame_turn = None
        # Set when a requested action can't be applied: the cue that the myopic
        # worker has run into a wall and it is time to rewind.
        self.blocked = False
        # One-shove-per-turn guard, keyed on the turn marker the host writes into
        # ``env`` before each live turn. Walking is unguarded — it is reversible, and
        # a worker that has to route itself needs several steps per decision.
        # Reading the marker instead of being armed by a host method call keeps the
        # guard working when the game lives in a worker process, where the host
        # cannot touch this object. It also stays inactive during deterministic fork
        # REPLAY, which re-runs a turn's actions back-to-back with no host turn
        # boundary and so writes no marker.
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
        """``["up", "right"]`` — directions ``move`` would accept from where the player
        stands: free floor, or an unlocked box with a free cell behind it. Only the
        simple one-action interface needs this. State derivation, never a plan."""
        r, c = self.player
        return [
            name
            for name, (dr, dc) in DIRS.items()
            if (r + dr, c + dc) not in self.walls
            and ((r + dr, c + dc) not in self.boxes or name in self.pushes_here())
        ]

    def pushes_here(self) -> list[str]:
        """``["right"]`` — directions ``push`` accepts from where the player stands:
        an unlocked box straight ahead with a free cell behind it."""
        r, c = self.player
        out = []
        for name, (dr, dc) in DIRS.items():
            ahead = (r + dr, c + dc)
            if ahead not in self.boxes or ahead in self.targets:
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

        Never put this set in a prompt: working out which pushes are *safe* is the
        worker's job, and handing over the answer is what the task is asking for.
        Reporting a box that already sits on one of these squares is a different
        thing — see ``doomed`` — because that mistake has already been made.
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

    def doomed(self) -> list[str]:
        """``["B2 (3,1)"]`` — unlocked boxes that can no longer reach any goal at all.

        A box on a dead square is a finished mistake: no sequence of pushes brings it
        to a goal, so the position cannot be solved however many pushes remain
        elsewhere. Worth publishing because that is exactly the state a rewind exists
        for, and because a branch that keeps shuffling its other boxes for another
        twenty turns is burning a turn budget on a board it has already lost.

        Derived from the walls, so it never guesses: every box named here is provably
        stranded, though other kinds of deadlock (boxes blocking each other) are not
        detected and will not show up.
        """
        dead = self.dead_cells()
        return [
            f"{bid} ({pos[0]},{pos[1]})"
            for bid, pos in self.box_items()
            if pos not in self.targets and pos in dead
        ]

    def legal_pushes(self) -> list[str]:
        """``["B1 right from (3,2)", "B2 up from (6,4)", ...]`` — box shoves physically
        possible right now: the box's landing cell is free and the player can walk to
        the far side, which is the square quoted. Says nothing about which shove is
        wise, or in what order, or whether the position is still winnable. State
        derivation, never a solver or a plan."""
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
                    out.append(f"{bid} {name} from ({stand[0]},{stand[1]})")
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
        self.env["legal_moves"] = self.legal_moves()
        self.env["pushes_here"] = self.pushes_here()
        self.env["doomed"] = self.doomed()
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
    def _begin_action(self) -> None:
        """Start this turn's frame buffer if this is the turn's first action.

        A turn walks several cells before it shoves, and the live grid wants the
        whole sequence, so the buffer follows the host's turn marker. Replay writes
        no marker, so there each call stands alone.
        """
        turn = self.env.get("turn")
        if turn is None or turn != self._frame_turn:
            self.turn_frames = []
        self._frame_turn = turn

    def _claim_push(self) -> None:
        """Spend this turn's single shove, or refuse a second one.

        Walks are unguarded, because they are reversible. The host stamps
        ``env["turn"]`` before each live turn. An unstamped run is replay, which
        re-runs a turn's actions back-to-back.
        """
        turn = self.env.get("turn")
        if turn is None:
            return
        if turn == self._turn:
            raise RuntimeError(
                "one push per turn — walking is free, but read the board before shoving again"
            )
        self._turn = turn

    def _reject(self, msg: str) -> NoReturn:
        """Refuse an action and end the turn, changing nothing.

        Raising rather than returning is the point. Everything a block does after a
        refused action was written for a position the man never reached, so letting
        execution continue turns one miscounted cell into a whole turn of nonsense —
        and, worse, into pushes aimed at the wrong box. The turn stops here, the board
        is untouched, and the message says where the man actually is so the next turn
        can be written against the truth.
        """
        self.blocked = True
        self._publish()
        raise IllegalAction(f"{msg} Nothing moved and you are at {self.player}.")

    def _record(self, label: str) -> None:
        """Snapshot the board after a sub-step: append to the full trace (disk +
        GIF) and this turn's frames (the live per-turn ``env["frames"]``). These
        are viewer-only, so they use the per-box identity glyphs."""
        render = self.render(ids=True)
        self.step_frames.append((label, render))
        self.turn_frames.append(render)

    def goto(self, row, col=None) -> str:
        """Walk the player to ``(row, col)`` by the shortest route around the boxes.

        Walking is free and reversible, so a turn may ``goto`` as often as it likes;
        the point of spelling a destination instead of a direction is that routing
        the man is bookkeeping, not strategy. Boxes are obstacles here — nothing this
        does can ever shove one, which leaves ``push`` as the only way to change what
        is still solvable. Reports why when it cannot get there and changes nothing.
        """
        self._begin_action()
        dest = row if col is None else (row, col)
        try:
            dest = (int(dest[0]), int(dest[1]))
        except (TypeError, ValueError, IndexError, KeyError):
            self._reject(f"UNKNOWN square {row!r} — call goto(row, col) with two integers.")
        if dest == self.player:
            return f"already standing on {dest}"
        if dest in self.walls:
            self._reject(f"ILLEGAL: {dest} is a wall.")
        if dest in self.boxes:
            self._reject(
                f"ILLEGAL: box {self._id_at(dest)} is on {dest} — walk to a free square "
                "beside it and push from there."
            )
        path = self._path(self.player, dest)
        if path is None:
            self._reject(f"ILLEGAL: no way to walk to {dest} — walls or boxes seal it off.")
        start = self.player
        # One frame per cell, so the viewer animates a walk instead of teleporting.
        for name in path:
            dr, dc = DIRS[name]
            self.player = (self.player[0] + dr, self.player[1] + dc)
            self.moves += 1
            self._record(f"walk {name}")
        self.blocked = False
        self._publish()
        return f"walked {start}->{dest} in {len(path)} cells"

    def move(self, directions) -> str:
        """Walk a route, one cell per step, shoving whatever box is straight ahead.

        Takes one direction or a whole list of them — ``move("up")`` or
        ``move(["down", "down", "right"])`` — and reports every step it took. This is
        Sokoban's classic action and the whole interface under ``--simple-moves``: it
        is strictly harder to use than ``goto`` + ``push``, because the route is the
        worker's to work out and a miscounted step shoves a box it never meant to
        touch, which is the point of having both. A shove still spends the turn's one
        push, so turns, scores and rewind depths mean the same in either mode.

        A refused step ends the route where it stood, and the refusal carries the
        steps that did land, so a turn's report is never a mystery.
        """
        steps = [directions] if isinstance(directions, str) else list(directions)
        done: list[str] = []
        for i, direction in enumerate(steps):
            try:
                done.append(self._step(direction))
            except IllegalAction as exc:
                walked = "".join(f"{line}\n" for line in done)
                raise IllegalAction(
                    f"{walked}step {i + 1} of {len(steps)} ({direction!r}) refused, so the "
                    f"rest of the route did not run. {exc}"
                ) from None
        return "\n".join(done)

    def _step(self, direction) -> str:
        """One cell of a ``move`` route: walk, or shove what is straight ahead."""
        self._begin_action()
        delta = _norm_dir(direction)
        if delta is None:
            self._reject(f"UNKNOWN direction {direction!r} (use up/down/left/right).")
        dr, dc = delta
        r, c = self.player
        ahead = (r + dr, c + dc)
        if ahead in self.boxes:
            return self.push(direction)
        if ahead in self.walls:
            self._reject(f"ILLEGAL: can't step {direction} — wall.")
        self.player = ahead
        self.moves += 1
        self.blocked = False
        self._record(f"walk {direction}")
        self._publish()
        return f"walk {direction}: {(r, c)}->{ahead}"

    def push(self, direction) -> str:
        """Shove the box directly ``direction`` of the player one cell, stepping into
        the square it left.

        The worker's one irreversible action, and the only one a turn is allowed. It
        needs the player already standing on the far side, which is what ``goto`` is
        for — a box against a wall can only be shoved along it, and a box in a corner
        is finished. Reports why when it cannot push and changes nothing.
        """
        self._begin_action()
        delta = _norm_dir(direction)
        if delta is None:
            self._reject(f"UNKNOWN direction {direction!r} (use up/down/left/right).")
        dr, dc = delta
        r, c = self.player
        ahead = (r + dr, c + dc)
        if ahead not in self.boxes:
            self._reject(
                f"ILLEGAL: nothing to push {direction} — {self._describe(ahead)} there. "
                "Stand beside a box first; legal_pushes names the square."
            )
        bid = self._id_at(ahead)
        if ahead in self.targets:
            self._reject(f"LOCKED: box {bid} is already on a goal.")
        beyond = (r + 2 * dr, c + 2 * dc)
        if beyond in self.walls or beyond in self.boxes:
            self._reject(
                f"ILLEGAL: box {bid} can't go {direction} — {self._describe(beyond)} behind it."
            )
        # Only a push that lands spends the turn's one shove: a worker that aimed at
        # the wrong box should get another go at pushing, not lose the turn to a refusal.
        self._claim_push()
        self.boxes.discard(ahead)
        self.boxes.add(beyond)
        self._pos_by_id[self._pos_by_id.index(ahead)] = beyond
        self.player = ahead
        self.moves += 1
        self.pushes += 1
        self.blocked = False
        self._record(f"push {direction} {bid}")
        self._publish()
        out = f"pushed {bid} {direction}: {ahead}->{beyond}"
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
