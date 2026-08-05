"""Render our ASCII Sokoban board to a pixel image using gym-sokoban tiles.

We keep our own pure-Python ``Sokoban`` class as the source of truth (custom
board + deterministic fork/replay); this module only turns a rendered board
string into a nice tiled PNG for the Gradio viewer. The 16x16 sprite tiles in
``assets/sokoban/`` are vendored from mpSchrader/gym-sokoban (MIT) — see
``assets/sokoban/ATTRIBUTION.md``.

Everything degrades gracefully: if Pillow or the tiles are missing, callers fall
back to the monospace board.
"""

from __future__ import annotations

import base64
import io
from functools import cache, lru_cache
from pathlib import Path

_ASSETS = Path(__file__).with_name("assets") / "sokoban"

# Our Sokoban.render() glyphs -> tile file name.
_GLYPH = {
    "#": "wall",
    " ": "floor",
    ".": "box_target",
    "$": "box",
    "*": "box_on_target",
    "@": "player",
    "+": "player_on_target",
}
_TILE_NAMES = frozenset(_GLYPH.values())

# Sokoban.render(ids=True) numbers each box instead of drawing a shared "$"/"*"
# (see BOX_GLYPHS / BOX_ON_GOAL_GLYPHS there). Same two crate tiles, recoloured
# per box, so a viewer can follow B1 from its start square to the goal it takes.
_BOX_IDS = "123456789"
_BOX_ON_GOAL_IDS = "ABCDEFGHI"
_BOX_COLORS = (
    (255, 190, 60),    # B1 amber
    (90, 190, 255),    # B2 sky
    (185, 130, 255),   # B3 violet
    (120, 225, 130),   # B4 green
    (255, 120, 170),   # B5 pink
    (90, 220, 205),    # B6 teal
    (255, 130, 80),    # B7 ember
    (205, 205, 235),   # B8 pale
    (210, 230, 90),    # B9 lime
)
_BOARD_CHARS = frozenset(_GLYPH) | frozenset(_BOX_IDS) | frozenset(_BOX_ON_GOAL_IDS)


def available() -> bool:
    """True if Pillow is importable and every tile is present on disk."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return _ASSETS.is_dir() and all((_ASSETS / f"{n}.png").exists() for n in _TILE_NAMES)


@cache
def _tile(name: str, size: int):
    from PIL import Image

    im = Image.open(_ASSETS / f"{name}.png").convert("RGB")
    return im if im.size == (size, size) else im.resize((size, size), Image.NEAREST)


@cache
def _tinted(name: str, size: int, rgb: tuple[int, int, int], locked: bool):
    """``name``'s tile recoloured towards ``rgb``, keeping the crate's shading."""
    from PIL import ImageDraw, ImageOps

    shadow = tuple(channel // 4 for channel in rgb)
    im = ImageOps.colorize(_tile(name, size).convert("L"), black=shadow, white=rgb)
    if locked:
        # The stock on-target tile marks the goal with a red border that the tint
        # flattens into the crate, so redraw it: a locked box has to stay legible
        # as locked whatever colour it carries.
        edge = max(1, size // 16)
        ImageDraw.Draw(im).rectangle(
            [0, 0, size - 1, size - 1], outline=(255, 255, 255), width=edge
        )
    return im


def _glyph_tile(ch: str, size: int):
    name = _GLYPH.get(ch)
    if name is not None:
        return _tile(name, size)
    index = _BOX_IDS.find(ch)
    if index >= 0:
        return _tinted("box", size, _BOX_COLORS[index % len(_BOX_COLORS)], False)
    index = _BOX_ON_GOAL_IDS.find(ch)
    if index >= 0:
        return _tinted("box_on_target", size, _BOX_COLORS[index % len(_BOX_COLORS)], True)
    return _tile("floor", size)


def _board_lines(text: str) -> list[str]:
    """Extract only the grid rows from a status block.

    ``game.status()`` wraps the board in a ``Current Grid:`` header plus
    ``Player/Box/Goal position`` lines (and ``push()`` may append a ``SOLVED`` or
    ``ILLEGAL`` tail); those lines contain non-glyph characters, so we keep only
    lines made purely of board glyphs.
    """
    rows = [ln.rstrip("\n") for ln in text.splitlines()]
    return [ln for ln in rows if ln and set(ln) <= _BOARD_CHARS]


def render_image(board_text: str, tile: int = 32):
    """Composite the board glyphs into a ``PIL.Image`` (``None`` if not a board)."""
    from PIL import Image

    rows = _board_lines(board_text)
    if not rows:
        return None
    width = max(len(r) for r in rows)
    img = Image.new("RGB", (width * tile, len(rows) * tile), (0, 0, 0))
    for r, line in enumerate(rows):
        for c in range(width):
            ch = line[c] if c < len(line) else " "
            img.paste(_glyph_tile(ch, tile), (c * tile, r * tile))
    return img


@lru_cache(maxsize=512)
def data_uri(board_text: str, tile: int = 32) -> str | None:
    """A ``data:image/png;base64,…`` URI for the board, or ``None`` if unavailable."""
    if not available():
        return None
    img = render_image(board_text, tile)
    if img is None:
        return None
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def save_gif(frames, path, *, tile: int = 32, ms: int = 450) -> bool:
    """Write an animated GIF, one board frame per sub-step, so the play-by-play
    (the man walking around before each shove) animates. ``frames`` is a list of
    board-render strings (see ``Sokoban.step_frames``). Returns ``False`` (writes
    nothing) if Pillow/tiles are missing or no frame is renderable."""
    if not available():
        return False
    imgs = [im for im in (render_image(f, tile) for f in frames) if im is not None]
    if not imgs:
        return False
    imgs[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=imgs[1:],
        duration=ms,
        loop=0,
    )
    return True


__all__ = ["available", "data_uri", "render_image", "save_gif"]
