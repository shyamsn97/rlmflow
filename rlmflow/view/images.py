"""Raster exports: a PNG per step, or the whole run as one GIF.

The figures are SVG, so turning them into pixels needs a rasteriser that is not
worth carrying for everyone: ``pip install rlmflow[image]`` brings CairoSVG and
Pillow. Everything else in ``rlmflow.view`` works without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlmflow.graph.nodes import AgentStart
from rlmflow.view.html import step_svg
from rlmflow.view.steps import timeline

_MISSING = (
    "PNG and GIF export need a rasteriser: pip install 'rlmflow[image]'. "
    "rlmflow.view.save_svg and save_html need nothing."
)


def _rasterise() -> Any:
    try:
        import cairosvg
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(_MISSING) from exc
    return cairosvg


def _image() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(_MISSING) from exc
    return Image


def save_frames(
    root: AgentStart,
    directory: str | Path,
    *,
    every: int = 1,
    scale: float = 1.0,
) -> list[Path]:
    """Write one PNG per step into ``directory``, named by step number.

    ``every`` thins a long run out — ``every=5`` keeps every fifth step.
    """
    cairosvg = _rasterise()
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    ordered = timeline(root)
    width = len(str(len(ordered)))
    written: list[Path] = []
    for index in range(0, len(ordered), max(every, 1)):
        path = out / f"step_{index + 1:0{width}d}.png"
        cairosvg.svg2png(
            bytestring=step_svg(root, ordered, index).encode("utf-8"),
            write_to=str(path),
            scale=scale,
        )
        written.append(path)
    return written


def save_gif(
    root: AgentStart,
    path: str | Path,
    *,
    every: int = 1,
    ms_per_frame: int = 120,
    hold_last_ms: int = 1600,
    scale: float = 1.0,
) -> Path:
    """Animate the run into a GIF, holding on the finished graph at the end."""
    cairosvg = _rasterise()
    Image = _image()
    import io

    ordered = timeline(root)
    frames = []
    for index in range(0, len(ordered), max(every, 1)):
        png = cairosvg.svg2png(
            bytestring=step_svg(root, ordered, index).encode("utf-8"), scale=scale
        )
        frames.append(Image.open(io.BytesIO(png)).convert("P", palette=Image.ADAPTIVE))
    if not frames:
        raise ValueError("nothing to animate: that graph has no nodes")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    durations = [ms_per_frame] * len(frames)
    durations[-1] = max(hold_last_ms, ms_per_frame)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return out
