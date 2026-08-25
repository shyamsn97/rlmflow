"""Reading a saved run (``rlmflow view``) and writing it out (``rlmflow render``).

Two surfaces because they answer different questions. Reading needs a path and
prints; rendering needs a path *and* somewhere to put the result, which is a
positional argument, not a flag that quietly means "also do this".
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rlmflow.cli.options import CliError


class ViewCLI:
    """Read a saved graph directory: the agent tree, then the timeline."""

    def show(
        self,
        source: str,
        step: int | None = None,
        tree: bool = False,
        frames_only: bool = False,
    ) -> None:
        """Print the agent tree, then the numbered timeline.

        Args:
          source: The checkpoint directory, e.g. runs/coding/graph.
          step: Print this step (1-based) with its content, instead of the timeline.
          tree: Print the tree only, no timeline.
          frames_only: Print the agent tree as it stood at every step.
        """
        from rlmflow import render_tree
        from rlmflow.view import render_steps, steps

        graph = load_graph(source)
        walked = steps(graph)

        if step is not None:
            if step < 1 or step > len(walked):
                raise CliError(f"--step {step} out of range (1..{len(walked)})")
            chosen = walked[step - 1]
            print(f"=== step {step}/{len(walked)} · {chosen.title} · +{chosen.elapsed:.2f}s ===")
            print(chosen.detail or chosen.summary or "(no content)")
            return

        if frames_only:
            for index, frame in enumerate(render_steps(graph), 1):
                print(f"=== step {index}/{len(walked)} ===\n{frame}\n")
            return

        print(render_tree(graph))
        if not tree:
            print(f"\n{len(walked)} steps")
            for walked_step in walked:
                print(
                    f"  {walked_step.index + 1:>4}  +{walked_step.elapsed:7.2f}s  "
                    f"{walked_step.title:<34}  {walked_step.summary}"
                )


class RenderCLI:
    """Turn a saved run into a figure, a page, or an animation."""

    def svg(self, source: str, out: str) -> None:
        """Write the graph figure as SVG.

        Args:
            source: The checkpoint directory.
            out: Where to write the .svg.
        """
        from rlmflow.view import save_svg

        print(f"wrote {save_svg(load_graph(source), Path(out))}")

    def html(self, source: str, out: str) -> None:
        """Write a single-file stepper: the graph per step, with node content.

        Args:
            source: The checkpoint directory.
            out: Where to write the .html.
        """
        from rlmflow.view import save_html

        print(f"wrote {save_html(load_graph(source), Path(out))}")

    def gif(
        self,
        source: str,
        out: str,
        *,
        every: int = 1,
        scale: float = 1.0,
        ms_per_frame: int = 120,
        hold_last_ms: int = 1600,
    ) -> None:
        """Animate the run, one frame per step (needs rlmflow[image]).

        Args:
          source: The checkpoint directory.
          out: Where to write the .gif.
          every: Keep every Nth step.
          scale: Multiply the figure's size by this.
          ms_per_frame: How long each step is on screen.
          hold_last_ms: How long to hold the finished graph at the end.
        """
        from rlmflow.view.images import save_gif

        graph = load_graph(source)
        with _rasteriser():
            written = save_gif(
                graph,
                Path(out),
                every=every,
                scale=scale,
                ms_per_frame=ms_per_frame,
                hold_last_ms=hold_last_ms,
            )
        print(f"wrote {written}")

    def frames(self, source: str, out: str, *, every: int = 1, scale: float = 1.0) -> None:
        """Write a PNG per step into a directory (needs rlmflow[image]).

        Args:
          source: The checkpoint directory.
          out: Directory to fill with PNGs.
          every: Keep every Nth step.
          scale: Multiply the figure's size by this.
        """
        from rlmflow.view.images import save_frames

        graph = load_graph(source)
        with _rasteriser():
            written = save_frames(graph, Path(out), every=every, scale=scale)
        print(f"wrote {len(written)} frames to {out}")

    def browser(self, source: str) -> None:
        """Open the run in the Gradio viewer (needs rlmflow[viewer]).

        Args:
          source: The checkpoint directory.
        """
        try:
            from rlmflow.view.app import open_viewer
        except ImportError as exc:
            raise CliError(str(exc)) from exc

        open_viewer(load_graph(source))


def load_graph(source: str | Path) -> Any:
    """Load a checkpoint directory, or say why it could not be read."""
    from rlmflow import persistence

    path = Path(source)
    if not path.exists():
        raise CliError(f"path not found: {path}")
    try:
        return persistence.load(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CliError(f"cannot read a graph from {path}: {exc}") from exc


@contextmanager
def _rasteriser() -> Iterator[None]:
    """Turn the missing-extra ImportError into one line of stderr.

    ``rlmflow.view.images`` imports cleanly and only reaches for cairosvg and
    Pillow when a frame is actually drawn, so the failure surfaces at call time.
    """
    try:
        yield
    except ImportError as exc:
        raise CliError(str(exc)) from exc


__all__ = ["RenderCLI", "ViewCLI", "load_graph"]
