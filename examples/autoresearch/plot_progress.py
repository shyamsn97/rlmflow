"""Karpathy-style autoresearch progress plot from a run ledger.

Usage:
    python plot_progress.py ../_runs/autoresearch/20260721-194137
    python plot_progress.py ../_runs/autoresearch/20260721-194137 --out progress.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_trials(ledger_path: Path) -> list[dict]:
    """Succeeded trials with ``val_bpb``, in completion order."""
    by_key: dict[str, dict] = {}
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("slug") or row.get("n"))
        prev = by_key.get(key)
        if prev is None or float(row.get("ts", 0)) >= float(prev.get("ts", 0)):
            by_key[key] = row
    trials = [
        t
        for t in by_key.values()
        if t.get("status") == "succeeded" and t.get("val_bpb") is not None
    ]
    trials.sort(key=lambda t: float(t.get("ts", 0)))
    for i, trial in enumerate(trials):
        trial["experiment"] = i
    return trials


def _slug_words(slug: str) -> str:
    return slug.replace("_", " ")


def label_for(trial: dict) -> str:
    """Readable kept label: ``parent + child`` when a parent exists."""
    own = _slug_words(str(trial.get("slug") or trial.get("name") or f"n{trial.get('n')}"))
    parent = trial.get("parent_slug")
    if not parent:
        return own
    return f"{_slug_words(str(parent))} + {own}"


def compute_kept(
    trials: list[dict],
    *,
    skip_slugs: set[str] | None = None,
) -> tuple[list[dict], list[dict], list[tuple[int, float]]]:
    """Discarded, kept, and running-best step points.

    ``skip_slugs`` still advance the running-best line, but are drawn as
    discarded (no green marker / label).
    """
    skip = skip_slugs or set()
    best = float("inf")
    discarded: list[dict] = []
    kept: list[dict] = []
    steps: list[tuple[int, float]] = []
    for trial in trials:
        bpb = float(trial["val_bpb"])
        slug = str(trial.get("slug") or "")
        if bpb < best:
            best = bpb
            steps.append((int(trial["experiment"]), best))
            if slug in skip:
                discarded.append(trial)
            else:
                kept.append(trial)
        else:
            discarded.append(trial)
    return discarded, kept, steps


def plot_matplotlib(
    trials: list[dict],
    *,
    title: str,
    out: Path,
    skip_slugs: set[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    discarded, kept, steps = compute_kept(trials, skip_slugs=skip_slugs)
    xs = [int(t["experiment"]) for t in trials]
    ys = [float(t["val_bpb"]) for t in trials]
    y_min, y_max = min(ys), max(ys)
    pad = max((y_max - y_min) * 0.12, 0.01)

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=160)
    ax.set_facecolor("#f7f7f7")
    fig.patch.set_facecolor("white")

    ax.scatter(
        [int(t["experiment"]) for t in discarded],
        [float(t["val_bpb"]) for t in discarded],
        s=28,
        c="#9ca3af",
        alpha=0.75,
        zorder=2,
    )
    ax.scatter(
        [int(t["experiment"]) for t in kept],
        [float(t["val_bpb"]) for t in kept],
        s=70,
        c="#16a34a",
        edgecolors="#14532d",
        linewidths=0.6,
        zorder=4,
    )

    if steps:
        step_x = [steps[0][0]]
        step_y = [steps[0][1]]
        for i in range(1, len(steps)):
            x, y = steps[i]
            step_x.extend([x, x])
            step_y.extend([steps[i - 1][1], y])
        step_x.append(xs[-1])
        step_y.append(steps[-1][1])
        ax.plot(step_x, step_y, color="#16a34a", linewidth=2.2, zorder=3)

    for trial in kept:
        ax.annotate(
            label_for(trial),
            xy=(int(trial["experiment"]), float(trial["val_bpb"])),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
            color="#15803d",
            rotation=28,
            rotation_mode="anchor",
            ha="left",
            va="bottom",
        )

    ax.set_title(
        f"{title}\n{len(trials)} Experiments, {len(kept)} Kept Improvements",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Validation BPB (lower is better)")
    ax.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_ylim(y_min - pad * 0.4, y_max + pad)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#9ca3af",
                markersize=7,
                label="Discarded",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#16a34a",
                markersize=9,
                label="Kept",
            ),
            Line2D([0], [0], color="#16a34a", linewidth=2.2, label="Running best"),
        ],
        loc="upper right",
        framealpha=0.95,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_svg(
    trials: list[dict],
    *,
    title: str,
    out: Path,
    skip_slugs: set[str] | None = None,
) -> None:
    import html

    discarded, kept, steps = compute_kept(trials, skip_slugs=skip_slugs)
    width, height = 1200, 700
    left, right, top, bottom = 80, 40, 70, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [int(t["experiment"]) for t in trials]
    ys = [float(t["val_bpb"]) for t in trials]
    x0, x1 = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad = max((y_max - y_min) * 0.12, 0.01)
    y0, y1 = y_min - pad * 0.4, y_max + pad

    def x_at(n: float) -> float:
        if x1 == x0:
            return left + plot_w / 2
        return left + (n - x0) / (x1 - x0) * plot_w

    def y_at(v: float) -> float:
        return top + (y1 - v) / (y1 - y0) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f7f7f7"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" '
        f'font-size="20" font-weight="bold">{html.escape(title)}</text>',
        f'<text x="{width / 2}" y="50" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13" fill="#374151">{len(trials)} Experiments, '
        f"{len(kept)} Kept Improvements</text>",
    ]
    for tick in range(5):
        value = y0 + tick * (y1 - y0) / 4
        y = y_at(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            'stroke="#d1d5db"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{value:.3f}</text>'
        )
    parts.append(
        f'<text x="18" y="{top + plot_h / 2}" text-anchor="middle" '
        f'transform="rotate(-90 18 {top + plot_h / 2})" font-family="sans-serif" '
        'font-size="13">Validation BPB (lower is better)</text>'
    )
    parts.append(
        f'<text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" '
        'font-family="sans-serif" font-size="13">Experiment #</text>'
    )

    if steps:
        poly = [f"{x_at(steps[0][0]):.1f},{y_at(steps[0][1]):.1f}"]
        for i in range(1, len(steps)):
            x, y = steps[i]
            poly.append(f"{x_at(x):.1f},{y_at(steps[i - 1][1]):.1f}")
            poly.append(f"{x_at(x):.1f},{y_at(y):.1f}")
        poly.append(f"{x_at(xs[-1]):.1f},{y_at(steps[-1][1]):.1f}")
        parts.append(
            f'<polyline points="{" ".join(poly)}" fill="none" stroke="#16a34a" '
            'stroke-width="2.5"/>'
        )

    for trial in discarded:
        parts.append(
            f'<circle cx="{x_at(int(trial["experiment"])):.1f}" '
            f'cy="{y_at(float(trial["val_bpb"])):.1f}" '
            'r="4" fill="#9ca3af" fill-opacity="0.75"/>'
        )
    for trial in kept:
        x = x_at(int(trial["experiment"]))
        y = y_at(float(trial["val_bpb"]))
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#16a34a" '
            'stroke="#14532d" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x + 6:.1f}" y="{y - 8:.1f}" font-family="sans-serif" '
            f'font-size="11" fill="#15803d" '
            f'transform="rotate(-28 {x + 6:.1f} {y - 8:.1f})">'
            f"{html.escape(label_for(trial))}</text>"
        )

    lx, ly = left + plot_w - 140, top + 18
    parts.extend(
        [
            f'<circle cx="{lx}" cy="{ly}" r="4" fill="#9ca3af"/>',
            f'<text x="{lx + 10}" y="{ly + 4}" font-family="sans-serif" '
            'font-size="12">Discarded</text>',
            f'<circle cx="{lx}" cy="{ly + 22}" r="6" fill="#16a34a"/>',
            f'<text x="{lx + 10}" y="{ly + 26}" font-family="sans-serif" '
            'font-size="12">Kept</text>',
            f'<line x1="{lx - 6}" y1="{ly + 44}" x2="{lx + 6}" y2="{ly + 44}" '
            'stroke="#16a34a" stroke-width="2.5"/>',
            f'<text x="{lx + 10}" y="{ly + 48}" font-family="sans-serif" '
            'font-size="12">Running best</text>',
        ]
    )
    parts.append("</svg>\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Tiny Autoresearch progress.")
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[1]
        / "_runs"
        / "autoresearch"
        / "20260721-194137",
        help="Run directory containing ledger.jsonl",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--title",
        default="Tiny Autoresearch - (TinyStories + GPT2)",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=["bigger_micro_batch_x2", "lr_up_3x"],
        help="Slugs to omit from kept/running-best (still shown as discarded).",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    ledger = run_dir / "ledger.jsonl"
    if not ledger.exists():
        raise SystemExit(f"ledger not found: {ledger}")

    trials = load_trials(ledger)
    if not trials:
        raise SystemExit("no succeeded trials with val_bpb in ledger")

    skip_slugs = set(args.skip or [])
    out = (args.out or (run_dir / "progress.png")).resolve()
    try:
        plot_matplotlib(trials, title=args.title, out=out, skip_slugs=skip_slugs)
        print(f"wrote {out}")
    except Exception as exc:  # noqa: BLE001
        svg_out = out.with_suffix(".svg") if out.suffix.lower() != ".svg" else out
        plot_svg(trials, title=args.title, out=svg_out, skip_slugs=skip_slugs)
        print(f"matplotlib unavailable ({type(exc).__name__}: {exc})")
        print(f"wrote {svg_out}")


if __name__ == "__main__":
    main()
