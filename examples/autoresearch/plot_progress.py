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


def _trial_slug(trial: dict) -> str:
    return str(trial.get("slug") or trial.get("name") or f"n{trial.get('n')}")


def lineage_for(trial: dict, by_slug: dict[str, dict]) -> list[str]:
    """Full ancestry chain (root ancestor .. this trial), following
    ``parent_slug`` all the way up — not just the immediate parent."""
    chain: list[str] = []
    seen: set[str] = set()
    cur: dict | None = trial
    while cur is not None:
        slug = _trial_slug(cur)
        chain.append(slug)
        if slug in seen:  # guard against a malformed cycle
            break
        seen.add(slug)
        parent = cur.get("parent_slug")
        cur = by_slug.get(str(parent)) if parent else None
    chain.reverse()
    return chain


def wrap_label(text: str, width: int = 32) -> list[str]:
    """Wrap a ``a + b + ...`` lineage label onto multiple lines so long chains
    don't run off the plot. Breaks on the ``+`` separators, never mid-word."""
    import textwrap

    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return lines or [text]


def label_for(trial: dict, by_slug: dict[str, dict] | None = None) -> str:
    """Readable kept label: the full lineage recipe (``a + b + ... + own``).

    Each hop is a change stacked on the previous solution, so the label traces
    the whole path back to the baseline rather than showing one parent hop.
    """
    if by_slug is None:
        return _slug_words(_trial_slug(trial))
    chain = lineage_for(trial, by_slug)
    return " + ".join(_slug_words(slug) for slug in chain)


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
    by_slug = {_trial_slug(t): t for t in trials}
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
            "\n".join(wrap_label(label_for(trial, by_slug))),
            xy=(int(trial["experiment"]), float(trial["val_bpb"])),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=7,
            color="#15803d",
            rotation=28,
            rotation_mode="anchor",
            ha="left",
            va="bottom",
            annotation_clip=False,
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
    by_slug = {_trial_slug(t): t for t in trials}
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
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#d1d5db"/>'
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
            f'<polyline points="{" ".join(poly)}" fill="none" stroke="#16a34a" stroke-width="2.5"/>'
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
        tx, ty = x + 6, y - 8
        lines = wrap_label(label_for(trial, by_slug))
        tspans = "".join(
            f'<tspan x="{tx:.1f}" dy="{0 if i == 0 else 9}">{html.escape(line)}</tspan>'
            for i, line in enumerate(lines)
        )
        parts.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" font-family="sans-serif" '
            f'font-size="8" fill="#15803d" '
            f'transform="rotate(-28 {tx:.1f} {ty:.1f})">{tspans}</text>'
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


def plot_tree_svg(
    trials: list[dict],
    *,
    title: str,
    out: Path,
    skip_slugs: set[str] | None = None,
) -> None:
    """Render the lineage *tree*: one node per trial, edges ``parent -> child``.

    A tidy left-to-right layout (depth on x, siblings spread on y). Kept
    improvements are green, everything else gray, and the overall best is ringed.
    """
    import html
    from collections import defaultdict

    by_slug = {_trial_slug(t): t for t in trials}
    _, kept, _ = compute_kept(trials, skip_slugs=skip_slugs)
    kept_slugs = {_trial_slug(t) for t in kept}
    best_slug = _trial_slug(min(trials, key=lambda t: float(t["val_bpb"])))

    # parent -> children (in experiment order); roots have no known parent.
    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    for trial in sorted(trials, key=lambda t: int(t["experiment"])):
        slug = _trial_slug(trial)
        parent = trial.get("parent_slug")
        if parent and str(parent) in by_slug:
            children[str(parent)].append(slug)
        else:
            roots.append(slug)

    depth = {slug: len(lineage_for(t, by_slug)) - 1 for slug, t in by_slug.items()}

    # Tidy layout: leaves take successive rows, parents center over children.
    ypos: dict[str, float] = {}
    counter = 0

    def assign(slug: str) -> None:
        nonlocal counter
        kids = children.get(slug, [])
        if not kids:
            ypos[slug] = counter
            counter += 1
            return
        for kid in kids:
            assign(kid)
        ypos[slug] = sum(ypos[kid] for kid in kids) / len(kids)

    for root in roots:
        assign(root)

    max_depth = max(depth.values(), default=0)
    rows = max(counter, 1)
    left, top = 30, 60
    col_w, row_h = 250, 30
    width = left * 2 + max_depth * col_w + 220
    height = top + rows * row_h + 40

    def x_at(slug: str) -> float:
        return left + depth[slug] * col_w

    def y_at(slug: str) -> float:
        return top + ypos[slug] * row_h + row_h / 2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" '
        f'font-size="18" font-weight="bold">{html.escape(title)} — lineage</text>',
    ]

    # Edges first (behind nodes).
    for parent, kids in children.items():
        px, py = x_at(parent), y_at(parent)
        for kid in kids:
            kx, ky = x_at(kid), y_at(kid)
            mx = (px + kx) / 2
            parts.append(
                f'<path d="M {px:.1f} {py:.1f} C {mx:.1f} {py:.1f} {mx:.1f} {ky:.1f} '
                f'{kx:.1f} {ky:.1f}" fill="none" stroke="#cbd5e1" stroke-width="1.5"/>'
            )

    for slug, trial in by_slug.items():
        x, y = x_at(slug), y_at(slug)
        kept_here = slug in kept_slugs
        fill = "#16a34a" if kept_here else "#9ca3af"
        radius = 7 if kept_here else 5
        if slug == best_slug:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius + 4}" fill="none" '
                'stroke="#f59e0b" stroke-width="2.5"/>'
            )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" '
            'stroke="#14532d" stroke-width="0.6"/>'
        )
        color = "#15803d" if kept_here else "#4b5563"
        label = f"{_slug_words(slug)}  {float(trial['val_bpb']):.3f}"
        parts.append(
            f'<text x="{x + radius + 5:.1f}" y="{y + 4:.1f}" font-family="sans-serif" '
            f'font-size="11" fill="{color}">{html.escape(label)}</text>'
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
        default=Path(__file__).resolve().parents[1] / "_runs" / "autoresearch" / "20260721-194137",
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

    tree_out = run_dir / "lineage.svg"
    plot_tree_svg(trials, title=args.title, out=tree_out, skip_slugs=skip_slugs)
    print(f"wrote {tree_out}")


if __name__ == "__main__":
    main()
