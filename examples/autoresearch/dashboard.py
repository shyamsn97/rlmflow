"""A live view of an autoresearch run: spend, trials, agents and recent activity.

Reads the run's checkpoint every few seconds, so start it beside ``run.py``:

    python examples/autoresearch/dashboard.py examples/_runs/autoresearch/<run>
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gradio as gr
import pandas as pd
from run import baseline, ranked, trial_rows

from rlmflow import Graph
from rlmflow.view import steps

ACTIVITY = 40


def load(out: Path) -> Graph | None:
    try:
        return Graph.load(out / "graph")
    except (OSError, ValueError, KeyError, TypeError):  # missing, or mid-save: try next tick
        return None


def seed(g: Graph, gpu: str | None) -> dict:
    """The baseline the planner was given, or the one for ``gpu``."""
    return baseline(gpu) if gpu else g.root.nodes()[0].input["baseline"]


def header(g: Graph, base: dict) -> str:
    usage = g.usage()
    rows = ranked(trial_rows(g))
    best = f"{rows[0]['val_bpb']:.4f}" if rows and rows[0].get("val_bpb") else "-"
    started = min((n.started_at for n in g.nodes() if n.started_at), default=time.time())
    return (
        f"**{g.root.status()}** · {(time.time() - started) / 60:.0f} min · "
        f"**${usage.cost_usd:.2f}** Claude Code · {usage.total / 1e6:.1f}M tokens · "
        f"{len(rows)} trials · best val_bpb **{best}** · "
        f"baseline {base['val_bpb']:.4f} ({base.get('gpu', '?')})"
    )


def trials(g: Graph, base: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = trial_rows(g)
    baseline = base["val_bpb"]
    table = pd.DataFrame(
        [
            {
                "n": r["n"],
                "val_bpb": r.get("val_bpb"),
                "vs baseline": None
                if r.get("val_bpb") is None
                else round(r["val_bpb"] - baseline, 4),
                "status": r.get("status"),
                "agent": r["agent"],
                "hypothesis": r.get("hypothesis"),
                "min": round((r.get("seconds") or 0) / 60, 1),
            }
            for r in sorted(rows, key=lambda r: r["n"], reverse=True)
        ],
        columns=["n", "val_bpb", "vs baseline", "status", "agent", "hypothesis", "min"],
    )
    best, points = baseline, [{"trial": -1, "val_bpb": baseline, "series": "baseline"}]
    for r in rows:
        if r.get("val_bpb") is None:
            continue
        best = min(best, r["val_bpb"])
        points += [
            {"trial": r["n"], "val_bpb": r["val_bpb"], "series": "trial"},
            {"trial": r["n"], "val_bpb": best, "series": "best so far"},
            {"trial": r["n"], "val_bpb": baseline, "series": "baseline"},
        ]
    return table, pd.DataFrame(points, columns=["trial", "val_bpb", "series"])


def agents(g: Graph) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "agent": a.path(),
                "config": a.config_name,
                "status": a.status(),
                "nodes": len(a.nodes()),
                "$": round(g.usage([a]).cost_usd, 2),
                "task": a.task[:80],
            }
            for a in g.agents()
        ],
        columns=["agent", "config", "status", "nodes", "$", "task"],
    )


def activity(g: Graph) -> str:
    return "\n".join(
        f"+{s.elapsed / 60:5.1f}m  {s.title:<36} {s.summary}" for s in steps(g)[-ACTIVITY:][::-1]
    )


def app(out: Path, every: float, gpu: str | None = None) -> gr.Blocks:
    with gr.Blocks(title=f"autoresearch · {out.name}") as demo:
        gr.Markdown(f"## autoresearch · `{out.name}`")
        status = gr.Markdown("waiting for the first checkpoint...")
        plot = gr.LinePlot(
            x="trial",
            y="val_bpb",
            color="series",
            # Every series up front: the plot keeps the ones it saw first.
            color_map={"trial": "#4c9be8", "best so far": "#2ca02c", "baseline": "#999999"},
            title="val_bpb (lower is better)",
        )
        trial_table = gr.Dataframe(label="trials, newest first", wrap=True)
        agent_table = gr.Dataframe(label="agents")
        log = gr.Textbox(label="recent activity, newest first", lines=20, max_lines=20)

        def refresh():
            g = load(out)
            if g is None:
                return (gr.skip(),) * 5
            base = seed(g, gpu)
            table, points = trials(g, base)
            return header(g, base), points, table, agents(g), activity(g)

        outputs = [status, plot, trial_table, agent_table, log]
        demo.load(refresh, outputs=outputs)
        gr.Timer(every).tick(refresh, outputs=outputs)
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path, help="the run directory (run.py's --out)")
    p.add_argument("--every", type=float, default=3.0, help="seconds between refreshes")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--baseline", default=None, help="compare with this GPU's baseline instead")
    args = p.parse_args()
    app(args.out.resolve(), args.every, args.baseline).launch(server_port=args.port)
