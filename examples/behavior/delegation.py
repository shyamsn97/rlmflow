"""Does the prompt delegate when it should, and stay local when it should not?

Prompt substring tests prove the strategy text is present, not that it changes
behavior, and the two failure modes pull in opposite directions: wording strong
enough to fan out three independent files can also spawn a child to add four
numbers. So each scenario here names the behavior it expects, runs a live model,
and grades the run from the trajectory rather than from the prose.

    python examples/behavior/delegation.py                    # the cheap scenarios
    python examples/behavior/delegation.py --repeat 3         # delegation is stochastic
    python examples/behavior/delegation.py --scenario boids   # the heavy one

Needs an API key for ``--model``. Exit status is 0 only when every selected
scenario reaches ``--min-pass-rate``. ``tests/test_delegation_behavior.py`` runs
the same scenarios under pytest, skipped unless ``RLMFLOW_LIVE_TESTS=1``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import random
import re
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import add_model_args, example_run_dir  # noqa: E402

from rlmflow import FILE_TOOLS, AgentStart, Flow, LocalRuntime  # noqa: E402
from rlmflow.llm import client_for  # noqa: E402

RUN_NAME = "delegation-behavior"

#: A child that answered in one turn with almost nothing is the shape of a
#: delegated one-liner: the coordination cost bought no work.
WASTED_TURNS = 1
WASTED_RESULT_CHARS = 400


def usable_child_result(result: Any, *, min_chars: int) -> bool:
    """Accept concise structured manifests as well as substantial prose/code."""
    if isinstance(result, (dict, list, tuple)):
        return bool(result)
    return len(str(result or "").strip()) >= min_chars


@dataclass
class Observed:
    """What one run did, measured from its graph."""

    answer: Any
    children: list[AgentStart]
    root_turns: int
    #: Children attached to a single launch action, i.e. launched in one turn.
    #: A batch above one is the `asyncio.gather` shape; three separate launches
    #: of one child each serialize instead.
    max_launch_batch: int
    wasted_children: int
    tokens: int
    seconds: float
    workdir: Path

    @property
    def text(self) -> str:
        return self.answer if isinstance(self.answer, str) else json.dumps(self.answer)


@dataclass(frozen=True)
class Scenario:
    name: str
    #: What this scenario asserts about topology, for the report's second column.
    expects: str
    query: str
    inputs: dict[str, str]
    grade: Callable[[Observed], list[str]]
    max_iters: int = 12
    max_depth: int = 1
    output_schema: dict[str, Any] | None = None
    with_files: bool = False
    cheap: bool = True


def observe(root: AgentStart, answer: Any, *, seconds: float, workdir: Path) -> Observed:
    children = [node for node in root.walk() if isinstance(node, AgentStart) and node is not root]
    # Children launched from one REPL block hang off that block's action, so
    # grouping by parent counts same-turn fan-out. Do not look for `AppendChild`:
    # that type is for controller-injected launches and never appears in a run
    # where the agent called `launch_subagent` itself.
    per_action: dict[str, int] = {}
    for child in children:
        if child.parent is not None:
            per_action[child.parent.id] = per_action.get(child.parent.id, 0) + 1
    wasted = [
        child
        for child in children
        if child.llm_turns() <= WASTED_TURNS
        and not usable_child_result(child.result(), min_chars=WASTED_RESULT_CHARS)
    ]
    return Observed(
        answer=answer,
        children=children,
        root_turns=root.llm_turns(),
        max_launch_batch=max(per_action.values(), default=0),
        wasted_children=len(wasted),
        tokens=root.usage.total,
        seconds=seconds,
        workdir=workdir,
    )


def expect_no_children(observed: Observed) -> list[str]:
    if not observed.children:
        return []
    names = ", ".join(child.config.name for child in observed.children)
    detail = f" ({observed.wasted_children} did one turn for a tiny result)"
    return [
        f"delegated {len(observed.children)} scope(s) to {names}"
        + (detail if observed.wasted_children else "")
    ]


def expect_fanout(observed: Observed, *, least: int = 2) -> list[str]:
    failures = []
    if len(observed.children) < least:
        failures.append(f"expected at least {least} children, got {len(observed.children)}")
    empty = [
        child.config.name
        for child in observed.children
        if not usable_child_result(child.result(), min_chars=200)
    ]
    if empty:
        failures.append(f"children returned nothing usable: {', '.join(empty)}")
    return failures


# --- Scenario 1: substantial independent work, so fan out ---------------------

EXPLAINER_TOPICS = """Topics:
1. How TCP congestion control reacts to packet loss.
2. Why B-trees suit on-disk indexes better than balanced binary search trees.
3. What causes catastrophic cancellation in floating-point arithmetic, and how
   to avoid it.
"""

EXPLAINER_QUERY = (
    "Write one technical explainer for each topic in INPUTS['topics']. Every "
    "explainer must be at least 500 words and must stand on its own. Return all "
    "three in a single answer, each under a heading naming its topic."
)


def grade_explainers(observed: Observed) -> list[str]:
    failures = expect_fanout(observed)
    text = observed.text.lower()
    missing = [term for term in ("congestion", "b-tree", "cancellation") if term not in text]
    if missing:
        failures.append(f"answer never covers: {', '.join(missing)}")
    if len(observed.text) < 4500:
        failures.append(f"answer is {len(observed.text)} chars, too short for three explainers")
    return failures


# --- Scenario 2: one deterministic lookup, so stay local ----------------------

GATEWAY_CONFIG = """# deployment.conf — staging
[gateway]
image = registry.internal/gateway:2.14.3
replicas = 4
bind_port = 8443
health_path = /healthz
request_timeout_ms = 30000

[gateway.tls]
cert = /etc/certs/gateway.pem
min_version = 1.2
alpn = h2,http/1.1

[postgres]
image = postgres:16
port = 5432
max_connections = 200
shared_buffers = 2GB
wal_level = replica

[redis]
image = redis:7
port = 6379
maxmemory = 4gb
maxmemory_policy = allkeys-lru

[worker]
image = registry.internal/worker:2.14.3
replicas = 12
metrics_port = 9102
queue = tasks.default
prefetch = 16
retry_backoff_ms = 250

[observability]
otlp_endpoint = http://collector.observability:4317
sample_rate = 0.05
log_level = info

[limits]
cpu = 2
memory = 4Gi
ephemeral_storage = 8Gi
"""

GATEWAY_QUERY = (
    "INPUTS['config'] is a deployment config. Report only the port the gateway "
    "service binds to, as a number."
)


def grade_gateway(observed: Observed) -> list[str]:
    failures = expect_no_children(observed)
    if "8443" not in observed.text:
        failures.append(f"wrong port in answer: {observed.text[:120]!r}")
    return failures


# --- Scenario 3: separable but trivial outputs, so stay local ------------------

_numbers = random.Random(20260817)
NUMBERS = [_numbers.randint(-500, 500) for _ in range(200)]
NUMBERS_TEXT = ", ".join(str(number) for number in NUMBERS)

STATS_QUERY = (
    "INPUTS['numbers'] is a comma-separated list of integers. Report its count, "
    "sum, minimum, and maximum."
)

STATS_SCHEMA = {
    "type": "object",
    "properties": {
        "count": {"type": "integer"},
        "sum": {"type": "integer"},
        "min": {"type": "integer"},
        "max": {"type": "integer"},
    },
    "required": ["count", "sum", "min", "max"],
}


def grade_stats(observed: Observed) -> list[str]:
    failures = expect_no_children(observed)
    expected = {
        "count": len(NUMBERS),
        "sum": sum(NUMBERS),
        "min": min(NUMBERS),
        "max": max(NUMBERS),
    }
    answer = observed.answer if isinstance(observed.answer, dict) else {}
    wrong = {key: answer.get(key) for key, want in expected.items() if answer.get(key) != want}
    if wrong:
        failures.append(f"wrong statistics {wrong}, expected {expected}")
    return failures


# --- Scenario 4: the boids task, opt-in because it is slow ---------------------


def boids_task() -> tuple[str, str]:
    """The query and requirements from the boids example, imported by path.

    That example is the case this whole strategy was written against, so the
    scenario reads its task rather than paraphrasing it.
    """
    path = examples_dir / "coding" / "boids" / "boids.py"
    spec = importlib.util.spec_from_file_location("boids_example", path)
    if spec is None or spec.loader is None:  # pragma: no cover - path is in-repo
        raise RuntimeError(f"cannot import the boids example from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TASK, module.CONTEXT


#: The modules the task specifies, and the global each one owns. The four
#: substantial ones are what delegation is for; `vec.js`, `index.html`, and
#: `style.css` are the glue a parent should keep.
BOIDS_GLOBALS = {
    "vec.js": "Vec",
    "spatial.js": "SpatialGrid",
    "rules.js": "Rules",
    "species.js": "SPECIES",
    "render.js": "Renderer",
}
BOIDS_SCRIPTS = (*BOIDS_GLOBALS, "main.js")
BOIDS_FILES = (*BOIDS_SCRIPTS, "index.html", "style.css")
#: One child per requested artifact is a valid decomposition when those artifacts
#: have explicit contracts. More children than artifacts is the concrete runaway
#: shape: at least one scope is duplicated or does not own a deliverable.
BOIDS_MOST_CHILDREN = len(BOIDS_FILES)


def defines(text: str, symbol: str) -> bool:
    pattern = rf"(\b(?:var|let|const|function|class)\s+{symbol}\b|\b{symbol}\s*=)"
    return re.search(pattern, text) is not None


def boids_scope_files(child: AgentStart) -> set[str]:
    """The explicitly assigned artifacts in a child scope.

    Prefer the output schema, then dedicated filename inputs, then the goal. Broad
    context inputs often mention every file and therefore cannot identify scope.
    """
    schema = child.config.output_schema or {}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    claimed = set(properties).intersection(BOIDS_FILES)
    if claimed:
        return claimed

    for key in ("filename", "filenames", "files"):
        raw = child.config.inputs.get(key)
        if not raw:
            continue
        try:
            values = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            values = raw
        if isinstance(values, str):
            values = [values]
        if isinstance(values, list):
            claimed.update(str(value) for value in values if str(value) in BOIDS_FILES)
    if claimed:
        return claimed

    return {name for name in BOIDS_FILES if re.search(rf"\b{re.escape(name)}\b", child.content)}


def missing_members(text: str, symbol: str, members: tuple[str, ...]) -> list[str]:
    """Return required JavaScript members that do not appear as definitions."""
    missing = []
    for member in members:
        patterns = (
            rf"\b{re.escape(member)}\s*\(",
            rf"\.{re.escape(member)}\s*=\s*function\b",
            rf"\b{re.escape(member)}\s*:\s*function\b",
        )
        if not any(re.search(pattern, text) for pattern in patterns):
            missing.append(f"{symbol}.{member}")
    return missing


def grade_boids(observed: Observed) -> list[str]:
    failures = expect_fanout(observed, least=2)
    if len(observed.children) > BOIDS_MOST_CHILDREN:
        failures.append(
            f"{len(observed.children)} children for {len(BOIDS_FILES)} requested artifacts: "
            "delegation ran beyond the available deliverables"
        )

    owners: dict[str, list[str]] = {}
    for child in observed.children:
        for name in boids_scope_files(child):
            owners.setdefault(name, []).append(child.config.path)
    duplicates = {name: paths for name, paths in owners.items() if len(paths) > 1}
    if duplicates:
        detail = "; ".join(f"{name}: {', '.join(paths)}" for name, paths in duplicates.items())
        failures.append(f"duplicate delegated artifact scopes: {detail}")

    present = {name: observed.workdir / name for name in BOIDS_FILES}
    missing = [name for name, path in present.items() if not path.exists()]
    if missing:
        failures.append(f"missing files: {', '.join(missing)}")

    empty = [
        name
        for name, path in present.items()
        if path.exists() and not path.read_text(errors="replace").strip()
    ]
    if empty:
        failures.append(f"empty files: {', '.join(empty)}")

    undefined = [
        f"{symbol} in {name}"
        for name, symbol in BOIDS_GLOBALS.items()
        if present[name].exists() and not defines(present[name].read_text(errors="replace"), symbol)
    ]
    if undefined:
        failures.append(f"contract not met: {', '.join(undefined)}")

    contracts = {
        "vec.js": ("Vec", ("add", "sub", "scale", "limit", "mag", "normalize", "dist")),
        "spatial.js": ("SpatialGrid", ("insert", "clear", "neighbors")),
        "rules.js": ("Rules", ("align", "cohere", "separate", "avoid")),
        "render.js": ("Renderer", ("draw",)),
    }
    missing_api = []
    for name, (symbol, members) in contracts.items():
        if present[name].exists():
            missing_api.extend(
                f"{member} in {name}"
                for member in missing_members(
                    present[name].read_text(errors="replace"),
                    symbol,
                    members,
                )
            )
    if missing_api:
        failures.append(f"contract members missing: {', '.join(missing_api)}")

    index = present["index.html"]
    if index.exists():
        html = index.read_text(errors="replace")
        if 'type="module"' in html:
            failures.append("index.html loads an ES module despite the constraint")
        loaded = re.findall(r'<script[^>]+src="([^"]+)"', html)
        absent = [name for name in BOIDS_SCRIPTS if name not in loaded]
        if absent:
            failures.append(f"index.html never loads: {', '.join(absent)}")
        elif loaded[-1] != "main.js" or loaded.index("vec.js") != 0:
            failures.append(f"script tags out of dependency order: {loaded}")

    main = present["main.js"]
    if main.exists():
        source = main.read_text(errors="replace")
        missing_integration = [
            symbol
            for symbol in ("SpatialGrid", "Rules", "SPECIES", "Renderer")
            if not re.search(rf"\b{symbol}\b", source)
        ]
        if missing_integration:
            failures.append(f"main.js never integrates: {', '.join(missing_integration)}")
    return failures


def scenarios() -> list[Scenario]:
    task, context = boids_task()
    return [
        Scenario(
            name="explainers",
            expects="delegates",
            query=EXPLAINER_QUERY,
            inputs={"topics": EXPLAINER_TOPICS},
            grade=grade_explainers,
        ),
        Scenario(
            name="gateway-port",
            expects="stays local",
            query=GATEWAY_QUERY,
            inputs={"config": GATEWAY_CONFIG},
            grade=grade_gateway,
            max_iters=6,
        ),
        Scenario(
            name="number-stats",
            expects="stays local",
            query=STATS_QUERY,
            inputs={"numbers": NUMBERS_TEXT},
            grade=grade_stats,
            max_iters=6,
            output_schema=STATS_SCHEMA,
        ),
        Scenario(
            name="boids",
            expects="delegates",
            query=task,
            inputs={"context": context},
            grade=grade_boids,
            max_iters=30,
            max_depth=1,
            with_files=True,
            cheap=False,
        ),
    ]


@dataclass
class Result:
    scenario: Scenario
    observed: Observed | None = None
    failures: list[str] = field(default_factory=list)
    graph: Path | None = None

    @property
    def ok(self) -> bool:
        return not self.failures


def run_scenario(
    scenario: Scenario,
    *,
    model: str = "gpt-5-mini",
    out_dir: Path,
    workers: int = 4,
    llm: Any | None = None,
    additional_model: str | None = None,
    additional_llm: Any | None = None,
) -> Result:
    """Run one scenario and grade its trajectory.

    ``llm`` overrides the client built from ``model``, which is how the offline
    tests exercise this harness without calling an API. ``additional_model``
    exposes another registered choice to the agent; ``additional_llm`` is its
    offline equivalent. The agent still selects a model on every launch.
    """
    run_dir = out_dir / scenario.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    workdir = run_dir / "workspace"
    workdir.mkdir(parents=True)

    additional_client = additional_llm
    if additional_client is None and additional_model is not None:
        additional_client = client_for(additional_model)
    additional_key = additional_model or "additional"

    flow = Flow(
        llm if llm is not None else client_for(model),
        llm_clients=(
            {additional_key: additional_client} if additional_client is not None else None
        ),
        runtime=LocalRuntime(working_directory=workdir) if scenario.with_files else None,
        tools=[FILE_TOOLS] if scenario.with_files else None,
        workers=workers,
    )
    root = flow.start(
        scenario.query,
        inputs=scenario.inputs,
        max_depth=scenario.max_depth,
        max_iters=scenario.max_iters,
        output_schema=scenario.output_schema,
    )

    async def drive() -> Any:
        try:
            return await flow.arun(root)
        finally:
            await flow.aclose()

    started = time.perf_counter()
    try:
        answer = asyncio.run(drive())
    finally:
        root.save(run_dir / "graph")

    observed = observe(root, answer, seconds=time.perf_counter() - started, workdir=workdir)
    return Result(
        scenario=scenario,
        observed=observed,
        failures=scenario.grade(observed),
        graph=run_dir / "graph",
    )


def summary_row(result: Result, *, model: str) -> dict[str, Any]:
    """One report line, recording the model so runs stay comparable."""
    seen = result.observed
    return {
        "scenario": result.scenario.name,
        "expects": result.scenario.expects,
        "model": model,
        "ok": result.ok,
        "failures": result.failures,
        "children": [child.config.name for child in seen.children] if seen else [],
        "max_launch_batch": seen.max_launch_batch if seen else 0,
        "wasted_children": seen.wasted_children if seen else 0,
        "root_turns": seen.root_turns if seen else 0,
        "tokens": seen.tokens if seen else 0,
        "seconds": round(seen.seconds, 2) if seen else 0.0,
    }


def report(results: list[Result], *, min_pass_rate: float) -> bool:
    """Print the scorecard and return whether every scenario cleared the bar."""
    header = f"{'scenario':<14}{'expects':<13}{'kids':>5}{'batch':>6}{'turns':>6}"
    header += f"{'tokens':>9}{'secs':>7}  verdict"
    print(f"\n{header}\n{'-' * len(header)}")
    for result in results:
        seen = result.observed
        cells = [
            f"{result.scenario.name:<14}",
            f"{result.scenario.expects:<13}",
            f"{len(seen.children) if seen else 0:>5}",
            f"{seen.max_launch_batch if seen else 0:>6}",
            f"{seen.root_turns if seen else 0:>6}",
            f"{seen.tokens if seen else 0:>9,}",
            f"{seen.seconds if seen else 0:>7.1f}",
            "  PASS" if result.ok else "  FAIL",
        ]
        print("".join(cells))
        for failure in result.failures:
            print(f"{'':<14}- {failure}")

    rates: dict[str, list[bool]] = {}
    for result in results:
        rates.setdefault(result.scenario.name, []).append(result.ok)

    print()
    passed = True
    for name, outcomes in rates.items():
        rate = sum(outcomes) / len(outcomes)
        clear = rate >= min_pass_rate
        passed = passed and clear
        note = "" if clear else f" (below {min_pass_rate:.0%})"
        print(f"{name:<14}{sum(outcomes)}/{len(outcomes)} passed{note}")
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    parser.add_argument(
        "--additional-model",
        default=None,
        help="Optional additional model the agent may select for subagents.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        metavar="NAME",
        help="Run only this scenario (repeatable). Default: every cheap scenario.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Runs per scenario.")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=1.0,
        help="Fraction of runs that must pass per scenario (default: all of them).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", default=str(example_run_dir(RUN_NAME)))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    available = {scenario.name: scenario for scenario in scenarios()}
    if args.scenario:
        unknown = [name for name in args.scenario if name not in available]
        if unknown:
            raise SystemExit(f"unknown scenario(s): {', '.join(unknown)}")
        selected = [available[name] for name in args.scenario]
    else:
        selected = [scenario for scenario in available.values() if scenario.cheap]

    out_dir = Path(args.out_dir).resolve()
    print(
        f"model: {args.model}\n"
        f"additional model: {args.additional_model or 'none'}\n"
        f"out: {out_dir}"
    )
    print(f"scenarios: {', '.join(scenario.name for scenario in selected)} x{args.repeat}")

    results = []
    for attempt in range(args.repeat):
        for scenario in selected:
            label = scenario.name if args.repeat == 1 else f"{scenario.name} #{attempt + 1}"
            print(f"\n=== {label} ({scenario.expects}) ===")
            results.append(
                run_scenario(
                    scenario,
                    model=args.model,
                    additional_model=args.additional_model,
                    out_dir=out_dir / f"run{attempt + 1}",
                    workers=args.workers,
                )
            )

    passed = report(results, min_pass_rate=args.min_pass_rate)
    summary = out_dir / "report.json"
    rows = [summary_row(result, model=args.model) for result in results]
    summary.write_text(json.dumps(rows, indent=2))
    print(f"\nreport: {summary}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
