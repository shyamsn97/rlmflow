# Benchmarks

The benchmark harness has one public interface:

```python
benchmark(
    name="delegation",
    description="...",
    environments=environments,
    runners=["prime-agent", "official-rlm", "jaz", "rlmflow-depth-1"],
    model="openai:gpt-5-mini",
    output_dir=Path("benchmarks/runs/delegation"),
)
```

An environment is one complete scored problem. A benchmark is a collection of environments run against each requested runner. Every runner receives the same task description and canonical observation bytes. Hidden answers and simulator state remain inside the environment.

## Benchmark catalog

- `smoke`: one deterministic synthetic needle/haystack environment.
- `delegation`: a 16-task frozen swarm-fanout matrix spanning 4–16 children, eight data formats, and four aggregation operations; pilot has one task.
- `long-context`: a fixed OOLONG TREC-coarse 128K slice; pilot has two questions and full has 25.
- `interactive`: MiniMaze, Sudoku Search, and MazeBench.

`pilot` and `full` are frozen manifests, not sampling requests. Needle and MazeBench expose their one deterministic environment in both splits. The other OOLONG datasets remain available as explicit datasets but are not part of the default benchmark.

See [Long-context prompt guidance](research/long_context_prompt_guidance.md) for the historical failure analysis and the experiments that led to the current phase-free prompt policy.

## CLI

There is one command:

```bash
python benchmarks/run_benchmark.py --list

python benchmarks/run_benchmark.py delegation --split full \
  --runners prime-agent,official-rlm,jaz,rlmflow-depth-1 \
  --model openai:gpt-5-mini \
  --executor docker --parallelism 4

python benchmarks/run_benchmark.py interactive --split pilot \
  --runners official-rlm,rlmflow-depth-1
```

`--runners` is repeatable and also accepts comma-separated values. Executors are `local`, `docker`, and `modal`. To resume, pass `--resume --output <existing-run>`.

The Makefile provides thin recipes:

```bash
make -f benchmarks/Makefile smoke
make -f benchmarks/Makefile delegation
make -f benchmarks/Makefile long-context
make -f benchmarks/Makefile interactive
```

## Outputs

Each run directory contains:

- `rows.jsonl`: one `EnvironmentReport` per environment and runner;
- `summary.json`: runner aggregates;
- `report.md` and `report.html`: the same per-problem metrics table;
- runner artifacts under `artifacts/`.

Every row records task and observation hashes, score/error, token counts, wall time, turns, agents, nodes, and environment metrics. Delegation rows include children, query calls, launch/retrieval/dedup evidence, and per-shard answers. Interactive rows include actions, forks, candidate selection, completion, and native task metrics.

Gold answers are never serialized into worker payloads. A description or observation hash mismatch is an infrastructure error and receives no semantic score.
