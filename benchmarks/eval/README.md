# Eval harness

The public interface is the environment-centered `benchmark(...)` callable.
Use the single CLI to run one of the four named benchmarks:

```bash
python benchmarks/run_benchmark.py --list
python benchmarks/run_benchmark.py long-context --split full \
  --runners official-rlm,rlmflow-depth-0,rlmflow-depth-1,rlmflow-depth-2 \
  --executor docker
```

Each environment owns its observation, hidden state, scoring, and report data.
Every runner receives the same canonical observation bytes. The default
long-context manifest contains two OOLONG TREC-coarse 128K questions for
`pilot` and ten for `full`.

See [`docs/benchmarks.md`](../../docs/benchmarks.md) for the catalog, runner
capabilities, output schema, and Makefile commands.
