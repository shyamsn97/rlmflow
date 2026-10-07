# benchmarks/

Four public evaluation suites plus direct dataset runs. Source of truth: [`docs/benchmarks.md`](../docs/benchmarks.md). Suite membership and runner matrices live in [`eval/suite.py`](eval/suite.py).

```bash
python benchmarks/run_benchmark.py --list
make -f benchmarks/Makefile smoke
make -f benchmarks/Makefile delegation
make -f benchmarks/Makefile long-context
make -f benchmarks/Makefile interactive
```

The default OOLONG benchmark is a fixed TREC-coarse 128K slice: two pilot
questions and 25 full questions. Delegation uses 16 frozen fan-out tasks. The
interactive benchmark runs MiniMaze, Sudoku Search, and MazeBench through one
environment report schema.
