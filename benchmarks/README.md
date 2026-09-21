# benchmarks/

Four public evaluation suites plus direct dataset runs. Source of truth: [`docs/benchmarks.md`](../docs/benchmarks.md). Suite membership and runner matrices live in [`eval/suite.py`](eval/suite.py).

```bash
python -m benchmarks.eval --list-suites
make -f benchmarks/Makefile eval-smoke
make -f benchmarks/Makefile eval-conformance
make -f benchmarks/Makefile eval-core
make -f benchmarks/Makefile eval-interactive
```

OOLONG-Pairs 32K and OOLONG Yahoo 128K form the paper-aligned core suite. The interactive suite runs the official MazeBench environment and writes replay-capable reports.
