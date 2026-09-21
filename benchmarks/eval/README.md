# Eval harness

Dataset, Model, Runner, Logger. Which suite or direct dataset to run: [`docs/benchmarks.md`](../../docs/benchmarks.md). Public suite membership and runner matrices: [`suite.py`](suite.py).

```bash
python -m benchmarks.eval --list-suites
make -f benchmarks/Makefile eval-smoke
```

```python
from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset


@dataset("my_dataset")
class MyDataset(Dataset): ...
```

Import the module from the package `__init__.py`. Add a `SuiteSpec` in `suite.py` only when the dataset belongs to one of the four public scientific questions; otherwise run it directly with an explicit `--runner`.
