# Architectures

Each shape in the top-level README's grid is a real run. Three are small scripts in this folder; the other three are the full examples that produced their graphs.

| Shape | Example | Run it |
| --- | --- | --- |
| Fan-out | [`needle/harnesses/`](../needle/harnesses/) | `python examples/needle/harnesses/run.py --all` |
| Recursive tree | [`tree.py`](tree.py) | `python examples/architectures/tree.py` |
| Pipeline | [`pipeline.py`](pipeline.py) | `python examples/architectures/pipeline.py` |
| Peers | [`coding/boids/`](../coding/boids/) | `python examples/coding/boids/boids.py --harness claude_code` |
| Review loop | [`review.py`](review.py) | `python examples/architectures/review.py` |
| Rewind and fork | [`shepherd/`](../shepherd/) | `python examples/shepherd/shepherd.py --trace` |

In the pipeline, a root awaits each stage and passes its result to the next as an argument; there is no dependency declaration, just `await`.

The scripts here default to `--model gpt-5-mini` and save to `examples/_runs/architectures/<shape>`. Render a saved run with:

```bash
rlmflow render gif examples/_runs/architectures/tree tree.gif
```
