# Patterns

Each shape in the top-level README's grid is a real run. Five are small scripts in this folder; the other three are the full examples that produced their graphs.

| Shape | Example | Run it |
| --- | --- | --- |
| Fan-out | [`needle/filesystem.py`](../needle/filesystem.py) | `python examples/needle/filesystem.py --all` |
| Recursive tree | [`tree.py`](tree.py) | `python examples/patterns/tree.py` |
| Pipeline | [`pipeline.py`](pipeline.py) | `python examples/patterns/pipeline.py` |
| Review loop | [`review.py`](review.py) | `python examples/patterns/review.py` |
| Peers, messaging | [`peers.py`](peers.py) | `python examples/patterns/peers.py` |
| Peers, shared board | [`board.py`](board.py) | `python examples/patterns/board.py` |
| Peers, coding | [`coding/boids/`](../coding/boids/) | `python examples/coding/boids/boids.py --harness claude_code` |
| Rewind and fork | [`shepherd/`](../shepherd/) | `python examples/shepherd/shepherd.py --trace` |

The scripts here default to `--model gpt-6.1-terra` and save to `examples/_runs/`. Render a saved run with:

```bash
rlmflow render gif examples/_runs/patterns/tree tree.gif
```

## Pipeline

A root awaits each stage and passes its result to the next as an argument; there is no dependency declaration, just `await`.

## Peers

`peers.py` and `board.py` put equal peers in one graph as top-level agents, with no root or manager.

In [`peers.py`](peers.py), three real-model peers can't await each other, so one asks the others for their values with `AGENTS[name].send(...)` and ends its turn. Their replies arrive as messages and reopen it; once it has both, it sends them the total. Each message is one of the receiver's nodes, with a `write` edge from the node whose code sent it, so the saved graph shows who told whom what.

[`board.py`](board.py) is a DeLM-style task board made of a few host functions over shared dicts and a lock, passed to the `Flow` as tools. A fixed pool of workers shares it:

| Board idea | Tool |
| --- | --- |
| a task | `board_post(id, task)`; the host posts the seeds, workers post follow-ups they find |
| open tasks | `board_open()` |
| claiming a task | `board_claim(id, worker)`; atomic, so exactly one worker wins |
| a finished task's result | `board_finish(id, result)`, read back with `board_results()` |
| workers | top-level agents; `--workers` of them, with that many model calls at once |

The tools run in the host process, so every worker sees the same board. They are `@tool(cache=True)`, so each call's result is saved on the node whose code made it: the saved run records what each worker saw, and a rebuilt REPL replays those results instead of calling the board again.

```bash
python examples/patterns/board.py \
  --model gpt-6.1-terra \
  --workers 3 \
  --task "API::Propose the public API and its invariants." \
  --task "Concurrency::Identify races and required locking." \
  --task "Tests::Design concurrency and termination tests."
```

`--task` is repeatable and accepts either `SUBJECT` or `SUBJECT::DETAIL`.
