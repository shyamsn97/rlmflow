# Collaboration

Both examples put equal peers in one graph as top-level agents. Neither has a
root or manager agent.

## Decentralized peers

[`decentralized_peers.py`](decentralized_peers.py) runs three real-model peers.
Peers can't await each other, so one asks the others for their values with
`AGENTS[name].send(...)` and ends its turn; their replies arrive as messages and
reopen it, and once it has both it sends them the total.
Messages are `message` nodes with `message` edges from the cell that sent them,
so the saved graph shows who told whom what.

    python examples/collaboration/decentralized_peers.py --model gpt-5-mini

## Shared task board

[`shared_board.py`](shared_board.py) is a DeLM-style board made of a few host
functions over shared dicts and a lock, passed to the `Flow` as tools. A fixed
pool of workers shares it, with no manager:

| Board idea | Tool |
| --- | --- |
| a task | `board_post(id, task)`; the host posts the seeds, workers post follow-ups they find |
| open tasks | `board_open()` |
| claiming a task | `board_claim(id, worker)`; atomic, so exactly one worker wins |
| a finished task's result | `board_finish(id, result)`, read back with `board_results()` |
| workers | top-level agents run `--workers` at a time |

The tools run in the host process, so every worker sees the same board. Each
call's result is saved on the `code` node that made it: the saved run records
what each worker saw, and a rebuilt REPL replays those results instead of
calling the board again.

```bash
python examples/collaboration/shared_board.py \
  --model gpt-5-mini \
  --workers 3 \
  --task "API::Propose the public API and its invariants." \
  --task "Concurrency::Identify races and required locking." \
  --task "Tests::Design concurrency and termination tests."
```

`--task` is repeatable and accepts either `SUBJECT` or `SUBJECT::DETAIL`.
