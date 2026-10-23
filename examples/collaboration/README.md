# Collaboration

Both examples put equal peers in one graph as top-level agents. Neither has a
root or manager agent.

## Decentralized peers

[`decentralized_peers.py`](decentralized_peers.py) runs three real-model peers.
One draws `peer` edges with `connect_agent`, asks the others for their values
with `ask_agent`, and sends them the total with `send_agent`.
Messages are `user_query` nodes with `from` edges to the action that sent them,
so the saved graph shows who told whom what.

    python examples/collaboration/decentralized_peers.py --model gpt-5-mini

## Shared task board

[`shared_board.py`](shared_board.py) is a DeLM-style board where the board is
the graph. Every task is an agent:

| Board idea | In the graph |
| --- | --- |
| a task | a top-level agent (`g.create_agent`) |
| follow-up work | a child from `launch_subagent(task=...)` |
| claiming a task | `Flow(workers=N)` running whichever agents are ready |
| a finished task's result | its `DoneOutput` (`submit(...)`) |
| shared context | `AGENTS`, every task's status and result |

Nothing lives outside the graph, so the saved run is the whole board and it
resumes or replays like any other graph.

```bash
python examples/collaboration/shared_board.py \
  --model gpt-5-mini \
  --workers 3 \
  --task "API::Propose the public API and its invariants." \
  --task "Concurrency::Identify races and required locking." \
  --task "Tests::Design concurrency and termination tests."
```

`--task` is repeatable and accepts either `SUBJECT` or `SUBJECT::DETAIL`.
