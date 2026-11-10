# benchmarks/

One table: three systems on the same tasks, with the same model, time limit and
dollar cap, each scored from what it leaves in its work directory.

| System | What runs |
| --- | --- |
| `claude_code` | `claude -p TASK` |
| `claude_code_rlmflow` | `claude -p "/rlmflow:run TASK"`; workers are Claude Code sessions |
| `rlmflow` | rlmflow's REPL harness as the root, with REPL workers |

| Task | What it is |
| --- | --- |
| `oolong` | Long-document QA: 15 OOLONG `trec_coarse` questions at 128K tokens ([`table/oolong.py`](table/oolong.py)) |
| `ale` | Algorithm engineering: ALE-Bench-Lite, 10 AtCoder Heuristic Contest problems ([`table/ale.py`](table/ale.py)) |

```bash
uv run --extra bench python -m benchmarks.table.run oolong rlmflow --ids 2
uv run python -m benchmarks.table.run ale claude_code --minutes 30 --budget 5
```

Each problem gets a fresh work directory under `--out` (default
`benchmarks/table/runs`), and its result is appended to `results.jsonl` there; a
finished problem is skipped on the next run. ALE-Bench runs in its own Python
(`ALE_PYTHON`, 3.11+) and needs Docker.
