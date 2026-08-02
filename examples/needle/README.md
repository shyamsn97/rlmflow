# Needle in a haystack

Two variants of the same search task:

- [`haystack.py`](haystack.py) — one massive in-memory `INPUTS["haystack"]` string; the agent must chunk and delegate.
- [`filesystem.py`](filesystem.py) — hundreds of files under `haystack/`; uses `FILE_TOOLS`.

```bash
python examples/needle/haystack.py --num-lines 1000000
python examples/needle/filesystem.py --num-files 500
python examples/needle/haystack.py --docker-image rlmflow:local
```

Runs save to `examples/_runs/needle-haystack/` and `examples/_runs/needle-filesystem/` by default.
Both examples render the live agent tree and checkpoint through a
`ConsumerGroup`; pass `--no-viz` to disable terminal redraws.
