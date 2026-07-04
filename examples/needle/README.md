# Needle in a haystack

Two variants of the same search task:

- [`haystack.py`](haystack.py) — one massive in-memory `INPUTS["haystack"]` string; the agent must chunk and delegate.
- [`filesystem.py`](filesystem.py) — hundreds of files under `haystack/`; uses `FILE_TOOLS`.
- [`minimal_filesystem.py`](minimal_filesystem.py) — the same filesystem search using the tiny `rflow.minimal` stack.

```bash
python examples/needle/haystack.py --num-lines 1000000 --no-viz
python examples/needle/filesystem.py --num-files 500 --no-viz
python examples/needle/minimal_filesystem.py --num-files 100
python examples/needle/haystack.py --docker-image rlmflow:local
```

Runs save to `examples/_runs/needle-haystack/` and `examples/_runs/needle-filesystem/` by default.
