# Examples cleanup and benchmark removal

Status: proposed, not started
Date: 2026-10-29

## Summary

Remove the benchmark harness from the repository, and cut the examples down to
four flagships plus a small set of API tours. Autoresearch (TinyStories on
Modal) is the search flagship, with Claude Code sessions as its trial agents.

## Why

The library's pitch is that recursive agent runs become execution graphs you can
inspect, fork, replay and resume, with any harness (rlmflow's REPL, Claude Code,
pi) as the root or a worker. Benchmark scores measure a method (RLM, DELM, a
hybrid), not the library. A strong score doesn't sell the library and a weak one
doesn't hurt it. Examples are what show the pitch.

The benchmark harness also costs more than it returns:

- `benchmarks/` is 89 tracked files and about 13k lines, roughly a second project.
- It blocks releases: `release.yml` needs the `benchmark-contract` job to pass.
- Cheap runs are too noisy to support a claim, and fair runs are expensive.
  Notes from reviewing the DELM paper (arXiv 2606.10662):
  - The headline Terminal-Bench 4.0 and DeepSWE v1.1 grid costs about $11k
    at the paper's own per-task prices.
  - The cheapest defensible comparison is its Table 9 (OOLONG plus LongBench-v2
    multi-doc; RLM vs. DELM vs. hybrid). That costs about $140 for one full run
    and isn't needed for a release.
  - A 40-question subset has about ±8 points of noise, more than the 4.5-point
    effect the paper reports.

Separately, the examples have grown into a mix of overlapping one-off scripts,
empty folders, and expensive demos. It's hard to see what the library is for
from the examples folder.

## Decision 1: remove benchmarks

Delete:

- `benchmarks/` (the whole directory)
- `tests/test_benchmark_fanout.py`, `tests/test_benchmark_metrics.py`
- the `benchmark-contract` job in `.github/workflows/ci.yml`
- the `benchmark-contract` job in `.github/workflows/release.yml`, and remove it
  from the release job's `needs`
- the forwarding recipe in the root `Makefile` (around lines 188–191)
- `docs/benchmarks.md` and its link in `docs/index.md`
- the README sections on the evaluation harness, the OOLONG-pairs pilot result,
  and the `make -f benchmarks/Makefile ...` commands
- benchmark references in `examples/README.md` and `docs/history/colocate.md`

Before deleting, tag the current commit `benchmarks-archive` so the harness,
runners (including the PrimeAgent runner) and the pilot result can be restored
with one checkout.

Add a CHANGELOG entry saying the harness was removed and where it's archived.

## Decision 2: four flagship examples

1. Needle: recursive decomposition. The harness matrix
   (`needle/harnesses/run.py`) is the proof that any harness can be the root
   or a worker. Make sure it runs cleanly and put its output table in the docs.
2. Shepherd (Sokoban): a worker jams on a Sokoban puzzle; a meta-agent
   inspects the failed run, rewinds it, forks recovery branches and picks the
   best one. This is the forking flagship and must stay. Keep as is, including
   `shepherd/sokoban.py` and the board/trace assets.
3. Autoresearch: a planner runs waves of parallel training trials on Modal GPUs
   and builds on the best results; the trial agents are Claude Code sessions.
   See below.
4. Boids: practical software construction. In the release check (2026-10-30)
   gpt-5 wrote the whole app in one reply, with no delegation and no REPL
   cells, so as written it doesn't show the library. Change the task so it
   requires splitting the work across agents, or demote it to a tour.

### Which harness each flagship runs on

Default to rlmflow's REPL harness. It works with any model (including Claude
models through the Anthropic client), needs no extra installs, and records every
step as a node with its full conversation, so a fork from any node is exact.

Claude Code and pi forks are coarser today. REPL state forks exactly for every
harness, because it is rebuilt from the recorded `rlmflow repl` cells. The
difference is the conversation: the REPL harness keeps it in the graph, while
Claude Code and pi keep it in their own session files. Forking at a turn
boundary resumes the session exactly (`fork_session`). Forking from a cell in
the middle of a turn currently starts a fresh session with a summary of the
cells, so the conversation is lost. File edits made with the session's own
tools aren't recorded, only snapshotted at turn ends.

Planned fix for Claude Code: the installed `claude-agent-sdk` (0.2.174) has
`resume_session_at` (resume from a given message) alongside `fork_session`.
Record, on each `code` node, the session message that ran that cell, and fork
with `resume=<session>, resume_session_at=<message>, fork_session=True`. That
makes Claude Code forks exact from any cell. To prototype:

- how cells map to messages (likely in order of the `rlmflow repl` Bash calls;
  failed and parallel calls are the edge cases);
- whether resuming at an assistant message whose tool result is cut off needs
  `resume_drops_turn`.

pi sessions are JSONL files, so the same fix there is truncating the copied
file at the matching tool call.

Shepherd stays on the REPL harness either way. It's built on that harness's
internals: it adds the board to each model call, writes the scripted jam as
model turns in the graph, and enforces one push per model turn. A Claude Code
turn makes many tool calls, so that guard doesn't carry over. Porting it would
be a redesign, not a config change. Exact Claude Code forking gets its own
small example instead.

| Flagship | Root | Workers |
| --- | --- | --- |
| Needle | REPL | REPL by default; the harness matrix runs every root/worker pair of REPL, Claude Code and pi |
| Shepherd (Sokoban) | REPL | REPL |
| Autoresearch | REPL (gpt-5 planner) | Claude Code (`--implementer-harness claude_code`) |
| Boids | REPL | Claude Code |

- Shepherd forks from arbitrary points, so it needs the REPL harness's exact
  forks.
- Autoresearch's trial agents edit `train.py` and submit a GPU job, which is
  file editing that Claude Code is good at. It's the clearest "bring your own
  harness" demo on a task with real results. The planner stays on the REPL
  harness so it keeps exact forks.
- Boids is real file editing too, so its workers should also be Claude Code
  once the task actually delegates. Workspace snapshots at turn ends capture
  the files.
- pi appears only in the Needle harness matrix. It shows a third,
  model-agnostic harness works, but it's an extra install, so no flagship
  depends on it.

### Autoresearch

Task: lower validation bits per byte (`val_bpb`) of a small GPT trained on
TinyStories for a fixed 3 minutes. Agents may edit only `train.py`. The cached
baseline scores 1.155.

How it runs: a gpt-5 planner on the REPL harness reads the trial ledger, plans
a wave of hypotheses, and launches one trial agent per hypothesis in parallel.
Each trial agent edits its copy of `train.py` and submits a Modal L4 job, about
5 minutes end to end. The planner builds the next wave on the best results.

Why it's the search flagship:

- It's real and it works: three runs of 8 trials each on 2026-10-30, 12–17
  minutes each.
- It's cheap: a few dollars of model calls plus about 35 GPU-minutes per run.
- It shows the library's pitch: parallel children, a planner that builds on
  its children's results, and Claude Code sessions as workers under a REPL
  planner.
- The output is easy to read: a ranked trial table, a score plot, and a
  lineage graph of which trial built on which.

Release-check results (8 trials each, one run per row; differences between rows
are search noise, not a harness comparison):

| Trial agents | Trials completed | Best val_bpb | Claude cost |
| --- | --- | --- | --- |
| REPL harness (gpt-5) | 8 of 8 | 0.920 | — |
| Claude Code, run 1 | 6 of 8 (Modal startup race, since fixed) | 0.837 | $1.92 |
| Claude Code, run 2 | 8 of 8 | 0.866 | $1.25 |

To do before release:

- Make Claude Code the documented default for trial agents, keeping
  `--implementer-harness repl` as the option that needs no Claude login.
- Document a small default run (`--parallel 4 --max-submissions 8`); the
  Makefile defaults (12 in parallel, 48 submissions) are much bigger.
- Replace the `--help` smoke entry with a documented live command, marked slow.
- Note in the README that Claude Code runs need `--agent-runtime subprocess` or
  `local`, not Docker.

Circle packing (AlphaEvolve's 26-circle problem) was the earlier proposal for
this slot. It's deferred: it would be new, untested code, and autoresearch
already runs.

## Decision 3: keep, slimmed down

- `graph/`: the offline, deterministic Node API tour. Keep.
- `control/`: keep delegation and branching; review injection and
  `streaming_until.py` and drop whatever duplicates the others.
- `sandboxes/`: Docker and Modal. Keep.
- `providers/`: DSPy, MCP, Tinker, Claude Code delegation. Keep; move
  `drop_in_llm.py` here.
- `collaboration/`: keep `shared_board.py` as the DELM-style board. If it
  can't already run Claude Code workers, add that; it shows DELM's idea, a
  coordination layer over existing harnesses, as a short script.
  Review `decentralized_peers.py` for overlap with it.
- Root scripts, from seven down to three:
  - `showcase.py` (quickstart)
  - `structured_output.py`
  - `skills.py`, with `dynamic_skills.py` merged in

## Decision 4: delete

- `summarizer.py`, `llm_query_fanout.py`: Needle already covers map-reduce
  and parallel fan-out.
- `behavior/`, `graph_queue/`: empty.
- `research/` (TextCraft RL training, `rao_openenv_gather.py`,
  `rao_textcraft.py`): read these before deleting. RL practitioners are a named
  audience in `docs/positioning.md`, so keep TextCraft if it's self-contained
  and runs. Delete the rest.
- `notebooks/coding_agent.ipynb`: keep only if it still runs against the
  current API.

## Follow-on updates

- `examples/README.md`: rewrite around the four flagships, then tours.
- Main README: flagship section and links; remove the benchmark sections.
- `examples/run_examples.py`: update the manifest so the smoke runner covers
  exactly the remaining examples.
- `Makefile`: remove targets for deleted examples, and check the
  showcase-asset recipes still point at existing paths.
- `uv.lock`: regenerate it. It's stale against `pyproject.toml`, so the
  README's `uv run --locked ...` commands fail as written.
- `docs/positioning.md`: no change needed.

## Order of work

1. Tag `benchmarks-archive`.
2. Remove benchmarks, the tests, the CI and release jobs, and the docs.
   Run the test suite and check CI is green.
3. Prune and merge examples; update the smoke-runner manifest and the
   examples README. Run `make examples` (the offline set).
4. Finish autoresearch as a flagship (the to-do list above); add its trial
   table, score plot and lineage graph to the docs.
5. Update the main README and CHANGELOG.

Steps 2 and 3 are each one PR; autoresearch is its own PR.

## Risks

- Someone relying on the benchmark harness loses it. Mitigation: the archive
  tag, plus a CHANGELOG note.
- The README's pilot result (rlmflow 0.51 vs. official RLM 0.34 pair-F1 on
  OOLONG-pairs) disappears from the docs. It was always labelled
  "not a release gate", so this is acceptable.
- Autoresearch needs a Modal account and GPU spend, and Claude Code trial
  agents need a Claude login. Keep `--implementer-harness repl` working so
  there's a path with only an OpenAI key and Modal.
- Scores vary run to run. The docs should show one run's real numbers and say
  it's a single search, not claim one setup beats another.

## Out of scope

- Any benchmark or performance claim against DELM, PrimeAgent or RLM. Revisit
  after release, if a paper or blog post needs one. At that point run the full
  LongBench-v2 multi-doc set (125 questions) plus OOLONG, all three arms,
  about $140.
- Circle packing as a second, CPU-only search example. Revisit after release.
