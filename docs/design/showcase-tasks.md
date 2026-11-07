# Benchmark table: Claude Code, Claude Code + rlmflow, rlmflow

Status: runners in `benchmarks/table/`; OOLONG running, ALE-Bench blocked on Docker disk
Date: 2026-11-04, revised 2026-11-05

## Summary

One results table for the README: three tasks, three systems, the same model in
every cell.

| Task | Claude Code | Claude Code + rlmflow | rlmflow harness |
| --- | --- | --- | --- |
| ALE-Bench-Lite (algorithm engineering) | | | |
| SWE-bench Verified, hard tasks (coding) | | | |
| Long-document QA (OOLONG) | | | |

The systems:

1. **Claude Code.** Vanilla Claude Code, with its own tools and native subagents.
2. **Claude Code + rlmflow.** The same Claude Code, with the rlmflow plugin. The
   session is the root agent of an rlmflow run and decides for itself how to use
   the library: launch agents, message them, fork them, read the graph.
3. **rlmflow harness.** rlmflow's own REPL harness as the root, on the same model
   through the Anthropic API, with the same agent tools.

This measures the library as an AI uses it, which is the main way it's meant to
be used. No arm is told which architecture to use: the guidance describes what
the library can do, and the agent decides. So the guidance is part of what's
being measured, and getting it right comes first (see "Prerequisite: guidance").

Showcases (Shepherd, Needle, Boids, the architecture gallery) stay separate from
this table. They show what the graph can express. The table shows whether
having it helps.

## Why these three tasks

Each task rewards a different thing the library offers, and each has a
deterministic scorer.

| Task | What should help | Scorer |
| --- | --- | --- |
| ALE-Bench-Lite | Parallel attempts, iterating on the best one, branching from it | Hidden test cases, ranked against contest participants |
| SWE-bench Verified, hard tasks | Splitting multi-file work across agents, and reviewing a fix before submitting | The task's own tests |
| Long-document QA (OOLONG) | Holding a long input as a variable and splitting it across fresh-context agents | OOLONG's scorer |

### ALE-Bench-Lite

[ALE-Bench](https://github.com/SakanaAI/ALE-Bench)'s 10-problem Lite set of
AtCoder Heuristic Contest problems: `ahc008`, `ahc011`, `ahc015`, `ahc016`,
`ahc024`, `ahc025`, `ahc026`, `ahc027`, `ahc039`, `ahc046`. These are open-ended
optimization problems. Every system gets the statement and an `./evaluate`
command that runs `main.cpp` (C++23, gcc 12.2) on the public test cases, as
contestants had. The final `main.cpp` is scored on the hidden cases and
ranked against the contest's participants. It replaces circle packing: ten
problems instead of one, and a rank against humans instead of a raw score.

The judge images are x86-only, so on Apple Silicon they run emulated. That's
fair between our columns, but solutions get fewer iterations within each time
limit than on AtCoder's judge, so the absolute ranks read low.

### SWE-bench Verified, hard tasks

15 to 20 tasks drawn from the roughly 40 that SWE-bench Verified's annotators
rated "1 to 4 hours" or more, chosen at random with fixed IDs before the first
run. Each is a real GitHub issue in a real repo, graded by the repo's own tests
in the task's Docker image.

Hard tasks only, because the easy ones tie. The DELM paper's Table 1 shows it:
on Claude Opus 4.6, mini-SWE-agent, Claude Code and DELM all land between 74.7%
and 78.0%, while on Gemini 3 Flash DELM beats Claude Code by 16 points (65.7% vs
49.3%). Gaps appear where the model can't already solve the task alone. The same
table puts Claude Code at $0.69 per task on Opus 4.6, so our Claude Code column
has a reference point.

### Long-document QA (OOLONG)

The first 15 of the fixed, seeded 25 OOLONG `trec_coarse` rows at 128K tokens
that the RLM authors evaluate on, scored with their scorer. The document is the
text **without labels** (`context_window_text`): each of about 3,000 questions
has to be classified to answer. With labels (`context_window_text_with_labels`,
which `benchmarks/eval`'s loader uses), every question is a grep. For Claude Code the input is a file in its working
directory; for the rlmflow systems it's also a REPL variable.
`benchmarks/eval/environments/oolong.py` already loads OOLONG.

### Deferred: Terminal-Bench

The 10 long-horizon Terminal-Bench 4.0 tasks from the DELM paper (Appendix B),
through [Harbor](https://docs.harborframework.com/agents/custom-agents), come
after the first pass. They cost about $12 per task per column on Sonnet, and
systems 2 and 3 need Harbor agents that install rlmflow in every task
container. DELM reports, with Claude Opus 5.5: Claude Code 81.67% at
$20.31/task, Claude Code with native subagents 80.00%, DELM with 2 agents
93.33%.

## Models and fairness

- **One model in every cell**, so the only difference is the system: Sonnet 5.5.
- **Workers use the same model** unless the agent itself chooses a cheaper
  config. If a system can choose cheaper workers, that's part of the system, but
  the cost column must include every worker.
- **Same budget per task** in every cell: a wall-clock limit and a dollar cap.
  Report cost, time and tokens next to the score.
- **Billing:** the Claude Code columns use the Claude Code login
  (`ANTHROPIC_API_KEY` unset) and report the cost Claude Code computes from
  tokens at list prices. The rlmflow harness column bills the API, priced with
  the same list prices (Sonnet 5.5: $2 in, $10 out per million tokens).
- **Workers are the root's kind.** Claude Code + rlmflow launches Claude Code
  workers (`benchmarks/table/flows.py`, loaded through `RLMFLOW_FLOW`); the
  plugin's default would launch gpt-5 REPL workers. The rlmflow harness
  launches REPL workers.
- **Fixed subsets and seeds**, committed before the first run. Three seeds where
  affordable, or per-task outcomes so the spread is visible. Our Aug 27 run
  showed one seed on ten tasks can flip a comparison
  (`docs/internal/delegation-ten-task-three-seed-findings-2026-08-27.md`).
- **Score the gold answers through our scorer first.** The same Aug 27 run had a
  scorer bug that inverted the comparison.
- **The guidance is frozen before the runs** and is the same for every task. No
  task-specific prompts, no hints about answers, and no tuning on the
  evaluation subsets.

## Prerequisite: guidance for AI drivers

System 2 is only as good as what Claude Code is told about rlmflow, and today it
isn't told much.

### What exists

- The plugin (`rlmflow/plugin/`): a `/rlmflow:run` command (11 lines), hooks for
  prompt submit, stop and session end, and an MCP server for the REPL.
- The session prompt (`rlmflow/harnesses/sessions.py`, `TOOLS_PROMPT` and
  neighbors). It covers `rlmflow repl`, `launch_agent`, handles and `await`,
  `llm_query`, `AGENTS`, `submit`, keeping state in REPL variables, and that
  cells are recorded.
- `docs/skills.md`: rlmflow's mechanism for putting `SKILL.md` files into agents'
  prompts.

### What's missing

1. **Messaging.** `send_agent` isn't mentioned in the session prompt, so a
   Claude Code root doesn't know it can message agents or reopen a finished one.
2. **Forking from the REPL.** There's no fork tool for agents. Forks happen only
   from host code (`g.fork`) or the `rlmflow fork` CLI on a saved run. Shepherd
   wrote its own. For ALE-Bench and recovering failed coding attempts, an agent needs
   something like `fork_agent(agent, at=step, message=...)` that branches an
   agent with its REPL state (and files, with `GitWorktrees`) and returns a
   handle.
3. **Reading the graph.** `AGENTS` lists the agents this one started. An agent
   can't read another agent's history to choose a step to fork from. Shepherd
   passed the transcript in by hand.
4. **When to use what.** Nothing tells a driver when a fan-out pays off, when
   peers do, when to fork rather than start over, or how many agents is too many.
   CAID found Commit0-Lite peaked at 4 engineers and dropped after; a team study
   found worker-to-worker messaging rarely happened and didn't raise reward. A
   short guide should say this plainly.
5. **The same guidance for both rlmflow systems.** The REPL harness and the
   Claude Code harness have separate prompts. The patterns guide should be one
   text both get, so systems 2 and 3 differ only in harness.

Done on Nov 5: items 1, 4 and 5. The session prompt now covers `send_agent` and
`AGENTS.print_graph`, and lives in `rlmflow/harnesses/prompts.py`. The
`orchestrate` skill (`rlmflow/plugin/skills/orchestrate/SKILL.md`) is in every
launching agent's prompt on both harnesses, and the plugin ships it, with a
section on building the same shapes from host code. Both harnesses take their
guidance (checking answers, keeping notes, the skill) from one `guidance()` in
`prompts.py`; only the mechanics of running code differ, and a test holds that.
Items 2 and
3 (forking and reading histories from the REPL) are still open.

### What to write

- An rlmflow skill shipped in the plugin (Claude Code plugins can carry skills)
  and also loaded into REPL-harness agents through the `SKILL.md` mechanism. It
  covers:
  - the agent tools: launch, await, send, fork, read the graph, submit;
  - the patterns from the README grid, with when each helps and when it doesn't;
  - cost discipline: few agents, cheap configs for mechanical work, stop when
    done.
- A `fork_agent` REPL tool and a read-only history view for agents (items 2 and
  3), designed separately.
- An `AGENTS.md` or `llms.txt` at the repo root for coding agents writing rlmflow
  code against the Python API, since humans and AIs should both be able to use
  it.

### Checking the guidance without overfitting

Validate on tasks that are not in the table: Needle, Shepherd, a few SWE-bench
Verified tasks outside the subset, and an ALE-Bench problem outside the Lite set. Freeze the guidance, then run the table.

## Order of work

1. Write the guidance and the `fork_agent` and history tools, and validate them on
   held-out tasks.
2. Build the three task runners, all on this machine (`benchmarks/table/`):
   ALE-Bench-Lite's judge (done, needs Docker disk), OOLONG (done, gold answers
   score 1), SWE-bench Verified (not started). `claude -p "/rlmflow:run ..."`
   works headless.
3. Pilot 2 tasks per cell on a cheap model to check the plumbing and measure
   cost.
4. Run the table on Sonnet 5.5, one seed, then more seeds where it's affordable.
5. Publish per-task results, cost and time, one rendered graph per rlmflow cell,
   and the failures, in `docs/design/example-results.md`.

## Rough cost

| Task | All three columns, one seed | Notes |
| --- | --- | --- |
| ALE-Bench-Lite, 10 problems | About $90 to $150 | $3 to $5 per problem per column |
| SWE-bench Verified, 20 hard tasks | About $40 to $60 | DELM's Table 1: Claude Code $0.69/task on Opus 4.6; hard tasks cost more |
| OOLONG, 15 tasks at 128K | Pilot: about $0.05 per problem on the Claude Code columns | More when a system reads the whole document |
| Pilot, 2 tasks per cell | About $20 | |

About $150 to $250 per seed in total.

Measure on the pilot before committing; Claude Code's per-task cost varies a lot
with effort settings.

## Considered and set aside

- **DELM and RLM as benchmark arms.** Claude Code can build a shared board or its
  own subagents, so a score for "DELM on rlmflow" measures DELM, not the library.
  They stay as examples: `examples/collaboration/shared_board.py` and Needle.
- **MettleBench rewind vs. continue** ([AgentRewind](https://arxiv.org/html/2608.14380)):
  AgentRewind already published that comparison, and its Docker setup is heavy.
- **BrowseComp-Plus:** needs a retrieval index over 100K documents; the 1K-document
  variant is near the ceiling.
- **Commit0-Lite:** frontier models already solve most of it.
- **LiveCodeBench review loop:** a known 2 to 16 point gain, nothing specific to us.
- **Aider Polyglot:** cheap and well known, but every task is one small file, so
  splitting work can't help, and top models score near the ceiling.
- **LongBench-v2 Multi-Doc:** its short documents fit in one context, and its long
  ones cost as much as OOLONG's longest.
- **Skipped outright:**
  - The full DELM Terminal-Bench and DeepSWE grid costs thousands of dollars.
  - PaperBench is judged by an LLM.
  - MLE-bench and RE-Bench need GPUs.
  - WebArena is heavy to set up.
  - MultiAgentBench prescribes roles and topologies.

## Open questions


## Sources

- DELM: [arXiv 2606.10662](https://arxiv.org/html/2606.10662), Tables 1 and 9, §3.1.
- Recursive Language Models: [arXiv 2512.24601](https://arxiv.org/html/2512.24601v3).
- Terminal-Bench 4.0: [announcement](https://www.tbench.ai/news/terminal-bench-4-0),
  [leaderboard](https://snorkel.ai/leaderboard/terminal-bench-4-0/); Harbor
  [custom agents](https://docs.harborframework.com/agents/custom-agents).
- LongBench-v2: [site](https://longbench2.github.io/).
- CAID: [arXiv 2603.21489](https://arxiv.org/html/2603.21489).
- Teams of coding agents: [arXiv 2610.17471](https://arxiv.org/abs/2610.17471).
- AgentRewind and MettleBench: [arXiv 2608.14380](https://arxiv.org/html/2608.14380).
