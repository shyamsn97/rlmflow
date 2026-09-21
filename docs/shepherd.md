# Recursive Shepherd: RLMs as meta-agents

<p align="center">
  <img
    src="static/shepherd/rewind.gif"
    alt="The selected live Shepherd trajectory: deliberate Sokoban jam, graph rewind, and recovery"
    width="620"
  />
</p>

[Shepherd: Enabling Programmable Meta-Agents via Reversible Agentic Execution Traces](https://arxiv.org/abs/2605.10913) (Yu et al., 2026) argues that an agent's execution should be a reversible object rather than a one-way transcript. Record the run as a Git-like trace, and you can program a second model against it — a *meta-agent* that inspects any earlier point, reverts to it, then plans and steers parallel alternative routes.

`rlmflow` already keeps a run as a durable graph of typed nodes, so Shepherd's four operations — observe, rewind, fork, re-instruct — are graph edits, and the meta-agent's forks are simply its own children.

Only the revert itself works differently. The paper checkpoints the worker's process and filesystem, so any program comes back byte-identical. `rlmflow` already tracks every action as a node, so the cheap route is to fork the graph and replay those actions — no snapshot to take, and exactly as faithful whenever the environment is deterministic.

This example is a live model recovery run. The initial eight-push jam is scripted so every attempt starts from the same failed trace. GPT-5 plans the recovery branches, GPT-5-mini plays them, the host selects the highest score, and the local deterministic Sokoban environment makes retained actions exactly replayable.

#### Irreversible Sokoban

For this example we augment our RLMs with tools to play Sokoban, where boxes can be pushed but never pulled. This means a single wrong shove can make the puzzle unsolvable.

For simplicity, we introduce two main tools in the worker's REPL:

```python
def goto(row, col=None):  # walk anywhere reachable; never disturbs a box
    print(game.goto(row, col))

def push(direction):      # shove what you stand against, once per turn
    print(game.push(direction))
```

So a turn can walk to a square, push a direction, or both:

```repl
goto(4, 3)
push("up")
```

```text
walked (3, 2)->(4, 3) in 2 cells
pushed B1 up: (3, 3)->(2, 3)
```

#### Live board context without coupling nodes to Flow

The worker needs the current board in every model request, but that state lives in its REPL rather than in the durable node graph. The example therefore uses a current-frontier `render_fn`. `Flow` passes its runtime to the renderer:

```python
def render_worker(runtime: Runtime, node: Node) -> list[dict[str, str]]:
    messages = default_render(runtime, node)
    content = board_prompt(runtime, node.parent_agent, simple=simple_moves)
    if content:
        messages.append({"role": "user", "content": content})
    return messages


flow = Flow(
    client,
    prompt_profiles={
        "worker": PromptProfile(render_fn=render_worker),
    },
)
```

`Node.render()` remains canonical and runtime-independent, while this renderer can read the live REPL `ENV` for the node currently being sent. The runtime argument makes that transient dependency explicit without coupling the node graph to `Flow`. `board_prompt` also stamps the upcoming turn in `ENV` so the game's one-push guard applies to the next live action; replay does not need that live-turn guard.

This run starts jammed on purpose: the worker shoves `B1` right eight times until it sits flat against the wall, where nothing can stand behind it.

<p align="center">
  <img
    src="static/shepherd/jam.gif"
    alt="The worker pushes one box right eight times until it is pinned against the far wall"
    width="520"
  />
</p>

The meta agent can now come up with distinct new pathways to explore and pick any of those eight spots to revert to, leading to some pretty diverse trajectories. This is done by giving the shepherd agent access to the following tools:

```python
@tool("Inspect the worker after rewinding pushes, without playing.", proxy=True)
async def preview(rewind: int) -> dict:
    """Board, boxes, goals and legal pushes as of `rewind` pushes ago."""

@tool("Record {rewind, order} recovery plans, then stop.", proxy=True)
def branch(specs: list[Plan]) -> str:
    """Each plan is one revert depth plus the box-to-goal order for that worker."""
```

`preview(k)` forks the graph, replays it to `k` pushes ago and returns that state; the failed run is untouched, so the shepherd can look at every earlier board for free. `branch(plans)` takes one `{rewind, order}` per attempt and ends the shepherd's turn: each plan becomes a worker of its own, rewound to that depth and carrying that order.

#### Step 1 — observe: `preview(k)`

The shepherd's first turn swept every depth — its own code, trimmed:

```repl
for k in range(1, max_k + 1):
    state = await preview(k)
    print(f"===== Preview k={k} =====")
    print(state["board"], state["boxes"], state["legal_pushes"][:10])
```

```text
===== Preview k=3 =====
            11
  012345678901
0 ############
1 #  .   .  ##
2 ### # # # ##
3 #     @$   #
4 #   # # # .#
5 #  $     $ #
6 #  ## # #  #
7 # .     .  #
8 ############
boxes (3): {'B1': [3, 7], 'B2': [5, 3], 'B3': [5, 9]}
legal_pushes (10): ['B1 up from (4,7)', 'B1 down from (2,7)', 'B1 left from (3,8)', ...]
```

#### Step 2 — re-instruct: `branch(plans)`

After previewing all eight depths and checking candidate goal entrances against the board, the shepherd submitted eight distinct box-to-goal pairings. This is the plan list from the current run:

```repl
plans = [
    {"rewind": 6, "order": ["B2->G4 entering from (6,2)", "B3->G5 entering from (7,9)"]},
    {"rewind": 8, "order": ["B2->G4 entering from (7,3)", "B3->G3 entering from (5,10)"]},
    {"rewind": 7, "order": ["B1->G1 entering from (2,3)", "B3->G5 entering from (7,7)"]},
    {"rewind": 6, "order": ["B1->G2 entering from (2,7)", "B2->G4 entering from (6,2)"]},
    {"rewind": 5, "order": ["B1->G2 entering from (1,6)", "B3->G3 entering from (5,10)"]},
    {"rewind": 4, "order": ["B1->G1 entering from (1,4)", "B2->G5 entering from (7,7)"]},
    {"rewind": 8, "order": ["B3->G2 entering from (1,8)", "B2->G4 entering from (6,2)"]},
    {"rewind": 5, "order": ["B1->G3 entering from (5,10)", "B2->G4 entering from (7,3)"]},
]
branch(plans)
```

After the host attached and ran those recovery children, their joined outcomes were:

```text
['stuck', 'solved', 'solved', 'solved', 'solved', 'solved', 'stuck', 'solved']
```

A plan is two decisions: where to revert back to, and what strategy to hand the worker instead. It contains no moves — those stay the worker's job.

#### Step 3 — rewind and fork

The host takes each plan's revert depth and turns it into a graph edit:

```python
cut = push_turns(worker)[-depth].prev
fork = cut.fork()
await flow.replay(fork)
```

`fork()` copies the graph and drops everything after the point it reverted to; `replay` re-runs the retained actions into a new REPL.

#### Step 4 — resume: each fork is a sub-agent

The forks launch as children of the shepherd, so the search is itself one recursive agent tree. Each child wakes in a rewound REPL with its instruction appended:

```text
Rewound 7 pushes. This is a fresh recovery attempt, so abandon the plan you
were following.
...
Lock the boxes in this order: B1->G1 entering from (2,3), then B3->G5 entering
from (7,7). The routes are yours to work out.
```

From there the subagents make their moves, all eight at once:

<table align="center">
  <tr>
    <td align="center"><img src="static/shepherd/branch0.gif" alt="branch0 reverted 6 pushes and got stuck" width="220" /><br /><sub>branch0 · reverted 6 · stuck, 14 pushes</sub></td>
    <td align="center"><img src="static/shepherd/branch1.gif" alt="branch1 reverted all 8 pushes and solved the board" width="220" /><br /><sub>branch1 · reverted 8 · solved, 14 pushes</sub></td>
    <td align="center"><img src="static/shepherd/branch2.gif" alt="branch2 reverted 7 pushes and solved the board in 11 pushes" width="220" /><br /><sub>branch2 · reverted 7 · <b>selected, solved in 11 pushes</b></sub></td>
    <td align="center"><img src="static/shepherd/branch3.gif" alt="branch3 reverted 6 pushes and solved the board" width="220" /><br /><sub>branch3 · reverted 6 · solved, 12 pushes</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="static/shepherd/branch4.gif" alt="branch4 reverted 5 pushes and solved the board" width="220" /><br /><sub>branch4 · reverted 5 · solved, 12 pushes</sub></td>
    <td align="center"><img src="static/shepherd/branch5.gif" alt="branch5 reverted 4 pushes and solved the board" width="220" /><br /><sub>branch5 · reverted 4 · solved, 34 pushes</sub></td>
    <td align="center"><img src="static/shepherd/branch6.gif" alt="branch6 reverted all 8 pushes and got stuck" width="220" /><br /><sub>branch6 · reverted 8 · stuck, 8 pushes</sub></td>
    <td align="center"><img src="static/shepherd/branch7.gif" alt="branch7 reverted 5 pushes and solved the board" width="220" /><br /><sub>branch7 · reverted 5 · solved, 23 pushes</sub></td>
  </tr>
</table>

#### Overall Results

Six of the eight branches solved the board. The shepherd selected `branch2`, which rewound seven pushes and solved in 11 total pushes for a score of 989. The full recovery tree contains 585 graph nodes; the selected path is the fresh `rewind.gif` at the top of this page.

<p align="center">
  <img
    src="shepherd_nodes.svg"
    alt="Every node of one shepherd run: a short shepherd trunk fanning into eight branch chains, each faded over the turns it inherited from the jam"
    width="920"
  />
</p>

#### Run it

```bash
export OPENAI_API_KEY=...
uv run --locked --python 3.12 --extra openai --extra image \
  python examples/shepherd/shepherd.py \
  --model gpt-5 \
  --fast-model gpt-5-mini \
  --branches 8 \
  --time-budget 1800 \
  --out-dir examples/_runs/shepherd

make showcase-shepherd-assets
```

The run writes the failed worker graph under `examples/_runs/shepherd/worker/`, the complete persistence-v5 recovery tree (including the selected branch) under `examples/_runs/shepherd/shepherd/`, and JSON/GIF board traces under `examples/_runs/shepherd/traces/`. The make target regenerates `docs/shepherd_graph.svg`, `docs/shepherd_nodes.svg`, `docs/static/shepherd/rewind.gif`, `jam.gif`, and every `branchN.gif` from those traces. See the [example README](https://github.com/shyamsn97/rlmflow/tree/main/examples/shepherd) for the browser and direct-renderer variants.
