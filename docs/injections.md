# Node injection

Controllers edit the caller-owned Node tree — while a run streams, or between streaming calls — then continue the same root.

## Append to a frontier

An agent's frontier is where its next step starts from, so appending there is how a controller gets a word in:

```python
from rlmflow import UserQuery

root.frontier.append(UserQuery(content="Controller stop request: finalize now."))

flow.run(root)
```

Any node type works. A `UserQuery` and an `ExecOutput` both reach the model as user turns, so the choice is about what the saved run should say happened — feedback from outside, or an observation the agent's own code could have made:

```python
from rlmflow import ExecOutput

worker = root.sub_agents[0]
worker.frontier.append(
    ExecOutput(content="Controller observation: finalize with current evidence.")
)
```

Injected nodes are ordinary durable nodes: they are saved, replayed, and projected into prompts like any other. What marks them is what they are and where they sit — a `UserQuery` the agent's own code did not produce — not a flag. They carry no timing, because no step ran to produce them.

## Only the frontier

`append` raises if the node it is called on is not its agent's frontier:

```python
stale = root.transcript()[1]
stale.append(UserQuery(content="too late"))   # ValueError: ... is not the frontier
```

This is the whole edit model. There is no insert, replace, or remove: history is append-only, and the way to change what an agent already did is to fork the run at that point and continue the copy.

```python
from rlmflow import ExecAction

action = next(n for n in root.transcript() if isinstance(n, ExecAction))
branch = action.fork()
branch.frontier.append(UserQuery(content="Try a different approach."))

async for _ in flow.run_streaming(branch):
    pass
```

`fork()` copies the whole tree, cuts everything after that node, and re-ids the copy, so the original is untouched and both can be run and saved side by side.

## Inject tools and values

The other injection point is the REPL namespace, which reaches agents that are already running:

```python
flow.inject("BUDGET_REMAINING", 4)
flow.add_tool(escalate)
flow.remove_tool("escalate")
```

`inject` takes any object, not just callables; `add_tool` is the same thing with the name read off the function. Both write into every open REPL as well as the flow's namespace, so a tool added mid-run is visible on the next step.

`finish`, `launch_subagent`, and `INPUTS` are reserved and cannot be injected over.

## Reactive control

There are two moments when an agent's frontier is yours to write to, and both come down to the same rule: an agent that is mid-step owns its frontier, and moving it out from under a running step makes that step's own append raise.

**Inside the loop, as the agent's node lands.** A pass submits its steps at the top and yields what landed at the bottom, so an agent whose node you are holding has not been given its next step yet:

```python
from rlmflow import ExecOutput, UserQuery

async for node in flow.run_streaming(root):
    agent = node.parent_agent
    if agent is not root and isinstance(node, ExecOutput):
        agent.frontier.append(UserQuery(content="Answer with the best evidence."))
```

This is the only way to reach a worker while its parent is still inside the block that launched it.

**Between streaming calls,** once the boundary has left nothing in flight:

```python
async for _ in flow.run_streaming(root, until="idle"):
    pass

root.frontier.append(UserQuery(content="Human feedback: verify the auth path."))

async for _ in flow.run_streaming(root):
    pass
```

The stream settles its pass before returning, so the edit cannot race a step that is landing, and the next prompt projection reads the changed tree.

The caveat is delegation. A boundary that fires on a child's node stops the stream while the parent's block is still running, and closing the stream cancels that step — so resuming re-runs the block and relaunches children the parent already has. The relaunch is refused by name and the parent reads that as an observation, so the run survives it, at the cost of a wasted turn. To edit between calls in a run that delegates, stop on the parent's own output:

```python
from rlmflow import DoneOutput, ExecOutput

def parent_idle(node, root):
    return isinstance(node, (ExecOutput, DoneOutput)) and node.parent_agent is root
```

Or edit inside the loop, which never stops the run at all.

See [`examples/control/injection/`](../examples/control/injection/) for runnable variants.
