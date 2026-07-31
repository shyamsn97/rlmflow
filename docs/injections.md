# Node injection

Controllers edit the caller-owned Node tree between streaming calls, then
continue the same root.

## Append an observation

```python
from rlmflow import ExecOutput

flow.append(
    root.tail("root.worker"),
    ExecOutput(
        content="Controller observation: finalize with current evidence.",
        output="Controller observation: finalize with current evidence.",
    ),
    injected=True,
)

flow.run(root)
```

A string is wrapped as `UserQuery`:

```python
flow.append(
    root.tail("root.worker"),
    "Controller stop request: finalize now.",
    injected=True,
)
```

Injected Nodes are ordinary durable Nodes with:

```python
node.metadata["injected"] = True
node.metadata["mutation"]["type"] == "append"
```

## Insert, replace, and remove

Structural edits use operation-specific surgery functions:

```python
from rlmflow import ExecOutput, surgery

inserted = surgery.insert(root, anchor_id, "note", mode="after")
replacement = surgery.insert(
    root,
    anchor_id,
    ExecOutput(content="replacement", output="replacement"),
    mode="replace",
    truncate="descendants",
)
removed = surgery.remove(root, node_id)
```

`mode` is `after`, `before`, or `replace`. `truncate="descendants"` drops the
displaced continuation; the default preserves and rehomes it.

Surgery validates before mutating and returns the affected Node. It stamps
mutation provenance in metadata.

## Reactive control

Stop at a boundary, edit, then resume:

```python
async for _ in flow.run_streaming(root, until="idle"):
    pass

flow.append(root.tail(), "Human feedback: verify the auth path.", injected=True)

async for _ in flow.run_streaming(root):
    pass
```

The first stream settles its active round before returning, so the edit cannot
race an in-flight transition. The next prompt projection reads the changed tree.

Do not mutate a tree from inside an unrelated concurrent task. Live framework
tools receive the stream-local publisher and should use the Flow/surgery
operation that owns their mutation.

See
[`examples/control/controller_injection.py`](../examples/control/controller_injection.py)
for a runnable example.
