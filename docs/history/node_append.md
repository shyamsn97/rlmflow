# `Node.append` is inplace

`append` mutates the tree and returns `None`, like `list.append`. A step constructs the next node, hangs it off the frontier, and returns **that** node.

```python
created = LLMOutput(
    content=reply,
    code=code_block(reply),
    usage=usage,
    prompt_id=agent.record_prompt(messages[0]["content"]),
)
node.append(created)
return created
```

`Flow.step` returns that created node. It is a local variable, not a side effect hidden in `append`'s return.

## Why not return the child

The old contract mutated **and** returned the argument:

```python
return node.append(LLMOutput(...))
```

Two Python readings, both wrong for a step:

1. **`list.append`** — inplace, returns `None`. Then `return node.append(...)` returns `None` and the step is broken.
1. **Fluent builder** — `append` returns `self`. Then the step returns the **old** frontier, which is dumb: `Flow.step` needs the node that was just created.

What the method actually did was a third thing: return the child. The step worked, and every call site still looked like it might be returning the parent. That was the bug. Not the mutation. The expression.

## What `append` does

- `self.children` gains `child`
- `child.parent` is `self`
- for a same-agent child, `agent.frontier` becomes `child` and `child.seq` is `self.seq + 1`
- the run index registers the new ids
- appending an `AgentStart` **does not** move the parent frontier (that is a branch, not a sequel)

`Flow.step` returns that child. If `append` ever returned the parent, every step would report that it created the node it started from.

Control-node injection is the same shape:

```python
created = FinalQuery()
node.append(created)
return created
```

## What we do not change

- Inplace tree mutation. There is no persistent `tree = tree.plus(child)`. The graph is the run.
- Frontier rules. Still only the current frontier may append. Still `AgentBusyError` if another step is in flight. Still `AgentStart` branches without moving the parent.
- `node.next` / `node.prev`. Those read the links `append` wrote.
- Persistence. Wire records already store `parent_id`; they never stored “append returned X.”
- `append_child` / `AppendChild.attach` stay factories: they return the attached subtree.

Fluent chains (`root.append(a).append(b)`) are gone. Construct the child, append it, keep the name you already have.

## Why not return `self`

Returning the parent would match some fluent APIs and would make `return node.append(child)` literally return the old node. The driver would then think the step produced the submitted frontier. Do not do that.

If a one-liner that yields the child is wanted, it gets a different name so it is not `append`. One method that both mutates and returns the child is how the old confusion started. There is no `advance` helper.
