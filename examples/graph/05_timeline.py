"""Every graph is its own timeline: replay the log to see each earlier state."""

from __future__ import annotations

from rlmflow import AgentStart, DoneOutput, Graph, LLMOutput, Node, ops, reads


def print_history(g: Graph, agent: AgentStart) -> None:
    print("  " + " -> ".join(node.type for node in reads.history(g, agent)))


def main() -> None:
    g = Graph()
    root = ops.spawn(g, "write hello.py")
    ops.append(g, root, LLMOutput(content='write_file("hello.py", "print(\\"hello\\")\\n")'))
    ops.append(g, root, DoneOutput(result="hello.py created"))

    # The log is append-only, so the prefix ending just before each node (with
    # the edges that landed alongside it) is a valid earlier graph.
    items = g.items()
    starts = [item.seq for item in items if isinstance(item, Node)][1:] + [items[-1].seq + 1]
    for step, end in enumerate(starts):
        cutoff = end - 1
        snapshot = g.as_of(cutoff)
        print(f"\nstep {step}")
        print_history(snapshot, snapshot.node(root.id))


if __name__ == "__main__":
    main()
