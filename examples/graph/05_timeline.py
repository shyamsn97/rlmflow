"""Every graph is its own timeline: replay the log to see each earlier state."""

from __future__ import annotations

from rlmflow import DONE, LLM_OUTPUT, Graph, Node


def print_history(g: Graph, agent: Node) -> None:
    print("  " + " -> ".join(node.type for node in g.history(agent)))


def main() -> None:
    g = Graph()
    root = g.spawn("write hello.py")
    g.append(root, Node(LLM_OUTPUT, text='```repl\nwrite_file("hello.py", "print(1)")\n```'))
    g.append(root, Node(DONE, result="hello.py created"))

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
