"""Forking a minimal Graph."""

from __future__ import annotations

from rflow.minimal import DoneOutput, Graph, UserQuery, render_tree


def seed_graph() -> Graph:
    graph = Graph(query="choose a release plan")
    graph.commit(UserQuery(content=graph.query))
    return graph


def main() -> None:
    base = seed_graph()
    conservative = base.fork()
    bold = base.fork()

    conservative.commit(DoneOutput(result="ship a small patch release"))
    bold.commit(DoneOutput(result="ship a major release candidate"))

    print("base:\n" + render_tree(base))
    print("\nconservative:\n" + render_tree(conservative))
    print("\nbold:\n" + render_tree(bold))


if __name__ == "__main__":
    main()
