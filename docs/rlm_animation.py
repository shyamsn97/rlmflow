"""rlmflow hero animation.

Graph-first README animation. This intentionally skips theory:
the video should immediately show what rlmflow offers - a recursive agent
run as a clean, typed execution graph.

Run from the repo root.

Quick preview (480p, fast)::

    manim -pql docs/rlm_animation.py RecursiveFlowHero

High quality MP4 (1080p, the canonical render)::

    manim -qh docs/rlm_animation.py RecursiveFlowHero
    cp media/videos/rlm_animation/1080p60/RecursiveFlowHero.mp4 docs/rlm_animation.mp4

GIF for README / PyPI previews::

    ffmpeg -y -i docs/rlm_animation.mp4 \
      -vf "fps=12,scale=960:-1:flags=lanczos" docs/rlm_animation.gif
"""

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Circle,
    Create,
    FadeIn,
    FadeOut,
    Flash,
    LaggedStart,
    Line,
    RegularPolygon,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    Text,
    TransformFromCopy,
    VGroup,
)

BG = "#0B0F14"
WHITE_C = "#E6EDF3"
DIM = "#6E7681"

Q_C = "#58A6FF"  # query
LLM_C = "#BC8CFF"
EXEC_C = "#FF9E64"
OUT_C = "#56D4DD"  # exec output
R_C = "#56D364"  # done
HOT = "#FFD60A"  # focus only
LABEL_C = "#3FB950"

FS_HEADER = 24
FS_BODY = 14
FS_CAP = 12
FS_SMALL = 10
FS_TINY = 8

CODE_FONT = "Menlo"


def typed_node(kind, label="", *, r=0.30, fs=FS_BODY):
    """Small typed graph node."""
    if kind in {"Q", "query"}:
        shape = Circle(radius=r, color=Q_C, fill_opacity=0.18, stroke_width=2)
    elif kind in {"L", "llm"}:
        shape = RegularPolygon(n=4, color=LLM_C, fill_opacity=0.18, stroke_width=2)
        shape.rotate(0.7853981633974483).scale(r)
    elif kind in {"E", "exec"}:
        shape = RoundedRectangle(
            corner_radius=0.015,
            width=r * 1.72,
            height=r * 1.72,
            color=EXEC_C,
            fill_opacity=0.18,
            stroke_width=2,
        )
    elif kind in {"O", "output"}:
        shape = RegularPolygon(n=3, color=OUT_C, fill_opacity=0.20, stroke_width=2)
        shape.rotate(-1.5707963267948966).scale(r)
    elif kind in {"R", "done"}:
        shape = RegularPolygon(n=6, color=R_C, fill_opacity=0.22, stroke_width=2)
        shape.scale(r)
    else:
        shape = Circle(radius=r, color=DIM, stroke_width=1)

    if not label:
        return VGroup(shape)

    txt = Text(label, font=CODE_FONT, font_size=fs, color=WHITE_C)
    max_w = r * 1.7
    if txt.width > max_w:
        txt.scale(max_w / txt.width)
    return VGroup(shape, txt)


def tiny_graph(specs, *, root_pos, scale=1.0):
    """Small static graph snapshot for the 'evolves every step' phase."""
    nodes = [typed_node(kind, "", r=0.15 * scale) for _label, kind, _children in specs]
    x_by_index: dict[int, float] = {}
    leaf_cursor = 0
    leaf_spacing = 0.62 * scale
    level_gap = 0.56 * scale

    def assign_x(idx):
        nonlocal leaf_cursor
        children = specs[idx][2]
        if not children:
            x_by_index[idx] = leaf_cursor * leaf_spacing
            leaf_cursor += 1
            return x_by_index[idx]
        child_xs = [assign_x(child_idx) for child_idx in children]
        x_by_index[idx] = sum(child_xs) / len(child_xs)
        return x_by_index[idx]

    assign_x(0)
    center_x = x_by_index[0]

    def layout(idx, depth):
        x = x_by_index[idx] - center_x
        nodes[idx].move_to(root_pos + RIGHT * x + DOWN * (depth * level_gap))
        for child_idx in specs[idx][2]:
            layout(child_idx, depth + 1)

    layout(0, 0)

    edges = VGroup()
    for idx, (_label, _kind, children) in enumerate(specs):
        for child_idx in children:
            edges.add(
                Line(
                    nodes[idx].get_bottom(),
                    nodes[child_idx].get_top(),
                    color=DIM,
                    stroke_width=1.1,
                ).set_opacity(0.75)
            )
    return VGroup(edges, *nodes)


def _pill(label, color):
    box = RoundedRectangle(
        corner_radius=0.12,
        width=1.42,
        height=0.42,
        color=color,
        stroke_width=1.3,
        fill_opacity=0.08,
    )
    text = Text(label, font=CODE_FONT, font_size=FS_CAP, color=color)
    return VGroup(box, text)


def code_card(lines, *, width=4.35, height=1.82, font_size=7, header_size=FS_TINY):
    box = RoundedRectangle(
        corner_radius=0.12,
        width=width,
        height=height,
        color=DIM,
        stroke_width=1.0,
        fill_opacity=0.10,
    )
    header = Text("python", font=CODE_FONT, font_size=header_size, color=DIM)
    header.next_to(box.get_top(), DOWN, buff=0.14).align_to(
        box.get_left() + RIGHT * 0.20, LEFT
    )
    rendered_lines = VGroup(
        *[
            Text(line, font=CODE_FONT, font_size=font_size, color=color)
            for line, color in lines
        ]
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.09)
    rendered_lines.next_to(header, DOWN, buff=0.14).align_to(
        box.get_left() + RIGHT * 0.20, LEFT
    )
    max_w = width - 0.35
    if rendered_lines.width > max_w:
        rendered_lines.set_width(max_w)
        rendered_lines.next_to(header, DOWN, buff=0.14).align_to(
            box.get_left() + RIGHT * 0.20, LEFT
        )
    return VGroup(box, header, rendered_lines)


def restore_typed_node_fills(graph):
    """Re-apply translucent fills after animations that flatten node styling."""
    for mob in graph.get_family():
        mob.set_opacity(1)
        if isinstance(mob, RoundedRectangle):
            mob.set_fill(opacity=0.18)
            mob.set_stroke(opacity=1)
        elif isinstance(mob, Circle):
            mob.set_fill(opacity=0.18)
            mob.set_stroke(opacity=1)
        elif isinstance(mob, RegularPolygon):
            mob.set_fill(opacity=0.20)
            mob.set_stroke(opacity=1)
        elif isinstance(mob, Line):
            mob.set_stroke(opacity=0.62)


def agent_stack(label, kinds, *, x, top_y, r=0.10, dy=0.30, fs=FS_TINY):
    """Renderer-style vertical stack for one agent transcript."""

    if label:
        label_mob = Text(label, font=CODE_FONT, font_size=fs, color=LABEL_C)
        label_mob.move_to(RIGHT * x + UP * (top_y + 0.24))
    else:
        label_mob = VGroup()
    nodes = [
        typed_node(kind, "", r=r).move_to(RIGHT * x + UP * (top_y - dy * index))
        for index, kind in enumerate(kinds)
    ]
    edges = VGroup(
        *[
            Line(
                nodes[index].get_bottom(),
                nodes[index + 1].get_top(),
                color=DIM,
                stroke_width=0.65,
            ).set_opacity(0.62)
            for index in range(len(nodes) - 1)
        ]
    )
    return VGroup(edges, *nodes, label_mob), nodes, edges, label_mob


def stack_ticks(nodes, edges, label):
    """Split an agent stack into growth ticks: (label+root), then edge+node."""
    ticks = [VGroup(label, nodes[0])]
    for i in range(1, len(nodes)):
        ticks.append(VGroup(edges[i - 1], nodes[i]))
    return ticks


def rendered_style_graph():
    """Deep symmetric recursive swarm: root -> agents -> subagents -> leaves."""

    branching = [3, 2, 2, 2]  # children spawned at depth 0, 1, 2, 3
    depth_count = len(branching) + 1  # 5 levels (root .. leaves)
    level_style = [
        dict(top_y=2.25, r=0.130, dy=0.24, fs=7),
        dict(top_y=1.32, r=0.108, dy=0.22, fs=6),
        dict(top_y=0.40, r=0.090, dy=0.20, fs=5),
        dict(top_y=-0.50, r=0.076, dy=0.185, fs=4),
        dict(top_y=-1.42, r=0.064, dy=0.165, fs=4),
    ]
    leaf_spacing = 0.50

    # 1) enumerate the tree with hierarchical labels.
    tree: list[dict] = []

    def add(label, depth, parent):
        idx = len(tree)
        tree.append({"label": label, "depth": depth, "parent": parent, "children": []})
        if parent is not None:
            tree[parent]["children"].append(idx)
        if depth < len(branching):
            for i in range(branching[depth]):
                # Match rlmflow agent paths: root.a, root.a.b, ...
                child_label = f"{label}.{chr(ord('a') + i)}"
                add(child_label, depth + 1, idx)
        return idx

    add("root", 0, None)

    # 2) place leaves evenly, center every parent over its children.
    leaf_cursor = [0]

    def assign_x(idx):
        node = tree[idx]
        if not node["children"]:
            node["x"] = leaf_cursor[0] * leaf_spacing
            leaf_cursor[0] += 1
        else:
            for child in node["children"]:
                assign_x(child)
            node["x"] = sum(tree[c]["x"] for c in node["children"]) / len(
                node["children"]
            )

    assign_x(0)
    center = tree[0]["x"]
    for node in tree:
        node["x"] -= center

    # 3) build each agent stack and bucket mobjects into reveal rings.
    rings = [[VGroup(), VGroup(), VGroup()] for _ in range(depth_count)]
    for idx, node in enumerate(tree):
        d = node["depth"]
        st = level_style[d]
        is_leaf = not node["children"]
        # A leaf answers; a parent's last node is the block that launched its
        # children, which is what they hang off.
        kinds = ["query", "llm", "done" if is_leaf else "exec"]
        show_label = d <= 1
        _stack, nodes, edges, label = agent_stack(
            node["label"] if show_label else "",
            kinds,
            x=node["x"],
            top_y=st["top_y"],
            r=st["r"],
            dy=st["dy"],
            fs=st["fs"],
        )
        node["nodes"] = nodes
        if node["parent"] is not None:
            parent_nodes = tree[node["parent"]]["nodes"]
            fan = Line(
                parent_nodes[-1].get_center(),
                nodes[0].get_center(),
                color=DIM,
                stroke_width=max(0.30, 0.62 - 0.08 * d),
            ).set_opacity(0.42 - 0.03 * d)
            rings[d][0].add(fan)
        rings[d][0].add(label, nodes[0])
        for k in range(1, len(nodes) - 1):
            rings[d][1].add(edges[k - 1], nodes[k])
        rings[d][2].add(edges[len(nodes) - 2], nodes[-1])

    stages = [ring for level in rings for ring in level]
    graph = VGroup(*stages)
    graph.scale(1.18)
    graph.move_to(DOWN * 0.18)
    return graph, stages


def node_type_legend(*, font_size=8, icon_r=0.062):
    """Small legend explaining typed node shapes."""
    items = [
        ("query", "query"),
        ("llm", "llm"),
        ("exec", "exec"),
        ("output", "output"),
        ("done", "done"),
    ]
    legend = VGroup()
    for text, kind in items:
        icon = typed_node(kind, "", r=icon_r)
        label = Text(text, font=CODE_FONT, font_size=font_size, color=DIM)
        label.next_to(icon, RIGHT, buff=0.08)
        legend.add(VGroup(icon, label))
    legend.arrange(RIGHT, buff=0.28)
    return legend


def fork_delegation_graph(x_center):
    """Multitree at a delegating block: root fans out to child agents."""

    r_root = 0.20
    r_child = 0.16
    dy_root = 0.46
    dy_child = 0.38

    root, root_nodes, root_edges, root_label = agent_stack(
        "root",
        ["query", "llm", "exec"],
        x=x_center,
        top_y=1.55,
        r=r_root,
        dy=dy_root,
        fs=FS_CAP,
    )
    fan_y = root_nodes[2].get_center()[1] - dy_root * 1.55
    agent0, agent0_nodes, agent0_edges, agent0_label = agent_stack(
        "root.a",
        ["query", "llm", "done"],
        x=x_center - 0.95,
        top_y=fan_y,
        r=r_child,
        dy=dy_child,
        fs=FS_CAP,
    )
    agent1, agent1_nodes, agent1_edges, agent1_label = agent_stack(
        "root.b",
        ["query", "llm", "done"],
        x=x_center,
        top_y=fan_y,
        r=r_child,
        dy=dy_child,
        fs=FS_CAP,
    )
    agent2, agent2_nodes, agent2_edges, agent2_label = agent_stack(
        "root.c",
        ["query", "llm", "done"],
        x=x_center + 0.95,
        top_y=fan_y,
        r=r_child,
        dy=dy_child,
        fs=FS_CAP,
    )
    fanout = VGroup(
        *[
            Line(
                root_nodes[2].get_center(),
                child.get_center(),
                color=DIM,
                stroke_width=1.1,
            ).set_opacity(0.55)
            for child in (agent0_nodes[0], agent1_nodes[0], agent2_nodes[0])
        ]
    )
    root_query = VGroup(root_label, root_nodes[0])
    root_llm = VGroup(root_edges[0], root_nodes[1])
    root_exec = VGroup(root_edges[1], root_nodes[2])
    child_query = VGroup(
        fanout,
        agent0_label,
        agent0_nodes[0],
        agent1_label,
        agent1_nodes[0],
        agent2_label,
        agent2_nodes[0],
    )
    child_llm = VGroup(
        agent0_edges[0],
        agent0_nodes[1],
        agent1_edges[0],
        agent1_nodes[1],
        agent2_edges[0],
        agent2_nodes[1],
    )
    child_tail = VGroup(
        agent0_edges[1],
        agent0_nodes[2],
        agent1_edges[1],
        agent1_nodes[2],
        agent2_edges[1],
        agent2_nodes[2],
    )
    graph = VGroup(
        root_query,
        root_llm,
        root_exec,
        child_query,
        child_llm,
        child_tail,
    )
    stages = [
        root_query,
        root_llm,
        root_exec,
        child_query,
        child_llm,
        child_tail,
    ]
    refs = {
        "exec": root_nodes[2],
        "children": VGroup(agent0, agent1, agent2, fanout),
        "root_nodes": root_nodes,
        "root_edges": root_edges,
        "root_label": root_label,
    }
    return graph, stages, refs


def fork_graph_refs(graph):
    """Resolve fork/append refs from a fork_delegation_graph instance."""
    return {
        "exec": graph[2][1],
        "children": VGroup(graph[3], graph[4], graph[5]),
        "root_label": graph[0][0],
    }


def simplified_style_graph(*, with_labels=True):
    """Smaller renderer-style graph for the opening and branch edit payoff."""

    def lbl(name):
        return name if with_labels else ""

    root, root_nodes, root_edges, root_label = agent_stack(
        lbl("root"),
        ["query", "llm", "exec"],
        x=0.0,
        top_y=2.05,
        r=0.13,
        dy=0.35,
    )

    agent_defs = [("root.a", -2.65), ("root.b", 0.0), ("root.c", 2.65)]
    agents = []
    fanout_edges = VGroup()
    child_query = VGroup()
    child_llm = VGroup()
    child_exec = VGroup()
    for name, ax in agent_defs:
        _stack, a_nodes, a_edges, a_label = agent_stack(
            lbl(name),
            ["query", "llm", "exec"],
            x=ax,
            top_y=0.52,
            r=0.11,
            dy=0.31,
        )
        agents.append(a_nodes)
        fanout_edges.add(
            Line(
                root_nodes[2].get_center(),
                a_nodes[0].get_center(),
                color=DIM,
                stroke_width=0.62,
            ).set_opacity(0.45)
        )
        child_query.add(a_label, a_nodes[0])
        child_llm.add(a_edges[0], a_nodes[1])
        child_exec.add(a_edges[1], a_nodes[2])
    child_query.add(fanout_edges)

    leaf_offsets = [-0.66, 0.66]
    leaf_queries = VGroup()
    leaf_llm = VGroup()
    leaf_exec = VGroup()
    leaf_finish = VGroup()
    for (name, ax), a_nodes in zip(agent_defs, agents):
        for i, off in enumerate(leaf_offsets):
            leaf_name = f"{name}.{chr(ord('a') + i)}"
            _stack, nodes, edges, label = agent_stack(
                lbl(leaf_name),
                ["query", "llm", "exec", "done"],
                x=ax + off,
                top_y=-1.06,
                r=0.082,
                dy=0.24,
                fs=6,
            )
            fan_line = Line(
                a_nodes[2].get_center(),
                nodes[0].get_center(),
                color=DIM,
                stroke_width=0.42,
            ).set_opacity(0.42)
            leaf_queries.add(fan_line, label, nodes[0])
            leaf_llm.add(edges[0], nodes[1])
            leaf_exec.add(edges[1], nodes[2])
            leaf_finish.add(edges[2], nodes[3])

    root_query = VGroup(root_label, root_nodes[0])
    root_llm = VGroup(root_edges[0], root_nodes[1])
    root_exec = VGroup(root_edges[1], root_nodes[2])
    graph = VGroup(
        root_query,
        root_llm,
        root_exec,
        child_query,
        child_llm,
        child_exec,
        leaf_queries,
        leaf_llm,
        leaf_exec,
        leaf_finish,
    )
    graph.scale(1.05)
    graph.move_to(DOWN * 0.02)
    stages = [
        root_query,
        root_llm,
        root_exec,
        child_query,
        child_llm,
        child_exec,
        leaf_queries,
        leaf_llm,
        leaf_exec,
        leaf_finish,
    ]
    return graph, stages, {}


def center_head(text, *, buff=0.62):
    """Section title — centered at the top."""
    return Text(
        text, font=CODE_FONT, font_size=FS_HEADER, color=WHITE_C
    ).to_edge(UP, buff=buff)


def center_sub(text, anchor, *, buff=0.20, color=DIM, font_size=FS_CAP):
    """Subtitle — centered under a section title."""
    return Text(text, font=CODE_FONT, font_size=font_size, color=color).next_to(
        anchor, DOWN, buff=buff
    )


def fade_clear(scene, *mobjs, run_time=0.55):
    grp = VGroup(*[m for m in mobjs if m is not None])
    if len(grp) > 0:
        scene.play(FadeOut(grp), run_time=run_time)


class RecursiveFlowHero(Scene):
    """Immediate graph-first animation."""

    def construct(self):
        self.camera.background_color = BG
        self._graph_first_animation()

    def _graph_first_animation(self):
        brand = Text(
            "rlmflow", font=CODE_FONT, font_size=26, color=WHITE_C
        ).to_corner(UP + LEFT, buff=0.30)
        self.play(FadeIn(brand), run_time=0.30)

        def reveal_stage(stage):
            return LaggedStart(
                *[FadeIn(item, shift=DOWN * 0.025) for item in stage],
                lag_ratio=0.08,
            )

        # Phase 1 — recursive agents as dynamic graphs.
        head = center_head("Recursive agents as dynamic graphs")
        self.play(FadeIn(head, shift=DOWN * 0.06), run_time=0.60)

        p1_graph, p1_stages, _ = simplified_style_graph(with_labels=True)
        p1_graph.scale(1.10)
        p1_graph.move_to(DOWN * 0.05)
        p1_legend = node_type_legend()
        p1_legend.to_edge(DOWN, buff=0.38)
        run_stack = VGroup(p1_graph, p1_legend)

        self.play(
            LaggedStart(
                *[reveal_stage(stage) for stage in p1_stages[:3]],
                lag_ratio=0.35,
            ),
            run_time=2.00,
        )
        self.play(
            LaggedStart(
                *[reveal_stage(stage) for stage in p1_stages[3:6]],
                lag_ratio=0.30,
            ),
            run_time=1.60,
        )
        self.play(
            LaggedStart(
                *[reveal_stage(stage) for stage in p1_stages[6:]],
                lag_ratio=0.25,
            ),
            run_time=1.80,
        )
        self.play(FadeIn(p1_legend, shift=UP * 0.04), run_time=0.55)
        self.wait(0.85)

        # Phase 2 — step through the live run: it is a graph, not a recording.
        step_title = center_head("Step through the live run")
        self.play(
            FadeOut(run_stack, shift=DOWN * 0.05),
            ReplacementTransform(head, step_title),
            run_time=0.70,
        )
        head = step_title

        snapshot_steps = [
            ([("root", "query", [])], "root receives the query"),
            (
                [("query", "query", [1]), ("llm", "llm", [])],
                "the llm proposes the next action",
            ),
            (
                [
                    ("query", "query", [1]),
                    ("llm", "llm", [2]),
                    ("exec", "exec", []),
                ],
                "the engine runs the block",
            ),
            (
                [
                    ("query", "query", [1]),
                    ("llm", "llm", [2]),
                    ("exec", "exec", [3, 6, 9]),
                    ("root.a", "query", [4]),
                    ("llm", "llm", [5]),
                    ("done", "done", []),
                    ("root.b", "query", [7]),
                    ("llm", "llm", [8]),
                    ("done", "done", []),
                    ("root.c", "query", [10]),
                    ("llm", "llm", [11]),
                    ("done", "done", []),
                ],
                "the block launches children, which run in parallel",
            ),
            (
                [
                    ("query", "query", [1]),
                    ("llm", "llm", [2]),
                    ("exec", "exec", [3, 6, 9, 12]),
                    ("root.a", "query", [4]),
                    ("llm", "llm", [5]),
                    ("done", "done", []),
                    ("root.b", "query", [7]),
                    ("llm", "llm", [8]),
                    ("done", "done", []),
                    ("root.c", "query", [10]),
                    ("llm", "llm", [11]),
                    ("done", "done", []),
                    ("output", "output", [13]),
                    ("done", "done", []),
                ],
                "the block returns their results, and the parent answers",
            ),
        ]

        timeline = Line(
            LEFT * 4.8 + DOWN * 3.28,
            RIGHT * 4.8 + DOWN * 3.28,
            color=DIM,
            stroke_width=1.2,
        )
        dots = VGroup(
            *[
                Circle(
                    radius=0.07, color=DIM, fill_opacity=0.35, stroke_width=1.0
                ).move_to(
                    LEFT * 4.8
                    + RIGHT * (9.6 * i / (len(snapshot_steps) - 1))
                    + DOWN * 3.28
                )
                for i in range(len(snapshot_steps))
            ]
        )
        self.play(Create(timeline), FadeIn(dots), run_time=0.40)

        active_graph = None
        active_label = None
        active_dot = None
        for i, (specs, label_text) in enumerate(snapshot_steps):
            graph = tiny_graph(specs, root_pos=UP * 1.85, scale=1.00)
            label = Text(label_text, font=CODE_FONT, font_size=FS_BODY, color=WHITE_C)
            label.move_to(DOWN * 2.62)
            dot = Circle(
                radius=0.11, color=HOT, fill_opacity=0.65, stroke_width=1.5
            ).move_to(dots[i])
            if active_graph is not None:
                self.play(
                    FadeOut(active_graph),
                    FadeOut(active_label),
                    FadeOut(active_dot),
                    run_time=0.40,
                )
            self.play(
                FadeIn(graph, shift=DOWN * 0.06),
                FadeIn(label, shift=UP * 0.04),
                FadeIn(dot),
                run_time=0.70,
            )
            self.wait(0.55)
            active_graph = graph
            active_label = label
            active_dot = dot

        self.play(
            FadeOut(
                VGroup(
                    timeline, dots, active_graph, active_label, active_dot
                ),
                shift=DOWN * 0.05,
            ),
            run_time=0.55,
        )

        # Phase 3 — multitree at a delegating block, fork it, steer the copy.
        fork_code = code_card(
            [
                ("action = next(n for n in root.transcript()", DIM),
                ("    if isinstance(n, ExecAction))", DIM),
                ("branch = action.fork()", DIM),
                ("branch.frontier.append(", DIM),
                ('    UserQuery(content="try it another way"))', DIM),
                ("async for node in flow.run_streaming(", DIM),
                ('    branch, until="done"):', DIM),
            ],
            width=5.35,
            height=3.10,
            font_size=9,
            header_size=FS_SMALL,
        )
        fork_code.to_edge(LEFT, buff=0.40)

        fork_title = center_head("Fork the run")

        fork_code_lines = fork_code[2]

        def fork_light(idx):
            return [
                fork_code_lines[i].animate.set_color(HOT if i == idx else DIM)
                for i in range(len(fork_code_lines))
            ]

        fr = 0.20
        fdy = 0.46
        orig_x = 2.05
        fork_x = 5.35

        def fnode(kind, pos, *, r=fr):
            return typed_node(kind, "", r=r).move_to(pos)

        def graph_edge(a, b):
            return Line(
                a.get_bottom(),
                b.get_top(),
                color=DIM,
                stroke_width=0.65,
            ).set_opacity(0.62)

        orig, tree_stages, _ = fork_delegation_graph(x_center=0)
        fork_tree = orig.copy()
        graph_anchor = UP * 0.15
        orig.move_to(graph_anchor)
        fork_tree.move_to(RIGHT * fork_x + graph_anchor)

        # Step 1 — a delegating multitree with child agents (graph only, no code).
        multitree_title = center_head("Every recursive run is an editable graph")
        self.play(ReplacementTransform(head, multitree_title), run_time=0.55)
        head = multitree_title
        for stage in tree_stages:
            self.play(reveal_stage(stage), run_time=0.50)
        self.wait(0.45)

        # Step 2 — fork copies the multitree; code appears, original slides left.
        self.play(ReplacementTransform(head, fork_title), run_time=0.45)
        head = fork_title
        fork_badge = VGroup(
            RoundedRectangle(
                corner_radius=0.10,
                width=1.65,
                height=0.44,
                color=HOT,
                stroke_width=1.3,
                fill_opacity=0.08,
            ),
            Text("forked run", font=CODE_FONT, font_size=FS_CAP, color=HOT),
        ).next_to(fork_tree[0][0], UP, buff=0.16)
        self.play(FadeIn(fork_code, shift=UP * 0.06), *fork_light(0), run_time=0.55)
        self.play(
            orig.animate.move_to(RIGHT * orig_x + graph_anchor),
            TransformFromCopy(orig, fork_tree),
            run_time=1.10,
        )
        restore_typed_node_fills(orig)
        restore_typed_node_fills(fork_tree)
        self.play(FadeIn(fork_badge, shift=UP * 0.05), run_time=0.40)
        self.wait(0.45)
        fork_refs = fork_graph_refs(fork_tree)

        # Step 3 — the fork cuts everything after the block; append a new query.
        inject_title = center_head("Fork at a node, cut what came after")
        exec_node = fork_refs["exec"]
        child_stages = VGroup(fork_tree[3], fork_tree[4], fork_tree[5])
        hi_box = RoundedRectangle(
            corner_radius=0.06,
            width=0.52,
            height=0.52,
            color=HOT,
            stroke_width=2.4,
            fill_opacity=0.0,
        ).move_to(exec_node)
        self.play(
            ReplacementTransform(head, inject_title),
            *fork_light(2),
            Create(hi_box),
            run_time=0.65,
        )
        head = inject_title
        self.wait(0.25)
        self.play(FadeOut(child_stages, shift=DOWN * 0.22), run_time=0.55)
        fork_tree.remove(fork_tree[5], fork_tree[4], fork_tree[3])
        self.remove(child_stages)
        restore_typed_node_fills(fork_tree)
        self.wait(0.25)

        append_title = center_head("Append to the frontier")
        inj_pos = exec_node.get_center() + DOWN * fdy
        inj = fnode("query", inj_pos)
        e_ei = graph_edge(exec_node, inj)
        self.play(
            ReplacementTransform(head, append_title),
            *fork_light(3),
            run_time=0.55,
        )
        head = append_title
        self.play(FadeIn(VGroup(e_ei, inj), shift=DOWN * 0.05), run_time=0.55)
        restore_typed_node_fills(inj)
        self.play(Flash(inj, color=Q_C, flash_radius=0.32), run_time=0.38)
        self.wait(0.35)

        # Step 4 — continue the fork: the model answers, and the run finishes.
        cont_title = center_head("Continue from the fork")
        _, tail_nodes, tail_edges, _ = agent_stack(
            "",
            ["llm", "done"],
            x=inj_pos[0],
            top_y=inj_pos[1] - fdy,
            r=fr,
            dy=fdy,
            fs=FS_CAP,
        )
        cr, cd = tail_nodes
        e_ir = graph_edge(inj, cr)
        e_rd = tail_edges[0]
        fork_tail = VGroup(e_ir, cr, e_rd, cd)
        fork_rhs = VGroup(fork_tree, fork_badge, e_ei, inj, fork_tail)
        self.play(
            ReplacementTransform(head, cont_title),
            *fork_light(5),
            FadeOut(hi_box),
            run_time=0.60,
        )
        head = cont_title
        self.play(
            LaggedStart(
                FadeIn(VGroup(e_ir, cr), shift=DOWN * 0.04),
                FadeIn(VGroup(e_rd, cd), shift=DOWN * 0.04),
                lag_ratio=0.40,
            ),
            run_time=1.10,
        )
        restore_typed_node_fills(fork_tail)
        self.play(Flash(cd, color=R_C, flash_radius=0.30), run_time=0.38)
        self.wait(0.80)

        fork_mobs = VGroup(fork_code, orig, fork_rhs)

        finale_title = center_head(
            "Nested agent trees -- one modular graph", buff=0.72
        )
        dense_graph, dense_stages = rendered_style_graph()
        self.play(
            FadeOut(head),
            FadeOut(fork_mobs),
            FadeIn(finale_title, shift=DOWN * 0.05),
            run_time=0.75,
        )
        for stage in dense_stages:
            self.play(reveal_stage(stage), run_time=0.18)
        self.play(Flash(dense_graph, color=R_C, flash_radius=0.35), run_time=0.28)
        self.wait(1.20)
