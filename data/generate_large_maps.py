"""
data/generate_large_maps.py
Generates larger benchmark maps + matching congestion scenarios, so that the
search landscape is big and rugged enough for convergence curves to be visible.

Usage (from the repo root):
    python data/generate_large_maps.py

Writes:
    data/maps/<name>.json
    data/scenarios/<name>_rush.json
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# name: (rows, cols, seed, n_waves, back_edge_prob, express_lines)
PRESETS = {
    "big_city": (9, 10, 6, 4, 0.25, 5),
    "megacity": (12, 14, 11, 4, 0.25, 7),
}


def node_id(r: int, c: int, rows: int, cols: int) -> str:
    if (r, c) == (0, 0):
        return "Home"
    if (r, c) == (rows - 1, cols - 1):
        return "Uni"
    return f"R{r}C{c}"


def build(name, rows, cols, seed, n_waves, back_p, n_express):
    rng = random.Random(seed)
    nodes, edges = [], {}

    for r in range(rows):
        for c in range(cols):
            nid = node_id(r, c, rows, cols)
            nodes.append({"id": nid, "x": float(c), "y": float(rows - 1 - r), "label": nid})

    def add(u, v, t):
        if u != v and (u, v) not in edges:
            edges[(u, v)] = round(t, 1)

    # Street grid: forward (east/south) streets always exist; backward ones
    # (west/north) sometimes, which makes long detours and loops possible.
    for r in range(rows):
        for c in range(cols):
            u = node_id(r, c, rows, cols)
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr < rows and cc < cols:
                    v = node_id(rr, cc, rows, cols)
                    add(u, v, rng.uniform(3.0, 7.0))
                    if rng.random() < back_p:
                        add(v, u, rng.uniform(3.0, 7.0))

    # Express links (diagonal hops of 2-3 blocks): fast but they get the
    # worst rush-hour congestion -> the classic "highway trap".
    express = []
    for _ in range(n_express * 3):
        if len(express) >= n_express:
            break
        r, c = rng.randrange(rows - 3), rng.randrange(cols - 3)
        k = rng.choice((2, 3))
        u, v = node_id(r, c, rows, cols), node_id(r + k, c + k, rows, cols)
        if (u, v) not in edges:
            add(u, v, 2.2 * k)  # ~4x faster than the 2k street blocks it skips
            express.append((u, v))

    # Congestion: several rush-hour "waves", each centred on a random district
    # at its own peak time, plus heavy peaks on all express links.
    peaks = []
    for w in range(n_waves):
        cr, cc = rng.uniform(0, rows - 1), rng.uniform(0, cols - 1)
        radius = rng.uniform(2.0, 3.5)
        mu = rng.uniform(450, 560)
        sigma = rng.uniform(15, 35)
        amp_max = rng.uniform(2.0, 4.0)
        for (u, v) in edges:
            nu = next(n for n in nodes if n["id"] == u)
            ur, uc = rows - 1 - nu["y"], nu["x"]
            d = ((ur - cr) ** 2 + (uc - cc) ** 2) ** 0.5
            if d <= radius:
                amp = amp_max * (1 - d / (radius + 0.5)) * rng.uniform(0.6, 1.0)
                peaks.append({"u": u, "v": v, "peak_time": round(mu + rng.uniform(-10, 10), 1),
                              "sigma": round(sigma, 1), "amplitude": round(amp, 2)})
    for (u, v) in express:
        peaks.append({"u": u, "v": v, "peak_time": round(rng.uniform(470, 530), 1),
                      "sigma": round(rng.uniform(20, 40), 1), "amplitude": round(rng.uniform(4.0, 7.0), 2)})
    # A light background wave on random streets so the landscape has no big flat areas.
    for (u, v) in edges:
        if rng.random() < 0.35:
            peaks.append({"u": u, "v": v, "peak_time": round(rng.uniform(430, 590), 1),
                          "sigma": round(rng.uniform(10, 30), 1), "amplitude": round(rng.uniform(0.3, 1.5), 2)})

    graph = {"nodes": nodes,
             "edges": [{"u": u, "v": v, "base_time": t} for (u, v), t in edges.items()]}
    scenario = {"scenario_name": f"{name} rush hour",
                "description": f"{n_waves} district rush-hour waves, congested express links, "
                               f"background noise on random streets ({rows}x{cols} grid).",
                "peaks": peaks}

    with open(os.path.join(HERE, "maps", f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=1)
    with open(os.path.join(HERE, "scenarios", f"{name}_rush.json"), "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=1)
    print(f"{name}: {len(nodes)} nodes, {len(edges)} edges, {len(peaks)} congestion peaks")


if __name__ == "__main__":
    for name, args in PRESETS.items():
        build(name, *args)
