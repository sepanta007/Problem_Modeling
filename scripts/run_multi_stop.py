#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms import (
    alt, astar, bfs, bidirectional_alt, bidirectional_dijkstra, build_arcflags,
    build_ch, combine_legs, dfs, dijkstra, greedy_best_first,
)
from src.heuristics import haversine_heuristic
from src.animate import animate_comparison
from src.interactive_map import build_route_map
from src.itinerary import build_itinerary, format_itinerary
from src.landmarks import build_landmarks
from src.network import load_network
from src.visualize import compare_searches

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


def geocode(query: str) -> tuple[float, float]:
    import osmnx as ox
    lat, lon = ox.geocode(query)
    return float(lat), float(lon)


def snap(net, lat: float, lon: float, label: str) -> int:
    node = net.nearest_node(lat, lon)
    R = 6_371_008.8
    dlat = math.radians(net.lat[node] - lat)
    dlon = math.radians(net.lon[node] - lon)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat)) * math.cos(math.radians(net.lat[node]))
         * math.sin(dlon / 2) ** 2)
    d = 2 * R * math.asin(math.sqrt(a))
    if d > 500:
        print(f"  ! {label}: snapped node is {d:,.0f} m away — address may be "
              f"outside the '{net.name}' extract.")
    return node


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--place", default="Paris, France")
    ap.add_argument("--stop", action="append", required=True,
                    help="an address; repeat for each waypoint (order matters)")
    ap.add_argument("--label", action="append", default=None,
                    help="pretty name for each stop (same order as --stop)")
    ap.add_argument("--landmarks", type=int, default=16)
    ap.add_argument("--strategy", default="farthest",
                    choices=["random", "planar", "farthest", "avoid"])
    ap.add_argument("--no-ch", action="store_true",
                    help="skip Contraction Hierarchies (avoids the heavy preprocessing)")
    ap.add_argument("--gif", action="store_true",
                    help="also render an animated GIF of the whole journey")
    ap.add_argument("--directions", action="store_true",
                    help="print the turn-by-turn route sheet (streets to follow)")
    ap.add_argument("--map", action="store_true",
                    help="save a pretty interactive HTML map of the route")
    ap.add_argument("--frames", type=int, default=60,
                    help="number of reveal frames for the GIF")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    if len(args.stop) < 2:
        ap.error("give at least two --stop addresses")
    labels = args.label or [f"stop {i}" for i in range(len(args.stop))]
    if len(labels) != len(args.stop):
        ap.error("number of --label must match number of --stop")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Loading {args.place!r} ...")
    net = load_network(place=args.place)
    print(net)
    print(f"Building {args.landmarks} '{args.strategy}' landmarks ...")
    lm = build_landmarks(net, k=args.landmarks, strategy=args.strategy, seed=1)
    print(f"  preprocessing took {lm.preprocess_s:.2f}s\n")

    # Geocode and snap every stop.
    nodes: list[int] = []
    for addr, lab in zip(args.stop, labels):
        lat, lon = geocode(addr)
        node = snap(net, lat, lon, lab)
        nodes.append(node)
        print(f"  {lab:<14} '{addr}' -> ({lat:.5f}, {lon:.5f})  [node {node}]")

    # The full line-up. Uninformed (BFS/DFS) and Greedy are NOT metric-optimal,
    # so their total distance is longer than the optimal 6.
    ch = None
    arcflags = None
    if not args.no_ch:
        print("\nBuilding Contraction Hierarchy + Arc Flags (one-off preproc.) ...")
        ch = build_ch(net)
        arcflags = build_arcflags(net, regions=32, seed=0)
        print(f"  CH: {ch.preprocess_s:.1f}s, {ch.n_shortcuts:,} shortcuts · "
              f"Arc Flags: {arcflags.preprocess_s:.1f}s")

    algos = {
        "BFS": lambda s, t: bfs(net, s, t),
        "DFS": lambda s, t: dfs(net, s, t),
        "Dijkstra": lambda s, t: dijkstra(net, s, t),
        "Greedy(haversine)": lambda s, t: greedy_best_first(
            net, s, t, haversine_heuristic, label="Greedy(haversine)"),
        "A*(haversine)": lambda s, t: astar(net, s, t, haversine_heuristic,
                                            label="A*(haversine)"),
        "ALT": lambda s, t: alt(net, s, t, lm),
        "Bi-Dijkstra": lambda s, t: bidirectional_dijkstra(net, s, t),
        "Bi-ALT": lambda s, t: bidirectional_alt(net, s, t, lm),
    }
    if arcflags is not None:
        algos["Arc Flags"] = lambda s, t: arcflags.query(s, t)
    if ch is not None:
        algos["CH"] = lambda s, t: ch.query(s, t)

    # Run every algorithm over all legs, and combine.
    combined = {}
    per_leg = {}
    for name, run in algos.items():
        legs = [run(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
        combined[name] = combine_legs(legs, algorithm=name)
        per_leg[name] = legs

    # ---- report --------------------------------------------------------
    leg_names = [f"{labels[i]}→{labels[i+1]}" for i in range(len(nodes) - 1)]
    total_km = combined["Dijkstra"].cost / 1000
    print(f"\nRoute: {' → '.join(labels)}")
    print(f"Total distance (optimal): {total_km:.2f} km "
          f"over {len(nodes)-1} legs\n")

    print(f"{'algorithm':<18}" + "".join(f"{ln:>22}" for ln in leg_names)
          + f"{'TOTAL settled':>16}{'total km':>10}{'ms':>8}")
    print("-" * (18 + 22 * len(leg_names) + 34))
    for name in algos:
        cells = "".join(f"{leg.settled:>13,} ({leg.cost/1000:>4.1f}km)"
                        for leg in per_leg[name])
        c = combined[name]
        tag = "  <- not optimal" if c.cost > total_km * 1000 + 1 else ""
        print(f"{name:<18}{cells}{c.settled:>16,}{c.cost/1000:>10.2f}"
              f"{c.elapsed_s*1000:>8.1f}{tag}")

    dij = combined["Dijkstra"].settled
    print("\nvs Dijkstra (total nodes explored):")
    for name in algos:
        c = combined[name]
        note = "  (but NOT optimal)" if c.cost > total_km * 1000 + 1 else ""
        print(f"  {name:<18}{dij/c.settled:>6.1f}x fewer nodes{note}")

    # ---- turn-by-turn route sheet --------------------------------------
    if args.directions:
        print("\n" + "=" * 70)
        print("TURN-BY-TURN ROUTE SHEET (streets to follow, optimal route)")
        print("=" * 70)
        for i in range(len(nodes) - 1):
            leg = per_leg["Dijkstra"][i]     # optimal path; identical for all optimal algos
            steps = build_itinerary(net, leg.path)
            print()
            print(format_itinerary(steps, labels[i], labels[i + 1]))

    # ---- interactive map -----------------------------------------------
    if args.map:
        legs_info = [
            {"path": per_leg["Dijkstra"][i].path,
             "from": labels[i], "to": labels[i + 1]}
            for i in range(len(nodes) - 1)
        ]
        html_out = os.path.join(RESULTS_DIR, "itinerary_map.html")
        build_route_map(net, legs_info, html_out,
                        title=" → ".join(labels))
        print(f"\nInteractive map -> {html_out}  (open it in a browser)")

    # ---- figure --------------------------------------------------------
    results = [combined[n] for n in algos]
    # Balanced grid: 9 algos -> 3x3, 10 -> 5x2, 8 -> 4x2, etc.
    ncols = 5 if len(results) == 10 else math.ceil(len(results) ** 0.5)
    out = os.path.join(RESULTS_DIR, "multi_stop_comparison.png")
    compare_searches(
        net, results, out, ncols=ncols,
        suptitle=f"Multi-stop route — {' → '.join(labels)}  ({args.place})",
        waypoints=nodes,
    )
    print(f"\nFigure written to {out}")

    if args.gif:
        anim_results = [combined[n] for n in algos]
        gif = os.path.join(RESULTS_DIR, "exploration.gif")
        print(f"Rendering GIF ({args.frames} frames @ {args.fps} fps) ...")
        animate_comparison(
            net, anim_results, gif, n_frames=args.frames, fps=args.fps, ncols=ncols,
            waypoints=nodes,
            suptitle=f"Exploration over time — {' → '.join(labels)}",
        )
        print(f"GIF written to {gif}  "
              f"({os.path.getsize(gif)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
