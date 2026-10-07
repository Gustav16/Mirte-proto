"""
path_planning_test.py

Offline comparison of path planners for MIRTE. No robot needed:

    python path_planning_test.py
    python path_planning_test.py --margin 0.05 --runs 50
    python path_planning_test.py --plot planners.png

It plans on the same LocalMap as mirte_rrt.py (circles around ArUco boxes) and
compares today's planner with ideas from Litteratur_på_pathfinding/:

  rrt       RRT + simplify_path, exactly as mirte_rrt.py does today (baseline)
  rrt_best  the same RRT run several times, keeping the cheapest path
            [LaValle & Kuffner, "Randomized Kinodynamic Planning", sec. 4:
             keep growing and choose the "best" solution by a cost functional]
  birrt     bidirectional RRT, one tree from the start and one from the goal
            [LaValle & Kuffner, Figure 7: RRT_BIDIRECTIONAL]
  astar     A* on an occupancy grid
            [Ferguson, Likhachev & Stentz, "A Guide to Heuristic-based Path
             Planning", Figure 1. They recommend deterministic search when the
             problem has few dimensions, as ours has (x, z)]
  wastar    A* with the heuristic inflated by eps = 2.5: faster, and at most
            eps times longer than optimal  [same paper, "Anytime Algorithms"]

  astar_safe  A* that first tries a wide safety margin round the boxes and
            narrows it until a path exists. This relies on A* being complete:
            it can say for certain that no path exists at a given margin,
            which RRT cannot  [same paper, "Path Planning"]

  birrt_opt  bidirectional RRT, then trajectory optimisation: all path points
            are moved at once to minimise a cost, argmin f(q_0:T)
            [Toussaint, "Path Planning", slides 6 and 54]. The cost is the
            time to drive the path while turning (no stops to rotate) plus a
            penalty for passing close to a box. Where nothing is in the way
            the result stays a straight line.

Every path is first shortened with the same shortcut smoothing
[LaValle & Kuffner, sec. 7 "Variational Optimization": simple path smoothing].

Each path is scored on length, turns, clearance and planning time, and on how
often MIRTE would hit a box when driving it. That is found by simulated drives
with the noise terms of the odometry motion model from the course slides
[motion_and_measurements, slide 6].
"""

import argparse
import heapq
import math
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../../Mirte/ku_mirte_python"))

# mirte_rrt.py and local_map.py import the robot library and OpenCV at the top.
# The planners need neither, so stand-ins are used when they are not installed.
try:
    import ku_mirte  # noqa: F401
except Exception:
    sys.modules["ku_mirte"] = types.SimpleNamespace(KU_Mirte=None)
try:
    import cv2  # noqa: F401
except Exception:
    sys.modules["cv2"] = types.SimpleNamespace(
        aruco=types.SimpleNamespace(getPredefinedDictionary=lambda d: None, DICT_6X6_250=0))

import matplotlib
matplotlib.use("Agg")

import local_map
import robot_models
from mirte_rrt import RRT, simplify_path


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

START = np.array([0.0, 0.0])
GOAL = np.array([0.0, 1.9])                 # same goal as mirte_rrt.main()
MAP_LOW, MAP_HIGH = (-1, 0), (1, 2)         # same area as mirte_rrt.main()

LINEAR_SPEED = 0.3                          # m/s,   from Execute_path
ANGULAR_SPEED = 0.7                         # rad/s, from Execute_path

GRID_RES = 0.05                             # A* cell size (m)
SAFE_MARGINS = (0.15, 0.10, 0.05)           # margins astar_safe tries, widest first (m)

# Trajectory optimisation (birrt_opt)
OPT_POINTS = 25                             # points on the optimised path
OPT_ITERATIONS = 300
SAFE_DIST = 0.15                            # wanted extra distance to a no-go zone (m)
W_SAFE = 4000.0                             # s per m^2 closer than SAFE_DIST, per metre of path
W_NO_GO = 50 * W_SAFE                       # same, inside a no-go zone or outside the map
W_EVEN = 100.0                              # keeps the points evenly spaced (numerics only)

# Physical contact: robot circle against the corner of a 25 cm box.
ROBOT_RADIUS = 0.22
BOX_CONTACT_RADIUS = 0.25 / math.sqrt(2)

# Odometry motion model noise (slide 6). These are guesses, not measurements:
# set them from measurements.py results when you have them.
#   variance of rot   = ALPHA1 * rot^2   + ALPHA2 * trans^2
#   variance of trans = ALPHA3 * trans^2 + ALPHA4 * rot^2
# simulate_drive() draws each error once per run (see its docstring).
ALPHA1, ALPHA2, ALPHA3, ALPHA4 = 0.005, 0.001, 0.002, 0.0005
MOTION_SAMPLES = 300

# Box layouts: map coordinates, x = right, z = forward, in metres.
SCENARIOS = {
    "open":    [],
    "gate":    [(-0.5, 1.25), (0.5, 1.25)],
    "blocked": [(-0.5, 1.25), (0.5, 1.25), (0.0, 0.7)],
    "slalom":  [(-0.35, 0.6), (0.35, 1.25)],
    "wall":    [(-0.8, 1.0), (-0.4, 1.0), (0.0, 1.0), (0.4, 1.0), (0.8, 1.0)],   # no path exists
}


# ---------------------------------------------------------------------------
# Map and path helpers
# ---------------------------------------------------------------------------

def make_map(boxes, margin=0.0):
    """LocalMap as mirte_rrt.py builds it, with an optional extra safety margin."""
    default_radius = local_map.LocalMap().landmark_radius
    m = local_map.LocalMap(low=MAP_LOW, high=MAP_HIGH, landmark_radius=default_radius + margin)
    m.landmarks = [[np.array(b, dtype=float), i + 1] for i, b in enumerate(boxes)]
    return m


def segment_free(m, a, b, step=0.02):
    """True if the straight line a -> b stays out of every no-go zone."""
    n = max(2, int(math.ceil(np.linalg.norm(b - a) / step)) + 1)
    return not any(m.in_collision(a + t * (b - a)) for t in np.linspace(0.0, 1.0, n))


def shortcut(path, m):
    """From each point, jump to the furthest later point with a free straight line."""
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not segment_free(m, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def path_length(path):
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])))


def count_stops(path, continuous=False):
    """Places where MIRTE has to stop or slow down to rotate.

    Turn-then-drive stops at every turn. When turning while driving, only a turn
    that cannot be finished within its leg at full speed counts.
    """
    heading, stops = math.pi / 2, 0
    for a, b in zip(path[:-1], path[1:]):
        target = math.atan2(b[1] - a[1], b[0] - a[0])
        turn = abs(math.atan2(math.sin(target - heading), math.cos(target - heading)))
        heading = target
        limit = ANGULAR_SPEED * np.linalg.norm(b - a) / LINEAR_SPEED if continuous else math.radians(2)
        stops += turn > limit
    return stops


def path_turns(path):
    """Turn angles (rad) that Execute_path would make. MIRTE starts facing +z."""
    heading, turns = math.pi / 2, []
    for a, b in zip(path[:-1], path[1:]):
        target = math.atan2(b[1] - a[1], b[0] - a[0])
        d = math.atan2(math.sin(target - heading), math.cos(target - heading))
        if abs(d) > 1e-6:
            turns.append(d)
        heading = target
    return turns


def drive_time(path, continuous=False):
    """Seconds to drive the path.

    continuous=False: Execute_path today, turn on the spot and then drive each leg.
    continuous=True:  turn while driving, so each leg takes the longer of the two.
    """
    heading, total = math.pi / 2, 0.0
    for a, b in zip(path[:-1], path[1:]):
        target = math.atan2(b[1] - a[1], b[0] - a[0])
        turn = abs(math.atan2(math.sin(target - heading), math.cos(target - heading)))
        heading = target
        t_drive, t_turn = np.linalg.norm(b - a) / LINEAR_SPEED, turn / ANGULAR_SPEED
        total += max(t_drive, t_turn) if continuous else t_drive + t_turn
    return float(total)


def clearance(path, m):
    """Smallest distance from the path to the edge of a no-go zone (m). Negative = inside."""
    if not m.landmarks:
        return float("nan")
    no_go = m.landmark_radius + m.mirte_radius
    best = float("inf")
    for a, b in zip(path[:-1], path[1:]):
        for t in np.linspace(0.0, 1.0, max(2, int(np.linalg.norm(b - a) / 0.01))):
            p = a + t * (b - a)
            best = min(best, min(np.linalg.norm(p - lm) for lm, _ in m.landmarks) - no_go)
    return best


# ---------------------------------------------------------------------------
# Planners. Each returns a path from start to goal (list of arrays) or None.
# ---------------------------------------------------------------------------

def plan_rrt(m, do_shortcut=True):
    """Baseline: the RRT and simplify_path from mirte_rrt.py, with main()'s settings."""
    rrt = RRT(start=START, goal=GOAL,
              robot_model=robot_models.PointMassModel(ctrl_range=[-0.1, 0.1]),
              map=m, expand_dis=0.4, path_resolution=0.1)
    path = rrt.planning(animation=False)
    if path is None:
        return None
    path = [np.asarray(p, dtype=float) for p in reversed(simplify_path(path, rrt))]
    return shortcut(path, m) if do_shortcut else path


def plan_rrt_best(m, tries=10):
    """Run the RRT several times and keep the path that is quickest to drive."""
    paths = [p for p in (plan_rrt(m) for _ in range(tries)) if p is not None]
    return min(paths, key=drive_time) if paths else None


def plan_birrt(m, step=0.2, max_iter=1000):
    """RRT_BIDIRECTIONAL (LaValle & Kuffner, Figure 7)."""
    low, high = m.map_area

    def extend(tree, x):
        """EXTEND (Figure 5): one step from the nearest vertex towards x."""
        near = min(range(len(tree)), key=lambda i: np.linalg.norm(tree[i][0] - x))
        p_near = tree[near][0]
        d = np.linalg.norm(x - p_near)
        p_new = x if d <= step else p_near + (x - p_near) * step / d
        if not segment_free(m, p_near, p_new):
            return "trapped", None
        tree.append((p_new, near))
        return ("reached" if d <= step else "advanced"), p_new

    def branch(tree):
        """Walk from the newest vertex back to the root."""
        out, i = [], len(tree) - 1
        while i is not None:
            out.append(tree[i][0])
            i = tree[i][1]
        return out

    tree_a, tree_b = [(START.copy(), None)], [(GOAL.copy(), None)]
    a_is_start = True
    for _ in range(max_iter):
        status, x_new = extend(tree_a, np.random.uniform(low, high))
        if status != "trapped" and extend(tree_b, x_new)[0] == "reached":
            half_a, half_b = branch(tree_a)[::-1], branch(tree_b)[1:]
            path = half_a + half_b
            return shortcut(path if a_is_start else path[::-1], m)
        tree_a, tree_b = tree_b, tree_a
        a_is_start = not a_is_start
    return None


def plan_astar(m, eps=1.0, stats=None):
    """A* on an 8-connected occupancy grid (Ferguson et al., Figure 1). eps > 1 = weighted A*."""
    low, high = np.asarray(m.map_area[0]), np.asarray(m.map_area[1])
    nx, nz = [int(round(v)) + 1 for v in (high - low) / GRID_RES]

    def centre(c):
        return low + np.array(c) * GRID_RES

    def cell(p):
        return tuple(int(round(v)) for v in (np.asarray(p) - low) / GRID_RES)

    free = np.array([[not m.in_collision(centre((i, j))) for j in range(nz)] for i in range(nx)])
    s, g = cell(START), cell(GOAL)
    if not (free[s] and free[g]):
        return None

    def h(c):
        return eps * GRID_RES * math.hypot(c[0] - g[0], c[1] - g[1])

    moves = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
    cost, parent, closed = {s: 0.0}, {s: None}, set()
    open_list = [(h(s), s)]
    while open_list:
        _, c = heapq.heappop(open_list)
        if c in closed:
            continue
        closed.add(c)
        if c == g:
            break
        for di, dj in moves:
            n = (c[0] + di, c[1] + dj)
            if not (0 <= n[0] < nx and 0 <= n[1] < nz) or not free[n] or n in closed:
                continue
            if di and dj and not (free[c[0] + di, c[1]] and free[c[0], c[1] + dj]):
                continue                                   # no cutting corners
            new_cost = cost[c] + GRID_RES * math.hypot(di, dj)
            if new_cost < cost.get(n, float("inf")):
                cost[n], parent[n] = new_cost, c
                heapq.heappush(open_list, (new_cost + h(n), n))
    if stats is not None:
        stats["expanded"] = len(closed)
    if g not in closed:
        return None                                        # searched everything: no path exists

    cells, c = [], g
    while c is not None:
        cells.append(c)
        c = parent[c]
    path = [START] + [centre(c) for c in cells[::-1][1:-1]] + [GOAL]
    return shortcut(path, m)


def resample(path, k):
    """k points evenly spaced along the path."""
    pts = np.array(path)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], k)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)


def optimisation_cost(pts, m, centres):
    """f(q_0:T): seconds to drive the path while turning, plus penalties near boxes.

    MIRTE turns while it drives. A leg takes len / LINEAR_SPEED unless the turn into
    it needs longer, so gentle bends are free and only sharp corners cost time.
    """
    seg = np.diff(pts, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    turn = np.diff(np.concatenate([[math.pi / 2], heading]))          # MIRTE starts facing +z
    turn = np.arctan2(np.sin(turn), np.cos(turn))
    t_drive = lens / LINEAR_SPEED
    extra = np.abs(turn) / ANGULAR_SPEED - t_drive                    # > 0: must slow down to turn
    cost = t_drive.sum() + (0.5 * (extra + np.sqrt(extra**2 + 1e-4))).sum()

    def penalty(p):
        """Penalty rate at each point: near a box, inside a no-go zone, outside the map."""
        out = W_NO_GO * ((np.maximum(0.0, m.map_area[0] - p) + np.maximum(0.0, p - m.map_area[1])) ** 2).sum(axis=1)
        if len(centres):
            no_go = m.landmark_radius + m.mirte_radius
            d = np.linalg.norm(p[:, None, :] - centres[None, :, :], axis=2)
            out += W_SAFE * (np.maximum(0.0, no_go + SAFE_DIST - d) ** 2).sum(axis=1)
            out += W_NO_GO * (np.maximum(0.0, no_go - d) ** 2).sum(axis=1)
        return out

    # integrate the penalty along the path, so moving points about cannot hide it
    at_pts, at_mid = penalty(pts), penalty((pts[:-1] + pts[1:]) / 2)
    cost += (lens * (at_pts[:-1] + 4 * at_mid + at_pts[1:]) / 6).sum()
    return cost + W_EVEN * ((lens - lens.mean()) ** 2).sum()


def optimise_path(path, m):
    """Trajectory optimisation (Toussaint, slide 6): gradient descent on all inner points.

    Start and goal stay fixed. The result stays on the same side of each box as the
    path it starts from. If the result is not collision-free, the input path is kept.
    """
    pts = resample(path, OPT_POINTS)
    centres = np.array([lm for lm, _ in m.landmarks], dtype=float).reshape(-1, 2)

    def f(x):
        return optimisation_cost(np.vstack([pts[:1], x.reshape(-1, 2), pts[-1:]]), m, centres)

    def gradient(x, h=1e-5):
        g = np.zeros_like(x)
        for i in range(len(x)):
            e = np.zeros_like(x)
            e[i] = h
            g[i] = (f(x + e) - f(x - e)) / (2 * h)
        return g

    x = pts[1:-1].ravel().copy()
    fx, step = f(x), 0.01
    for _ in range(OPT_ITERATIONS):
        g = gradient(x)
        g2 = float(g @ g)
        if g2 < 1e-10:
            break
        # backtracking line search: halve the step until the cost really drops
        while step > 1e-9 and f(x - step * g) > fx - 1e-4 * step * g2:
            step *= 0.5
        if step <= 1e-9:
            break
        x = x - step * g
        fx, step = f(x), step * 2.0

    out = [p for p in np.vstack([pts[:1], x.reshape(-1, 2), pts[-1:]])]
    if all(segment_free(m, a, b) for a, b in zip(out[:-1], out[1:])):
        return out
    return path


def plan_birrt_opt(m):
    """Bidirectional RRT, then trajectory optimisation (Toussaint, slides 54 and 6)."""
    path = plan_birrt(m)
    return None if path is None else optimise_path(path, m)


def plan_astar_safe(m):
    """A* with the widest safety margin that still leaves a path."""
    for margin in SAFE_MARGINS:
        wide = local_map.LocalMap(low=m.map_area[0], high=m.map_area[1],
                                  landmark_radius=m.landmark_radius + margin)
        wide.landmarks = m.landmarks
        path = plan_astar(wide)
        if path is not None:
            return path
    return plan_astar(m)


PLANNERS = {
    # name:      (function, uses random numbers, driven as one continuous curve)
    "rrt":        (plan_rrt, True, False),
    "rrt_best":   (plan_rrt_best, True, False),
    "birrt":      (plan_birrt, True, False),
    "birrt_opt":  (plan_birrt_opt, True, True),
    "astar":      (plan_astar, False, False),
    "wastar":     (lambda m: plan_astar(m, eps=2.5), False, False),
    "astar_safe": (plan_astar_safe, False, False),
}


# ---------------------------------------------------------------------------
# Driving the path with the odometry motion model (slide 6)
# ---------------------------------------------------------------------------

def simulate_drive(path, boxes, rng, n=MOTION_SAMPLES):
    """Drive the path n times without looking, with noisy motion.

    The noise terms are those of the odometry motion model (slide 6), but each
    error is drawn once per run and then applies to the whole drive:
      turn error      = e_rot * turn            e_rot   ~ N(0, ALPHA1)
      heading drift   = e_curve * distance      e_curve ~ N(0, ALPHA2)
      distance error  = e_trans * distance      e_trans ~ N(0, ALPHA3)
                      + e_slip * |turn|         e_slip  ~ N(0, ALPHA4)
    For a single turn-and-drive leg this gives the variances on the slide. Drawing
    the errors per leg instead would make a path cut into many short legs look more
    accurate than the same path as one long leg.

    Returns (share of runs that touch a box, median distance from the goal in m).
    """
    e_rot = rng.normal(0, math.sqrt(ALPHA1), n)
    e_curve = rng.normal(0, math.sqrt(ALPHA2), n)
    e_trans = rng.normal(0, math.sqrt(ALPHA3), n)
    e_slip = rng.normal(0, math.sqrt(ALPHA4), n)

    pos = np.tile(path[0], (n, 1))
    theta = np.full(n, math.pi / 2)            # actual heading of each simulated run
    planned = math.pi / 2                      # heading MIRTE believes it has
    hit = np.zeros(n, dtype=bool)
    contact = ROBOT_RADIUS + BOX_CONTACT_RADIUS
    centres = [np.asarray(b, dtype=float) for b in boxes]

    for a, b in zip(path[:-1], path[1:]):
        trans = float(np.linalg.norm(b - a))
        target = math.atan2(b[1] - a[1], b[0] - a[0])
        rot = math.atan2(math.sin(target - planned), math.cos(target - planned))
        planned = target

        theta = theta + rot * (1 + e_rot)
        steps = max(1, int(math.ceil(trans / 0.02)))
        ds = (trans * (1 + e_trans) + abs(rot) * e_slip) / steps
        for _ in range(steps):
            theta = theta + e_curve * ds / 2
            pos = pos + np.stack([np.cos(theta), np.sin(theta)], axis=1) * ds[:, None]
            theta = theta + e_curve * ds / 2
            for c in centres:
                hit |= np.linalg.norm(pos - c, axis=1) < contact

    return float(hit.mean()), float(np.median(np.linalg.norm(pos - path[-1], axis=1)))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def evaluate(name, boxes, margin, runs, rng):
    """Run one planner on one layout. Returns averaged results as a dict, plus an example path."""
    planner, is_random, continuous = PLANNERS[name]
    m = make_map(boxes, margin)
    m_true = make_map(boxes)                   # clearance is measured against the real no-go zones
    rows, example = [], None
    for _ in range(runs if is_random else 1):
        t0 = time.perf_counter()
        path = planner(m)
        ms = (time.perf_counter() - t0) * 1000.0
        if path is None:
            rows.append(dict(found=0, ms=ms))
            continue
        p_hit, end_err = simulate_drive(path, boxes, rng)
        turns = path_turns(path)
        rows.append(dict(found=1, ms=ms, length=path_length(path),
                         turns=count_stops(path, continuous),
                         turning=math.degrees(sum(abs(t) for t in turns)),
                         time=drive_time(path, continuous),
                         clear=clearance(path, m_true), p_hit=p_hit, end_err=end_err))
        example = example or path
    out = dict(found=np.mean([r["found"] for r in rows]), ms=np.mean([r["ms"] for r in rows]))
    good = [r for r in rows if r["found"]]
    for k in ("length", "turns", "turning", "time", "clear", "p_hit", "end_err"):
        out[k] = float(np.mean([r[k] for r in good])) if good else float("nan")
    return out, example


def save_plot(examples, margin, filename):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(1, len(examples), figsize=(3.6 * len(examples), 4.2))
    for ax, (scenario, paths) in zip(np.atleast_1d(axes), examples.items()):
        m = make_map(SCENARIOS[scenario], margin)
        for lm, lm_id in m.landmarks:
            ax.add_patch(Circle(lm, m.landmark_radius + m.mirte_radius, color="red", alpha=0.12))
            ax.add_patch(Circle(lm, BOX_CONTACT_RADIUS, color="0.3"))
        for name, path in paths.items():
            if path is not None:
                p = np.array(path)
                ax.plot(p[:, 0], p[:, 1], "-o", markersize=3, linewidth=1.5, label=name)
        ax.plot(*START, "k^")
        ax.plot(*GOAL, "kx")
        ax.set_xlim(MAP_LOW[0], MAP_HIGH[0])
        ax.set_ylim(MAP_LOW[1], MAP_HIGH[1])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(scenario)
        ax.set_xlabel("x (m)")
    np.atleast_1d(axes)[0].set_ylabel("z (m)")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower left")
    fig.savefig(filename, dpi=130, bbox_inches="tight")
    print(f"\nSaved plot to: {filename}")


def main():
    ap = argparse.ArgumentParser(description="Compare path planners for MIRTE offline.")
    ap.add_argument("--runs", type=int, default=30, help="runs per layout for the random planners")
    ap.add_argument("--margin", type=float, default=0.0, help="extra safety margin on the box radius (m)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", metavar="FILE", help="save one example path per planner to this PNG")
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    examples = {}

    print(f"goal ({GOAL[0]:.1f}, {GOAL[1]:.1f}), margin {args.margin:.2f} m, {args.runs} runs for random planners, "
          f"{MOTION_SAMPLES} simulated drives per path")
    for scenario, boxes in SCENARIOS.items():
        print(f"\n{scenario}: {len(boxes)} boxes")
        print(f"  {'planner':10s} {'found':>6s} {'plan ms':>8s} {'length':>7s} {'stops':>6s} {'turning':>8s} "
              f"{'drive s':>8s} {'clearance':>10s} {'hits box':>9s} {'end error':>10s}")
        examples[scenario] = {}
        for name in PLANNERS:
            r, example = evaluate(name, boxes, args.margin, args.runs, rng)
            examples[scenario][name] = example
            if r["found"] == 0:
                print(f"  {name:10s} {r['found']:6.0%} {r['ms']:8.1f}   no path found")
                continue
            clear = "     -    " if math.isnan(r["clear"]) else f"{r['clear'] * 100:7.1f} cm"
            print(f"  {name:10s} {r['found']:6.0%} {r['ms']:8.1f} {r['length']:6.2f}m {r['turns']:6.1f} "
                  f"{r['turning']:7.0f}° {r['time']:8.1f} {clear} {r['p_hit']:9.1%} {r['end_err'] * 100:7.1f} cm")

    if args.plot:
        save_plot(examples, args.margin, args.plot)


if __name__ == "__main__":
    main()
