"""
track_planning_test.py

Offline test of path planning on a rally-style track with boxes along it.
No robot needed:

    python track_planning_test.py
    python track_planning_test.py --runs 50 --plot /tmp/tracks.png
    python track_planning_test.py --margin 0.15

MIRTE is treated as non-holonomic: it drives forward and rotates, never sideways.

Pipeline
  1. Global RRT: two trees, one from the start and one from the goal, growing
     towards each other  [LaValle & Kuffner, Figure 7; Toussaint, slide 54].
     All boxes are known from the start.
  2. Shortcut smoothing: remove waypoints that are not needed
     [LaValle & Kuffner, sec. 7].
  3. Dubins-style smoothing: every corner is replaced by an arc, so the path
     is straight legs joined by arcs  [Toussaint, slides 74-76]. At full speed
     MIRTE can turn on a radius of LINEAR_SPEED / ANGULAR_SPEED (about 0.43 m),
     so it drives such a path without stopping to rotate. Where there is no
     room for that radius, a smaller one is used and MIRTE slows down there.
  The result is a list of drive commands (linear speed, angular speed, seconds),
  the arguments of mirte.drive().

Compared methods
  rrt, stop and turn     steps 1-2, driven as today: stop at each corner, turn, drive
  rrt + margin, stop     the same, planned with PLAN_MARGIN extra clearance
  rrt + margin, arcs     steps 1-3 (the margin is what gives the arcs room)
  astar + margin, arcs   A* on a grid instead of the RRT, then steps 2-3
                         [Ferguson, Likhachev & Stentz, Figure 1]

Driving
  Every method is driven MOTION_SAMPLES times on every track with noisy motion
  [motion_and_measurements, slide 6] to see how often MIRTE would hit a box or
  leave the track. Driving the whole track without looking lets the drift grow
  until it decides everything, so MIRTE can also look where it is every few
  metres (LOOK_INTERVALS). After a look it knows its pose up to a small
  measuring error and steers back onto the route: steps 2-3 are run again
  from the pose it measured, over the route points it has not passed yet.
  Step 1 is not run again, because the boxes have not moved: the old route is
  repaired instead of planned from scratch [Ferguson, Likhachev & Stentz,
  "Incremental Replanning Algorithms"]. A look is taken while driving and is
  assumed to cost no time.

  The motion noise and the measuring error are guesses, not measurements. The
  crash rates compare the methods with each other; they do not predict how
  often the real MIRTE crashes.
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

# local_map.py imports the robot library and OpenCV at the top. Only its radii
# are used here, so stand-ins are used when the two are not installed.
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


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

LINEAR_SPEED = 0.3                          # m/s,   from Execute_path
ANGULAR_SPEED = 0.7                         # rad/s, from Execute_path
TURN_RADIUS = LINEAR_SPEED / ANGULAR_SPEED  # tightest arc at full speed (m)

PLAN_MARGIN = 0.10                          # extra clearance when planning (m)
ARC_MARGIN = 0.02                           # clearance an arc must keep (m)
MIN_RADIUS = 0.05                           # below this MIRTE turns on the spot (m)
RRT_STEP = 0.3                              # m
RRT_MAX_ITER = 4000
GRID_RES = 0.05                             # A* cell size (m)

# Physical contact: robot circle against the corner of a 25 cm box.
ROBOT_RADIUS = 0.22
BOX_CONTACT_RADIUS = 0.25 / math.sqrt(2)

# Motion noise, see drive(). These are guesses, not measurements:
# set them from measurements.py results when you have them.
ALPHA1, ALPHA2, ALPHA3, ALPHA4 = 0.005, 0.001, 0.002, 0.0005
MOTION_SAMPLES = 300                        # simulated drives per method, track and look interval

# Looking where MIRTE is (re-localisation), see simulate_drives(). Also guesses.
LOOK_INTERVALS = (None, 2.0, 1.0, 0.5)      # metres driven between two looks, None = never look
LOOK_POS_ERROR = 0.05                       # standard deviation in x and in z of a measured position (m)
LOOK_ANGLE_ERROR = math.radians(3)          # standard deviation of a measured heading (rad)

# Tracks: centre line and boxes in metres, x = right, z = forward.
# MIRTE starts at the first centre-line point, facing along the track,
# and the goal is the last centre-line point.
TRACKS = {
    "chicane": dict(width=1.6, centre=[(0, 0), (0, 6)],
                    boxes=[(-0.35, 1.2), (0.35, 2.4), (-0.35, 3.6), (0.35, 4.8)]),
    "bends":   dict(width=1.6, centre=[(0, 0), (0, 3), (3, 3), (3, 6)],
                    boxes=[(-0.3, 1.5), (1.5, 2.7), (3.3, 4.5)]),
    "rally":   dict(width=1.6, centre=[(0, 0), (0, 2.5), (2, 4.5), (2, 7), (0, 9)],
                    boxes=[(-0.3, 1.3), (0.8, 3.7), (2.3, 5.8), (0.8, 7.8)]),
}


# ---------------------------------------------------------------------------
# Track map
# ---------------------------------------------------------------------------

class TrackMap:
    """A track with boxes. Has the map_area and in_collision() that the planners use."""

    def __init__(self, track, margin=0.0):
        defaults = local_map.LocalMap()
        self.no_go = defaults.landmark_radius + defaults.mirte_radius   # as in LocalMap
        self.mirte_radius = defaults.mirte_radius
        self.centre = np.array(track["centre"], dtype=float)
        self.half_width = track["width"] / 2.0
        self.boxes = np.array(track["boxes"], dtype=float).reshape(-1, 2)
        self.margin = margin
        self.start, self.goal = self.centre[0], self.centre[-1]
        first = self.centre[1] - self.centre[0]
        self.start_heading = math.atan2(first[1], first[0])
        self.map_area = [self.centre.min(axis=0) - self.half_width,
                         self.centre.max(axis=0) + self.half_width]

    def _distances(self, pts):
        """Distance from each point to the centre line and to the nearest box centre."""
        pts = np.atleast_2d(pts)
        a, ab = self.centre[:-1], np.diff(self.centre, axis=0)
        t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(axis=2) / (ab * ab).sum(axis=1)[None], 0.0, 1.0)
        d_centre = np.linalg.norm(pts[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=2).min(axis=1)
        d_box = (np.linalg.norm(pts[:, None, :] - self.boxes[None], axis=2).min(axis=1)
                 if len(self.boxes) else np.full(len(pts), np.inf))
        return d_centre, d_box

    def clearance(self, pts):
        """Distance (m) from each point to where MIRTE's centre must not be:
        a box's no-go zone, or closer than the robot radius to the track edge."""
        d_centre, d_box = self._distances(pts)
        return np.minimum(self.half_width - self.mirte_radius - d_centre, d_box - self.no_go)

    def in_collision(self, pos):
        return int(self.clearance(pos)[0] < self.margin)

    def free(self, a, b, margin=None, step=0.02):
        """True if the straight line a -> b keeps the margin (the planning margin if none is given)."""
        n = max(2, int(math.ceil(np.linalg.norm(b - a) / step)) + 1)
        pts = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
        return bool(np.all(self.clearance(pts) >= (self.margin if margin is None else margin)))

    def crashed(self, pts):
        """True where the robot body touches a box or sticks out over the track edge."""
        d_centre, d_box = self._distances(pts)
        return (d_box < ROBOT_RADIUS + BOX_CONTACT_RADIUS) | (d_centre > self.half_width - ROBOT_RADIUS)


# ---------------------------------------------------------------------------
# Step 1: global planners. Each returns a path from start to goal or None.
# ---------------------------------------------------------------------------

def plan_birrt(m):
    """Bidirectional RRT: one tree from the start, one from the goal."""
    low, high = m.map_area

    class Tree:
        def __init__(self, root):
            self.pts = np.empty((2 * RRT_MAX_ITER + 2, 2))
            self.parent = [None]
            self.pts[0] = root

        def extend(self, x):
            """One step from the nearest vertex towards x: 'trapped', 'advanced' or 'reached'."""
            n = len(self.parent)
            near = int(np.argmin(np.linalg.norm(self.pts[:n] - x, axis=1)))
            d = np.linalg.norm(x - self.pts[near])
            new = x if d <= RRT_STEP else self.pts[near] + (x - self.pts[near]) * RRT_STEP / d
            if not m.free(self.pts[near], new):
                return "trapped"
            self.pts[n] = new
            self.parent.append(near)
            return "reached" if d <= RRT_STEP else "advanced"

        def branch(self):
            """Points from the newest vertex back to the root."""
            out, i = [], len(self.parent) - 1
            while i is not None:
                out.append(self.pts[i].copy())
                i = self.parent[i]
            return out

    tree_a, tree_b = Tree(m.start), Tree(m.goal)
    a_is_start = True
    for _ in range(RRT_MAX_ITER):
        if len(tree_a.parent) + len(tree_b.parent) > 2 * RRT_MAX_ITER:
            break
        if tree_a.extend(np.random.uniform(low, high)) != "trapped":
            x_new = tree_a.pts[len(tree_a.parent) - 1]
            status = "advanced"
            while status == "advanced":                    # let the other tree grow towards it
                status = tree_b.extend(x_new)
            if status == "reached":
                path = tree_a.branch()[::-1] + tree_b.branch()[1:]
                return path if a_is_start else path[::-1]
        tree_a, tree_b = tree_b, tree_a
        a_is_start = not a_is_start
    return None


def plan_astar(m):
    """A* on an 8-connected grid."""
    low = np.asarray(m.map_area[0])
    nx, nz = [int(round(v)) + 1 for v in (np.asarray(m.map_area[1]) - low) / GRID_RES]
    ii, jj = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    centres = low + np.stack([ii, jj], axis=-1) * GRID_RES
    free = (m.clearance(centres.reshape(-1, 2)) >= m.margin).reshape(nx, nz)

    def cell(p):
        return tuple(int(round(v)) for v in (np.asarray(p) - low) / GRID_RES)

    s, g = cell(m.start), cell(m.goal)
    if not (free[s] and free[g]):
        return None
    moves = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
    cost, parent, closed = {s: 0.0}, {s: None}, set()
    open_list = [(0.0, s)]
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
                heapq.heappush(open_list, (new_cost + GRID_RES * math.hypot(n[0] - g[0], n[1] - g[1]), n))
    if g not in closed:
        return None
    cells, c = [], g
    while c is not None:
        cells.append(c)
        c = parent[c]
    return [m.start] + [centres[c] for c in cells[::-1][1:-1]] + [m.goal]


# ---------------------------------------------------------------------------
# Step 2: shortcut smoothing
# ---------------------------------------------------------------------------

def shortcut(path, m):
    """From each point, jump to the furthest later point with a free straight line."""
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not m.free(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


# ---------------------------------------------------------------------------
# Step 3: Dubins-style smoothing (straight legs joined by arcs), as drive commands
# ---------------------------------------------------------------------------
#
# A command is (linear speed m/s, angular speed rad/s, seconds, ahead). The first
# three are the arguments of mirte.drive(); angular speed > 0 turns left. 'ahead'
# is not for the robot: it lists the path points MIRTE has not passed yet, which
# is what is left of the route if MIRTE looks where it is during that command.

def add_command(commands, dist, turn, ahead):
    """Append the command that drives dist metres while turning by turn radians,
    as fast as the two speed limits allow. dist = 0 turns on the spot."""
    seconds = max(dist / LINEAR_SPEED, abs(turn) / ANGULAR_SPEED)
    if seconds > 1e-9:
        commands.append((dist / seconds, turn / seconds, seconds, ahead))


def arc_points(pos, heading, turn, r):
    """Points 2 cm apart on the arc of radius r that starts at pos along heading and turns by turn radians."""
    centre = pos + math.copysign(r, turn) * np.array([-math.sin(heading), math.cos(heading)])
    n = max(3, int(math.ceil(r * abs(turn) / 0.02)) + 1)
    angles = heading - math.copysign(math.pi / 2, turn) + np.linspace(0.0, turn, n)
    return centre + r * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def turn_towards(pos, heading, target, r):
    """How far MIRTE must turn on an arc of radius r before it can drive straight to target:
    the 'turn, then straight' start of a Dubins curve [Toussaint, slide 74].
    None if the target lies inside the turning circle or needs more than half a turn."""
    to = target - pos
    ahead = to[0] * math.cos(heading) + to[1] * math.sin(heading)
    side = to[1] * math.cos(heading) - to[0] * math.sin(heading)       # > 0: target is to the left
    root = ahead ** 2 + side ** 2 - 2 * r * abs(side)
    if root < 0 or ahead + math.sqrt(root) <= 0:
        return None
    return math.copysign(2 * math.atan2(abs(side), ahead + math.sqrt(root)), side)


def set_off(pos, heading, route, m, radius):
    """The turn that takes MIRTE from its pose towards the route.

    MIRTE aims for the furthest route point it can drive straight to with the
    planning margin. If there is none it keeps going for the next route point.
    With radius > 0 it makes the turn while driving, on an arc that starts along
    its heading. For the next route point a tighter arc is tried if that one does
    not fit. With radius = 0, or if no arc fits, it turns on the spot.

    Returns (commands, path): the path starts where the turn ends and continues
    along the route, with MIRTE facing the second path point.
    """
    commands = []
    for j in range(len(route) - 1, -1, -1):                # furthest route point first
        r = radius
        while r > MIN_RADIUS:                              # turn while driving
            turn = turn_towards(pos, heading, route[j], r)
            if turn is not None:
                arc = arc_points(pos, heading, turn, r)
                clear = m.clearance(arc)
                slack = min(ARC_MARGIN, clear[0])          # MIRTE may already be closer than an arc should be
                if np.all(clear >= slack) and m.free(arc[-1], route[j], m.margin if j else slack):
                    add_command(commands, r * abs(turn), turn, route[j:])
                    return commands, [arc[-1]] + route[j:]
            r = r * 0.7 if j == 0 else 0.0                 # tighter arcs only for the next route point
        if j == 0 or (radius == 0 and m.free(pos, route[j])):   # turn on the spot
            to = route[j] - pos
            turn = math.atan2(to[1], to[0]) - heading
            add_command(commands, 0.0, math.atan2(math.sin(turn), math.cos(turn)), route[j:])
            return commands, [pos] + route[j:]


def round_corners(path, m, radius):
    """Drive commands for the path, with each corner replaced by an arc of the given radius.
    MIRTE stands at the first path point and faces the second.

    A corner gets a smaller radius if its two legs are too short for the full one,
    or if the full arc would come closer than ARC_MARGIN to a box or the track edge.
    radius = 0 keeps every corner sharp: MIRTE stops there and turns on the spot.
    """
    pts = []
    for p in path:
        p = np.asarray(p, dtype=float)
        if not pts or np.linalg.norm(p - pts[-1]) > 1e-9:               # a repeated point is not a corner
            pts.append(p)
    legs = [b - a for a, b in zip(pts[:-1], pts[1:])]
    lens = [float(np.linalg.norm(v)) for v in legs]
    units = [v / l for v, l in zip(legs, lens)]
    last = len(legs) - 1
    # a leg between two corners gives half its length to each of them
    room = [l if i in (0, last) else l / 2 for i, l in enumerate(lens)]

    commands, cursor = [], pts[0]
    for c in range(1, len(pts) - 1):
        u, w = units[c - 1], units[c]
        turn = math.atan2(u[0] * w[1] - u[1] * w[0], float(u @ w))       # > 0: left turn
        if abs(turn) < 1e-6:
            continue
        tan_half = math.tan(abs(turn) / 2)
        r = min(radius, min(room[c - 1], room[c]) / tan_half)
        arc = None
        while r > MIN_RADIUS:
            start = pts[c] - u * r * tan_half                            # where the arc leaves the first leg
            arc = arc_points(start, math.atan2(u[1], u[0]), turn, r)
            if np.all(m.clearance(arc) >= ARC_MARGIN):
                break
            r, arc = r * 0.7, None
        if arc is None:                                                  # sharp corner: stop and turn
            add_command(commands, np.linalg.norm(pts[c] - cursor), 0.0, pts[c:])
            add_command(commands, 0.0, turn, pts[c + 1:])
            cursor = pts[c]
        else:
            add_command(commands, np.linalg.norm(arc[0] - cursor), 0.0, pts[c:])
            add_command(commands, r * abs(turn), turn, pts[c + 1:])
            cursor = arc[-1]
    add_command(commands, np.linalg.norm(pts[-1] - cursor), 0.0, pts[-1:])
    return commands


def drive_commands(pos, heading, route, m, radius):
    """All commands that take MIRTE from its pose along the route to the goal."""
    commands, path = set_off(pos, heading, route, m, radius)
    return commands + round_corners(path, m, radius)


def first_part(commands, dist):
    """The commands for the first dist metres, and the path points still ahead of MIRTE after them."""
    head = []
    for lin, ang, seconds, ahead in commands:
        if lin * seconds >= dist:
            return head + [(lin, ang, dist / lin, ahead)], ahead
        head.append((lin, ang, seconds, ahead))
        dist -= lin * seconds


# ---------------------------------------------------------------------------
# Driving the commands with noisy motion
# ---------------------------------------------------------------------------

def drive(commands, pos, heading, errors=(0.0, 0.0, 0.0, 0.0)):
    """Where MIRTE really goes when it carries out the commands from a pose.

    errors = (e_rot, e_curve, e_trans, e_slip) are the noise terms of the odometry
    motion model (slide 6):
      turn error      = e_rot * turn            e_rot   ~ N(0, ALPHA1)
      heading drift   = e_curve * distance      e_curve ~ N(0, ALPHA2)
      distance error  = e_trans * distance      e_trans ~ N(0, ALPHA3)
                      + e_slip * |turn|         e_slip  ~ N(0, ALPHA4)
    For a single turn-and-drive leg this gives the variances on the slide. Each error
    is drawn once per simulated drive and then applies to the whole drive. Drawing
    them per command instead would make an arc, which is many short steps, look
    more accurate than a sharp corner. Without errors MIRTE follows the commands exactly.

    Returns the positions it passes, about 2 cm apart, and its final heading.
    """
    e_rot, e_curve, e_trans, e_slip = errors
    steps = [max(1, int(math.ceil(lin * seconds / 0.02))) for lin, _, seconds, _ in commands]
    dist = np.repeat([lin * seconds / k for (lin, _, seconds, _), k in zip(commands, steps)], steps)
    turn = np.repeat([ang * seconds / k for (_, ang, seconds, _), k in zip(commands, steps)], steps)
    dist = dist * (1 + e_trans) + np.abs(turn) * e_slip
    turn = turn * (1 + e_rot) + e_curve * dist
    theta = heading + np.cumsum(turn)
    middle = theta - turn / 2                              # heading half-way through each step
    points = pos + np.cumsum(dist[:, None] * np.stack([np.cos(middle), np.sin(middle)], axis=1), axis=0)
    return points, float(theta[-1])


def simulate_drives(route, m, radius, look, n, rng):
    """Drive the route n times with noisy motion, looking where MIRTE is every `look` metres.

    Between two looks MIRTE drives blind: it carries out its commands and the motion
    errors add up. At a look it measures its pose with a small error (LOOK_POS_ERROR,
    LOOK_ANGLE_ERROR) and makes new commands from that pose for the route points it
    has not passed. It does not look on a last stretch shorter than 1.5 * look, and
    with look = None it never looks.

    Returns one row per drive: (crashed, distance from the goal at the end, seconds, stops).
    A stop is a turn on the spot along the way; the turn before setting off is not counted.
    """
    first = drive_commands(m.start, m.start_heading, route, m, radius)
    rows = []
    for _ in range(n):
        errors = rng.normal(0.0, np.sqrt([ALPHA1, ALPHA2, ALPHA3, ALPHA4]))
        pos, heading = m.start, m.start_heading            # where MIRTE really is
        commands, seconds = first, 0.0
        stops = -int(first[0][0] == 0)                     # the turn before setting off is not a stop
        while True:
            ahead = None                                   # route points left after this stretch, if any
            if look and sum(c[0] * c[2] for c in commands) > 1.5 * look:
                commands, ahead = first_part(commands, look)
            points, heading = drive(commands, pos, heading, errors)
            pos = points[-1]
            seconds += sum(c[2] for c in commands)
            stops += sum(c[0] == 0 for c in commands)
            crashed = bool(m.crashed(points).any())
            if crashed or ahead is None:
                break
            commands = drive_commands(pos + rng.normal(0.0, LOOK_POS_ERROR, 2),
                                      heading + rng.normal(0.0, LOOK_ANGLE_ERROR), ahead, m, radius)
        rows.append((crashed, float(np.linalg.norm(pos - m.goal)), seconds, stops))
    return np.array(rows)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

METHODS = {
    # name:                 (planner, plans with margin, corner radius, uses random numbers)
    "rrt, stop and turn":   (plan_birrt, False, 0.0, True),
    "rrt + margin, stop":   (plan_birrt, True, 0.0, True),
    "rrt + margin, arcs":   (plan_birrt, True, TURN_RADIUS, True),
    "astar + margin, arcs": (plan_astar, True, TURN_RADIUS, False),
}


def evaluate(track, method, margin, runs, rng):
    """Plan `runs` paths and drive them. Returns the numbers for the tables and one planned path to plot."""
    planner, with_margin, radius, random = METHODS[method]
    m = TrackMap(track, margin if with_margin else 0.0)
    routes, plans, ms = [], [], []
    for _ in range(runs if random else 1):
        t0 = time.perf_counter()
        path = planner(m)
        if path is not None:
            routes.append(shortcut(path, m)[1:])
            plans.append(drive_commands(m.start, m.start_heading, routes[-1], m, radius))
        ms.append((time.perf_counter() - t0) * 1000.0)
    out = dict(found=len(plans) / len(ms), ms=float(np.mean(ms)))
    if not plans:
        return out, None

    traces = [np.vstack([m.start, drive(commands, m.start, m.start_heading)[0]]) for commands in plans]
    out["length"] = np.mean([sum(c[0] * c[2] for c in commands) for commands in plans])
    out["time"] = np.mean([sum(c[2] for c in commands) for commands in plans])
    out["stops"] = np.mean([sum(c[0] == 0 for c in commands[1:]) for commands in plans])
    out["slow"] = np.mean([sum(0 < c[0] < LINEAR_SPEED - 1e-6 for c in commands) for commands in plans])
    out["clear"] = np.mean([m.clearance(trace).min() for trace in traces])

    out["drives"] = {}
    per_route = max(1, MOTION_SAMPLES // len(routes))
    for look in LOOK_INTERVALS:
        rows = np.vstack([simulate_drives(route, m, radius, look, per_route, rng) for route in routes])
        done = rows[rows[:, 0] == 0]
        out["drives"][look] = dict(crash=rows[:, 0].mean(),
                                   end_err=np.median(done[:, 1]) if len(done) else float("nan"),
                                   time=done[:, 2].mean() if len(done) else float("nan"),
                                   stops=done[:, 3].mean() if len(done) else float("nan"))
    return out, traces[0]


def look_name(look):
    return "never" if look is None else f"{look:g} m"


def save_plot(examples, filename):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    fig, axes = plt.subplots(1, len(examples), figsize=(5.0 * len(examples), 7.5))
    for ax, (name, traces) in zip(np.atleast_1d(axes), examples.items()):
        m = TrackMap(TRACKS[name])
        for a, b in zip(m.centre[:-1], m.centre[1:]):                    # the track at its true width
            side = np.array([a[1] - b[1], b[0] - a[0]]) * m.half_width / np.linalg.norm(b - a)
            ax.add_patch(Polygon([a - side, a + side, b + side, b - side], color="0.88", zorder=0))
        for p in m.centre:
            ax.add_patch(Circle(p, m.half_width, color="0.88", zorder=0))
        for b in m.boxes:
            ax.add_patch(Circle(b, m.no_go, color="red", alpha=0.12))
            ax.add_patch(plt.Rectangle(b - 0.125, 0.25, 0.25, color="0.3"))
        for method, trace in traces.items():
            if trace is not None:
                ax.plot(trace[:, 0], trace[:, 1], linewidth=1.6, label=method)
        ax.plot(*m.start, "k^")
        ax.plot(*m.goal, "kx")
        ax.set_xlim(m.map_area[0][0], m.map_area[1][0])
        ax.set_ylim(m.map_area[0][1], m.map_area[1][1])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(name)
        ax.set_xlabel("x (m)")
    np.atleast_1d(axes)[0].set_ylabel("z (m)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9)
    fig.savefig(filename, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to: {filename}")


def main():
    ap = argparse.ArgumentParser(description="Compare path planning methods for MIRTE on a track.")
    ap.add_argument("--runs", type=int, default=20, help="planned paths per track for the RRT methods")
    ap.add_argument("--margin", type=float, default=PLAN_MARGIN, help="extra clearance when planning (m)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", metavar="FILE", help="save one example path per method to this PNG")
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    examples, results = {}, {}

    print(f"turn radius at full speed {TURN_RADIUS:.2f} m, planning margin {args.margin:.2f} m, "
          f"{args.runs} planned paths for RRT methods,\n"
          f"{MOTION_SAMPLES} simulated drives per method, track and look interval. "
          f"A look measures the pose to {LOOK_POS_ERROR * 100:.0f} cm and "
          f"{math.degrees(LOOK_ANGLE_ERROR):.0f} degrees.\n"
          f"The motion noise and the measuring error are guesses: compare the methods, do not read the "
          f"crash rates as predictions.")
    for name, track in TRACKS.items():
        examples[name] = {}
        for method in METHODS:
            results[name, method], examples[name][method] = evaluate(track, method, args.margin, args.runs, rng)

        print(f"\n{name}: {len(track['boxes'])} boxes, width {track['width']} m")
        print(f"  the planned paths\n  {'method':21s} {'found':>6s} {'plan ms':>8s} {'length':>7s} {'stops':>6s} "
              f"{'slow arcs':>10s} {'drive s':>8s} {'clearance':>10s}")
        for method in METHODS:
            r = results[name, method]
            if r["found"] == 0:
                print(f"  {method:21s} {r['found']:6.0%} {r['ms']:8.0f}   no path found")
                continue
            print(f"  {method:21s} {r['found']:6.0%} {r['ms']:8.0f} {r['length']:6.2f}m {r['stops']:6.1f} "
                  f"{r['slow']:10.1f} {r['time']:8.1f} {r['clear'] * 100:7.1f} cm")
        print(f"  driven with noisy motion (end error, stops and drive s are of the drives that reach the goal)\n"
              f"  {'method':21s} {'look every':>10s} {'crashes':>8s} {'end error':>10s} {'stops':>6s} {'drive s':>8s}")
        for method in METHODS:
            for i, (look, d) in enumerate(results[name, method].get("drives", {}).items()):
                print(f"  {'' if i else method:21s} {look_name(look):>10s} {d['crash']:8.1%} "
                      f"{d['end_err'] * 100:7.1f} cm {d['stops']:6.1f} {d['time']:8.1f}")

    print("\nmean of the tracks: crashes (drive s)\n  " + f"{'method':21s}"
          + "".join(f" {'look every ' + look_name(look):>18s}" for look in LOOK_INTERVALS))
    for method in METHODS:
        missing = [name for name in TRACKS if "drives" not in results[name, method]]
        if missing:
            print(f"  {method:21s} no path found on: {', '.join(missing)}")
            continue
        drives = [results[name, method]["drives"] for name in TRACKS]
        print(f"  {method:21s}" + "".join(
            f" {np.mean([d[look]['crash'] for d in drives]):11.1%}"
            f" ({np.nanmean([d[look]['time'] for d in drives]):4.1f})" for look in LOOK_INTERVALS))

    if args.plot:
        save_plot(examples, args.plot)


if __name__ == "__main__":
    main()
