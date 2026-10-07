"""Plots of one MIRTE self-localisation run (Exercise 5).

selflocalize.py fills a `log` dict while it runs (see LOG FORMAT) and saves it with save_log().
save_run_plots() turns it into one PNG that shows what the particle filter believed at each stop
and which drive decisions it made. Re-plot a saved run with:

    python run_plots.py path/to/run.pkl [out.png]

LOG FORMAT (positions in cm, angles in rad, heading 0 = +x, counter-clockwise positive):
    log = {
        "landmarks": {4: (0.0, 0.0), 2: (121.0, 0.0)},   # id -> marker position (front face of the box)
        "goal": (60.5, 0.0),
        "camera_offset": 14.0,                           # camera is this far in front of the robot centre
        "true_start": None or (x, y, theta),             # optional, e.g. tape-measured
        "true_end": None or (x, y, theta),               # optional
        "steps": [{                                      # one per decision stop, in order
            "time": 12.3,                                # s since start
            "phase": "scan" | "look" | "drive" | "done" | "failed",
            "particles": (N, 3) array of x, y, theta     # (or a list of particle.Particle)
            "estimate": (x, y, theta),                   # (or a particle.Particle)
            "z": {id: (dist_cm, angle_rad)},             # camera -> marker, angle positive = to the left
            "move": (turn_rad, dist_cm),                 # commanded after this stop: turn, then drive straight
        }, ...],
    }
"""
import math
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BOX_SIZE = 29.0     # cm, boxes are 29 x 29 cm, the marker sits on the face at y = 0
PHASE_COLOURS = {"scan": "tab:orange", "look": "tab:cyan", "drive": "tab:blue", "done": "tab:green",
                 "failed": "tab:red"}
LANDMARK_COLOURS = ["tab:red", "tab:green", "tab:purple", "tab:brown"]
M = 0.01            # cm -> m, the maps are drawn in metres


def save_log(log, path):
    with open(path, "wb") as f:
        pickle.dump(log, f)


def load_log(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Reading the log (tolerant of missing keys and particle.Particle objects)
# ---------------------------------------------------------------------------

def as_pose(p):
    """(x, y, theta) as floats, from a tuple/array or a particle.Particle. None stays None."""
    if p is None:
        return None
    if hasattr(p, "getX"):
        return float(p.getX()), float(p.getY()), float(p.getTheta())
    return float(p[0]), float(p[1]), float(p[2])


def as_particles(ps):
    """(N, 3) array of x, y, theta, from an array or a list of particle.Particle."""
    if ps is None or len(ps) == 0:
        return np.zeros((0, 3))
    if hasattr(ps[0], "getX"):
        return np.array([as_pose(p) for p in ps])
    return np.asarray(ps, dtype=float).reshape(-1, 3)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def clean_steps(log):
    """The steps with every field present and in a fixed type."""
    steps = []
    for s in log.get("steps", []):
        est = as_pose(s.get("estimate"))
        parts = as_particles(s.get("particles"))
        if est is None:     # no estimate logged: use the particle mean
            est = (*parts[:, :2].mean(axis=0), math.atan2(np.sin(parts[:, 2]).sum(), np.cos(parts[:, 2]).sum()))
        spread = float(np.median(np.hypot(parts[:, 0] - est[0], parts[:, 1] - est[1]))) if len(parts) else np.nan
        steps.append(dict(time=s.get("time", np.nan), phase=s.get("phase", "?"), particles=parts, estimate=est,
                          z=dict(s.get("z") or {}), move=tuple(s.get("move") or (0.0, 0.0)), spread=spread))
    return steps


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_world(ax, log, title):
    for marker_id, (x, y) in log["landmarks"].items():
        ax.add_patch(plt.Rectangle(((x - BOX_SIZE / 2) * M, y * M), BOX_SIZE * M, BOX_SIZE * M, color="0.6"))
        ax.text(x * M, (y + BOX_SIZE + 4) * M, f"box {marker_id}", ha="center", fontsize=8)
    gx, gy = log["goal"]
    ax.plot(gx * M, gy * M, "x", color="green", markersize=11, markeredgewidth=2.5, label="goal (midpoint)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_title(title, fontsize=10)


def draw_pose(ax, pose, colour, label, length=0.15, marker="o"):
    x, y, th = pose
    ax.plot(x * M, y * M, marker, color=colour, markersize=7, label=label)
    ax.plot([x * M, x * M + length * math.cos(th)], [y * M, y * M + length * math.sin(th)], color=colour, linewidth=2)


def draw_particles(ax, parts, label):
    if len(parts):
        ax.scatter(parts[:, 0] * M, parts[:, 1] * M, s=3, color="tab:blue", alpha=0.25, linewidths=0, label=label)


def draw_rays(ax, log, step):
    """Measured landmarks, drawn from the estimated camera position: they should end on the markers."""
    x, y, th = step["estimate"]
    off = log.get("camera_offset", 0.0)
    cx, cy = x + off * math.cos(th), y + off * math.sin(th)
    ids = list(log["landmarks"])
    for marker_id, (dist, angle) in sorted(step["z"].items()):
        colour = LANDMARK_COLOURS[ids.index(marker_id) % 4] if marker_id in ids else "black"
        ex, ey = cx + dist * math.cos(th + angle), cy + dist * math.sin(th + angle)
        ax.plot([cx * M, ex * M], [cy * M, ey * M], "--", color=colour, linewidth=1.3,
                label=f"seen box {marker_id}: {dist:.0f} cm, {math.degrees(angle):+.0f}°")
        ax.plot(ex * M, ey * M, "o", color=colour, markersize=5, markerfacecolor="none")
    if not step["z"]:
        ax.text(0.02, 0.02, "no landmarks seen at this stop", transform=ax.transAxes, fontsize=8, color="0.3")


def pose_error(true, est):
    return math.hypot(true[0] - est[0], true[1] - est[1]), math.degrees(wrap(true[2] - est[2]))


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

def save_run_plots(log, filename):
    """One PNG: first localisation, decisions along the way, final pose, and a per-stop timeline."""
    log = dict(log)
    log.setdefault("landmarks", {4: (0.0, 0.0), 2: (121.0, 0.0)})
    log.setdefault("goal", tuple(np.mean(list(log["landmarks"].values()), axis=0)))
    steps = clean_steps(log)
    if not steps:
        print("run_plots: the log has no steps, nothing to plot")
        return
    true_start, true_end = as_pose(log.get("true_start")), as_pose(log.get("true_end"))
    first = next((s for s in steps if s["phase"] == "drive"), steps[-1])
    last = steps[-1]

    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(2, 3, height_ratios=[2.2, 1], hspace=0.45, wspace=0.2)
    maps = [fig.add_subplot(grid[0, i]) for i in range(3)]

    # 1. first localisation: the cloud and estimate the first drive decision was based on
    ax = maps[0]
    draw_world(ax, log, f"1. First localisation (stop {steps.index(first) + 1}, {first['phase']})")
    draw_particles(ax, first["particles"], "particles")
    draw_rays(ax, log, first)
    draw_pose(ax, first["estimate"], "magenta", "estimate")
    if true_start is not None:
        draw_pose(ax, true_start, "red", "true start", marker="s")

    # 2. every stop's estimate, coloured by the decision made there, and the commanded moves
    ax = maps[1]
    draw_world(ax, log, "2. Decisions along the way (estimate at each stop)")
    est = np.array([s["estimate"] for s in steps])
    ax.plot(est[:, 0] * M, est[:, 1] * M, ":", color="0.5", linewidth=1, zorder=1, label="estimate path")
    label_at = None
    for i, s in enumerate(steps):
        x, y, th = s["estimate"]
        colour = PHASE_COLOURS.get(s["phase"], "black")
        ax.plot(x * M, y * M, "o", color=colour, markersize=7, zorder=3,
                label=s["phase"] if s["phase"] not in [t["phase"] for t in steps[:i]] else None)
        ax.plot([x * M, (x + 10 * math.cos(th)) * M], [y * M, (y + 10 * math.sin(th)) * M], color=colour, lw=2)
        turn, dist = s["move"]
        new_th = th + turn
        if abs(turn) > 0.05:     # turn on the spot: small arc from the old to the new heading
            a = np.linspace(th, new_th, 20)
            ax.plot((x + 7 * np.cos(a)) * M, (y + 7 * np.sin(a)) * M, color="black", linewidth=1)
        if dist > 0:
            ax.annotate("", xy=((x + dist * math.cos(new_th)) * M, (y + dist * math.sin(new_th)) * M),
                        xytext=(x * M, y * M), zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, shrinkA=0, shrinkB=0))
        # stop numbers; stops at the same spot (scanning) share one label "a-b"
        if label_at is None or math.hypot(x - label_at[1], y - label_at[2]) > 5:
            label_at = [ax.text((x + 4) * M, (y - 4) * M, f"{i + 1}", fontsize=8, va="top"), x, y, i + 1]
        else:
            label_at[0].set_text(f"{label_at[3]}-{i + 1}")
    ax.plot([], [], color="black", marker=r"$\rightarrow$", linestyle="none", label="commanded move")
    if true_start is not None:
        draw_pose(ax, true_start, "red", "true start", marker="s")

    # 3. final
    ax = maps[2]
    gx, gy = log["goal"]
    fx, fy, _ = last["estimate"]
    title = f"3. Final (stop {len(steps)}, {last['phase']}): estimate {math.hypot(fx - gx, fy - gy):.1f} cm from goal"
    if true_end is not None:
        d, a = pose_error(true_end, last["estimate"])
        title += f"\ntrue error {d:.1f} cm, {a:+.0f}°;  true end {math.hypot(true_end[0] - gx, true_end[1] - gy):.1f} cm from goal"
    draw_world(ax, log, title)
    draw_particles(ax, last["particles"], "particles")
    draw_pose(ax, last["estimate"], "magenta", "final estimate")
    if true_end is not None:
        draw_pose(ax, true_end, "red", "true end", marker="s")

    # the same view in all three maps: boxes, goal, estimates, true poses and the bulk of the clouds
    pts = [np.array(list(log["landmarks"].values())) + [0, BOX_SIZE + 8], est[:, :2], np.array([log["goal"]])]
    pts += [np.array([p[:2]]) for p in (true_start, true_end) if p is not None]
    for s in (first, last):
        if len(s["particles"]):
            pts.append(np.percentile(s["particles"][:, :2], [1, 99], axis=0))
    pts = np.vstack(pts) * M
    lo, hi = pts.min(axis=0) - 0.15, pts.max(axis=0) + 0.15
    for ax in maps:
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)    # below: never covers data
    maps[0].set_ylabel("y (m)")

    # 4. timeline per stop, background = phase
    n = np.arange(1, len(steps) + 1)
    heading = np.degrees(np.unwrap(wrap(est[:, 2])))    # no jumps at +-180 deg
    lines = [fig.add_subplot(grid[1, i]) for i in range(3)]
    lines[0].plot(n, est[:, 0], "o-", label="x")
    lines[0].plot(n, est[:, 1], "s-", label="y")
    lines[0].set_ylabel("estimate (cm)")
    lines[0].legend(fontsize=8)
    lines[1].plot(n, heading, "o-", color="tab:purple")
    lines[1].set_ylabel("heading (deg)")
    lines[2].plot(n, [s["spread"] for s in steps], "o-", color="tab:blue", label="particle spread")
    lines[2].set_ylabel("median particle distance\nto estimate (cm)")
    seen = lines[2].twinx()
    seen.bar(n, [len(s["z"]) for s in steps], width=0.4, color="0.3", alpha=0.3)
    seen.set_ylabel("landmarks seen (bars)")
    seen.set_ylim(0, max(2, len(log["landmarks"])) * 2.5)     # keep the bars in the lower part
    seen.set_yticks(range(len(log["landmarks"]) + 1))
    for ax in lines:
        for i, s in enumerate(steps):
            ax.axvspan(i + 0.5, i + 1.5, color=PHASE_COLOURS.get(s["phase"], "white"), alpha=0.12, linewidth=0)
        ax.set_xlim(0.5, len(steps) + 0.5)
        ax.set_xlabel("stop")
        ax.set_xticks(n[::max(1, len(n) // 15)])
        ax.grid(True, alpha=0.3)
    lines[1].set_title("Timeline (background colour = decision at that stop, as in panel 2)", fontsize=10)

    fig.savefig(filename, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved run plots to: {filename}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python run_plots.py path/to/run.pkl [out.png]")
        sys.exit(1)
    pkl = sys.argv[1]
    save_run_plots(load_log(pkl), sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(pkl)[0] + ".png")
