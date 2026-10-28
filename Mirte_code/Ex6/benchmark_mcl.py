#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import cProfile
import csv
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time
import traceback
import warnings

import numpy as np

BASE_DIR = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description="Offline profiler for MIRTE's MCL")
    parser.add_argument("--particles", type=int, default=1000,
                        help="Antal partikler, normalt 1000")
    parser.add_argument("--repeats", type=int, default=2,
                        help="Gentagelser per scenarie, normalt 2")
    parser.add_argument("--seed", type=int, default=20261028,
                        help="Seed for sammenlignelige testdata")
    parser.add_argument("--module", default="mcl",
                        help="Modul til test, normalt mcl (senere mcl_fast)")
    parser.add_argument("--output", default=str(BASE_DIR / "mcl_benchmark_results"),
                        help="Overordnet mappe til resultater")
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction,
                        default=True, help="cProfile af en drejning og en observation")
    parser.add_argument("--top", type=int, default=35,
                        help="Antal funktioner i profileringsrapporten")
    args = parser.parse_args()
    if args.particles <= 0 or args.repeats <= 0:
        parser.error("--particles og --repeats skal vaere positive")
    if args.top <= 0:
        parser.error("--top skal vaere positiv")
    return args


def make_map():
    # Samme globale maalestok (centimeter) og sikkerhedsafstande som Ex6.
    from local_map import LocalMap
    landmarks = [
        [np.array([0.0, 0.0]), 60.0, 6],
        [np.array([0.0, 300.0]), 60.0, 3],
        [np.array([400.0, 0.0]), 20.0, 1],
        [np.array([400.0, 300.0]), 20.0, 4],
    ]
    # Et lille antal syntetiske forhindringer, saa kollisionstjek er repraesentative.
    obstacles = [
        [np.array([160.0, 200.0]), 7],
        [np.array([275.0, 255.0]), 8],
    ]
    return LocalMap(
        landmarks=landmarks, obstacles=obstacles,
        low=(-120.0, -120.0), high=(420.0, 520.0),
        landmark_radius=20, mirte_radius=22, obstacale_radius=20,
    )


def make_prior(count, seed, m, localized):
    import particle as pcl
    np.random.seed(seed)
    if not localized:
        # Bruger samme uniform init som brugerens originale main.py.
        return pcl.initialize_particles(count, m)

    # Samlet prior efter en vellykket delvis lokalisering.
    # Kun benchmarkdata, ikke simuleret ground truth.
    x = np.random.normal(300.0, 25.0, count)
    y = np.random.normal(125.0, 25.0, count)
    theta = np.random.normal(0.7, 0.15, count)
    return [pcl.Particle(float(xi), float(yi), float(ti), 1.0 / count)
            for xi, yi, ti in zip(x, y, theta)]


def make_measurements(ids, m):
    # Observationer, der er matematisk konsistente med en fast referencepose.
    # De tvinges ikke gennem fysisk kamera/ROS og udtrykker ikke virkelig sensor-FOV.
    import particle as pcl
    reference = pcl.Particle(300.0, 125.0, 0.7)
    positions = {lm_id: pos for pos, _, lm_id in m.landmarks}
    return {lm_id: tuple(float(v) for v in pcl.expected_measurement(
        reference, positions[lm_id], camera_offset=0.0)) for lm_id in ids}


def scenarios(m):
    z1 = make_measurements((4,), m)
    z2 = make_measurements((4, 1), m)
    return [
        {"name": "01_stille_uden_landmarks", "u": (0.0, 0.0), "z": {}, "localized": False},
        {"name": "02_drejning_uden_landmarks", "u": (0.0, math.radians(40)),
         "z": {}, "localized": False},
        {"name": "03_koersel_uden_landmarks", "u": (25.0, 0.0),
         "z": {}, "localized": False},
        {"name": "04_et_landmark", "u": (25.0, 0.10),
         "z": z1, "localized": False},
        {"name": "05_to_landmarks", "u": (25.0, 0.10),
         "z": z2, "localized": True},
    ]


def run_once(case, repetition, args, m, cls):
    # Nulstil baade data og global NumPy random state foer hver uafhaengig test.
    seed = args.seed + 1000 * (int(case["name"].split("_")[0])) + repetition
    prior = make_prior(args.particles, seed, m, case["localized"])
    model = cls(prior)
    np.random.seed(seed + 100000)

    # Fange almindelig debug-print fra mcl(), uden at lave hundreder af linjer output.
    debug_text = io.StringIO()
    row = {"scenario": case["name"], "repeat": repetition + 1,
           "particles": args.particles, "landmarks_seen": len(case["z"]),
           "seed": seed, "seconds": "", "status": "ok", "message": ""}

    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(debug_text), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            cls.update_mcl(model, list(case["u"]), dict(case["z"]), m, args.particles)
        row["seconds"] = round(time.perf_counter() - start, 6)
        if len(model.particles) != args.particles:
            row["status"] = "error"
            row["message"] = "Forkert antal partikler efter MCL-opdatering"
        elif caught:
            row["status"] = "warning"
            row["message"] = "; ".join(str(w.message) for w in caught[:3])
    except Exception as exc:
        row["seconds"] = round(time.perf_counter() - start, 6)
        row["status"] = "error"
        row["message"] = "%s: %s" % (type(exc).__name__, exc)
        debug_text.write("\n" + traceback.format_exc())
    return row, debug_text.getvalue()


def run_profile(case, args, m, cls, target):
    """Detailed function breakdown; cProfile times must not be used as wall-clock baseline."""
    profile = cProfile.Profile()
    seed = args.seed + 50000 + int(case["name"].split("_")[0])
    model = cls(make_prior(args.particles, seed, m, case["localized"]))
    np.random.seed(seed + 100000)
    debug_text = io.StringIO()
    status = "ok"
    try:
        with contextlib.redirect_stdout(debug_text):
            profile.enable()
            try:
                cls.update_mcl(model, list(case["u"]), dict(case["z"]), m,
                               args.particles)
            finally:
                profile.disable()
    except Exception:
        status = "error"
        debug_text.write("\n" + traceback.format_exc())
    profile.dump_stats(str(target.with_suffix(".prof")))
    stream = io.StringIO()
    stream.write("MCL detailed profile: " + case["name"] + "\n")
    stream.write("Status: " + status + "\n")
    stream.write("cProfile instrumentation adds overhead; do NOT interpret these numbers as actual robot time.\n\n")
    for sort_key, title in (("cumulative", "SORTED BY CUMULATIVE TIME"),
                            ("tottime", "SORTED BY SELF TIME")):
        stream.write("\n" + title + "\n")
        stats = pstats.Stats(profile, stream=stream)
        stats.strip_dirs().sort_stats(sort_key).print_stats(args.top)
    stream.write("\nORIGINAL MCL PRINT OUTPUT\n" + debug_text.getvalue() + "\n")
    target.with_suffix(".txt").write_text(stream.getvalue(), encoding="utf-8")
    return status


def main():
    args = arguments()
    sys.path.insert(0, str(BASE_DIR))
    mod = importlib.import_module(args.module)
    cls = mod.MCL

    source_path = Path(mod.__file__).resolve()
    version_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    output = Path(args.output).expanduser().resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output / ("run_%s_%d" % (timestamp, os.getpid()))
    run_dir.mkdir(parents=True, exist_ok=False)

    m = make_map()
    cases = scenarios(m)

    print("\nMIRTE MCL OFFLINE BENCHMARK")
    print("MCL file: %s" % source_path)
    print("MCL SHA256: %s" % version_hash)
    print("Particles: %d | repetitions: %d" % (args.particles, args.repeats))
    print("Result directory: %s" % run_dir)
    print("No main.py, motor, camera or ROS2 is started.\n", flush=True)

    rows = []
    logs = []
    for case in cases:
        print("Testing %-32s" % case["name"], flush=True)
        for rep in range(args.repeats):
            row, debug = run_once(case, rep, args, m, cls)
            rows.append(row)
            logs.append("=== %s run %d ===\n%s\n" % (case["name"], rep + 1, debug))
            print("  %d/%d: %8.3f s  [%s]%s" % (
                rep + 1, args.repeats, row["seconds"], row["status"],
                (" " + row["message"]) if row["message"] else ""), flush=True)

    with (run_dir / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "scenario", "repeat", "particles", "landmarks_seen", "seed",
            "seconds", "status", "message"])
        writer.writeheader()
        writer.writerows(rows)

    (run_dir / "mcl_prints.txt").write_text("\n".join(logs), encoding="utf-8")

    summary = []
    for case in cases:
        values = [r["seconds"] for r in rows
                  if r["scenario"] == case["name"] and r["status"] == "ok"]
        summary.append({
            "scenario": case["name"], "observations": len(case["z"]),
            "runs_ok": len(values), "runs_total": args.repeats,
            "median_s": round(statistics.median(values), 6) if values else None,
            "mean_s": round(statistics.mean(values), 6) if values else None,
            "min_s": round(min(values), 6) if values else None,
            "max_s": round(max(values), 6) if values else None,
        })

    profiles = {}
    if args.profile:
        print("\nDetailed cProfile (slower because of profiler overhead):", flush=True)
        for case_index in (1, 4):
            case = cases[case_index]
            prefix = run_dir / ("profile_" + case["name"])
            print("  " + case["name"], flush=True)
            profiles[case["name"]] = run_profile(case, args, m, cls, prefix)

    report = {
        "mcl_module": args.module,
        "mcl_file": str(source_path),
        "mcl_sha256": version_hash,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "particles": args.particles,
        "repeats": args.repeats,
        "seed": args.seed,
        "units": "cm, radians, seconds",
        "map_type": "synthetic 4-landmark map, 2 synthetic obstacles",
        "note": "Measurement values are artificial; performance benchmark only, not physical localization accuracy.",
        "profile_status": profiles,
        "summary": summary,
    }
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nRESULTS (median wall time, not cProfile time)")
    print("%-34s %12s %10s" % ("Scenario", "Median (s)", "OK / total"))
    for item in summary:
        median = "%.3f" % item["median_s"] if item["median_s"] is not None else "ERROR"
        print("%-34s %12s %5d / %-4d" % (
            item["scenario"], median, item["runs_ok"], item["runs_total"]))
    print("\nFiles: %s" % run_dir)
    print("  timings.csv       - timings for all cases")
    print("  summary.json      - baseline and system info")
    print("  mcl_prints.txt    - native MCL debug output")
    if args.profile:
        print("  profile_*.txt     - functions ranked by cumulative/self time")
        print("  profile_*.prof    - raw cProfile data")
    if any(r["status"] != "ok" for r in rows):
        print("\nWARNING: Some runs reported warnings/errors; inspect timings.csv and mcl_prints.txt")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
