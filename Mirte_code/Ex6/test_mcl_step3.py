#!/usr/bin/env python3
"""Safety checks for MIRTE Ex6, Step 3a: faster MCL collision tests.

Run in Ex6 with mcl.py (untouched baseline), mcl_step3.py, and
benchmark_mcl.py in the same folder:

    python3 test_mcl_step3.py

No robot/motor/camera/ROS2 started. No files modified.
"""

import contextlib
import hashlib
import io
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Baseline measured on MIRTE during step 2. Refuse to compare against a
# different baseline silently, because that could hide changes unrelated
# to the collision optimization.
BASELINE_SHA256 = 'e9f076e7d75bd04b7cb69caa909920f983b3de0ff0b1d6951a5dfe8da8166341'


def verify_reference():
    base = HERE / 'mcl.py'
    if not base.is_file():
        raise RuntimeError('Missing original Ex6/mcl.py')
    digest = hashlib.sha256(base.read_bytes()).hexdigest()
    if digest != BASELINE_SHA256:
        raise RuntimeError(
            'mcl.py differs from the step 2 baseline!\n'
            'Expected SHA256: %s\nActual SHA256: %s\n'
            'Keep the original mcl.py until Step 3 tests pass.'
            % (BASELINE_SHA256, digest)
        )
    print('[OK] Original mcl.py matches benchmark baseline: %s' % digest[:12])


def load():
    from mcl import MCL as Original
    from mcl_step3 import MCL as Optimized
    import benchmark_mcl as bench
    import particle as pcl
    return Original, Optimized, bench, pcl


def exact_collision_equivalence(Original, Optimized, bench):
    m = bench.make_map()
    a = Optimized([], camera_offset=0.0)
    rng = np.random.default_rng(12345)
    random_points = rng.uniform(low=(-200.0, -200.0), high=(500.0, 600.0), size=(8000, 2))

    # Explicit boundary and landmark/obstacle tangency cases.
    low, high = m.map_area
    boundary_points = [
        [low[0], low[1]], [high[0], high[1]],
        [low[0] - 1e-6, low[1]], [high[0] + 1e-6, high[1]],
        [low[0], high[1]], [high[0], low[1]],
        [0, 0], [400, 300], [np.nan, 0], [0, np.nan],
    ]
    for center, radius, _ in m.landmarks:
        rr = m.mirte_radius + radius
        boundary_points.extend([[center[0] + rr, center[1]],
                                [center[0] - rr, center[1]],
                                [center[0], center[1] + rr]])
    for center, _ in m.obstacles:
        rr = m.mirte_radius + m.obstacale_radius
        boundary_points.extend([[center[0] + rr, center[1]],
                                [center[0] - rr, center[1]]])

    count = 0
    for xy in list(random_points) + boundary_points:
        expected = bool(m.in_collision(xy))
        actual = bool(a._in_collision_fast(xy, m))
        if expected != actual:
            raise AssertionError('Collision mismatch for %r: old=%r, new=%r' % (xy, expected, actual))
        count += 1

    # Empty landmarks/obstacles + different radii to catch forgotten values.
    from local_map import LocalMap
    for scale in (0.5, 1.0, 2.0):
        special = LocalMap(landmarks=[], obstacles=[], mirte_radius=22 * scale,
                           obstacale_radius=20 * scale,
                           low=(-100.0, -100.0), high=(100.0, 100.0))
        for xy in random_points[:500]:
            if bool(special.in_collision(xy)) != bool(a._in_collision_fast(xy, special)):
                raise AssertionError('Empty-map mismatch: %r' % xy)
            count += 1
    print('[OK] %d collision checks: same results as LocalMap.in_collision()' % count)


def collect(model):
    return np.array([[p.getX(), p.getY(), p.getTheta(), p.getWeight()]
                     for p in model.particles], dtype=float)


def check_mcl_replay(Original, Optimized, bench, pcl):
    m = bench.make_map()
    n = 150
    cases = bench.scenarios(m)
    for index, case in enumerate(cases, 1):
        seed = 70000 + index
        initial = bench.make_prior(n, seed, m, case['localized'])
        poses = [(p.getX(), p.getY(), p.getTheta(), p.getWeight()) for p in initial]
        results = []
        for cls in (Original, Optimized):
            prior = [pcl.Particle(*pose) for pose in poses]
            model = cls(prior)
            np.random.seed(seed + 100000)
            with contextlib.redirect_stdout(io.StringIO()):
                cls.update_mcl(model, list(case['u']), dict(case['z']), m, n)
            results.append((collect(model), model.W_fast, model.W_slow))
        old, new = results
        if not np.array_equal(old[0], new[0], equal_nan=True):
            delta = np.max(np.abs(old[0] - new[0]))
            raise AssertionError('%s: particle states differ; max delta %r' % (case['name'], delta))
        if old[1:] != new[1:]:
            raise AssertionError('%s: W_fast/W_slow differ' % case['name'])
        print('[OK] %-33s same particles and MCL weights' % case['name'])


def check_no_external_changes():
    # Deliberately no writes. Inform users which files were read.
    print('[OK] Checks completed without changing mcl.py, particle.py or local_map.py')


def main():
    print('MIRTE MCL STEP 3a - VALIDATION (offline)')
    verify_reference()
    Original, Optimized, bench, pcl = load()
    exact_collision_equivalence(Original, Optimized, bench)
    check_mcl_replay(Original, Optimized, bench, pcl)
    check_no_external_changes()
    print('\nALL TESTS PASSED. Now compare benchmark timings before deploying.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('\n[FAILED] %s: %s' % (type(exc).__name__, exc), file=sys.stderr)
        sys.exit(1)
