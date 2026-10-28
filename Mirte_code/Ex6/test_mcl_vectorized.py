#!/usr/bin/env python3
"""Offline validation for MIRTE Ex6 vectorized MCL (no robot/ROS2).

Place alongside mcl.py (step-3 baseline), mcl_vectorized.py, particle.py,
local_map.py and benchmark_mcl.py, then run:
    python3 test_mcl_vectorized.py

IMPORTANT: Different RNG draw ordering means complete motion trajectories cannot
be compared bit-for-bit; verify equations, Gaussian noise statistics and collision
constraints instead. Fixed-state measurement weights can be compared directly.
"""
import contextlib
import hashlib
import io
import math
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STEP3_SHA256 = "19f3c172925c54ef5215c9bfb4569ef171b1ab1618aaa39068cb322bb19add4b"


def models():
    import mcl
    import mcl_vectorized
    import benchmark_mcl as bench
    import particle as pcl
    return mcl.MCL, mcl_vectorized.MCL, bench, pcl


def assert_baseline():
    target = HERE / 'mcl.py'
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != STEP3_SHA256:
        raise AssertionError(
            'Current mcl.py is not the tested step-3 baseline. SHA256=%s. '
            'Do not overwrite it; compare against your saved backup first.' % digest)
    print('[OK] Original mcl.py is unchanged step-3 reference (%s)' % digest[:12])


def test_collision(cls, bench):
    m = bench.make_map()
    model = cls([])
    rng = np.random.default_rng(20261028)
    positions = rng.uniform([-200, -200], [500, 600], (9000, 2))
    critical = [[np.nan, 0], [0, np.nan], [-120, -120], [420, 520],
                [-120.000001, 0], [420.000001, 0], [0, 0], [400, 300]]
    for center, radius, _ in m.landmarks:
        r = m.mirte_radius + radius
        critical.extend([[center[0] + r, center[1]],
                         [center[0], center[1] + r],
                         [center[0] - r, center[1]]])
    positions = np.concatenate([positions, np.array(critical)])
    fast = model._collision_mask_batch(positions, m)
    reference = np.array([model._in_collision_fast(pos, m) for pos in positions])
    assert np.array_equal(fast, reference), 'Collision mismatch'
    print('[OK] %d collision checks: batch == scalar' % len(positions))


def test_measurement(cls_original, cls_fast, bench):
    m = bench.make_map()
    for offset in (0.0, 10.0):
        a = cls_original([], camera_offset=offset)
        b = cls_fast([], camera_offset=offset)
        prior = bench.make_prior(1000, 207, m, False)
        poses = np.array([(p.getX(), p.getY(), p.getTheta()) for p in prior])
        for ids in ((), (4,), (4, 1)):
            z = bench.make_measurements(ids, m)
            expected = np.array([a.measurement_model(z, p, m) for p in prior])
            actual = b._measurement_batch(z, poses, m)
            np.testing.assert_allclose(expected, actual, atol=1e-16, rtol=1e-12)
    print('[OK] Observation likelihood agrees (0/1/2 landmarks; camera offset 0/10 cm)')


def test_motion_distributions(cls_original, cls_fast, pcl):
    from local_map import LocalMap
    m = LocalMap(landmarks=[], obstacles=[], low=(-1e9, -1e9), high=(1e9, 1e9))
    count = 12000
    initial = [pcl.Particle(0., 0., 1.0) for _ in range(count)]
    cases = [(25.0, 0.1), (0.0, math.radians(40)), (0.0, 0.0)]
    for u in cases:
        old = cls_original(initial)
        new = cls_fast(initial)
        np.random.seed(102)
        with contextlib.redirect_stdout(io.StringIO()):
            a = np.array([old.sample_motion_model(u, p) for p in initial])
        np.random.seed(202)
        b = new._sample_motion_batch(u, m)
        # Independent Gaussian draws have different sequence/order but same law.
        arr_a = np.array([(p.getX(), p.getY(), p.getTheta()) for p in a])
        diffs = np.abs(arr_a[:, :2].mean(axis=0) - b[:, :2].mean(axis=0))
        assert np.max(diffs) < 0.35, (u, 'mean error', diffs)
        stddiff = np.abs(arr_a[:, :2].std(axis=0) - b[:, :2].std(axis=0))
        assert np.max(stddiff) < 0.35, (u, 'std error', stddiff)
        assert np.all(np.isfinite(b))
    print('[OK] Motion sampling: mean/std preserved for move/turn/stationary (12,000 particles)')


def test_deterministic_resampling(cls_original, cls_fast, bench, pcl):
    m = bench.make_map()
    z = bench.make_measurements((4, 1), m)
    count = 500
    base_prior = bench.make_prior(count, 891, m, True)
    base = [(p.getX(), p.getY(), p.getTheta(), p.getWeight()) for p in base_prior]
    obj1 = cls_original([pcl.Particle(*v) for v in base])
    obj2 = cls_fast([pcl.Particle(*v) for v in base])
    obj1.sample_motion_model_with_map = lambda u, p, map_: p
    obj2._sample_motion_batch = lambda u, map_: np.array([v[:3] for v in base])
    outputs = []
    for model, cls in [(obj1, cls_original), (obj2, cls_fast)]:
        np.random.seed(431)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.update_mcl(model, (25., 0.1), z, m, count)
        outputs.append((np.array([(p.getX(), p.getY(), p.getTheta(), p.getWeight())
                                  for p in model.particles]), model.W_fast, model.W_slow))
    expected, actual = outputs
    np.testing.assert_allclose(expected[0], actual[0], atol=1e-12, rtol=1e-11)
    np.testing.assert_allclose(expected[1:], actual[1:], atol=1e-12, rtol=1e-11)
    print('[OK] Same resampling/fast-slow weights with identical motion proposals')

    # Augmented MCL: force a non-zero reinjection probability and verify that
    # the unchanged sensor-reset sampler is invoked with equivalent results.
    count = 250
    z = bench.make_measurements((4,), m)
    prior = bench.make_prior(count, 345, m, True)
    base = [(p.getX(), p.getY(), p.getTheta(), p.getWeight()) for p in prior]
    results = []
    for cls in (cls_original, cls_fast):
        model = cls([pcl.Particle(*v) for v in base])
        model.W_fast = 0.002
        model.W_slow = 0.05
        model.sample_motion_model_with_map = lambda u, p, map_: p
        model._sample_motion_batch = lambda u, map_: np.array([v[:3] for v in base])
        np.random.seed(786)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            cls.update_mcl(model, (25.0, 0.1), z, m, count)
        assert float(log.getvalue().split('p_random: ')[-1].split()[0]) > 0.5, 'Expected nonzero sensor-reset probability'
        results.append((np.array([(p.getX(), p.getY(), p.getTheta(), p.getWeight())
                                  for p in model.particles]), model.W_fast, model.W_slow))
    old, new = results
    np.testing.assert_allclose(old[0], new[0], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(old[1:], new[1:], rtol=1e-11, atol=1e-12)
    print('[OK] Augmented sensor resetting preserved (nonzero p_random)')


def test_five_scenarios(cls, bench):
    m = bench.make_map()
    for case in bench.scenarios(m):
        for seed in (101, 202):
            prior = bench.make_prior(1000, seed, m, case['localized'])
            model = cls(prior)
            np.random.seed(seed + 555)
            with contextlib.redirect_stdout(io.StringIO()):
                cls.update_mcl(model, case['u'], case['z'], m, 1000)
            assert len(model.particles) == 1000
            states = np.array([(p.getX(), p.getY(), p.getTheta(), p.getWeight())
                               for p in model.particles], dtype=float)
            assert np.all(np.isfinite(states)), (case['name'], 'non-finite particles')
            # The original update_mcl deliberately leaves particles untouched
            # for stationary motion with no measurements, including ones
            # initialized in collision. Only require valid positions if a
            # motion/measurement update actually ran.
            if case['u'] != (0.0, 0.0) or case['z']:
                assert np.array_equal(model._collision_mask_batch(states[:, :2], m),
                                      np.zeros(1000, dtype=bool)), (case['name'], 'collision')
            assert np.all((states[:, 2] >= 0) & (states[:, 2] < 2 * np.pi))
        print('[OK] %-31s 1000 finite particles, two runs' % case['name'])


def main():
    print('MIRTE MCL VECTORISERING - OFFLINE VERIFIKATION')
    assert_baseline()
    a, b, bench, pcl = models()
    test_collision(b, bench)
    test_measurement(a, b, bench)
    test_motion_distributions(a, b, pcl)
    test_deterministic_resampling(a, b, bench, pcl)
    test_five_scenarios(b, bench)
    print('\nALL TESTS PASSED - now benchmark on MIRTE before installation.')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('[FAILED] %s: %s' % (type(e).__name__, e), file=sys.stderr)
        raise
