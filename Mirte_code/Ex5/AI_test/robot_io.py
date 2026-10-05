"""Small hardware boundary. Importing modules does not start ROS or motors."""
import importlib
import math
import os
from pathlib import Path
import sys

def KU_Mirte():
    root = Path(__file__).resolve().parent
    paths = [os.environ.get('KU_MIRTE_PYTHON_PATH', '')]
    paths += [str(p / 'Mirte' / 'ku_mirte_python') for p in root.parents]
    paths += [str(root / '../../../Mirte/ku_mirte_python')]
    for path in paths:
        if path and path not in sys.path:
            sys.path.append(path)
    try:
        return importlib.import_module('ku_mirte').KU_Mirte()
    except ModuleNotFoundError as exc:
        raise RuntimeError('KU_Mirte unavailable. Set KU_MIRTE_PYTHON_PATH to the directory containing ku_mirte.py.') from exc

def stop(mirte):
    """Always stop; propagate a stop failure rather than silently ignoring it."""
    mirte.stop()

def front_clearance(mirte, threshold):
    """Sonar distances must be in metres. Missing/invalid data is an error."""
    if threshold is None:
        return math.inf
    sonar = mirte.sonar
    values = [float(sonar[key]) for key in ('front_left', 'front_right')]
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise RuntimeError('Invalid front sonar data; motion cancelled.')
    return min(values)

def move(mirte, translation, rotation, linear_speed, angular_speed,
         sonar_stop_distance=0.25):
    """Blocking rotate-then-translate. Returned motion matches MCL.predict.

    Distances/angles are commanded, not measured odometry. Speeds need physical
    calibration. There is no continuous-drive API dependency.
    """
    translation, rotation = float(translation), float(rotation)
    if not all(math.isfinite(v) for v in (translation, rotation, linear_speed, angular_speed)):
        raise ValueError('Motion values must be finite.')
    if linear_speed <= 0 or angular_speed <= 0 or translation < 0:
        raise ValueError('Positive speeds and forward/nonnegative translation required.')
    try:
        if abs(rotation) > 1e-8:
            mirte.drive(0.0, math.copysign(angular_speed, rotation),
                        abs(rotation) / angular_speed, blocking=True)
            stop(mirte)
        if translation > 1e-8:
            if sonar_stop_distance is not None and front_clearance(mirte, sonar_stop_distance) <= sonar_stop_distance:
                raise RuntimeError('Sonar safety stop.')
            mirte.drive(linear_speed, 0.0, translation / linear_speed, blocking=True)
            stop(mirte)
    finally:
        stop(mirte)
    return translation, rotation
