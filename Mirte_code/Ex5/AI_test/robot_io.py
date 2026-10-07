"""Small hardware boundary. Importing modules does not start ROS or motors."""
import importlib
import inspect
import math
import os
from pathlib import Path
import sys
import ex5_config as cfg

def KU_Mirte():
    root = Path(__file__).resolve().parent
    paths = [os.environ.get('KU_MIRTE_PYTHON_PATH', '')]
    paths += [str(p / 'Mirte' / 'ku_mirte_python') for p in root.parents]
    paths += [str(root / '../../../Mirte/ku_mirte_python')]
    for path in paths:
        if path and path not in sys.path:
            sys.path.append(path)
    try:
        robot = importlib.import_module('ku_mirte').KU_Mirte()
    except ModuleNotFoundError as exc:
        raise RuntimeError('KU_Mirte unavailable. Set KU_MIRTE_PYTHON_PATH to the directory containing ku_mirte.py.') from exc
    try:
        configure_drive(robot)
    except Exception:
        stop(robot)
        raise
    return robot

def configure_drive(mirte):
    """Apply the SAME driver factors as ContinuousDrive, once at creation."""
    values=(cfg.DRIVE_SPEED_MODIFIER,cfg.DRIVE_TURN_MODIFIER)
    if not all(math.isfinite(v) and v>0 for v in values):
        raise ValueError('Driving modifiers must be finite and positive.')
    method=getattr(mirte,'set_driving_modifier',None)
    if not callable(method):
        raise RuntimeError('Driver lacks set_driving_modifier(speed,turn); check ku_mirte.py.')
    method(*values)

def velocity_to_command(linear,angular):
    """Convert physical target velocities to commands using measured gains.

    Neutral defaults preserve targets. Compensation belongs here; MCL keeps
    integrating the physical target velocities, never driver modifiers.
    """
    values=(linear,angular,cfg.DRIVE_LINEAR_GAIN,cfg.DRIVE_LEFT_GAIN,
            cfg.DRIVE_RIGHT_GAIN,cfg.DRIVE_DRIFT_RAD_PER_M)
    if not all(math.isfinite(v) for v in values) or min(values[2:5])<=0:
        raise ValueError('Finite velocities and positive response gains required.')
    corrected=angular-cfg.DRIVE_DRIFT_RAD_PER_M*linear
    gain=cfg.DRIVE_LEFT_GAIN if corrected>=0 else cfg.DRIVE_RIGHT_GAIN
    return linear/cfg.DRIVE_LINEAR_GAIN,corrected/gain

def stop(mirte):
    """Always stop; propagate a stop failure rather than silently ignoring it."""
    mirte.stop()

def require_continuous_api(mirte):
    """Verify replacement support before starting motors.

    Driver contract: drive(v,w,None,blocking=False,interrupt=True) must replace
    the current velocity immediately, without internally stopping/queuing it.
    Introspection checks argument support, not hardware implementation.
    """
    signature=inspect.signature(mirte.drive)
    try:
        signature.bind(0.0,0.0,None,blocking=False,interrupt=True)
    except TypeError as exc:
        raise RuntimeError('Smooth driving requires drive(v,w,None,blocking=False,interrupt=True). Supply the KU_Mirte driver to adapt its continuous-command API.') from exc

def set_velocity(mirte,linear,angular):
    """Replace an active velocity; never call stop between control ticks."""
    command_v,command_w=velocity_to_command(float(linear),float(angular))
    send_drive_command(mirte,command_v,command_w)

def send_drive_command(mirte,linear,angular):
    """Raw command for calibration; physical targets use set_velocity."""
    if not math.isfinite(linear) or not math.isfinite(angular):
        raise ValueError('Velocity must be finite.')
    mirte.drive(float(linear),float(angular),None,blocking=False,interrupt=True)

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
            mirte.drive(*velocity_to_command(0.0, math.copysign(angular_speed, rotation)),
                        abs(rotation) / angular_speed, blocking=True)
            stop(mirte)
        if translation > 1e-8:
            if sonar_stop_distance is not None and front_clearance(mirte, sonar_stop_distance) <= sonar_stop_distance:
                raise RuntimeError('Sonar safety stop.')
            mirte.drive(*velocity_to_command(linear_speed, 0.0), translation / linear_speed, blocking=True)
            stop(mirte)
    finally:
        stop(mirte)
    return translation, rotation
