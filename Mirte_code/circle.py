"""Drive in a circle. THIS MOVES THE ROBOT.

A circle is just linear + angular velocity at the same time:

    radius   = speed / turn_rate          (so turn_rate = speed / radius)
    lap time = 2*pi / turn_rate

Run it on the robot:
    python3 circle.py              # one lap, default 0.30 m radius
    python3 circle.py 0.4          # 0.40 m radius
    python3 circle.py 0.4 2        # 0.40 m radius, 2 laps

WHY THE DEFAULTS LOOK LIKE THIS
The robot barely turns below ~1.0 rad/s (measured: 27% of commanded at
0.5 rad/s, 97% at 1.0). Since turn_rate = speed / radius, keeping the turn
rate healthy forces either a small radius or a high speed. The defaults pick
turn_rate = 1.0 and derive the speed from the radius you ask for.

Space needed: a clear circle of 2 x radius, plus a bit -- 0.30 m radius
wants roughly 1 x 1 m.
"""
import math
import os
import signal
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../Mirte/ku_mirte_python'))

from ku_mirte import KU_Mirte

RADIUS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30   # metres
LAPS = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
TURN_RATE = 1.0            # rad/s -- the rate we measured at 97% accuracy

SPEED = TURN_RATE * RADIUS                 # v = omega * r
LAP_TIME = 2 * math.pi / TURN_RATE         # seconds for one full circle
DURATION = LAP_TIME * LAPS

MIN_SPEED = 0.12           # below this the robot silently does not move
MAX_SPEED = 0.50           # sanity cap so a big radius can't launch it

print(f"circle: radius {RADIUS:.2f} m, {LAPS:g} lap(s)")
print(f"  speed     {SPEED:.2f} m/s")
print(f"  turn rate {TURN_RATE:.2f} rad/s")
print(f"  duration  {DURATION:.1f} s")

if SPEED < MIN_SPEED:
    print(f"\nREFUSING: {SPEED:.2f} m/s is below the {MIN_SPEED} m/s deadband -- "
          f"the robot would not move at all.\n"
          f"Use a radius of at least {MIN_SPEED / TURN_RATE:.2f} m.")
    sys.exit(1)
if SPEED > MAX_SPEED:
    print(f"\nREFUSING: {SPEED:.2f} m/s is faster than the {MAX_SPEED} m/s cap.\n"
          f"Use a radius under {MAX_SPEED / TURN_RATE:.2f} m.")
    sys.exit(1)

robot = KU_Mirte()

# Hard time limit: if anything hangs, stop the robot and quit rather than
# leaving it driving. Running on the robot there is nothing else to save you.
def _watchdog(signum, frame):
    print("\n!!! WATCHDOG -- stopping", flush=True)
    try:
        robot.stop()
    except Exception:
        pass
    os._exit(2)

signal.signal(signal.SIGALRM, _watchdog)
signal.alarm(int(DURATION) + 45)

sonar = robot.get_sonar_ranges() or {}
front = [sonar.get('front_left'), sonar.get('front_right')]
if any(v is not None and v < 0.30 for v in front):
    print(f"\nABORT: obstacle ahead ({front})")
    sys.exit(1)

print("\nstarting in 3 s -- Ctrl-C to abort")
for i in (3, 2, 1):
    print(f"  {i}...")
    time.sleep(1)

try:
    # The whole circle is ONE command: forward and turning simultaneously.
    robot.drive(SPEED, TURN_RATE, DURATION, blocking=True)
finally:
    robot.stop()          # always stop, even if the above raised
    print("stopped")

sys.stdout.flush()
os._exit(0)               # KU_Mirte.__del__ can deadlock on rclpy.shutdown()
