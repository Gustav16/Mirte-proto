"""Measure raw driver response. Offline unless --run is supplied.

Every trial ends with stop. Measurements are entered AFTER motors stop.
No old calibration is silently activated or extrapolated to lower speeds.
"""
import argparse
import json
import math
from pathlib import Path
import time
import ex5_config as cfg
from robot_io import KU_Mirte,require_continuous_api,send_drive_command,front_clearance,stop


def measured_response(mode,speed,elapsed,measurement):
    if mode not in ('linear','left','right'):raise ValueError('Unknown trial mode.')
    if not all(math.isfinite(x) and x>0 for x in (speed,elapsed)):
        raise ValueError('Positive finite speed and elapsed time required.')
    if not math.isfinite(measurement) or measurement<0:raise ValueError('Invalid measurement.')
    signed_speed=-speed if mode=='right' else speed
    actual=measurement/elapsed if mode=='linear' else math.radians(measurement)/elapsed
    return {'mode':mode,'command_speed':signed_speed,'active_seconds':elapsed,
            'measurement_m_or_degrees':measurement,'measured_speed_m_s_or_rad_s':actual,
            'response_gain':actual/speed,'speed_modifier':cfg.DRIVE_SPEED_MODIFIER,
            'turn_modifier':cfg.DRIVE_TURN_MODIFIER,
            'note':'Gain for this speed, direction, floor, battery and modifier setup only.'}


def run_trial(robot,mode,speed,duration,poll=.05):
    if mode not in ('linear','left','right'):raise ValueError('Unknown trial mode.')
    if not all(math.isfinite(x) and x>0 for x in (speed,duration,poll)) or duration>10:
        raise ValueError('Positive finite values; duration at most 10 seconds.')
    if mode=='linear' and speed>.35 or mode!='linear' and speed>1.5:
        raise ValueError('Trial speed exceeds the old scripts reference range.')
    try:
        require_continuous_api(robot)
        if front_clearance(robot,.30)<=.30:raise RuntimeError('Insufficient front clearance before trial.')
        v=speed if mode=='linear' else 0.
        w=0. if mode=='linear' else (speed if mode=='left' else -speed)
        send_drive_command(robot,v,w)
        started=time.monotonic()
        while time.monotonic()-started<duration:
            if front_clearance(robot,.30)<=.30:raise RuntimeError('Sonar interrupted trial; do not use it as calibration.')
            remaining=duration-(time.monotonic()-started)
            if remaining>0:time.sleep(min(poll,remaining))
        return time.monotonic()-started
    finally:stop(robot)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['linear','left','right'],default='linear')
    p.add_argument('--speed',type=float,default=None)
    p.add_argument('--seconds',type=float,default=2.)
    p.add_argument('--run',action='store_true',help='Start one physical motor trial.')
    p.add_argument('--output',type=Path,default=Path('drive_measurements.jsonl'))
    args=p.parse_args();speed=args.speed if args.speed is not None else (.15 if args.mode=='linear' else 1.5)
    if not math.isfinite(speed) or speed<=0 or not math.isfinite(args.seconds) or not 0<args.seconds<=10:
        p.error('Use positive finite speed and seconds <= 10.')
    expected=speed*args.seconds
    if args.mode!='linear':expected=math.degrees(expected)
    print('Raw trial:',args.mode,'speed=',speed,'seconds=',args.seconds,'nominal distance/angle=',expected)
    print('Driver modifiers:',cfg.DRIVE_SPEED_MODIFIER,cfg.DRIVE_TURN_MODIFIER)
    if not args.run:
        print('Offline: no hardware or motors started. --run executes one trial in an open test area.');return
    robot=KU_Mirte()
    elapsed=run_trial(robot,args.mode,speed,args.seconds)
    measurement=float(input('Motors stopped. Measured distance in metres: ' if args.mode=='linear' else
                            'Motors stopped. Measured rotation in degrees (positive magnitude): '))
    record=measured_response(args.mode,speed,elapsed,measurement)
    if args.mode=='linear':
        drift=float(input('Measured heading change in degrees (+ left, - right): '))
        if not math.isfinite(drift):raise ValueError('Finite heading change required.')
        record['heading_change_deg']=drift
        record['drift_rad_per_m']=math.radians(drift)/measurement if measurement>0 else None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record,indent=2));print('Saved:',args.output)
    print('Repeat trials. Enter gains in ex5_config only for a verified operating range; no automatic activation.')


if __name__=='__main__':main()
