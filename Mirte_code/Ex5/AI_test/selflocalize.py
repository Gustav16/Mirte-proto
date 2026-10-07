"""Inspect MCL particles, inspired by the colleague's selflocalize.py.

Default: synthetic offline demo. --robot reads camera only; never sends motor
commands. GUI is opt-in; a final particle plot is always saved as PNG.
All coordinates are metres, theta=0 faces +z, positive theta points left.
"""
import argparse
import math
from pathlib import Path
import time
import numpy as np
import ex5_config as cfg
from mcl import MCL
from aruco_measurements import observe_mirte
from robot_io import KU_Mirte
from simulate_mcl import observations_for_pose
from continuous_control import integrate_pose


def plot_particles(localizer,ax):
    ax.clear();p=localizer.particles;weights=localizer.weights
    ax.scatter(p[:,0],p[:,1],c=weights,cmap='jet',s=5,alpha=.5)
    for i,(x,z) in localizer.landmarks.items():
        ax.scatter([x],[z],marker='s',color='black');ax.annotate(f'ID {i}',(x,z))
    x,z,theta=localizer.estimate_pose()
    ax.scatter([x],[z],color='magenta')
    ax.arrow(x,z,-.15*math.sin(theta),.15*math.cos(theta),color='magenta',head_width=.03)
    ax.set_xlim(*localizer.x_bounds);ax.set_ylim(*localizer.z_bounds)
    ax.set_xlabel('x right (m)');ax.set_ylabel('z forward (m)')
    ax.set_aspect('equal',adjustable='box');ax.grid(True)
    ax.set_title(f'MCL: {len(p)} particles | confident={localizer.confident()} | injected={localizer.last_random_particles}')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot',action='store_true',help='Stationary real camera; NO motor commands.')
    parser.add_argument('--gui',action='store_true')
    parser.add_argument('--augmented',action='store_true',help='Enable experimental fast/slow recovery.')
    parser.add_argument('--frames',type=int,default=100)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--output',type=Path,default=Path('selflocalize_particles.png'))
    args=parser.parse_args(argv)
    if args.frames<1:parser.error('frames must be positive')
    import matplotlib
    if not args.gui:matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    np.random.seed(args.seed);f=MCL(cfg.LANDMARKS,augmented=args.augmented)
    robot=KU_Mirte() if args.robot else None
    truth=np.array([.35,-.85,math.radians(18)]);seeded=False
    fig,ax=plt.subplots(figsize=(7,6));failure=None
    if args.gui:plt.ion()
    try:
        for tick in range(args.frames):
            if args.gui and not plt.fignum_exists(fig.number):break
            if robot is None:
                if seeded:
                    truth=integrate_pose(truth,.015,.02,.15);f.predict_twist(.015,.02,.15)
                obs=observations_for_pose(truth,noisy=True)
            else:
                # This program is stationary; no keyboard key fakes motor motion.
                obs=observe_mirte(robot,allowed_ids=f.landmarks)
            if not seeded:
                seeded=f.initialize_from_observations(obs)
                if not seeded and tick%10==0:print('Waiting for both configured landmarks in one stationary frame.')
            else:
                f.correct(obs);f.resample()
            if tick%10==0:print('pose m/rad:',f.estimate_pose(),'std m:',f.position_std(),'confident:',f.confident())
            if args.gui:
                plot_particles(f,ax);plt.pause(.001)
            if robot is not None:time.sleep(.15)
    except KeyboardInterrupt:pass
    except Exception as exc:failure=exc
    finally:
        plot_particles(f,ax);args.output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(args.output,dpi=180,bbox_inches='tight');plt.close(fig)
        print('Saved:',args.output)
    if failure is not None:raise failure
    return {'pose':f.estimate_pose().tolist(),'seeded':seeded,'confident':bool(f.confident()),
            'simulated_error_m':None if robot is not None else float(np.linalg.norm(f.estimate_pose()[:2]-truth[:2]))}


if __name__=='__main__':main()
