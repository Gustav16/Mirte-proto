#!/usr/bin/env python3
"""MIRTE visualization: import from existing main.py with minimal changes.

In existing main.py, add after its module imports:
    import mirte_visualizer as vis

After setting up global_map, BEFORE initializing particles:
    vis.start(globals())

Optional status updates at decision points:
    vis.note("Scanning for next target")

At the end of main.py (or automatically via atexit):
    vis.stop()

Run the robot as usual: python3 main.py
The visualizer captures the existing camera, MCL, RRT, and drive calls via
in-memory wrappers, and starts a SECOND process for creating PNG files.
No files except mirte_debug outputs are changed by the visualizer.
This is observation only, not motor control, collision avoidance, or ground truth.

Standalone legacy options (without editing main.py):
    python3 mirte_visualizer.py --background
    python3 mirte_visualizer.py --demo
"""

from __future__ import annotations

import argparse
import atexit
import subprocess
import uuid
import copy
import json
import math
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # No X11 required on the robot / SSH.
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np


C = dict(robot='#159447', uncertain='#cd850d', particle='#3874ca',
         landmark='#7d51b0', obstacle='#ce404c', clearance='#d69a2a',
         path='#0098a8', tree='#a0afbd', history='#169657',
         goal='#dc8a00', ink='#263548', paper='#f4f7fb')


def xy(value):
    a = np.asarray(value, dtype=float).ravel()
    return [float(a[0]), float(a[1])]


def pose(value):
    if value is None:
        return None
    if hasattr(value, 'getX'):
        return [float(value.getX()), float(value.getY()), float(value.getTheta())]
    return [float(v) for v in value[:3]]


def points_dict(d):
    return {str(k): xy(v) for k, v in (d or {}).items()}


def robot_to_world(local_point, world_pose):
    """Camera [right,forward] -> world [x,y]; not R(theta)@[right,forward]."""
    right, forward = xy(local_point)
    x, y, t = world_pose
    return [x + forward * math.cos(t) + right * math.sin(t),
            y + forward * math.sin(t) - right * math.cos(t)]


def parse_map(m):
    if m is None:
        return {'landmarks': [], 'obstacles': [], 'bounds': None, 'robot_radius': 22, 'obstacle_radius': 20}
    landmarks, obstacles = [], []
    for item in getattr(m, 'landmarks', []) or []:
        try:
            p, r, i = item
            landmarks.append([xy(p), float(r), str(i)])
        except (ValueError, TypeError):
            continue
    for item in getattr(m, 'obstacles', []) or []:
        try:
            p, i = item
            obstacles.append([xy(p), str(i)])
        except (ValueError, TypeError):
            continue
    bounds = getattr(m, 'map_area', None)
    return {'landmarks': landmarks, 'obstacles': obstacles,
            'bounds': [xy(bounds[0]), xy(bounds[1])] if bounds is not None else None,
            'robot_radius': float(getattr(m, 'mirte_radius', 22)),
            'obstacle_radius': float(getattr(m, 'obstacale_radius', 20))}


class Recorder:
    def __init__(self, output: Path, period=5, publisher=False):
        self.output = output
        self.closed = False
        self.frames_dir = output / 'frames'
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.start = time.monotonic()
        self.period = period
        self.publisher = publisher
        self.session = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.render_lock = threading.Lock()
        self.start_state = None
        self.start_saved = False
        self.stop = threading.Event()
        self.frame_id = 0
        self.globals = {}
        self.status = {
            'known': {}, 'global_map': None, 'particles': [], 'estimate': None,
            'converged': False, 'sigma': None, 'history': [],
            'local_landmarks': {}, 'local_obstacles': {},
            'seen_landmarks': {}, 'seen_obstacles': {},
            'local_last': None, 'plan': None, 'target': None,
            'decision': 'Starting', 'events': [], 'last_update': 0,
        }
        # Fresh log for each run. The file is supplementary; source files untouched.
        (self.output / 'events.jsonl').write_text('', encoding='utf-8')
        self.renderer = threading.Thread(target=self._publish_worker if publisher else self._render_worker,
                                         daemon=True, name='MIRTE-telemetry' if publisher else 'MIRTE-map-render')
        self.renderer.start()

    def _sync_main(self):
        g = self.globals
        if not g:
            return
        target = g.get('target')
        if target is not None:
            self.status['target'] = getattr(target, 'id', None)
        elif 'target' in g:
            self.status['target'] = None
        known = g.get('landmarks')
        if isinstance(known, dict):
            self.status['known'] = points_dict(known)
        gm = g.get('global_map')
        if gm is not None:
            self.status['global_map'] = parse_map(gm)

    def event(self, kind, message, **values):
        with self.lock:
            self._sync_main()
            elapsed = round(time.monotonic() - self.start, 2)
            entry = {'elapsed_s': elapsed, 'time': datetime.now().isoformat(timespec='seconds'),
                     'type': kind, 'message': str(message),
                     'target': self.status['target'], **values}
            self.status['events'].append(entry)
            self.status['events'] = self.status['events'][-12:]
            self.status['last_update'] = elapsed
            with (self.output / 'events.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def prior(self, particles):
        with self.lock:
            self.status['particles'] = [[float(p.getX()), float(p.getY()), float(p.getTheta())]
                                        for p in particles]
        self.event('start', f'MCL prior initialized with {len(particles)} particles')
        with self.lock:
            self.start_state = copy.deepcopy(self.status)

    def observations(self, z, landmarks, obstacles):
        with self.lock:
            self.status['local_landmarks'] = points_dict(landmarks)
            self.status['local_obstacles'] = points_dict(obstacles)
            # Do not world-project here: a turn may have happened since the
            # last MCL update. Anchor only after MCL has processed this frame.
            target = getattr(self.globals.get('target'), 'id', None)
            if target is not None and target in landmarks:
                self.status['decision'] = f'Target {target} visible'
        self.event('camera', f'Seen landmark IDs {list(landmarks)}, obstacle IDs {list(obstacles)}')

    def _anchor_observations(self):
        s = self.status
        if not s['converged'] or s['estimate'] is None:
            return
        for i, p in s['local_landmarks'].items():
            s['seen_landmarks'][i] = robot_to_world(p, s['estimate'])
        for i, p in s['local_obstacles'].items():
            s['seen_obstacles'][i] = robot_to_world(p, s['estimate'])

    def mcl(self, mcl):
        particles = getattr(mcl, 'particles', [])
        try:
            samples = [[float(p.getX()), float(p.getY()), float(p.getTheta())]
                       for p in particles]
            est = pose(mcl.estimate_pose())
            variance = [float(v) for v in type(mcl).pose_variance(mcl.estimate_pose(), particles)]
        except (TypeError, ValueError, AttributeError, ZeroDivisionError):
            return
        with self.lock:
            self.status['particles'] = samples
            self.status['estimate'] = est
            self.status['sigma'] = [math.sqrt(max(0, v)) for v in variance]
            # Matches thresholds in uploaded main.py; main.has_converged overrides later.
            thresholds = (25., 25., math.radians(15.))
            self.status['converged'] = all(s < lim for s, lim in zip(self.status['sigma'], thresholds))
            history = self.status['history']
            # HISTORY is only MCL estimate (not measured ground truth).
            if not history or math.dist(history[-1], est[:2]) >= 2:
                history.append(est[:2])
                if len(history) > 2500:
                    del history[:-2500]
            self._anchor_observations()
        self.event('mcl', 'Particle filter updated', converged=self.status['converged'])

    def convergence(self, converged):
        with self.lock:
            self.status['converged'] = bool(converged)
            self._anchor_observations()

    def rrt(self, rrt, result, frame):
        edges = []
        for node in getattr(rrt, 'node_list', []):
            if getattr(node, 'parent', None) is not None:
                edges.append([xy(node.parent.pos), xy(node.pos)])
        with self.lock:
            prev = self.status.get('plan') or {}
            attempt = prev.get('attempt', 0) + 1 if prev.get('frame') == frame else 1
            self.status['plan'] = {
                'frame': frame, 'attempt': attempt, 'edges': edges,
                'map': parse_map(rrt.map), 'start': xy(rrt.start.pos),
                'goal': xy(rrt.end.pos), 'path': None,
                'raw_path': [xy(p) for p in reversed(result)] if result is not None else None,
                'success': result is not None,
                'pose_at_plan': copy.deepcopy(self.status['estimate'])
            }
            self.status['decision'] = f'RRT {frame}: ' + ('route found' if result is not None else 'no route')
        self.event('rrt', self.status['decision'], tree_edges=len(edges), attempt=attempt)

    def selected_path(self, path, frame):
        with self.lock:
            if self.status['plan'] is None:
                self.status['plan'] = {'frame': frame, 'edges': [], 'map': None,
                                       'start': None, 'goal': None, 'attempt': 0,
                                       'pose_at_plan': self.status['estimate']}
            # Existing RRT code returns goal FIRST.
            self.status['plan']['path'] = [xy(p) for p in reversed(path)] if path is not None else None
            self.status['decision'] = f'{frame.title()} route planned' if path is not None else f'{frame.title()} RRT failed'
        self.event('plan', self.status['decision'])

    def execute(self, target, frame):
        with self.lock:
            self.status['decision'] = f'Driving {frame} plan toward target {target}'
        self.event('drive', self.status['decision'])

    def scan(self, angle):
        with self.lock:
            self.status['decision'] = f'Scanning / turning {math.degrees(angle):+.0f} degrees'
        self.event('turn', self.status['decision'])

    def _publish(self, closed=False):
        """Send a copy of the navigation state to the independent plotter.

        This is a file-based IPC channel; it NEVER accesses robot hardware.
        os.replace ensures the viewer reads either a complete JSON or nothing.
        """
        with self.lock:
            self._sync_main()
            state = copy.deepcopy(self.status)
            start = copy.deepcopy(self.start_state)
        message = {
            'session': self.session,
            'closed': bool(closed),
            'elapsed_s': time.monotonic() - self.start,
            'state': state,
            'start_state': start,
        }
        tmp = self.output / f'_telemetry_{os.getpid()}.tmp'
        tmp.write_text(json.dumps(message, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, self.output / 'telemetry.json')

    def _publish_worker(self):
        # 0.5 s telemetry is lightweight and does not render on the robot thread.
        while not self.stop.wait(.5):
            try:
                self._publish()
            except Exception as e:
                print(f'[VISUALIZER] Telemetry update failed: {e}', file=sys.stderr)

    def _render_worker(self):
        # Rendering never calls MIRTE, the camera, MCL or motor commands.
        next_frame = time.monotonic() + self.period
        while not self.stop.wait(0.3):
            try:
                self.save_start()
                if time.monotonic() >= next_frame:
                    self.save()
                    next_frame = time.monotonic() + self.period
            except Exception as e:
                print(f'[VISUALIZER] Could not save image: {e}', file=sys.stderr)

    def save_start(self):
        if self.publisher:
            self._publish()
            return
        with self.lock:
            if self.start_state is None or self.start_saved:
                return
            self.start_saved = True
            first = copy.deepcopy(self.start_state)
        with self.render_lock:
            fig = create_figure(first, elapsed=0.0)
            try:
                fig.savefig(self.output / 'start.png', dpi=105)
            finally:
                plt.close(fig)

    def save(self, final=False):
        if self.publisher:
            self._publish(closed=final)
            return
        with self.render_lock:
            return self._save_locked(final=final)

    def _save_locked(self, final=False):
        with self.lock:
            self._sync_main()
            s = copy.deepcopy(self.status)
            self.frame_id += 1
            index = self.frame_id
        fig = create_figure(s, elapsed=time.monotonic()-self.start)
        name = self.frames_dir / f'map_{index:05d}.png'
        try:
            fig.savefig(name, dpi=105)
            temp = self.output / '_latest_tmp.png'
            fig.savefig(temp, dpi=105)
            os.replace(temp, self.output / 'latest.png')
            state = self.output / 'latest_state.json'
            tmp_state = self.output / '_latest_state_tmp.json'
            tmp_state.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding='utf-8')
            os.replace(tmp_state, state)
        finally:
            plt.close(fig)
        if final:
            print(f'[VISUALIZER] Saved final map: {self.output / "latest.png"}')

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.renderer.join(timeout=12)
        if not self.publisher:
            self.save_start()
        self.event('stop', 'Visualizer closed (original program stopped)')
        self.save(final=True)


def circle(ax, p, r, col, dash=False, alpha=.16, z=2):
    ax.add_patch(Circle(xy(p), r, facecolor='none' if dash else col, edgecolor=col,
                        linestyle='--' if dash else '-', linewidth=1.25,
                        alpha=.8 if dash else alpha, zorder=z))


def label(ax, p, text, color=C['ink'], shift=(5, 6)):
    ax.annotate(str(text), xy(p), xytext=shift, textcoords='offset points',
                fontsize=8, color=color, zorder=8)


def draw_route(ax, plan, world=False):
    if not plan:
        return
    if (plan.get('frame') == 'global') != world:
        return
    edges = plan.get('edges', [])
    if edges:
        ax.add_collection(LineCollection(edges, colors=C['tree'], linewidths=.7, alpha=.65, zorder=2))
    path = plan.get('path') or plan.get('raw_path')
    if path:
        p = np.asarray(path)
        ax.plot(p[:,0], p[:,1], color=C['path'], linewidth=2.8, zorder=6)
        ax.scatter(p[:,0], p[:,1], c=C['path'], s=13, zorder=6)
    if plan.get('start') is not None:
        ax.scatter(*plan['start'], c=C['robot'], marker='D', s=45, zorder=7)
    if plan.get('goal') is not None:
        ax.scatter(*plan['goal'], c=C['goal'], marker='*', s=125, zorder=8)


def draw_world(ax, s):
    ax.set_title('GLOBAL MAP  |  cm', loc='left', fontweight='bold', fontsize=12)
    ax.set_facecolor('white')
    wm = s.get('global_map') or {}
    b = wm.get('bounds') or [[-120, -120], [420, 520]]
    ax.set_xlim(b[0][0], b[1][0]); ax.set_ylim(b[0][1], b[1][1])
    known = s.get('known') or {}
    if len(known) >= 3:
        a = np.asarray(list(known.values()))
        ax.add_patch(Rectangle((min(a[:,0]),min(a[:,1])), np.ptp(a[:,0]),np.ptp(a[:,1]),
                               fill=False, edgecolor='#afb9c7', linestyle=':', linewidth=1.5))
    gm_radii = {str(item[2]): item[1] for item in wm.get('landmarks', [])}
    rr = wm.get('robot_radius', 22)
    for i, p in known.items():
        rad = gm_radii.get(i, 20)
        circle(ax, p, rad + rr, C['clearance'], dash=True)
        circle(ax, p, 20, C['landmark'])
        ax.scatter(*p, marker='s', c=C['landmark'], s=36, zorder=5)
        label(ax, p, f'L{i}' + (' (target)' if str(i)==str(s['target']) else ''), C['landmark'])
    # These are visualization-only accumulated observations, not inserted in RRT.
    for i,p in s.get('seen_obstacles', {}).items():
        circle(ax, p, wm.get('obstacle_radius',20), C['obstacle'], dash=True)
        ax.scatter(*p, marker='o', facecolors='none', edgecolors=C['obstacle'], s=40, zorder=4)
        label(ax,p,f'O{i} seen',C['obstacle'])
    for i,p in s.get('seen_landmarks', {}).items():
        ax.scatter(*p, marker='x', c=C['landmark'], s=36, zorder=5)
    plan = s.get('plan')
    if plan and plan.get('frame')=='global':
        pmap = plan.get('map') or {}
        for p, i in pmap.get('obstacles', []):
            circle(ax,p,pmap.get('obstacle_radius',20)+pmap.get('robot_radius',22),C['clearance'],dash=True)
            circle(ax,p,pmap.get('obstacle_radius',20),C['obstacle'])
        draw_route(ax, plan,world=True)
    if s['particles']:
        particles=np.asarray(s['particles'])
        step=max(1,len(particles)//850)
        ax.scatter(particles[::step,0], particles[::step,1], s=5,c=C['particle'],alpha=.18,zorder=1)
    if len(s['history']) > 1:
        h=np.asarray(s['history'])
        ax.plot(h[:,0],h[:,1],color=C['history'],linestyle='--',linewidth=1.8,alpha=.75,zorder=3)
    if s['estimate'] is not None:
        x,y,t = s['estimate']
        col=C['robot'] if s['converged'] else C['uncertain']
        circle(ax,[x,y],rr,col,alpha=.14)
        ax.scatter(x,y,c=col,s=46,zorder=9)
        ax.arrow(x,y,35*math.cos(t),35*math.sin(t),head_width=13,head_length=14,
                 length_includes_head=True,fc=col,ec=col,zorder=9)
        label(ax,[x,y],'MCL estimate'+('' if s['converged'] else ' (?)'),col,shift=(9,-18))
    ax.set_xlabel('World X (cm)'); ax.set_ylabel('World Y (cm)')
    ax.set_aspect('equal',adjustable='box');ax.grid(True,alpha=.18)


def draw_local(ax,s):
    plan=s.get('plan')
    has_local_plan=bool(plan and plan.get('frame')=='local')
    ax.set_title('LOCAL MAP  |  right / forward (cm)' + ('  [plan-start frame]' if has_local_plan else '  [now]'),
                 loc='left',fontweight='bold',fontsize=11)
    ax.set_facecolor('white')
    if has_local_plan:
        m=plan.get('map') or {}
        bounds=m.get('bounds')
        landmarks=m.get('landmarks',[])
        obstacles=m.get('obstacles',[])
        rr=m.get('robot_radius',22)
        orad=m.get('obstacle_radius',20)
    else:
        bounds=None
        landmarks=[[p,20,i] for i,p in s['local_landmarks'].items()]
        obstacles=[[p,i] for i,p in s['local_obstacles'].items()]
        rr=(s.get('global_map') or {}).get('robot_radius',22)
        orad=(s.get('global_map') or {}).get('obstacle_radius',20)
    pts=[p for p,_,_ in landmarks]+[p for p,_ in obstacles]
    if bounds:
        xmin,ymin=bounds[0]; xmax,ymax=bounds[1]
    else:
        side=max([110]+[abs(p[0])+70 for p in pts])
        ymax=max([170]+[p[1]+65 for p in pts])
        xmin,ymin,xmax,ymax=-side,-40,side,ymax
    ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)
    for p,r,i in landmarks:
        circle(ax,p,r+rr,C['clearance'],dash=True)
        circle(ax,p,20,C['landmark'])
        ax.scatter(*p,c=C['landmark'],marker='s',s=40,zorder=5)
        label(ax,p,'L'+str(i),C['landmark'])
    for p,i in obstacles:
        circle(ax,p,orad+rr,C['clearance'],dash=True)
        circle(ax,p,orad,C['obstacle'])
        ax.scatter(*p,c=C['obstacle'],s=32,zorder=5)
        label(ax,p,'O'+str(i),C['obstacle'])
    if has_local_plan:
        draw_route(ax,plan,world=False)
    else:
        circle(ax,(0,0),rr,C['robot'])
        ax.arrow(0,0,0,35,head_width=12,head_length=12,
                 length_includes_head=True,fc=C['robot'],ec=C['robot'],zorder=8)
        label(ax,(0,0),'MIRTE (now)',C['robot'])
    ax.set_xlabel('Robot right / plan X (cm)');ax.set_ylabel('Robot forward / plan Y (cm)')
    ax.set_aspect('equal',adjustable='box');ax.grid(True,alpha=.18)


def draw_status(ax,s,elapsed):
    ax.axis('off')
    def write(y,string,fontsize=9,color=C['ink'],bold=False):
        ax.text(0,y,str(string),transform=ax.transAxes,ha='left',va='top',fontsize=fontsize,
                color=color,fontweight='bold' if bold else 'normal',clip_on=False)
    write(.99,'NAVIGATION STATUS',12,bold=True)
    converged=s['converged']
    write(.91,'MCL CONVERGED' if converged else 'MCL NOT CONVERGED',10,
          C['robot'] if converged else C['uncertain'],True)
    write(.84,f'Elapsed: {elapsed:.1f} s')
    write(.79,f'Target ID: {s["target"] if s["target"] is not None else "-"}')
    write(.74,'Action: '+str(s['decision'])[:36])
    if s['estimate']:
        x,y,t=s['estimate']
        write(.66,f'Estimated X, Y: {x:.0f}, {y:.0f} cm')
        write(.61,f'Heading: {math.degrees(t)%360:.0f} deg')
    else:
        write(.66,'Estimated position: unknown')
    if s['sigma']:
        sx,sy,st=s['sigma']
        write(.54,'Particle spread (1 sigma):',bold=True)
        write(.49,f'X: {sx:.1f} cm / Y: {sy:.1f} cm')
        write(.44,f'Angle: {math.degrees(st):.1f} deg')
    write(.37,'Camera landmarks: '+', '.join(s['local_landmarks'])[:23])
    write(.32,'Camera obstacles: '+', '.join(s['local_obstacles'])[:23])
    p=s['plan']
    write(.25,f'RRT: {p["frame"] if p else "-"} / nodes: {len(p["edges"])+1 if p else "-"}')
    write(.20,'Last events:',bold=True)
    for j,e in enumerate(reversed(s['events'][-3:])):
        msg=str(e.get('message',''))
        write(.16-j*.05,msg[:37],fontsize=8)
    write(-.035,'* Traces are MCL estimates, not ground truth.',fontsize=7,color='#607286')


def create_figure(s,elapsed=0):
    fig=plt.figure(figsize=(16,7.1),facecolor=C['paper'])
    gs=fig.add_gridspec(1,3,width_ratios=[1.2,1.1,.75],
                        left=.05,right=.975,top=.86,bottom=.16,wspace=.26)
    ax1,ax2,ax3=[fig.add_subplot(gs[0,i]) for i in range(3)]
    fig.suptitle('MIRTE  |  robot localization, observations & route planning',
                 fontsize=16,fontweight='bold',y=.975,color=C['ink'])
    draw_world(ax1,s);draw_local(ax2,s);draw_status(ax3,s,elapsed)
    legend=[Line2D([],[],marker='o',color=C['robot'],ls='none',label='MCL estimate'),
            Line2D([],[],marker='.',color=C['particle'],ls='none',label='Particles'),
            Line2D([],[],marker='s',color=C['landmark'],ls='none',label='Landmark'),
            Line2D([],[],marker='o',color=C['obstacle'],ls='none',label='Obstacle'),
            Line2D([],[],color=C['clearance'],ls='--',label='Safety clearance'),
            Line2D([],[],color=C['tree'],label='RRT explored'),
            Line2D([],[],color=C['path'],linewidth=3,label='Selected path'),
            Line2D([],[],color=C['history'],linestyle='--',label='MCL history')]
    fig.legend(handles=legend,loc='lower center',ncol=4,frameon=False,fontsize=9,
               bbox_to_anchor=(.5,.005))
    return fig


def install_hooks(rec):
    """Wrap only functions in memory. The original calls and return values are preserved."""
    import camera, drive, mcl, particle, planning, rrt

    orig_initialize=particle.initialize_particles
    def initialize(*args,**kwargs):
        out=orig_initialize(*args,**kwargs)
        try: rec.prior(out)
        except Exception as e: print(f'[VISUALIZER] Prior logging: {e}', file=sys.stderr)
        return out
    particle.initialize_particles=initialize

    orig_cam=camera.get_observations
    def observations(*args,**kwargs):
        out=orig_cam(*args,**kwargs)
        try:
            if isinstance(out,tuple) and len(out)==3:
                rec.observations(*out)
        except Exception as e:
            print(f'[VISUALIZER] Camera logging: {e}',file=sys.stderr)
        return out
    camera.get_observations=observations

    orig_update=mcl.MCL.update_mcl
    def update(*args,**kwargs):
        out=orig_update(*args,**kwargs)
        try:
            rec.mcl(args[0])
        except Exception as e:
            print(f'[VISUALIZER] MCL logging: {e}',file=sys.stderr)
        return out
    mcl.MCL.update_mcl=staticmethod(update)

    orig_conv=mcl.MCL.has_converged
    def has_converged(*args,**kwargs):
        out=orig_conv(*args,**kwargs)
        try: rec.convergence(out)
        except Exception: pass
        return out
    mcl.MCL.has_converged=staticmethod(has_converged)

    current_frame={'type':'global'}
    orig_rrt=rrt.RRT.planning
    def rrt_plan(self,*args,**kwargs):
        out=orig_rrt(self,*args,**kwargs)
        try: rec.rrt(self,out,current_frame['type'])
        except Exception as e: print(f'[VISUALIZER] RRT logging: {e}',file=sys.stderr)
        return out
    rrt.RRT.planning=rrt_plan

    orig_planner=planning.rrt_plan
    def plan(*args,**kwargs):
        local=bool(args[3] if len(args)>3 else kwargs.get('local_plan',False))
        current_frame['type']='local' if local else 'global'
        out=orig_planner(*args,**kwargs)
        try: rec.selected_path(out,current_frame['type'])
        except Exception as e: print(f'[VISUALIZER] Path logging: {e}',file=sys.stderr)
        return out
    planning.rrt_plan=plan

    orig_exec=drive.Execute_path
    def execute(*args,**kwargs):
        target=args[9] if len(args)>9 else kwargs.get('target')
        global_plan=args[11] if len(args)>11 else kwargs.get('global_plan',False)
        try: rec.execute(getattr(target,'id','?'),'global' if global_plan else 'local')
        except Exception as e: print(f'[VISUALIZER] Execution logging: {e}', file=sys.stderr)
        return orig_exec(*args,**kwargs)
    drive.Execute_path=execute

    orig_turn=drive.turn
    def turn(*args,**kwargs):
        try: rec.scan(args[0])
        except Exception: pass
        return orig_turn(*args,**kwargs)
    drive.turn=turn


def run_robot(main_file, recorder):
    path=main_file.resolve()
    if not path.is_file():
        raise FileNotFoundError('Could not find original main.py: '+str(path))
    sys.path.insert(0,str(path.parent))
    # Give unmodified original main.py the same import / __file__ semantics.
    namespace={'__name__':'__main__','__file__':str(path),'__package__':None}
    recorder.globals=namespace
    install_hooks(recorder)
    print('[VISUALIZER] Starting UNMODIFIED main.py:',path,flush=True)
    code=compile(path.read_bytes(),str(path),'exec')
    exec(code,namespace)


def start_web(output,port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    html='''<!DOCTYPE html><html><head><meta charset="utf-8"><title>MIRTE live map</title>
<style>body{font:16px system-ui;background:#eef2f8;color:#263548;text-align:center;margin:20px}
img{width:min(98vw,1600px);box-shadow:0 3px 18px #aab;border-radius:8px}</style></head>
<body><h2>MIRTE - navigation map</h2><p>Refresh every 5 seconds; robot program continues independently of the browser.</p>
<img id="map" alt="Map is being generated" src="/latest.png">
<script>setInterval(()=>{document.getElementById('map').src='/latest.png?t='+Date.now()},5000)</script>
</body></html>'''
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith('/latest.png'):
                filename=output/'latest.png'
                if not filename.exists():
                    self.send_error(404,'Map not generated yet');return
                data=filename.read_bytes()
                self.send_response(200);self.send_header('Content-Type','image/png')
                self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)))
                self.end_headers();self.wfile.write(data)
            elif self.path=='/':
                data=html.encode('utf-8')
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
                self.end_headers();self.wfile.write(data)
            else: self.send_error(404)
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True,name='MIRTE-web')
    thread.start()
    print(f'[VISUALIZER] Optional browser: http://localhost:{port} (SSH tunnel from laptop)',flush=True)
    return server


def demo(rec):
    """Hardware-free smoke test; synthetic data are NOT real MIRTE observations."""
    rec.globals={
        'landmarks':{3:(0,0),6:(0,300),4:(400,0),1:(300,400)},
        'target':type('Goal',(),{'id':1})(),
    }
    with rec.lock:
        rec.status['global_map']={'landmarks': [[[0,0],60,'3'],[[0,300],60,'6'],[[400,0],60,'4'],[[300,400],20,'1']],
            'obstacles':[], 'bounds':[[-120,-120],[420,520]], 'robot_radius':22,'obstacle_radius':20}
        rng=np.random.default_rng(12)
        pts=np.column_stack([rng.normal(120,30,650),rng.normal(135,35,650),rng.normal(.65,.3,650)])
        rec.status['particles']=pts.tolist()
        rec.status['estimate']=[120,135,.75]
        rec.status['converged']=True
        rec.status['sigma']=[22,22,9*math.pi/180]
        rec.status['history']=[[60,85],[75,92],[95,105],[120,135]]
    rec.observations({}, {1:[40,155],3:[-60,85]}, {11:[10,85],12:[-40,110]})
    rec.selected_path([[35,170],[35,130],[-35,85],[0,0]],'local')
    with rec.lock:
        rec.status['plan'].update({'start':[0,0], 'goal':[35,170],
          'map':{'landmarks': [[[40,155],20,'1'],[[-60,85],60,'3']],
                 'obstacles':[[[10,85],'11'],[[-40,110],'12']],
                 'bounds':[[-110,-35],[110,225]],'robot_radius':22,'obstacle_radius':20},
          'edges':[[[0,0],[-25,40]],[[-25,40],[-35,85]],[[-35,85],[25,135]],
                   [[0,0],[30,30]],[[30,30],[58,100]],[[25,135],[35,170]]]})
    rec.event('demo','Synthetic demonstration only: no robot connected')
    rec.save(final=True)


def watch_telemetry(output: Path, period: float, web=False, port=8765):
    """Independent plotter process. Only reads telemetry.json; never imports MIRTE."""
    output.mkdir(parents=True, exist_ok=True)
    frames = output / 'frames'
    frames.mkdir(exist_ok=True)
    server = None
    if web:
        try:
            server = start_web(output, port)
        except OSError as e:
            print('[VISUALIZER] Browser preview disabled:', e, file=sys.stderr)
    start_saved = False
    frame_number = 0
    last_render = 0.0
    session = None
    start_wait = time.monotonic()
    try:
        while True:
            path = output / 'telemetry.json'
            if not path.is_file():
                if time.monotonic() - start_wait > 45:
                    raise TimeoutError('No telemetry received from robot process')
                time.sleep(.25)
                continue
            try:
                snap = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                time.sleep(.2)
                continue
            # A force-killed main.py cannot publish 'closed'. Avoid an
            # orphan plotter process waiting forever on old telemetry.
            try:
                stale_for = time.time() - path.stat().st_mtime
                if stale_for > 30:
                    print('[VISUALIZER] Telemetry stale for 30 seconds; stopping watcher.', flush=True)
                    break
            except OSError:
                continue
            if session is None:
                session = snap['session']
            elif snap['session'] != session:
                # The old main ended and a new run has begun.
                session = snap['session']
                start_saved = False
                frame_number = 0
                last_render = 0
            start_state = snap.get('start_state')
            if start_state is not None and not start_saved:
                f = create_figure(start_state, elapsed=0)
                try:
                    f.savefig(output / 'start.png', dpi=105)
                    start_saved = True
                finally:
                    plt.close(f)
            now = time.monotonic()
            closed = bool(snap.get('closed', False))
            if frame_number == 0 or closed or now - last_render >= period:
                state = snap['state']
                f = create_figure(state, elapsed=snap.get('elapsed_s', 0))
                try:
                    frame_number += 1
                    f.savefig(frames / f'map_{frame_number:05d}.png', dpi=105)
                    tmp = output / '_latest_render.png'
                    f.savefig(tmp, dpi=105)
                    os.replace(tmp, output / 'latest.png')
                    state_tmp = output / '_latest_state.tmp'
                    state_tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
                    os.replace(state_tmp, output / 'latest_state.json')
                    last_render = time.monotonic()
                finally:
                    plt.close(f)
            if closed:
                print('[VISUALIZER] Final map:', output / 'latest.png', flush=True)
                break
            time.sleep(.25)
    finally:
        if server:
            server.shutdown()


def spawn_viewer(output: Path, period: float, web: bool, port: int):
    """Launch the renderer as a separate Linux process, away from main.py."""
    cmd = [sys.executable, str(Path(__file__).resolve()), '--watch',
           '--output', str(output), '--period', str(period), '--port', str(port)]
    if web:
        cmd.append('--web')
    logfile = (output / 'visualizer.log').open('a', encoding='utf-8')
    try:
        proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent),
                                stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT,
                                start_new_session=True)
    finally:
        logfile.close()
    print(f'[VISUALIZER] Background renderer PID: {proc.pid}', flush=True)
    print(f'[VISUALIZER] Map: {output / "latest.png"}', flush=True)
    return proc



# ---------------------------------------------------------------------------
# Importable API for the original main.py (normal way to run the robot).
# ---------------------------------------------------------------------------
_ACTIVE_RECORDER = None
_ACTIVE_VIEWER = None


def start(main_globals=None, output=None, period=5.0, web=False, port=8765):
    """Start telemetry and an independent background map-rendering process.

    Call ONCE from main.py after the world map is defined and before the
    particle filter is initialized. The original navigation still runs as main.

    Parameters:
        main_globals: globals() from main.py (gives access to target and map)
        output: optional path; defaults to Ex6/mirte_debug on MIRTE
        period: seconds between PNG screenshots (default five)
        web: optional built-in HTTP preview at localhost:8765
    """
    global _ACTIVE_RECORDER, _ACTIVE_VIEWER
    if _ACTIVE_RECORDER is not None:
        return _ACTIVE_RECORDER
    if period <= 0:
        raise ValueError('period must be > 0')
    output = Path(output).resolve() if output else Path('/home/mirte/Mirte-proto/Mirte_code/Ex6/mirte_debug')
    rec = Recorder(output, period=period, publisher=True)
    rec.globals = main_globals if main_globals is not None else {}
    try:
        # The wrappers only observe returns and do not overwrite source files.
        # Install BEFORE main calls initialize_particles/update_mcl/rrt_plan.
        install_hooks(rec)
        rec._publish()
        viewer = spawn_viewer(output, period, web, port)
    except Exception:
        rec.close()
        raise
    _ACTIVE_RECORDER = rec
    _ACTIVE_VIEWER = viewer
    atexit.register(stop)
    rec.event('start', 'Visualizer attached to main.py')
    return rec


def note(message):
    """Optional explicit decision text shown in the status panel/log."""
    rec = _ACTIVE_RECORDER
    if rec is None:
        return
    try:
        with rec.lock:
            rec.status['decision'] = str(message)
        rec.event('decision', str(message))
    except Exception as exc:
        print(f'[VISUALIZER] Decision logging failed: {exc}', file=sys.stderr)


def stop():
    """Publish final map. Safe to call more than once and on normal program exit."""
    global _ACTIVE_RECORDER, _ACTIVE_VIEWER
    rec = _ACTIVE_RECORDER
    if rec is None:
        return
    _ACTIVE_RECORDER = None
    _ACTIVE_VIEWER = None
    try:
        rec.close()
    except Exception as exc:
        print(f'[VISUALIZER] Could not finalize: {exc}', file=sys.stderr)


def main():
    parser=argparse.ArgumentParser(description='One-file map visualizer for the ORIGINAL MIRTE main.py')
    parser.add_argument('--main',default='main.py',help='Existing main.py to run unchanged')
    parser.add_argument('--background', action='store_true', help='Run unmodified main in foreground, map rendering in another process')
    parser.add_argument('--watch', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output',default='/home/mirte/Mirte-proto/Mirte_code/Ex6/mirte_debug',help='Directory for PNG and logs')
    parser.add_argument('--period',type=float,default=5.0,help='Save a frame every N seconds')
    parser.add_argument('--web',action='store_true',help='Offer live image over localhost:8765')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--demo',action='store_true',help='Test plotting without robot (synthetic data)')
    args=parser.parse_args()
    if args.period <= 0: parser.error('--period must be greater than zero')
    output=Path(args.output).resolve()
    if args.watch:
        watch_telemetry(output, args.period, args.web, args.port)
        return
    rec=Recorder(output,args.period,publisher=args.background)
    server=None
    viewer=None
    try:
        if args.background:
            # Publish immediately so the viewer has a valid session to read.
            rec._publish()
            viewer = spawn_viewer(output, args.period, args.web, args.port)
        if args.demo:
            demo(rec)
        else:
            if args.web and not args.background:
                try: server=start_web(output,args.port)
                except OSError as e: print('[VISUALIZER] Web server skipped:',e,file=sys.stderr)
            run_robot(Path(args.main),rec)
    except KeyboardInterrupt:
        print('\n[VISUALIZER] Ctrl+C received; storing final map.',flush=True)
    except Exception as e:
        rec.event('error',f'Original main.py stopped: {e}')
        print('[VISUALIZER] Original program ended with an exception:',file=sys.stderr)
        traceback.print_exc()
        raise
    finally:
        rec.close()
        if server: server.shutdown()


if __name__=='__main__':
    main()
