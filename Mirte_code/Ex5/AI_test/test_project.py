"""Offline regression tests; never instantiate physical KU_Mirte."""
import contextlib
import io
import math
import importlib
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import cv2
import ex5_config as cfg
from mcl import MCL,wrap_angle
from local_map import LocalMap,marker_pose_to_box
import aruco_measurements as aruco
import mirte_rrt_smooth as planner
import path_smoothing
import path_follower
import between
from framebuffer import FrameBuffer
from grid_occ import GridOccupancyMap
from robot_models import PointMassModel,MirteModel
from robot_io import move
from simulate_mcl import observations_for_pose,run_controller,stationary_sweep,SimulatedMirte


def box(x,z,i=1,angle=0):
    # Base front-facing marker; yaw around the camera vertical axis.
    ry,_=cv2.Rodrigues(np.array([0.,angle,0.]));rx,_=cv2.Rodrigues(np.array([math.pi,0.,0.]))
    r,_=cv2.Rodrigues(ry@rx)
    b=marker_pose_to_box([x*1000,0,z*1000],r,[0,.14]);b['id']=i;return b


class Tests(unittest.TestCase):
    def setUp(self):np.random.seed(7)
    def test_01_all_modules_import(self):
        for p in pathlib.Path(__file__).parent.glob('*.py'):
            if p.stem!='test_project':importlib.import_module(p.stem)
    def test_02_mcl_prior_is_not_localization(self):
        f=MCL(cfg.LANDMARKS);f.correct([])
        self.assertFalse(f.confident());self.assertFalse(f.has_measurements)
    def test_03_mcl_preserves_weights(self):
        f=MCL(cfg.LANDMARKS,number_of_particles=2)
        f.particles=np.array([[0.,-1.,0.],[.1,-1.,0.]])
        obs=observations_for_pose([0,-1,0],False);f.correct(obs);old=f.weights.copy()
        self.assertFalse(f.resample());f.correct(obs)
        expected=old**2;expected/=expected.sum();np.testing.assert_allclose(f.weights,expected)
    def test_04_weighted_pose_without_observations(self):
        f=MCL({},number_of_particles=2);f.particles=np.array([[0.,0.,0.],[1.,0.,0.]])
        f.weights=np.array([.9,.1]);np.testing.assert_allclose(f.correct([]),[.1,0,0])
        self.assertAlmostEqual(f.position_std()[0],.3)
    def test_05_wrap_and_heading_convention(self):
        self.assertAlmostEqual(float(wrap_angle(3*math.pi)),math.pi)
        f=MCL({},initial_pose=[0,0,math.pi/2],initial_std=[0,0,0]);f.predict(1,0)
        self.assertLess(f.estimate_pose()[0],-.95);self.assertLess(abs(f.estimate_pose()[1]),.02)
    def test_06_stationary_sweep(self):
        r=stationary_sweep(100);self.assertEqual(r['over_10cm'],0)
    def test_07_invalid_mcl_inputs(self):
        with self.assertRaises(ValueError):MCL({1:[float('nan'),0]})
        with self.assertRaises(TypeError):MCL(LocalMap())
        f=MCL(cfg.LANDMARKS);self.assertEqual(f.valid_observations([(1,float('nan'),0),(1,-1,0),(99,1,0)]),[])
    def test_08_opencv_detection(self):
        image=np.full((480,640),255,np.uint8);image[195:285,275:365]=cv2.aruco.generateImageMarker(aruco.ARUCO_DICT,1,90)
        obs=aruco.detect_observations_from_image(image,[1]);self.assertEqual(obs[0][0],1)
        self.assertAlmostEqual(obs[0][1],1.134,delta=.015)
        self.assertEqual(aruco.detect_observations_from_image(image,[10]),[])
    def test_09_blank_and_wrong_resolution(self):
        self.assertEqual(aruco.detect_observations_from_image(np.full((480,640),255,np.uint8)),[])
        with self.assertRaises(ValueError):aruco.detect_observations_from_image(np.zeros((100,100),np.uint8))
    def test_10_camera_geometry(self):
        a=aruco.pose_to_robot_geometry([200,0,1000],[math.pi,0,0]);b=box(.2,1)
        np.testing.assert_allclose(a['box_center'],b['box_center']);np.testing.assert_allclose(a['marker_position'],[.2,1.14])
    def test_11_buffer_ownership(self):
        b=FrameBuffer((1,1,1));self.assertIsNone(b.get_frame())
        b.new_frame(np.ones((1,1,1),np.uint8));held=b.get_frame()
        b.new_frame(np.full((1,1,1),2,np.uint8));b.new_frame(np.full((1,1,1),3,np.uint8))
        self.assertEqual(held.item(),1)
    def test_12_grid_dimensions_and_boundaries(self):
        m=GridOccupancyMap();self.assertEqual(m.n_grids,[40,40]);self.assertEqual(m.in_collision([1.99,1.99]),0)
        self.assertEqual(m.in_collision([2.,1]),1)
        m=GridOccupancyMap(low=(-1,-3),high=(1,5));m.populate(3)
        self.assertTrue(m.grid.any())
    def test_13_zero_pointmass_motion(self):
        out=PointMassModel([-.1,.1]).inverse_dyn(np.zeros(2),np.zeros(2),2)
        self.assertTrue(np.all(np.isfinite(out)));np.testing.assert_equal(out,np.zeros((2,2)))
    def test_14_stateless_mirte_branches(self):
        m=MirteModel([-.1,.1]);x=np.array([0.,0.,0.]);a=m.inverse_dyn(x,[1.,0.],3)
        m.inverse_dyn(x,[-1.,0.],3);b=m.inverse_dyn(x,[1.,0.],3)
        np.testing.assert_equal(a,b)
    def test_15_particle_von_mises_wrap(self):
        import particle
        p=particle.Particle(theta=.3)
        with patch('random_numbers.rand_von_mises',lambda mu,k:mu):particle.add_uncertainty_von_mises([p],0,10)
        self.assertAlmostEqual(p.theta,.3)
    def test_16_exact_segment_collision(self):
        m=LocalMap(mirte_radius=.001,clearance_margin=0);b=box(0,.2)
        m.set_boxes({1:b});self.assertFalse(m.segment_is_free([0,0],[0,1]))
        self.assertFalse(m.segment_is_free(b['box_center'],b['box_center']))
    def test_17_tilted_passage_margin(self):
        m=LocalMap();m.set_boxes({1:box(-.905/2,1,1,-.2),2:box(.905/2,1,2,-.2)})
        goal,width=between.passage_between_ids(m,1,2);self.assertIsNone(goal);self.assertLess(width,.49)
    def test_18_passage_and_target(self):
        m=LocalMap();m.set_boxes({1:box(-.6,1,1),10:box(.6,1,10)})
        goal,width=between.passage_between_ids(m,1,10);self.assertAlmostEqual(width,.8)
        self.assertFalse(m.in_collision(goal));self.assertIsNotNone(between.goal_in_front_of_box(m,1,.4))
    def test_19_two_frame_map_consistency(self):
        count=[0]
        def capture(self,robot):
            count[0]+=1;ids=[1,2] if count[0]==1 else [1,3]
            self.set_boxes({i:box(i*.1,1+count[0]*.01,i) for i in ids});return self.landmarks
        with patch.object(LocalMap,'get_map_from_mirte',capture),patch('time.sleep',lambda _:None):m=planner.make_map(None)
        self.assertEqual(m.get_visible_box_ids(),[1]);self.assertEqual([i for _,i in m.landmarks],[1])
        np.testing.assert_allclose(m.landmarks[0][0],m.boxes[1]['box_center'])
    def test_20_observation_does_not_mutate_map(self):
        m=LocalMap();m.set_boxes({1:box(0,1)})
        class Robot:
            def get_image_compressed(self):return None
        self.assertEqual(planner.get_aruco_observations(Robot(),m),[]);self.assertEqual(m.get_visible_box_ids(),[1])
    def test_21_rrt_detour_valid(self):
        m=LocalMap();m.set_boxes({1:box(0,.9)})
        with contextlib.redirect_stdout(io.StringIO()):path=planner.plan_path(m,[0,2])
        self.assertIsNotNone(path)
        self.assertTrue(all(m.segment_is_free(a,b) for a,b in zip(path[:-1],path[1:])))
    def test_22_bad_path_rejected(self):
        m=LocalMap();m.set_boxes({1:box(0,.9)})
        with self.assertRaises(ValueError):path_smoothing.smooth_path([[0,0],[0,2]],m)
    def test_23_sonar_fails_closed(self):
        b=SimulatedMirte();b.sonar={}
        with self.assertRaises(KeyError):move(b,.1,0,.2,.6)
        self.assertFalse(any(v!=0 for v,w,t in b.commands))
        b.sonar={'front_left':.1,'front_right':2}
        with self.assertRaises(RuntimeError):move(b,.1,0,.2,.6)
    def test_24_controller_without_markers(self):
        r=run_controller(visible=False);self.assertFalse(r['reported_success']);self.assertEqual(r['translations'],0)
    def test_25_controller_with_field_of_view(self):
        for seed in range(5):
            r=run_controller(seed);self.assertTrue(r['reported_success'],r['log'])
            self.assertLessEqual(r['physical_simulated_goal_error_m'],cfg.GOAL_TOLERANCE_M)
    def test_26_controller_with_motor_drift(self):
        for seed in range(5):
            r=run_controller(seed,.9,1.1)
            self.assertTrue(r['reported_success'],r['log']);self.assertLessEqual(r['physical_simulated_goal_error_m'],cfg.GOAL_TOLERANCE_M)
    def test_27_local_path_integration(self):
        m=LocalMap();m.set_boxes({1:box(-.6,2,1),10:box(.6,2,10)})
        b=SimulatedMirte(pose=[0,0,0]);markers={i:box['marker_position'] for i,box in m.boxes.items()}
        def observation(robot,allowed_ids=None):
            out=[];x,z,theta=b.pose
            for i,(lx,lz) in markers.items():
                dx,dz=lx-x,lz-z;out.append((i,math.hypot(dx,dz),float(wrap_angle(math.atan2(-dx,dz)-theta))))
            return out
        with patch.object(planner,'observe_mirte',observation),patch('time.sleep',lambda _:None),contextlib.redirect_stdout(io.StringIO()):
            success=planner.drive_path(b,m,[[0,0],[0,.3],[.25,.3]])
        self.assertTrue(success);self.assertLess(np.linalg.norm(b.pose[:2]-[.25,.3]),.10)
    def test_28_local_path_markers_lost(self):
        m=LocalMap();m.set_boxes({1:box(-.6,2,1)})
        b=SimulatedMirte(pose=[0,0,0])
        with patch.object(planner,'observe_mirte',lambda *a,**k:[]),patch('time.sleep',lambda _:None),contextlib.redirect_stdout(io.StringIO()):
            success=planner.drive_path(b,m,[[0,0],[0,1]])
        self.assertFalse(success);self.assertLessEqual(np.linalg.norm(b.pose[:2]),.32)
    def test_29_plot_and_json_geometry(self):
        from visualize_local_map import plot_local_map,save_map_json
        import json
        m=LocalMap();m.set_boxes({1:box(0,1)})
        with tempfile.TemporaryDirectory() as folder:
            path=pathlib.Path(folder)/'map.json';save_map_json(m,path);data=json.loads(path.read_text())
            self.assertIn('corners',data['boxes']['1']);plot_local_map(m)
    def test_30_camera_horizontal_bearing(self):
        import camera
        c=camera.Camera.__new__(camera.Camera);c.arucoDict=None;c.arucoMarkerLength=.15
        c.intrinsic_matrix=np.eye(3);c.distortion_coeffs=np.zeros(5)
        with patch.object(camera,'detect_markers',lambda *a:([],np.array([[1]]),[])),patch.object(camera,'estimate_markers',lambda *a:(np.zeros((1,1,3)),np.array([[[.2,.3,1.]]]),None)):
            _,_,angles=c.detect_aruco_objects(np.zeros((1,1,3)))
        self.assertAlmostEqual(angles[0],math.atan2(-.2,1))


    def test_31_frozen_camera_is_not_fresh_evidence(self):
        image=np.full((480,640),255,np.uint8);image[195:285,275:365]=cv2.aruco.generateImageMarker(aruco.ARUCO_DICT,1,90)
        class Robot:
            def get_image_compressed(self):return image.copy()
        robot=Robot();self.assertEqual(len(aruco.observe_mirte(robot,[1])),1)
        self.assertEqual(aruco.observe_mirte(robot,[1]),[])
    def test_32_initial_search_rotation(self):
        r=run_controller(pose=(.6,-1.8,math.pi))
        self.assertTrue(r['reported_success'],r['log']);self.assertLess(r['physical_simulated_goal_error_m'],.1)
    def test_33_different_start_positions(self):
        for pose in [(0.35,-.85,math.radians(18)),(.1,-1.5,math.radians(40)),(.7,-1.2,-1.)]:
            r=run_controller(pose=pose)
            if r['reported_success']:self.assertLess(r['physical_simulated_goal_error_m'],.1)
            else:self.assertIn('stopped without confirming goal',r['log'])


if __name__=='__main__':unittest.main(verbosity=2)
