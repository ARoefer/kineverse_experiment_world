#!/usr/bin/env python3
import rospy
import kineverse.gradients.gradient_math as gm
import tf2_ros
import tf2_kdl
import numpy  as np
import pandas as pd
import math

from kineverse_msgs.msg             import ValueMap as ValueMapMsg
from kineverse.model.geometry_model import GeometryModel, Path, Frame
from kineverse.ros.tf_publisher     import ModelTFBroadcaster_URDF
from kineverse.visualization.ros_visualizer import ROSVisualizer

from kineverse_experiment_world.qp_state_model import QPStateModel, QPSolverException
from kineverse_experiment_world.nobilia_shelf  import create_nobilia_shelf
from kineverse_experiment_world.msg import TransformStampedArray as TransformStampedArrayMsg

def np_frame3_quaternion(x, y, z, qx, qy, qz, qw):
    a  = [qx, qy, qz, qw]
    mf = np.array
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    qw2 = qw * qw
    return mf([[qw2 + qx2 - qy2 - qz2, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy, x],
               [2 * qx * qy + 2 * qw * qz, qw2 - qx2 + qy2 - qz2, 2 * qy * qz - 2 * qw * qx, y],
               [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, qw2 - qx2 - qy2 + qz2, z],
               [0, 0, 0, 1]])

def np_6d_pose_feature(x, y, z, qx, qy, qz, qw):
    axis_norm = np.sqrt(qx**2 + qy**2 + qz**2)
    angle = np.arctan2(axis_norm, qw) * 2
    # print(f'Angle of quat: qw -> {math.acos(qw) * 2}, {math.asin(axis_norm) * 2}. From arctan2: {angle}')
    if np.abs(angle) < 1e-4:
        return [x, y, z, 0, 0, 0]
    rotation_vector = (np.array([qx, qy, qz]) / axis_norm) * angle
    # print(f'Rotation vector: {rotation_vector} Norm: {np.sqrt(np.sum(rotation_vector**2))}')
    return np.hstack(([x, y, z], rotation_vector))

def generate_6d_feature(pose):
    rotation = gm.rotation_vector_from_matrix(pose)
    position = gm.pos_of(pose)
    return gm.matrix_wrapper([position[0], position[1], position[2], rotation[0], rotation[1], rotation[2]])


class Kineverse6DQPTracker(object):
    def __init__(self, km, paths_observables, transition_rules=None):
        """Initializes the tracker to estimate variables based 
           on the given paths. The paths are assumed to point to
           6D poses.
        
        Args:
            km (ArticulationModel): Articulation model to used
            paths_observables (list): List of paths to poses that will be observed
            noise_estimation_observations (int): Number of observations to collect before initializing the EKFs
            estimate_init_steps (int): Number of steps used to initialize the estimate of an ekf
        """
        poses = {p: km.get_data(p) for p in paths_observables}
        poses = {str(p): pose.pose if isinstance(pose, Frame) else pose for p, pose in poses.items()}
        # obs_features = {p: generate_6d_feature(pose) for p, pose in poses.items()}
        obs_features = {p: pose for p, pose in poses.items()}

        # Identify tracking pools. All poses within one pool share at least one DoF
        tracking_pools = [(gm.free_symbols(feature), [(p, feature)]) for p, feature in obs_features.items() 
                                                                     if len(gm.free_symbols(feature)) != 0]

        final_pools = []
        while len(tracking_pools) > 0:
            syms, feature_list = tracking_pools[0]
            initial_syms_size = len(syms)
            y = 1
            while y < len(tracking_pools):
                o_syms, o_feature_list = tracking_pools[y]
                if len(syms.intersection(o_syms)) != 0:
                    syms = syms.union(o_syms)
                    feature_list += o_feature_list
                    del tracking_pools[y]
                else:
                    y += 1

            # If the symbol set has not been enlarged, 
            # there is no more overlap with other pools
            if len(syms) == initial_syms_size:
                final_pools.append((syms, feature_list))
                tracking_pools = tracking_pools[1:]
            else:
                tracking_pools[0] = (syms, feature_list)

        tracking_pools  = final_pools
        self.estimators = [QPStateModel(km, dict(features),
                                        transition_rules=transition_rules) for symbols, features in tracking_pools]

        print(f'Generated {len(self.estimators)} Estimators:\n  '
               '\n  '.join(str(e) for e in self.estimators))

        self.observation_names = [str(p) for p in paths_observables]

    def process_control(self, controls):
        for estimator in self.estimators:
            estimator.set_command(controls)

    def process_observation(self, observation):
        """Processes a single observation
        
        Args:
            observation (dict): Dictionary of the observations
        """
        for estimator in self.estimators:
            estimator.update(observation)

    def get_estimated_state(self):
        out = {}
        for estimator in self.estimators:
            out.update(estimator.state())
        return out


class ROSQPEManager(object):
    def __init__(self, tracker : Kineverse6DQPTracker, 
                       model=None, model_path=None, reference_frame='world', urdf_param='/qp_description_check', update_freq=30):
        self.tracker = tracker 
        self.last_observation = {}
        self.last_update      = None
        self.reference_frame  = reference_frame

        self.vis = ROSVisualizer('~vis', reference_frame)

        if model is not None:
            self.broadcaster = ModelTFBroadcaster_URDF(urdf_param, model_path, model, Path('ekf'))
        else:
            self.broadcaster = None

        self.tf_buffer = tf2_ros.Buffer()
        self.listener  = tf2_ros.TransformListener(self.tf_buffer)

        self.pub_state        = rospy.Publisher('~state_estimate', ValueMapMsg, queue_size=1, tcp_nodelay=True)
        self.sub_observations = rospy.Subscriber('~observations',  TransformStampedArrayMsg, callback=self.cb_obs, queue_size=1)
        self.sub_controls     = rospy.Subscriber('~controls',      ValueMapMsg, callback=self.cb_controls, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1 / update_freq), self.cb_update)

    def cb_obs(self, transform_stamped_array_msg):
        ref_frames = {}
        
        for trans in transform_stamped_array_msg.transforms:
            if trans.header.frame_id != self.reference_frame:
                if trans.header.frame_id not in ref_frames:
                    try:
                        ref_trans = self.tf_buffer.lookup_transform(self.reference_frame, trans.header.frame_id, rospy.Time(0))
                        ref_frames[trans.header.frame_id] = ref_trans
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                        print(f'Exception raised while looking up {trans.header.frame_id} -> {self.reference_frame}:\n{e}')
                        continue
                else:
                    ref_trans = tf2_kdl.transform_to_kdl(ref_frames[trans.header.frame_id])

                kdl_trans = tf2_kdl.transform_to_kdl(trans)
                kdl_trans = ref_trans * kdl_trans
                dir(kdl_trans)
            else:
                matrix = np_frame3_quaternion(trans.transform.translation.x, 
                                              trans.transform.translation.y, 
                                              trans.transform.translation.z,
                                              trans.transform.rotation.x,
                                              trans.transform.rotation.y,
                                              trans.transform.rotation.z,
                                              trans.transform.rotation.w)
                self.last_observation[trans.child_frame_id] = matrix
                # self.last_observation[trans.child_frame_id] = np_6d_pose_feature(trans.transform.translation.x, 
                #                                                                  trans.transform.translation.y, 
                #                                                                  trans.transform.translation.z,
                #                                                                  trans.transform.rotation.x,
                #                                                                  trans.transform.rotation.y,
                #                                                                  trans.transform.rotation.z,
                #                                                                  trans.transform.rotation.w)
        try:
            self.tracker.process_observation(self.last_observation)
        except QPSolverException as e:
            print(f'Solver crashed during observation update. Skipping observation...')
            return
        self.vis.begin_draw_cycle('observations')
        # temp_poses = [gm.frame3_axis_angle(feature[3:] / np.sqrt(np.sum(feature[3:]**2)), np.sqrt(np.sum(feature[3:]**2)), feature[:3]) for feature in self.last_observation.values()]
        self.vis.draw_poses('observations', np.eye(4), 0.2, 0.01, self.last_observation.values())
        # self.vis.draw_poses('observations', np.eye(4), 0.2, 0.01, temp_poses)
        self.vis.render('observations')

    def cb_controls(self, map_message):
        self.last_control = {gm.Symbol(str_symbol): v for str_symbol, v in zip(map_message.symbol, map_message.value) 
                                                      if str_symbol in self.str_controls}
        self.tracker.process_control(self.last_control)

    def cb_update(self, *args):
        if len(self.last_observation) == 0:
            return

        est_state = self.tracker.get_estimated_state()

        state_msg = ValueMapMsg()
        state_msg.header.stamp = rospy.Time.now()
        state_msg.symbol, state_msg.value = zip(*est_state.items())
        state_msg.symbol = [str(s) for s in state_msg.symbol]
        self.pub_state.publish(state_msg)

        if self.broadcaster is not None:
            self.broadcaster.update_state(est_state)
            self.broadcaster.publish_state()




if __name__ == '__main__':
    rospy.init_node('kineverse_ar_tracker')

    km = GeometryModel()

    # try:
    #     location_x   = rospy.get_param('/nobilia_x')
    #     location_y   = rospy.get_param('/nobilia_y')
    #     location_z   = rospy.get_param('/nobilia_z')
    #     location_yaw = rospy.get_param('/nobilia_yaw')
    # except KeyError as e:
    #     print(f'Failed to get nobilia localization params. Original error:\n{e}')
    #     exit()

    shelf_location = gm.point3(*[gm.Position(f'nobilia_location_{x}') for x in 'xyz'])

    shelf_yaw = gm.Position('nobilia_location_yaw')
    # origin_pose = gm.frame3_rpy(0, 0, 0, shelf_location)
    origin_pose = gm.frame3_rpy(0, 0, shelf_yaw, shelf_location)

    create_nobilia_shelf(km, Path('nobilia'), origin_pose)

    km.clean_structure()
    km.dispatch_events()

    shelf   = km.get_data('nobilia')
    tracker = Kineverse6DQPTracker(km, 
                                    [Path(f'nobilia/markers/{name}') for name in shelf.markers.keys()], # 
                                    # [Path(f'nobilia/markers/body')],
                                    # [Path(f'nobilia/markers/top_panel')],
                                    #{x: x for x in gm.free_symbols(origin_pose)},
                                   )

    node = ROSQPEManager(tracker, shelf, Path('nobilia'), 'world')

    # reference_frame = rospy.get_param('~reference_frame', 'world')
    # grab_from_tf    = rospy.get_param('~grab_from_tf', False)
    # grab_rate       = rospy.get_param('~grab_rate', 20.0)

    while not rospy.is_shutdown():
        rospy.sleep(0.1)



