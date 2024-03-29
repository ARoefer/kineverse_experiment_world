#!/usr/bin/env python3
import rospy
import kineverse.gradients.gradient_math as gm
import tf2_ros
import numpy  as np
import pandas as pd
import math

from kineverse_msgs.msg             import ValueMap as ValueMapMsg
from kineverse.model.geometry_model import GeometryModel, Path, Frame
from kineverse.ros.tf_publisher     import ModelTFBroadcaster_URDF
from kineverse.visualization.ros_visualizer import ROSVisualizer

from kineverse_experiment_world.ekf_model     import EKFModel
from kineverse_experiment_world.nobilia_shelf import create_nobilia_shelf
from kineverse_experiment_world.msg           import TransformStampedArray as TransformStampedArrayMsg
from kineverse_experiment_world.utils         import np_frame3_quaternion


def np_6d_pose_feature(x, y, z, qx, qy, qz, qw):
    axis_norm = np.sqrt(qx**2 + qy**2 + qz**2)
    angle = np.arctan2(axis_norm, qw) * 2
    print(f'Angle of quat: qw -> {math.acos(qw) * 2}, {math.asin(axis_norm) * 2}. From arctan2: {angle}')
    if np.abs(angle) < 1e-4:
        return [x, y, z, 0, 0, 0]
    rotation_vector = (np.array([qx, qy, qz]) / axis_norm) * angle
    print(f'Rotation vector: {rotation_vector} Norm: {np.sqrt(np.sum(rotation_vector**2))}')
    return np.hstack(([x, y, z], rotation_vector))

def generate_6d_feature(pose):
    rotation = gm.rotation_vector_from_matrix(pose)
    position = gm.pos_of(pose)
    return gm.matrix_wrapper([position[0], position[1], position[2], rotation[0], rotation[1], rotation[2]])


class Kineverse6DEKFTracker(object):
    def __init__(self, km, paths_observables,
                       transition_rules=None, 
                       noise_estimation_observations=100, estimate_init_steps=50):
        """Initializes the tracker to estimate variables based 
           on the given paths. The paths are assumed to point to
           6D poses.
        
        Args:
            km (ArticulationModel): Articulation model to used
            paths_observables (list): List of paths to poses that will be observed
            noise_estimation_observations (int): Number of observations to collect before initializing the EKFs
            estimate_init_steps (int): Number of steps used to initialize the estimate of an ekf
        """
        poses = {obs_name: km.get_data(p) for p, obs_name in paths_observables.items()}
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

        tracking_pools = final_pools
        self.ekfs = [EKFModel(dict(features), 
                              km.get_constraints_by_symbols(symbols), 
                              transition_rules=transition_rules) for symbols, features in tracking_pools]

        print(f'Generated {len(self.ekfs)} EKFs:\n  '
               '\n  '.join(str(e) for e in self.ekfs))

        self.observation_names = list(poses.keys()) # [str(p) for p in paths_observables]
        self.controls          = set(sum([e.ordered_controls for e in self.ekfs], []))

        self.estimates   = [None] * len(self.ekfs)
        self.obs_counter = 0

        self._n_estimation_observations = noise_estimation_observations
        self._estimation_observations   = []
        self._ekfs_initialized = False
        self._estimate_init_steps = estimate_init_steps


    def compute_observation_covariance(self, observations):
        """Computes the observational covariance from a set of observations
        
        Args:
            observations ([dict]): Dictionary of noisy observations
        """
        for ekf in self.ekfs:
            ekf.generate_R(observations)
        self._ekfs_initialized = True

    def process_update(self, observation, control, dt=0.05):
        """Processes a single observation
        
        Args:
            observation (dict): Dictionary of the observations
        """
        if not self._ekfs_initialized and len(self._estimation_observations) < self._n_estimation_observations:
            self._estimation_observations.append(observation)
        else:
            if not self._ekfs_initialized:
                self.compute_observation_covariance(self._estimation_observations)

            self.obs_counter += 1

            for x, ekf in enumerate(self.ekfs):
                # Currently only using a single estimate
                # Proper particle filtering with a clever initialization
                # would be preferable.
                # For now: initialize the estimate with gradient descent 
                # on the observation
                if self.estimates[x] is None:
                    self.estimates[x] = ekf.spawn_particle()
                    estimate = self.estimates[x]

                    obs_vector = ekf.gen_obs_vector(observation)
                    # obs_delta = obs_vector.reshape((len(obs_vector), 1)) - ekf.h_fn.call2(estimate.state)
                    # df = pd.DataFrame(data=np.hstack((ekf.h_prime_fn.call2(estimate.state), obs_delta)), 
                    #                   index=[str(l) for l in ekf.obs_labels],
                    #                   columns=[str(c) for c in ekf.ordered_controls] + ['obs_delta'])
                    # df.to_csv(f'ekf_{x}_preinit.csv')

                    # deltas = []

                    last_error = 1000
                    while True:
                        h_prime   = ekf.h_prime_fn.call2(estimate.state)
                        obs_delta = obs_vector.reshape((len(obs_vector), 1)) - ekf.h_fn.call2(estimate.state)

                        jjt   = h_prime.dot(h_prime.T)
                        jjt_e = jjt.dot(obs_delta)
                        alpha_numerator = obs_delta.T.dot(jjt_e)
                        temp_denominator = jjt_e.T.dot(jjt_e)
                        alpha = np.abs(alpha_numerator) / np.abs(temp_denominator + (temp_denominator == 0))
                        # print(x, alpha, obs_delta.T.dot(jjt_e), jjt_e.T.dot(jjt_e), np.linalg.cond(jjt))

                        estimate.state += (h_prime.T.dot(obs_delta) * alpha).reshape(estimate.state.shape)
                        error_norm = np.sqrt((obs_delta**2).sum())
                        # deltas.append(np.hstack(([[error_norm, last_error - error_norm]], obs_delta.T)))
                        if last_error - error_norm < 1e-6:
                            break

                        last_error = error_norm


                    # pd.DataFrame(data=np.vstack(deltas), columns=['error_norm', 'error_delta'] + [str(l) for l in ekf.obs_labels]).to_csv(f'ekf_{x}_error_progression.csv')

                    # obs_vector = ekf.gen_obs_vector(observation)
                    # obs_delta = obs_vector.reshape((len(obs_vector), 1)) - ekf.h_fn.call2(estimate.state)
                    # df = pd.DataFrame(data=np.hstack((ekf.h_prime_fn.call2(estimate.state), obs_delta)), 
                    #                   index=[str(l) for l in ekf.obs_labels],
                    #                   columns=[str(c) for c in ekf.ordered_controls] + ['obs_delta'])
                    # df.to_csv(f'ekf_{x}_postinit.csv')
                else:
                    estimate    = self.estimates[x]
                    # print(f'EKF {x} cov:\n{estimate.cov}')
                    estimate.state, estimate.cov = ekf.predict(estimate.state.flatten(), 
                                                               estimate.cov, 
                                                               ekf.gen_control_vector(control), dt=0.05)
                    obs_vector  = ekf.gen_obs_vector(observation)
                    estimate.state, estimate.cov = ekf.update(estimate.state, 
                                                              estimate.cov, 
                                                              obs_vector)

    def get_estimated_state(self):
        if self._ekfs_initialized is None:
            return None
        return dict(sum([list(zip(ekf.ordered_vars, estimate.state.flatten())) for ekf, estimate 
                                                                               in zip(self.ekfs, self.estimates) if estimate is not None], []))

class ROSEKFManager(object):
    def __init__(self, tracker : Kineverse6DEKFTracker, 
                       model=None, model_path=None, urdf_param='/ekf_description_check'):
        self.tracker = tracker 
        self.last_observation = {}
        self.last_update      = None
        self.last_control     = {s: 0.0 for s in self.tracker.controls}
        self.str_controls     = {str(s) for s in self.tracker.controls}

        self.vis = ROSVisualizer('~vis', 'world')

        if model is not None:
            self.broadcaster = ModelTFBroadcaster_URDF(urdf_param, model_path, model, Path('ekf'))
        else:
            self.broadcaster = None

        self.pub_state        = rospy.Publisher('~state_estimate', ValueMapMsg, queue_size=1, tcp_nodelay=True)
        self.sub_observations = rospy.Subscriber('~observations',  TransformStampedArrayMsg, callback=self.cb_obs, queue_size=1)
        self.sub_controls     = rospy.Subscriber('~controls',      ValueMapMsg, callback=self.cb_controls, queue_size=1)

    def cb_obs(self, transform_stamped_array_msg):
        for trans in transform_stamped_array_msg.transforms:
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

        self.vis.begin_draw_cycle('observations')
        # temp_poses = [gm.frame3_axis_angle(feature[3:] / np.sqrt(np.sum(feature[3:]**2)), np.sqrt(np.sum(feature[3:]**2)), feature[:3]) for feature in self.last_observation.values()]
        self.vis.draw_poses('observations', np.eye(4), 0.2, 0.01, self.last_observation.values())
        self.vis.render('observations')
        self.try_update()

    def cb_controls(self, map_message):
        self.last_control = {gm.Symbol(str_symbol): v for str_symbol, v in zip(map_message.symbol, map_message.value) 
                                                      if str_symbol in self.str_controls}
        self.try_update()

    def try_update(self):
        if min(p in self.last_observation for p in self.tracker.observation_names) is False:
            return

        if len(self.tracker.controls) > 0 and min(s in self.last_control for s in self.tracker.controls) is False:
            return

        now = rospy.Time.now()
        dt  = 0.05 if self.last_update is None else (now - self.last_update).to_sec()

        self.tracker.process_update(self.last_observation, self.last_control, dt)

        est_state = self.tracker.get_estimated_state()
        if len(est_state) == 0:
            return

        state_msg = ValueMapMsg()
        state_msg.header.stamp = now
        state_msg.symbol, state_msg.value = zip(*est_state.items())
        state_msg.symbol = [str(s) for s in state_msg.symbol]
        self.pub_state.publish(state_msg)

        # print('Performed update of estimate and published it.')

        # self.last_control = {}
        self.last_observation = {}
        
        if self.broadcaster is not None:
            self.broadcaster.update_state(self.tracker.get_estimated_state())
            self.broadcaster.publish_state()


if __name__ == '__main__':
    rospy.init_node('kineverse_ar_tracker')

    km = GeometryModel()

    try:
        location_x   = rospy.get_param('/nobilia_x')
        location_y   = rospy.get_param('/nobilia_y')
        location_z   = rospy.get_param('/nobilia_z')
        location_yaw = rospy.get_param('/nobilia_yaw')
    except KeyError as e:
        print(f'Failed to get nobilia localization params. Original error:\n{e}')
        exit()

    reference_frame = rospy.get_param('~reference_frame', 'world')
    grab_from_tf    = rospy.get_param('~grab_from_tf', False)
    grab_rate       = rospy.get_param('~grab_rate', 20.0)

    shelf_location = gm.point3(location_x, location_y, location_z)

    # origin_pose = gm.frame3_rpy(0, 0, 0, shelf_location)
    origin_pose = gm.frame3_rpy(0, 0, location_yaw, shelf_location)

    create_nobilia_shelf(km, Path('nobilia'), origin_pose, parent_path=Path(reference_frame))

    km.clean_structure()
    km.dispatch_events()

    shelf   = km.get_data('nobilia')
    tracker = Kineverse6DEKFTracker(km, 
                                    # [Path(f'nobilia/markers/{name}') for name in shelf.markers.keys()], # 
                                    # [Path(f'nobilia/markers/body')],
                                    {Path(f'nobilia/markers/top_panel'): 'obs_shelf_body'},
                                    #{x: x for x in gm.free_symbols(origin_pose)},
                                    noise_estimation_observations=20,
                                    estimate_init_steps=1000)

    node = ROSEKFManager(tracker, shelf, Path('nobilia'))




    tf_buffer = tf2_ros.Buffer()
    listener  = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Duration(1 / grab_rate )

    while not rospy.is_shutdown():
        if grab_from_tf:
            start = rospy.Time.now()

            msg = TransformStampedArrayMsg()
            for obs in tracker.observation_names:
                try:
                    trans = tf_buffer.lookup_transform(reference_frame, obs, rospy.Time(0))
                    msg.transforms.append(trans)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    print(f'Exception raised while looking up {obs} -> {reference_frame}:\n{e}')
                    break
            else:
                node.cb_obs(msg)

            
            time_remaining = rate - (rospy.Time.now() - start)
            if time_remaining > rospy.Duration(0):
                rospy.sleep(time_remaining)
        else:
            rospy.sleep(1.0)



