#!/usr/bin/env python3
import rospy
import kineverse.gradients.gradient_math as gm
import tf2_ros
import kineverse     as kv
import kineverse.ros as kv_ros
import numpy         as np

from kineverse          import gm
from kineverse_msgs.msg import ValueMap as ValueMapMsg

from kineverse_experiment_world.qp_state_model import QPStateModel, QPSolverException
from kineverse_experiment_world.utils          import load_localized_model

from kineverse_experiment_world import ROSQPEManager

from another_aruco.msg import TransformStampedArray as TransformStampedArrayMsg


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
    def __init__(self, km, paths_observables, transition_rules=None, num_samples=10):
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
        poses = {str(p): pose.pose if isinstance(pose, kv.Frame) else pose for p, pose in poses.items()}
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
                                        transition_rules=transition_rules,
                                        num_samples=num_samples) for symbols, features in tracking_pools]

        print(f'Generated {len(self.estimators)} Estimators:\n  '
               '\n  '.join(str(e) for e in self.estimators))

        self.observation_names = [str(p) for p in paths_observables]

    def get_controls(self):
        return kv.utils.union([e.command_vars for e in self.estimators])

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


if __name__ == '__main__':
    rospy.init_node('kineverse_ar_tracker')

    if not rospy.has_param('~model'):
        print('Missing required parameter ~model. Set it to a urdf path or "nobilia"')
        exit(1)

    reference_frame = rospy.get_param('~reference_frame', 'world')
    model_path      = rospy.get_param('~model')
    observations    = rospy.get_param('~features', None)
    num_samples     = rospy.get_param('~samples', 5)

    if type(observations) is not dict:
        print(type(observations))
        print('Required parameter ~features needs to be a dict mapping tracked model paths to observation names')
        exit(1)

    km = kv.GeometryModel()

    model_name = load_localized_model(km, model_path, reference_frame)

    km.clean_structure()
    km.dispatch_events()

    model   = km.get_data(model_name)
    tracker = Kineverse6DQPTracker(km, observations.keys(), num_samples=num_samples)

    node = ROSQPEManager(tracker,
                         model,
                         kv.Path(model_name),
                         reference_frame,
                         observation_alias=observations)

    while not rospy.is_shutdown():
        rospy.sleep(0.1)



