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


class ROSQPEManager(object):
    def __init__(self, tracker, 
                       model=None, model_path=None, 
                       reference_frame='world', urdf_param='/qp_description_check', update_freq=30,
                       observation_alias=None):
        self.tracker          = tracker 
        self.last_observation = 0
        self.last_update      = 0
        self.reference_frame  = reference_frame
        self.observation_aliases = {o: o for o in tracker.observation_names}
        if observation_alias is not None:
            for path, alias in observation_alias.items():
                if path in self.observation_aliases:
                    self.observation_aliases[alias] = path

        self.str_controls = {str(s) for s in self.tracker.get_controls()}

        self.vis = kv_ros.ROSVisualizer('~vis', reference_frame)

        if model is not None:
            self.broadcaster = kv_ros.ModelTFBroadcaster_URDF(urdf_param, model_path, model, kv.Path('ekf'))
        else:
            self.broadcaster = None

        self.tf_buffer = tf2_ros.Buffer()
        self.listener  = tf2_ros.TransformListener(self.tf_buffer)

        self.pub_state        = rospy.Publisher('~state_estimate', ValueMapMsg, queue_size=1, tcp_nodelay=True)
        self.sub_observations = rospy.Subscriber('~observations',  TransformStampedArrayMsg, callback=self.cb_obs, queue_size=1)
        self.sub_controls     = rospy.Subscriber('~controls',      ValueMapMsg, callback=self.cb_controls, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1 / update_freq), self.cb_update)

    def cb_obs(self, transform_stamped_array_msg):
        print('OBS')
        ref_frames = {}
        
        num_valid_obs = 0
        last_observation = {}

        for trans in transform_stamped_array_msg.transforms:
            if trans.child_frame_id not in self.observation_aliases:
                continue

            matrix = np_frame3_quaternion(trans.transform.translation.x, 
                                          trans.transform.translation.y, 
                                          trans.transform.translation.z,
                                          trans.transform.rotation.x,
                                          trans.transform.rotation.y,
                                          trans.transform.rotation.z,
                                          trans.transform.rotation.w)

            if trans.header.frame_id != self.reference_frame:
                if trans.header.frame_id not in ref_frames:
                    try:
                        ref_trans = self.tf_buffer.lookup_transform(self.reference_frame, trans.header.frame_id, rospy.Time(0))
                        np_ref_trans = np_frame3_quaternion(ref_trans.transform.translation.x, 
                                                            ref_trans.transform.translation.y, 
                                                            ref_trans.transform.translation.z,
                                                            ref_trans.transform.rotation.x,
                                                            ref_trans.transform.rotation.y,
                                                            ref_trans.transform.rotation.z,
                                                            ref_trans.transform.rotation.w)
                        ref_frames[trans.header.frame_id] = np_ref_trans
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                        print(f'Exception raised while looking up {trans.header.frame_id} -> {self.reference_frame}:\n{e}')
                        break
                else:
                    np_ref_trans = ref_frames[trans.header.frame_id]

                matrix = np_ref_trans.dot(matrix)
            
            last_observation[self.observation_aliases[trans.child_frame_id]] = matrix
            num_valid_obs += 1
        else:
            try:
                if num_valid_obs == 0:
                    return

                self.tracker.process_observation(last_observation)
                self.last_observation += 1
            except QPSolverException as e:
                print(f'Solver crashed during observation update. Skipping observation... Error:\n{e}')
                return
        self.vis.begin_draw_cycle('observations')
        # temp_poses = [gm.frame3_axis_angle(feature[3:] / np.sqrt(np.sum(feature[3:]**2)), np.sqrt(np.sum(feature[3:]**2)), feature[:3]) for feature in self.last_observation.values()]
        self.vis.draw_poses('observations', np.eye(4), 0.2, 0.01, last_observation.values())
        # self.vis.draw_poses('observations', np.eye(4), 0.2, 0.01, temp_poses)
        self.vis.render('observations')

    def cb_controls(self, map_message):
        self.last_control = {gm.Symbol(str_symbol): v for str_symbol, v in zip(map_message.symbol, map_message.value) 
                                                      if  str_symbol    in self.str_controls}
        self.tracker.process_control(self.last_control)

    def cb_update(self, *args):
        if self.last_observation == self.last_update:
            return

        self.last_update = self.last_observation

        est_state = self.tracker.get_estimated_state()

        state_msg = ValueMapMsg()
        state_msg.header.stamp = rospy.Time.now()
        state_msg.symbol, state_msg.value = zip(*est_state.items())
        state_msg.symbol = [str(s) for s in state_msg.symbol]
        self.pub_state.publish(state_msg)

        if self.broadcaster is not None:
            self.broadcaster.update_state(est_state)
            self.broadcaster.publish_state()