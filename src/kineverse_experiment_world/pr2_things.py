import rospy
import tf2_ros
import math
import kineverse.gradients.gradient_math as gm

from collections           import namedtuple
from multiprocessing       import RLock
from kineverse.model.paths import Path

from geometry_msgs.msg        import Twist                as TwistMsg
from pr2_controllers_msgs.msg import Pr2GripperCommand    as Pr2GripperCommandMsg, \
                                     JointControllerState as JointControllerStateMsg
from sensor_msgs.msg          import JointState           as JointStateMsg

from kineverse_experiment_world.robot_interfaces import GripperWrapper
from kineverse_experiment_world.utils            import np_frame3_quaternion, \
                                                        np_vector3

BaseSymbols = namedtuple('BaseSymbols', ['xp', 'yp', 'ap', 'xv', 'yv', 'av'])

class PR2VelCommandProcessor(RobotCommandProcessor):
    def __init__(self, robot_prefix,
                       joint_topic,
                       joint_vel_symbols,
                       base_topic=None,
                       base_symbols=None,
                       reference_frame='odom'):
        self.joint_aliases = {s: str(Path(gm.erase_type(s))[-1]) for s in joint_vel_symbols}
        self._robot_cmd_msg      = JointStateMsg()
        self._robot_cmd_msg.name = list(control_alias.keys())
        self._state_cb = None
        self.robot_prefix = robot_prefix

        if base_symbols is not None:
            self.base_aliases     = base_symbols
            self.tf_buffer        = tf2_ros.Buffer()
            self.listener         = tf2_ros.TransformListener(tf_buffer)
            self.reference_frame  = reference_frame
            self._base_cmd_msg    = TwistMsg()
            self.pub_base_command = rospy.Publisher(base_topic, JointStateMsg, queue_size=1, tcp_nodelay=True)
        else:
            self.base_aliases = None

        self.pub_joint_command = rospy.Publisher(joint_topic, JointStateMsg, queue_size=1, tcp_nodelay=True)
        self.sub_robot_js    = rospy.Subscriber('/joint_states',  JointStateMsg, callback=self.cb_robot_js, queue_size=1)


    def send_command(self, cmd):
        self._robot_cmd_msg.name, self._robot_cmd_msg.velocity = zip(*[(self.control_alias[s], v) for s, v in cmd.items()
                                                                                                  if s in self.control_alias])
        if self.base_aliases is not None:
            if self.base_aliases.xv in cmd and \
               self.base_aliases.yv in cmd and \
               self.base_aliases.av in cmd:
                try:
                    trans = self.tf_buffer.lookup_transform('base_footprint',
                                                            self.reference_frame,
                                                            rospy.Time(0))
                    rotation = trans.transform.rotation
                    tf = np_frame3_quaternion(0, 0, 0, rotation.x,
                                                       rotation.y,
                                                       rotation.z,
                                                       rotation.w)
                    local_control = tf.dot(np_vector3(cmd[self.base_aliases.xv],
                                                      cmd[self.base_aliases.yv],
                                                      0)).sum(axis=1)
                    self._base_cmd_msg.linear.x  = local_control[0]
                    self._base_cmd_msg.linear.y  = local_control[1]
                    self._base_cmd_msg.angular.z = cmd[self.base_aliases.av]
                    self.pub_base_command.publish(self._base_cmd_msg)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    print(f'Exception raised while looking up {self.reference_frame} -> base_footprint:\n{e}')
                    continue

        self.pub_joint_command.publish(self._robot_cmd_msg)

    def register_state_cb(self, cb):
        self._state_cb = cb

    def cb_robot_js(self, js_msg):
        if self._state_cb is None:
            return

        state_update = {}
        for x, name in enumerate(js_msg.name):
            if len(js_msg.position) > x:
                state_update[gm.Position(self.robot_prefix + (name,))] = js_msg.position[x]
            if len(js_msg.velocity) > x:
                state_update[gm.Velocity(self.robot_prefix + (name,))] = js_msg.velocity[x]

        if self.base_aliases is not None:
            try:
                trans = self.tf_buffer.lookup_transform(self.reference_frame,
                                                        'base_footprint',
                                                        rospy.Time(0))
                state_update[self.base_aliases.ap] = math.acos(trans.rotation.w) * 2
                state_update[self.base_aliases.xp] = trans.translation.x
                state_update[self.base_aliases.yp] = trans.translation.y
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                print(f'Exception raised while looking up {self.reference_frame} -> base_footprint:\n{e}')
                continue

        self._state_cb(state_update)



class PR2GripperWrapper(GripperWrapper):
    def __init__(self, topic):
        self._error_deltas = []
        self._last_error   = None
        self._last_error_stamp = None
        self._error_lock   = RLock()

        self.pub_gripper_command = rospy.Publisher(f'{topic}/command', Pr2GripperCommandMsg, queue_size=1, tcp_nodelay=True)
        self.sub_gripper_state   = rospy.Subscriber(f'{topic}/state', JointControllerStateMsg, callback=self._cb_gripper_feedback, queue_size=1)

    def _cb_gripper_feedback(self, feedback_msg):
        if self._last_error is not None:
            delta = (feedback_msg.error - self._last_error) / (feedback_msg.header.stamp - self._last_error_stamp).to_sec()
            with self._error_lock:
                self._error_deltas.append(delta)
                if len(self._error_deltas) > 10:
                    self._error_deltas = self._error_deltas[-10:]
        self._last_error_stamp = feedback_msg.header.stamp
        self._last_error = feedback_msg.error

    def wait(self):
        while True:
            if len(self._error_deltas) == 10:
                with self._error_lock:
                    if abs(sum(self._error_deltas)) / 10 <= 0.005:
                        break
                rospy.sleep(0.1)

    def get_latest_error(self):
        with self._error_lock:
            return self._last_error

    def set_gripper_position(self, position, effort=50):
        cmd = Pr2GripperCommandMsg()
        cmd.position = position
        cmd.max_effort = effort
        self.pub_gripper_command.publish(cmd)

    def sync_set_gripper_position(self, position, effort=50):
        self.set_gripper_position(position, effort)
        rospy.sleep(0.1)
        self.wait()
