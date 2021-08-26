import rospy

from multiprocessing import RLock

from pr2_controllers_msgs.msg import Pr2GripperCommand    as Pr2GripperCommandMsg, \
                                     JointControllerState as JointControllerStateMsg

from kineverse_experiment_world.robot_interfaces import GripperWrapper

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
        