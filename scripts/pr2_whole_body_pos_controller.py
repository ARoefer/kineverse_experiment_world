#!/usr/bin/env python3
import rospy

from sensor_msgs.msg import JointState as JointStateMsg
from std_msgs.msg    import Float64    as Float64Msg

class JointController(object):
	def __init__(self, name, command_topic, timeout=rospy.Duration(0.2)):
		self.name = name
		self.pub = rospy.Publisher(command_topic, Float64Msg, queue_size=1, tcp_nodelay=True)
		self.timeout = timeout
		self.last_command_stamp = rospy.Time(0)
		self.stopped = False

	def send_command(self, cmd_val):
		cmd_msg = Float64Msg()
		cmd_msg.data = cmd_val
		self.pub.publish(cmd_msg)
		self.last_command_stamp = rospy.Time.now()
		self.stopped = False

	def check_watchdog(self):
		if not self.stopped:
			now = rospy.Time.now()
			if now - self.last_command_stamp >= self.timeout:
				# self.send_command(0)
				self.stopped = True

if __name__ == '__main__':
	rospy.init_node('whole_body_velocity_controller')

	joints = [
			  # 'l_elbow_flex_joint',
			  # 'l_forearm_roll_joint',
			  # 'l_shoulder_lift_joint',
			  # 'l_shoulder_pan_joint',
			  # 'l_upper_arm_roll_joint',
			  # 'l_wrist_flex_joint',
			  # 'l_wrist_roll_joint',
			  'r_elbow_flex_joint',
			  'r_forearm_roll_joint',
			  'r_shoulder_lift_joint',
			  'r_shoulder_pan_joint',
			  'r_upper_arm_roll_joint',
			  'r_wrist_flex_joint',
			  'r_wrist_roll_joint',
			  'torso_lift_joint']

	controllers = {j: JointController(j, f'/{j[:-6]}_position_controller/command') for j in joints}

	def command_watchdog(*args):
		for j, c in controllers.items():
			c.check_watchdog()


	def process_js_command(cmd):
		if len(cmd.name) != len(cmd.position):
			print(f'Received invalid command. |name| = {len(cmd.name)} |position| = {len(cmd.position)}')
			return

		for name, pos in zip(cmd.name, cmd.position):
			if name in controllers:
				controllers[name].send_command(pos)
			else:
				print(f'Could not find joint "{name}"')

	watchdog_timer = rospy.Timer(rospy.Duration(0.2), command_watchdog)
	sub_js_command = rospy.Subscriber('~command', JointStateMsg, callback=process_js_command, queue_size=1)

	print('Controller is ready')

	while not rospy.is_shutdown():
		rospy.sleep(1.0)

	for j, c in controllers.items():
		c.send_command(0)