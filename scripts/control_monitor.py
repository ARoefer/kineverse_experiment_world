#!/usr/bin/env python
import rospy
import sys


from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import RLock

from sensor_msgs.msg import JointState as JointStateMsg


if __name__ == '__main__':
	parser = ArgumentParser(description='Monitors a robot state topic and control topic and plots control vs state')
	parser.add_argument('-o', '--out', default='control_monitor', help='Tensorboard log file name')

	args = parser.parse_args([a for a in sys.args[1:] if ':=' not in a])

	writer = SummaryWriter(args.out)

	rospy.init_node('control_monitor')

	last_command = {}
	log_lock = RLock()

	def cb_state(state_msg):
		with log_lock:
			for name, velocity in zip(state_msg.name, state_msg.velocity):
				if name in last_command:
					delta = last_command[name] - velocity
					writer.add_scalar(f'{name} delta', delta)

	def cb_control(control_msg):
		with log_lock:
			for name, velocity in zip(state_msg.name, state_msg.velocity):
				last_command[name] = velocity

	sub_state   = rospy.Subscriber('/joint_states', JointStateMsg, callback=cb_state,   queue_size=1)
	sub_control = rospy.Subscriber('/control',      JointStateMsg, callback=cb_control, queue_size=1)

	print('Monitoring topics... Connect to http://localhost:6006 to view plots')

	while not rospy.is_shutdown():
		rospy.sleep(0.1)
