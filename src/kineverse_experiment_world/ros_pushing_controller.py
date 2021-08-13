import rospy


from sensor_msgs.msg    import JointState as JointStateMsg
from kineverse_msgs.msg import ValueMap   as ValueMapMsg


class ROSPushingController(object):
	def __init__(self, km, eef_path, ext_path, visualizer):
		self.controller = PushingController(km,
                                            eef_path,
                                            ext_path,
                                            robot_controlled_symbols,
                                            start_state,
                                            cam_path,
                                            use_geom_circulation,
                                            visualizer,
                                            weights)

		self.pub_robot_command    = rospy.Publisher('~robot_command', JointStateMsg, queue_size=1, tcp_nodelay=True)
		self.pub_external_command = rospy.Publisher('~external_command', ValueMapMsg, queue_size=1, tcp_nodelay=True)

		self.sub_robot_js    = rospy.Subscriber('/joint_states', JointStateMsg, callback=self.cb_robot_js, queue_size=1)
		self.sub_external_js = rospy.Subscriber('~external_js',  ValueMapMsg, callback=self.cb_external_js, queue_size=1)


	def cb_robot_js(self, js_msg):
		pass

	def cb_external_js(self, value_msg):
		pass