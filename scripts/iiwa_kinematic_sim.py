import rospy
import numpy as np
import kineverse.gradients.gradient_math as gm

from kineverse.model.geometry_model import GeometryModel, Path
from kineverse.operations.urdf_operations import load_urdf
from kineverse.urdf_fix             import load_urdf_str

from kineverse_tools.kinematic_sim import KineverseKinematicSim

from std_msgs.msg    import Float64MultiArray as Float64MultiArrayMsg
from sensor_msgs.msg import JointState        as JointStateMsg

if __name__ == '__main__':
	rospy.init_node('iiwa_kinematic_sim')

	rosparam_description = rospy.get_param('/robot_description', None)
	if rosparam_description is None:
		with open(res_pkg_path('package://kineverse_experiment_world/urdf/iiwa_wsg_50.urdf'), 'r') as f:
			rosparam_description = f.read()

	urdf = load_urdf_str(rosparam_description)

	km = GeometryModel()

	load_urdf(km, Path('iiwa'), urdf)

	km.clean_structure()
	km.dispatch_events()

	sim = KineverseKinematicSim(km, Path('iiwa'), 
								state_topic='/iiwa/joint_states')

	sorted_names = sorted(sim.state_info.keys())
	js_msg = JointStateMsg()
	js_msg.name = sorted_names

	def cb_pos_array_command(msg):
		js_msg.header.stamp = rospy.Time.now()
		js_msg.position = msg.data
		print(f'JS names {js_msg.name}')
		print(f'JS positions {js_msg.position}')
		sim.process_command(js_msg)

	init_msg = Float64MultiArrayMsg()
	init_msg.data = [np.deg2rad(x) for x in [-18, 25.04, 3, -98.2, 0.38, 60.2, 35.85]]
	cb_pos_array_command(init_msg)

	sub_pos_array_cmd = rospy.Subscriber('/iiwa/PositionController/command',
										 Float64MultiArrayMsg,
										 callback=cb_pos_array_command,
										 queue_size=1)

	print(f'Running kinematics sim for iiwa. Order for array commands is:'
		  '\n  {}'.format('\n  '.join(sorted_names)))

	while not rospy.is_shutdown():
		rospy.sleep(0.1)