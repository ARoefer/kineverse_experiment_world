#!/usr/bin/env python
import rospy
import kineverse.gradients.gradient_math as gm

from kineverse.model.geometry_model import GeometryModel, Path
from kineverse_tools.kinematic_sim import KineverseKinematicSim

from kineverse_experiment_world.nobilia_shelf import create_nobilia_shelf

if __name__ == '__main__':
	rospy.init_node('nobilia_kinematic_sim')

	km = GeometryModel()

	create_nobilia_shelf(km, Path('nobilia'), gm.translation3(2.0, 0, 0))

	km.clean_structure()
	km.dispatch_events()

	sim = KineverseKinematicSim(km, Path('nobilia'), '/nobilia_description')

	print(f'Running kinematics sim for nobilia shelf')

	while not rospy.is_shutdown():
		rospy.sleep(1)