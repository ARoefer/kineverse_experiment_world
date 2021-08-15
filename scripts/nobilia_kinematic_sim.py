#!/usr/bin/env python
import rospy
import kineverse.gradients.gradient_math as gm

from kineverse.model.geometry_model import GeometryModel, Path
from kineverse_tools.kinematic_sim  import KineverseKinematicSim

from kineverse_msgs.msg import ValueMap as ValueMapMsg

from kineverse_experiment_world.nobilia_shelf import create_nobilia_shelf

if __name__ == '__main__':
    rospy.init_node('nobilia_kinematic_sim')

    root_transform = rospy.get_param('~root_transform', gm.eye(4))
    if root_transform is not None and not gm.is_matrix(root_transform):
        if type(root_transform) is not list and type(root_transform) is not tuple:
            print(f'root_transform parameter is expected to be a list, but is {type(root_transform)}')
            exit(1)
        if len(root_transform) == 6:
            root_transform = gm.frame3_rpy(root_transform[3],
                                           root_transform[4],
                                           root_transform[5],
                                           root_transform[:3])
        elif len(root_transform) == 7:
            root_transform = gm.frame3_rpy(*root_transform)
        else:
            print('root_transform needs to encode transform eiter as xyzrpy, or as xyzqxqyqzqw')
            exit(1)

    km = GeometryModel()

    create_nobilia_shelf(km, Path('nobilia'), root_transform)

    km.clean_structure()
    km.dispatch_events()

    sim = KineverseKinematicSim(km, Path('nobilia'), '/nobilia_description', command_type=ValueMapMsg)

    print(f'Running kinematics sim for nobilia shelf')

    while not rospy.is_shutdown():
        rospy.sleep(1)