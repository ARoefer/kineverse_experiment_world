#!/usr/bin/env python
import rospy
import kineverse.gradients.gradient_math as gm

from kineverse.model.geometry_model       import GeometryModel, Path
from kineverse.operations.urdf_operations import load_urdf
from kineverse.urdf_fix                   import load_urdf_str, \
                                                 load_urdf_file
from kineverse.utils                      import res_pkg_path

from kineverse_tools.kinematic_sim import KineverseKinematicSim

from kineverse_experiment_world.utils import insert_omni_base, \
                                             insert_diff_base

if __name__ == '__main__':
    rospy.init_node('urdf_kinematic_sim')

    if not rospy.has_param('~description'):
        print('Parameter description is needed. It can either point to a file or the parameter server.')
        exit(1)
    
    special_base = rospy.get_param('~base_joint', None)
    if special_base is not None and special_base not in {'omni', 'static', 'diff'}:
        print(f'Base joint parameter refers to unknown type "{special_base}". Supported are [static (default), omni, diff]')
        exit(1)

    root_transform = rospy.get_param('~root_transform', None)
    if root_transform is not None:
        if type(root_transform) is not list and type(root_transform) is not tuple:
            print('root_transform parameter is expected to be a list.')
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

    description = rospy.get_param('~description')
    if description[-5:].lower() == '.urdf' or description[-4:].lower() == '.xml':
        urdf_model = load_urdf_file(description)
    else:
        if not rospy.has_param(description):
            print(f'Description is supposed to be located at "{description}" but that parameter does not exist')
        urdf_model = load_urdf_str(rospy.get_param(description))


    name = rospy.get_param('~name', urdf_model.name)
    reference_frame = rospy.get_param('~reference_frame', 'world')

    km = GeometryModel()

    load_urdf(km,
              Path(name),
              urdf_model,
              reference_frame,
              joint_prefix='',
              root_transform=root_transform)

    km.clean_structure()

    if special_base is not None:
        if special_base == 'omni':
            insert_omni_base(km,
                             Path(name),
                             urdf_model.get_root(),
                             reference_frame)
        elif special_base == 'diff':
            insert_diff_base(km,
                             Path(name),
                             urdf_model.get_root(),
                             reference_frame)

    km.dispatch_events()

    sim = KineverseKinematicSim(km, Path(name), f'/{name}/description')

    print(f'Running kinematics sim for {name}')

    while not rospy.is_shutdown():
        rospy.sleep(0.5)