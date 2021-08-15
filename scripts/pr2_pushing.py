import rospy

import kineverse.gradients.gradient_math as gm

from kineverse.model.paths                    import Path, CPath
from kineverse.model.geometry_model           import GeometryModel, \
                                                     Path
from kineverse.operations.urdf_operations     import load_urdf
from kineverse.urdf_fix                       import load_urdf_str, load_urdf_file
from kineverse.visualization.bpb_visualizer   import ROSBPBVisualizer
from kineverse.operations.basic_operations    import ExecFunction
from kineverse.operations.special_kinematics  import create_diff_drive_joint_with_symbols, \
                                                     create_omnibase_joint_with_symbols, \
                                                     DiffDriveJoint, \
                                                     CreateAdvancedFrameConnection

from kineverse_experiment_world.nobilia_shelf import create_nobilia_shelf
from kineverse_experiment_world.utils import insert_omni_base

from kineverse_experiment_world.ros_pushing_controller import ROSPushingBehavior

if __name__ == '__main__':
    rospy.init_node('pr2_pushing')

    use_base = rospy.get_param('~use_base', False)

    if not rospy.has_param('/robot_description'):
        print('PR2 will be loaded from parameter server. It is currently not there.')
        exit(1)

    urdf_model = load_urdf_file('package://iai_pr2_description/robots/pr2_calibrated_with_ft2.xml')
    # urdf_model = load_urdf_str(rospy.get_param('/robot_description'))
    if urdf_model.name.lower() != 'pr2':
        print(f'The loaded robot is not the PR2. Its name is "{urdf_model.name}"')
        exit(1)

    km = GeometryModel()

    load_urdf(km, Path('pr2'), urdf_model)
    km.clean_structure()

    if use_base:
        insert_omni_base(km, Path('pr2'), urdf_model.get_root(), 'world')
        visualizer = ROSBPBVisualizer('~vis', base_frame='world')
    else:
        visualizer = ROSBPBVisualizer('~vis', base_frame=urdf_model.get_root())


    shelf_location = gm.point3(*[gm.Position(f'nobilia_location_{x}') for x in 'xyz'])
    shelf_yaw = gm.Position('nobilia_location_yaw')
    # origin_pose = gm.frame3_rpy(0, 0, 0, shelf_location)
    origin_pose = gm.frame3_rpy(0, 0, shelf_yaw, shelf_location)
    create_nobilia_shelf(km, Path('nobilia'), origin_pose)
    
    km.clean_structure()
    km.dispatch_events()

    eef_link = rospy.get_param('~eef_link', 'r_gripper_r_finger_tip_link')

    joint_symbols = [j.position for j in km.get_data(f'pr2/joints').values() 
                                if hasattr(j, 'position') and gm.is_symbol(j.position)]
    robot_controlled_symbols = {gm.DiffSymbol(j) for j in joint_symbols}
    if use_base:
        base_joint = km.get_data(base_joint_path)
        robot_controlled_symbols |= {gm.get_diff(x) for x in [base_joint.x_pos, base_joint.y_pos, base_joint.a_pos]}

    eef_path = Path(f'pr2/links/{eef_link}')
    cam_path = Path('pr2/links/head_mount_link')

    resting_pose = {'l_elbow_flex_joint' : -2.1213,
                    'l_shoulder_lift_joint': 1.2963,
                    'l_wrist_flex_joint' : -1.05,
                    # 'r_shoulder_pan_joint': -1.2963,
                    'r_shoulder_lift_joint': 1.2963,
                    # 'r_upper_arm_roll_joint': -1.2,
                    'r_elbow_flex_joint' : -2.1213,
                    'r_wrist_flex_joint' : -1.05,
                    'torso_lift_joint'   : 0.16825}
    resting_pose = {gm.Position(Path(f'pr2/{n}')): v for n, v in resting_pose.items()}

    behavior = ROSPushingBehavior(km,
                                  Path('pr2'),
                                  eef_path,
                                  [Path('nobilia/links/handle')],
                                  robot_controlled_symbols,
                                  {s: str(Path(gm.erase_type(s))[-1]) for s in robot_controlled_symbols},
                                  cam_path,
                                  resting_pose=resting_pose,
                                  visualizer=visualizer)

    while not rospy.is_shutdown():
        rospy.sleep(0.3)