#!/usr/bin/env python3
import rospy

import kineverse.gradients.gradient_math as gm

from multiprocessing import RLock

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
from kineverse_experiment_world.utils import insert_omni_base, load_localized_model

from kineverse_experiment_world.ros_opening_controller import ROSOpeningBehavior
from kineverse_experiment_world.pr2_things             import PR2GripperWrapper, \
                                                              PR2VelCommandProcessor, \
                                                              BaseSymbols


if __name__ == '__main__':
    rospy.init_node('pr2_opening')

    if not rospy.has_param('~model'):
        print('Parameter ~model needs to be set to either a urdf path, or "nobilia"')
        exit(1)

    if not rospy.has_param('~links'):
        print('Parameter ~links needs to be set to a dict of paths in the model that shall be operated and related grasp poses')
        exit(1)

    model_path = rospy.get_param('~model')
    body_paths = rospy.get_param('~links')

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

    reference_frame = rospy.get_param('~reference_frame', 'base_footprint')
    use_base = reference_frame != 'base_footprint'

    if use_base:
        insert_omni_base(km, Path('pr2'), urdf_model.get_root(), reference_frame, lin_vel=0.05)
        base_joint_path = Path(f'pr2/joints/to_{reference_frame}')
        km.dispatch_events()
    else:
        reference_frame = urdf_model.get_root()

    visualizer = ROSBPBVisualizer('~vis', base_frame=reference_frame)

    model_name = load_localized_model(km, model_path, reference_frame)
    
    km.clean_structure()
    km.dispatch_events()

    # Robot stuff
    eef_link = rospy.get_param('~eef_link', 'r_gripper_tool_frame')

    joint_symbols = [j.position for j in km.get_data(f'pr2/joints').values() 
                                if hasattr(j, 'position') and gm.is_symbol(j.position)]
    robot_controlled_symbols = {gm.DiffSymbol(j) for j in joint_symbols if 'torso' not in str(j)}
    base_symbols = None
    if use_base:
        base_joint   = km.get_data(base_joint_path)
        base_symbols = BaseSymbols(base_joint.x_pos, base_joint.y_pos, base_joint.a_pos,
                                   gm.DiffSymbol(base_joint.x_pos),
                                   gm.DiffSymbol(base_joint.y_pos),
                                   gm.DiffSymbol(base_joint.a_pos))
        robot_controlled_symbols |= {gm.DiffSymbol(x) for x in [base_joint.x_pos, base_joint.y_pos, base_joint.a_pos]}

    eef_path = Path(f'pr2/links/{eef_link}')
    cam_path = Path('pr2/links/head_mount_link')

    resting_pose = {#'l_elbow_flex_joint' : -2.1213,
                    #'l_shoulder_lift_joint': 1.2963,
                    #'l_wrist_flex_joint' : -1.16,
                    'r_shoulder_pan_joint': -1.0,
                    'r_shoulder_lift_joint': 0.9,
                    'r_upper_arm_roll_joint': -1.2,
                    'r_elbow_flex_joint' : -2.1213,
                    'r_wrist_flex_joint' : -1.05,
                    'r_forearm_roll_joint': 0,
                    'r_wrist_roll_joint': 0,
                    # 'torso_lift_joint'   : 0.16825
                    }
    resting_pose = {gm.Position(Path(f'pr2/{n}')): v for n, v in resting_pose.items()}

    # Object stuff
    grasp_poses = {}
    for p, pose_params in body_paths.items():
        if not km.has_data(p):
            print(f'Link {p} is not part of articulation model')
            exit(1)
        if type(pose_params) is not list and type(pose_params) is not tuple:
            print(f'Grasp pose for {p} is not a list.')
            exit(1)
        if len(pose_params) == 6:
            grasp_poses[Path(p)] = gm.frame3_rpy(*pose_params[-3:], gm.point3(*pose_params[:3]))
        elif len(pose_params) == 7:
            grasp_poses[Path(p)] = gm.frame3_quaternion(*pose_params)
        else:
            print(f'Grasp pose of {p} has {len(pose_params)} parameters but only 6 or 7 are permissable.')
            exit(1)

    monitoring_threshold = rospy.get_param('~m_threshold', 0.6)

    pr2_commander = PR2VelCommandProcessor(Path('pr2'),
                                           '/pr2_vel_controller/command',
                                           robot_controlled_symbols,
                                           '/base_controller/command',
                                           base_symbols,
                                           reference_frame=reference_frame)

    gripper = PR2GripperWrapper('/r_gripper_controller')

    print('\n'.join(str(s) for s in robot_controlled_symbols))

    behavior = ROSOpeningBehavior(km,
                                  pr2_commander,
                                  gripper,
                                  Path('pr2'),
                                  eef_path,
                                  grasp_poses,
                                  robot_controlled_symbols,
                                  cam_path,
                                  resting_pose=resting_pose,
                                  visualizer=None,
                                  monitoring_threshold=monitoring_threshold,
                                  acceptance_threshold=min(monitoring_threshold + 0.3, 1),
                                  control_mode='vel')

    while not rospy.is_shutdown():
        rospy.sleep(0.3)