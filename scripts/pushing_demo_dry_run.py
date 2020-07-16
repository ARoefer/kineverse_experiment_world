#!/usr/bin/env python
import os
import argparse
import rospy
import numpy as np

from kineverse.gradients.gradient_math             import *
from kineverse.model.paths                         import Path
from kineverse.model.frames                        import Frame
from kineverse.model.geometry_model                import GeometryModel, \
                                                          contact_geometry, \
                                                          generate_contact_model
from kineverse.motion.integrator                   import CommandIntegrator, DT_SYM
from kineverse.motion.min_qp_builder               import TypedQPBuilder as TQPB, \
                                                          GeomQPBuilder  as GQPB, \
                                                          Constraint, \
                                                          generate_controlled_values, \
                                                          depth_weight_controlled_values, \
                                                          SoftConstraint as SC, \
                                                          PID_Constraint as PIDC, \
                                                          ControlledValue, \
                                                          PANDA_LOGGING
from kineverse.operations.basic_operations         import CreateComplexObject
from kineverse.operations.urdf_operations          import load_urdf
from kineverse.operations.special_kinematics       import create_diff_drive_joint_with_symbols, \
                                                          create_omnibase_joint_with_symbols, \
                                                          DiffDriveJoint
from kineverse.time_wrapper                        import Time
from kineverse.urdf_fix                            import urdf_filler, \
                                                          hacky_urdf_parser_fix
from kineverse.utils                               import res_pkg_path
from kineverse.visualization.bpb_visualizer        import ROSBPBVisualizer
from kineverse.visualization.plotting              import draw_recorders,  \
                                                          split_recorders, \
                                                          convert_qp_builder_log, \
                                                          filter_contact_symbols
from kineverse.visualization.trajectory_visualizer import TrajectoryVisualizer

from kineverse_experiment_world.push_demo_base     import generate_push_closing

from trajectory_msgs.msg import JointTrajectory as JointTrajectoryMsg

from urdf_parser_py.urdf import URDF

tucked_arm = {'wrist_roll_joint'   : 0.0,
              'shoulder_pan_joint' : 1.32,
              'elbow_flex_joint'   : 1.72,
              'forearm_roll_joint' : 0.0,
              'upperarm_roll_joint': -0.2,
              'wrist_flex_joint'   : 1.66,
              'shoulder_lift_joint': 1.4,
              'torso_lift_joint'   : 0.2}
arm_poses  = {'l_elbow_flex_joint' : -2.1213,
              'l_shoulder_lift_joint': 1.2963,
              'l_wrist_flex_joint' : -1.05,
              'r_elbow_flex_joint' : -2.1213,
              'r_shoulder_lift_joint': 1.2963,
              'r_wrist_flex_joint' : -1.05,
              'torso_lift_joint'   : 0.16825}

# robot = 'pr2'
robot = 'fetch'

use_omni =  False
# use_geom_circulation = None
# use_geom_circulation = 'linear'
# use_geom_circulation = 'cubic'
use_geom_circulation = 'cross'

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    rospy.init_node('kineverse_sandbox')

    parser = argparse.ArgumentParser(description='Plans motions for closing doors and drawers in the IAI kitchen environment using various robots.')
    parser.add_argument('--robot', '-r', default='pr2', help='Name of the robot to use. [ pr2 | fetch ]')
    parser.add_argument('--omni', type=str2bool, default=True, help='To use an omnidirectional base or not.')
    parser.add_argument('--nav', type=str, default='cross', help='Heuristic for navigating object geometry. [ cross |  linear | cubic ]')
    parser.add_argument('--vis-plan', type=str2bool, default=False, help='Visualize trajector while planning.')
    
    args = parser.parse_args()
    robot    = args.robot
    use_omni = args.omni
    use_geom_circulation = args.nav

    plot_dir = res_pkg_path('package://kineverse/test/plots')

    if robot != 'pr2':
        with open(res_pkg_path('package://{r}_description/robots/{r}.urdf'.format(r=robot)), 'r') as urdf_file:
            urdf_str = urdf_file.read()
    else:
        with open(res_pkg_path('package://iai_pr2_description/robots/pr2_calibrated_with_ft2.xml'), 'r') as urdf_file:
            urdf_str = urdf_file.read()

    with open(res_pkg_path('package://iai_kitchen/urdf_obj/IAI_kitchen.urdf'), 'r') as urdf_file:
        urdf_kitchen_str = urdf_file.read() 

    urdf_model    = urdf_filler(URDF.from_xml_string(hacky_urdf_parser_fix(urdf_str)))
    kitchen_model = urdf_filler(URDF.from_xml_string(hacky_urdf_parser_fix(urdf_kitchen_str)))
    
    traj_pup   = rospy.Publisher('/{}/commands/joint_trajectory'.format(urdf_model.name), JointTrajectoryMsg, queue_size=1)
    kitchen_traj_pup = rospy.Publisher('/{}/commands/joint_trajectory'.format(kitchen_model.name), JointTrajectoryMsg, queue_size=1)


    # KINEMATIC MODEL
    km = GeometryModel()
    load_urdf(km, Path(robot), urdf_model)
    load_urdf(km, Path('kitchen'), kitchen_model)

    km.clean_structure()
    km.apply_operation_before('create world', 'create {}'.format(robot), CreateComplexObject(Path('world'), Frame('')))

    if robot == 'pr2' or use_omni:
        base_op = create_omnibase_joint_with_symbols(Path('world/pose'), 
                                                   Path('{}/links/{}/pose'.format(robot, urdf_model.get_root())),
                                                   Path('{}/joints/to_world'.format(robot)),
                                                   vector3(0,0,1),
                                                   1.0, 0.6, Path(robot))
    else:
        base_op = create_diff_drive_joint_with_symbols(Path('world/pose'), 
                                                   Path('{}/links/{}/pose'.format(robot, urdf_model.get_root())),
                                                   Path('{}/joints/to_world'.format(robot)),
                                                   0.12 * 0.5,
                                                   0.3748,
                                                   17.4, Path(robot))
    km.apply_operation_after('connect world {}'.format(urdf_model.get_root()), 'create {}/{}'.format(robot, urdf_model.get_root()), base_op)
    km.clean_structure()
    km.dispatch_events()

    visualizer = ROSBPBVisualizer('/vis_pushing_demo', base_frame='world')
    traj_vis   = TrajectoryVisualizer(visualizer)

    traj_vis.add_articulated_object(Path(robot),     km.get_data(robot))
    traj_vis.add_articulated_object(Path('kitchen'), km.get_data('kitchen'))


    # GOAL DEFINITION
    eef_path = Path('{}/links/gripper_link/pose'.format(robot)) if robot != 'pr2' else Path('pr2/links/r_gripper_r_finger_tip_link/pose')
    eef_pose = km.get_data(eef_path)
    eef_pos  = pos_of(eef_pose)

    cam_pose    = km.get_data('{}/links/head_camera_link/pose'.format(robot)) if robot != 'pr2' else km.get_data('pr2/links/head_mount_link/pose')
    cam_pos     = pos_of(cam_pose)
    cam_forward = x_of(cam_pose)
    cam_to_eef  = eef_pos - cam_pos

    parts = ['iai_fridge_door_handle', #]
             'fridge_area_lower_drawer_handle',#]
             'oven_area_area_left_drawer_handle',#]
             'oven_area_area_middle_lower_drawer_handle',
             'oven_area_area_middle_upper_drawer_handle',
             'oven_area_area_right_drawer_handle',
             'oven_area_oven_door_handle',
             'sink_area_dish_washer_door_handle',
             'sink_area_left_bottom_drawer_handle',
             'sink_area_left_middle_drawer_handle',
             'sink_area_left_upper_drawer_handle',
             'sink_area_trash_drawer_handle'
             ]
    
    # QP CONFIGURTION
    base_joint    = km.get_data('{}/joints/to_world'.format(robot))
    base_link     = km.get_data('{}/links/{}'.format(robot, urdf_model.get_root())) 
    joint_symbols = [j.position for j in km.get_data('{}/joints'.format(robot)).values() if hasattr(j, 'position') and type(j.position) is Symbol]
    
    robot_controlled_symbols = {DiffSymbol(j) for j in joint_symbols}
    integration_rules        = None
    if isinstance(base_joint, DiffDriveJoint):
        robot_controlled_symbols |= {base_joint.l_wheel_vel, base_joint.r_wheel_vel}

        # print(pos_of(km.get_data('fetch/links/base_link/pose'))[0][base_joint.l_wheel_vel])
        # exit()

        integration_rules = {
                      base_joint.x_pos: base_joint.x_pos + DT_SYM * (base_joint.r_wheel_vel * cos(base_joint.a_pos) * base_joint.wheel_radius * 0.5 + base_joint.l_wheel_vel * cos(base_joint.a_pos) * base_joint.wheel_radius * 0.5),
                      base_joint.y_pos: base_joint.y_pos + DT_SYM * (base_joint.r_wheel_vel * sin(base_joint.a_pos) * base_joint.wheel_radius * 0.5 + base_joint.l_wheel_vel * sin(base_joint.a_pos) * base_joint.wheel_radius * 0.5),
                      base_joint.a_pos: base_joint.a_pos + DT_SYM * (base_joint.r_wheel_vel * (base_joint.wheel_radius / base_joint.wheel_distance) + base_joint.l_wheel_vel * (- base_joint.wheel_radius / base_joint.wheel_distance))}
    else:
        robot_controlled_symbols |= {get_diff(x) for x in [base_joint.x_pos, base_joint.y_pos, base_joint.a_pos]}
    
    n_iter    = []
    total_dur = []

    for part in parts:
        print('Planning trajectory for "{}"'.format(part))
        kitchen_path = Path('kitchen/links/{}/pose'.format(part))
        obj_pose = km.get_data(kitchen_path)

        controlled_symbols = robot_controlled_symbols.union({DiffSymbol(j) for j in obj_pose.free_symbols})
        
        start_state = {s: 0.4 for s in obj_pose.free_symbols}

        constraints, geom_distance, coll_world, debug_draw = generate_push_closing(km, 
                                                                                   start_state, 
                                                                                   controlled_symbols,
                                                                                   eef_pose,
                                                                                   obj_pose,
                                                                                   eef_path[:-1],
                                                                                   kitchen_path[:-1],
                                                                                   use_geom_circulation)

        start_state.update({s: 0.0 for s in coll_world.free_symbols})
        start_state.update({s: 0.4 for s in obj_pose.free_symbols})
        controlled_values, constraints = generate_controlled_values(constraints, controlled_symbols)
        controlled_values = depth_weight_controlled_values(km, controlled_values, exp_factor=1.1)
        
        if isinstance(base_joint, DiffDriveJoint):
          controlled_values[str(base_joint.l_wheel_vel)].weight = 0.001 
          controlled_values[str(base_joint.r_wheel_vel)].weight = 0.001 

        # CAMERA STUFF
        cam_to_obj = pos_of(obj_pose) - cam_pos
        look_goal  = 1 - (dot(cam_to_obj, cam_forward) / norm(cam_to_obj))

        # GOAL CONSTAINT GENERATION
        goal_constraints = {'reach_point': PIDC(geom_distance, geom_distance, 1, k_i=0.01),
                            'look_at_obj':   SC(   -look_goal,    -look_goal, 1, look_goal)}
        goal_constraints.update({'open_object_{}'.format(x): PIDC(s, s, 1) for x, s in enumerate(obj_pose.free_symbols)})

        in_contact = less_than(geom_distance, 0.01)
        
        if robot == 'pr2':
          start_state.update({Position(Path(robot) + (k,)): v  for k, v in arm_poses.items()})
        else:
          start_state.update({Position(Path(robot) + (k,)): v  for k, v in tucked_arm.items()})

        if args.vis_plan:
            qpb = GQPB(coll_world, constraints, goal_constraints, controlled_values, visualizer=visualizer)
        else:
            qpb = GQPB(coll_world, constraints, goal_constraints, controlled_values) #, visualizer=visualizer)
        qpb._cb_draw = debug_draw
        integrator = CommandIntegrator(qpb,
        #integrator = CommandIntegrator(TQPB(constraints, goal_constraints, controlled_values),
                                       integration_rules,
                                       start_state=start_state,
                                       recorded_terms={'distance': geom_distance,
                                                       'gaze_align': look_goal,
                                                       'in contact': in_contact,
                                                       'goal': goal_constraints.values()[0].expr,
                                                       'location_x': base_joint.x_pos,
                                                       'location_y': base_joint.y_pos,
                                                       'rotation_a': base_joint.a_pos})


        # RUN
        int_factor = 0.1
        integrator.restart('{} Cartesian Goal Example'.format(robot))
        try:
            start = Time.now()
            integrator.run(int_factor, 500)
            total_dur.append((Time.now() - start).to_sec())
            n_iter.append(integrator.current_iteration + 1)
        except Exception as e:
            print(e)

        # DRAW
        print('Drawing recorders')
        draw_recorders([filter_contact_symbols(integrator.recorder, integrator.qp_builder), integrator.sym_recorder], 4.0/9.0, 8, 4).savefig('{}/{}_sandbox_{}_plots.png'.format(plot_dir, robot, part))
        if PANDA_LOGGING:
            rec_w, rec_b, rec_c, recs = convert_qp_builder_log(integrator.qp_builder)
            draw_recorders([rec_b, rec_c] + [r for _, r in sorted(recs.items())], 1, 8, 4).savefig('{}/{}_sandbox_{}_constraints.png'.format(plot_dir, robot, part))

        if not args.vis_plan:
            traj_vis.visualize(integrator.recorder.data, hz=50)
            pass

        if rospy.is_shutdown():
          break

    traj_vis.shutdown()

    print('Mean Iter: {}\nMean Iter D: {}\nIter D SD: {}'.format(np.mean(n_iter), 
                                                                 np.mean([d / float(i) for d, i in zip(total_dur, n_iter)]),
                                                                 np.std([d / float(i) for d, i in zip(total_dur, n_iter)])))
