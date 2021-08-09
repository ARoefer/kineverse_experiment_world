#!/usr/bin/env python
import os
import argparse
import rospy
import numpy as np
import matplotlib.pyplot as plt

import kineverse.gradients.gradient_math as gm
from kineverse.model.paths                         import Path, CPath
from kineverse.model.frames                        import Frame
from kineverse.model.geometry_model                import GeometryModel, \
                                                          contact_geometry, \
                                                          generate_contact_model, \
                                                          closest_distance_constraint_world
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
from kineverse.motion.utils                        import cart_force_to_q_forces
from kineverse.operations.basic_operations         import ExecFunction
from kineverse.operations.urdf_operations          import load_urdf
from kineverse.operations.special_kinematics       import create_diff_drive_joint_with_symbols, \
                                                          create_omnibase_joint_with_symbols, \
                                                          DiffDriveJoint, \
                                                          CreateAdvancedFrameConnection
from kineverse.time_wrapper                        import Time

from kineverse.urdf_fix                            import urdf_filler, \
                                                          hacky_urdf_parser_fix
from kineverse.utils                               import res_pkg_path
from kineverse.visualization.bpb_visualizer        import ROSBPBVisualizer
from kineverse.visualization.plotting              import draw_recorders,  \
                                                          split_recorders, \
                                                          convert_qp_builder_log, \
                                                          filter_contact_symbols, \
                                                          ColorGenerator
from kineverse.visualization.trajectory_visualizer import TrajectoryVisualizer

from kineverse_experiment_world.push_demo_base     import generate_push_closing
from kineverse_experiment_world.nobilia_shelf      import create_nobilia_shelf

from trajectory_msgs.msg import JointTrajectory as JointTrajectoryMsg

from urdf_parser_py.urdf import URDF

# TERMINAL COMMANDS TO RUN WITH DIFFERENT ROBOTS:
# HSR: ../scripts/pushing_demo_dry_run.py --vis-plan=during --robot=hsr --link=hand_l_distal_link --camera=head_l_stereo_camera_link
# DONBOT: ./scripts/pushing_demo_dry_run.py --vis-plan=during --robot=iai_donbot --link=gripper_finger_right_link
# BOXY: ./scripts/pushing_demo_dry_run.py --vis-plan=during --robot=iai_boxy --link=right_gripper_finger_right_link  --camera=head_mount_kinect2_rgb_optical_frame
# FETCH: ./scripts/pushing_demo_dry_run.py --nav=linear --robot=fetch --vis-plan=during --omni=False
# PR2:

  # 'pr2' : {'l_elbow_flex_joint' : -2.1213,
  #             'l_shoulder_lift_joint': 1.2963,
  #             'l_wrist_flex_joint' : -1.05,
  #             'r_shoulder_pan_joint': -1.2963,
  #             'r_shoulder_lift_joint': 0,
  #             'r_upper_arm_roll_joint': -1.2,
  #             'r_elbow_flex_joint' : -2.1213,
  #             'r_wrist_flex_joint' : -1.05,
  #             'torso_lift_joint'   : 0.30}

start_poses = {
  'fetch' : {'wrist_roll_joint'   : 0.0,
              'shoulder_pan_joint' : 1.0,
              'elbow_flex_joint'   : 1.72,
              'forearm_roll_joint' : 0.0,
              'upperarm_roll_joint': -0.2,
              'wrist_flex_joint'   : 1.66,
              'shoulder_lift_joint': 1.4,
              'torso_lift_joint'   : 0.2},
  'pr2' : {'l_elbow_flex_joint' : -2.1213,
              'l_shoulder_lift_joint': 1.2963,
              'l_wrist_flex_joint' : -1.05,
              # 'r_shoulder_pan_joint': -1.2963,
              'r_shoulder_lift_joint': 1.2963,
              # 'r_upper_arm_roll_joint': -1.2,
              'r_elbow_flex_joint' : -2.1213,
              'r_wrist_flex_joint' : -1.05,
              'torso_lift_joint'   : 0.16825},
  'iai_boxy': {'neck_shoulder_pan_joint': -1.57,
               'neck_shoulder_lift_joint': -1.88,
               'neck_elbow_joint': -2.0,
               'neck_wrist_1_joint': 0.139999387693,
               'neck_wrist_2_joint': 1.56999999998,
               'neck_wrist_3_joint': 0,
               'triangle_base_joint': -0.24,
               'left_arm_0_joint': 0.68,
               'left_arm_1_joint': 1.08,
               'left_arm_2_joint': -0.13,
               'left_arm_3_joint': -1.35,
               'left_arm_4_joint': 0.3,
               'left_arm_5_joint': 0.7,
               'left_arm_6_joint': -0.01,
               'right_arm_0_joint': 0.68,
               'right_arm_1_joint': -1.08,
               'right_arm_2_joint': 0.13,
               'right_arm_3_joint': 1.35,
               'right_arm_4_joint': -0.3,
               'right_arm_5_joint': -0.7,
               'right_arm_6_joint': 0.01},
              }
torque_limits = {
    'pr2': {'l_shoulder_pan_joint' : 30,
            'l_shoulder_lift_joint': 30,
            'l_upper_arm_roll_joint': 30,
            'l_elbow_flex_joint' : 30,
            'l_forearm_roll_joint' : 30,
            'l_wrist_flex_joint' : 10,
            'l_wrist_roll_joint' : 10,

            'r_shoulder_pan_joint' : 30,
            'r_shoulder_lift_joint': 30,
            'r_upper_arm_roll_joint': 30,
            'r_forearm_roll_joint' : 30,
            'r_elbow_flex_joint' : 30,
            'r_wrist_flex_joint' : 10,
            'r_wrist_roll_joint' : 10,

            'torso_lift_joint'   : 10000}
}


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def wait_to_continue(dt, cmd):
    bla = raw_input('Press enter for next step.')

if __name__ == '__main__':
    rospy.init_node('kineverse_sandbox')

    # Terminal args
    parser = argparse.ArgumentParser(description='Plans motions for closing doors and drawers in the IAI kitchen environment using various robots.')
    parser.add_argument('--robot', '-r', default='pr2', help='Name of the robot to use. Will look for package://ROBOT_description/robot/ROBOT.urdf')
    parser.add_argument('--omni', type=str2bool, default=True, help='To use an omnidirectional base or not.')
    parser.add_argument('--nav', type=str, default='linear', help='Heuristic for navigating object geometry. [ cross |  linear | cubic ]')
    parser.add_argument('--vis-plan', type=str, default='after', help='Visualize trajector while planning. [ during | after | none ]')
    parser.add_argument('--link', type=str, default=None, help='Link of the robot to use for actuation.')
    parser.add_argument('--camera', type=str, default=None, help='Camera link of the robot.')

    args = parser.parse_args()
    robot    = args.robot
    use_omni = args.omni
    use_geom_circulation = args.nav
    
    if args.link is not None:
      robot_link = args.link
    elif robot == 'pr2':
      robot_link = 'r_gripper_r_finger_tip_link'
    else:
      robot_link = 'gripper_link' # 'r_gripper_finger_link'

    camera_link = args.camera
    if camera_link is None:
      if args.robot == 'fetch':
        camera_link = 'head_camera_rgb_optical_frame'
      elif args.robot == 'pr2':
        camera_link = 'head_mount_kinect_rgb_optical_frame'

    plot_dir = res_pkg_path('package://kineverse/test/plots')

    # Loading of models
    if robot != 'pr2':
        with open(res_pkg_path('package://{r}_description/robots/{r}.urdf'.format(r=robot)), 'r') as urdf_file:
            urdf_str = hacky_urdf_parser_fix(urdf_file.read())
    else:
        with open(res_pkg_path('package://iai_pr2_description/robots/pr2_calibrated_with_ft2.xml'), 'r') as urdf_file:
            urdf_str = hacky_urdf_parser_fix(urdf_file.read())

    with open(res_pkg_path('package://iai_kitchen/urdf_obj/IAI_kitchen.urdf'), 'r') as urdf_file:
        urdf_kitchen_str = hacky_urdf_parser_fix(urdf_file.read())

    urdf_model    = urdf_filler(URDF.from_xml_string(hacky_urdf_parser_fix(urdf_str)))
    kitchen_model = urdf_filler(URDF.from_xml_string(hacky_urdf_parser_fix(urdf_kitchen_str)))
    
    # 
    traj_pup   = rospy.Publisher('/{}/commands/joint_trajectory'.format(urdf_model.name), JointTrajectoryMsg, queue_size=1)
    kitchen_traj_pup = rospy.Publisher('/{}/commands/joint_trajectory'.format(kitchen_model.name), JointTrajectoryMsg, queue_size=1)


    # KINEMATIC MODEL
    km = GeometryModel()
    load_urdf(km, Path(robot), urdf_model)
    load_urdf(km, Path('kitchen'), kitchen_model)
    create_nobilia_shelf(km, Path('nobilia'), gm.translation3(1.2, 0, 0.8))

    km.clean_structure()
    km.apply_operation_before('create world', 'create {}'.format(robot), ExecFunction(Path('world'), Frame, ''))

    base_joint_path = Path(f'{robot}/joints/to_world')

    # Insert base to world kinematic
    if robot == 'pr2' or use_omni:
        base_op = ExecFunction(base_joint_path,
                               create_omnibase_joint_with_symbols,
                                    CPath('world/pose'),
                                    CPath('{}/links/{}/pose'.format(robot, urdf_model.get_root())),
                                    gm.vector3(0,0,1),
                                    1.0, 0.6, CPath(robot))
    else:
        base_op = ExecFunction(base_joint_path,
                               create_diff_drive_joint_with_symbols,
                                    CPath('world/pose'),
                                    CPath('{}/links/{}/pose'.format(robot, urdf_model.get_root())),
                                    0.12 * 0.5,
                                    0.3748,
                                    17.4, CPath(robot))
    km.apply_operation_after(f'create {base_joint_path}',
                             f'create {robot}/{urdf_model.get_root()}', base_op)
    km.apply_operation_after(f'connect world {urdf_model.get_root()}',
                             f'create {base_joint_path}',
                             CreateAdvancedFrameConnection(base_joint_path,
                                                           Path('world'),
                                                           Path('{}/links/{}'.format(robot, urdf_model.get_root()))))
    km.clean_structure()
    km.dispatch_events()

    # Visualization of the trajectory
    visualizer = ROSBPBVisualizer('/vis_pushing_demo', base_frame='world')
    traj_vis   = TrajectoryVisualizer(visualizer)

    traj_vis.add_articulated_object(Path(robot),     km.get_data(robot))
    traj_vis.add_articulated_object(Path('kitchen'), km.get_data('kitchen'))


    # GOAL DEFINITION
    eef_path = Path(f'{robot}/links/{robot_link}/pose')
    eef_pose = km.get_data(eef_path)
    eef_pos  = gm.pos_of(eef_pose)

    print('EEF free symbols:\n  {}'.format('\n  '.join(sorted([str(s) for s in gm.free_symbols(eef_pose)]))))

    cam_pose    = km.get_data('{}/links/head_camera_link/pose'.format(robot)) if robot != 'pr2' else km.get_data('pr2/links/head_mount_link/pose')
    cam_pos     = gm.pos_of(cam_pose)
    cam_forward = gm.x_of(cam_pose)
    cam_to_eef  = eef_pos - cam_pos

    kitchen_parts = ['iai_fridge_door_handle', #]
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
    base_joint    = km.get_data(f'{robot}/joints/to_world')
    base_link     = km.get_data(f'{robot}/links/{urdf_model.get_root()}')
    joint_symbols = [j.position for j in km.get_data(f'{robot}/joints').values() if hasattr(j, 'position') and gm.is_symbol(j.position)]

    robot_controlled_symbols = {gm.DiffSymbol(j) for j in joint_symbols}
    integration_rules        = None
    if isinstance(base_joint, DiffDriveJoint):
        robot_controlled_symbols |= {base_joint.l_wheel_vel, base_joint.r_wheel_vel}

        # print(pos_of(km.get_data('fetch/links/base_link/pose'))[0][base_joint.l_wheel_vel])
        # exit()

        integration_rules = {
                      base_joint.x_pos: base_joint.x_pos + DT_SYM * (base_joint.r_wheel_vel * gm.cos(base_joint.a_pos) * base_joint.wheel_radius * 0.5 + base_joint.l_wheel_vel * gm.cos(base_joint.a_pos) * base_joint.wheel_radius * 0.5),
                      base_joint.y_pos: base_joint.y_pos + DT_SYM * (base_joint.r_wheel_vel * gm.sin(base_joint.a_pos) * base_joint.wheel_radius * 0.5 + base_joint.l_wheel_vel * gm.sin(base_joint.a_pos) * base_joint.wheel_radius * 0.5),
                      base_joint.a_pos: base_joint.a_pos + DT_SYM * (base_joint.r_wheel_vel * (base_joint.wheel_radius / base_joint.wheel_distance) + base_joint.l_wheel_vel * (- base_joint.wheel_radius / base_joint.wheel_distance))}
    else:
        robot_controlled_symbols |= {gm.get_diff(x) for x in [base_joint.x_pos, base_joint.y_pos, base_joint.a_pos]}

    active_robot_world = km.get_active_geometry(joint_symbols)
    # c_avoidance_constraints = {f'collision avoidance {name}': SC.from_constraint(closest_distance_constraint_world(gm.get_data(Path(name) + ('pose',)),
    #                                                                              Path(name),
    #                                                                              0.03), 100) for name in active_robot_world.names}

    print('\n'.join([str(s) for s in robot_controlled_symbols]))

    n_iter    = []
    total_dur = []

    for part in [Path(f'nobilia/links/handle/pose')]: # + [Path(f'kitchen/links/{p}/pose') for p in kitchen_parts]:
        print('Planning trajectory for "{}"'.format(part))
        kitchen_path = part
        obj_pose = km.get_data(kitchen_path)

        printed_exprs = {}

        controlled_symbols = robot_controlled_symbols.union({gm.DiffSymbol(j) for j in gm.free_symbols(obj_pose)})

        start_state = {s: 0.4 for s in gm.free_symbols(obj_pose)}

        # Generate push problem
        constraints, geom_distance, coll_world, p_internals = generate_push_closing(km,
                                                                                   start_state,
                                                                                   controlled_symbols,
                                                                                   eef_pose,
                                                                                   obj_pose,
                                                                                   eef_path[:-1],
                                                                                   kitchen_path[:-1],
                                                                                   use_geom_circulation)

        start_state.update({s: 0.0 for s in gm.free_symbols(coll_world)})
        start_state.update({s: 2.2 for s in gm.free_symbols(obj_pose)})
        controlled_values, constraints = generate_controlled_values(constraints, controlled_symbols)
        controlled_values = depth_weight_controlled_values(km, controlled_values, exp_factor=1.0)

        # print(len(controlled_symbols))

        if isinstance(base_joint, DiffDriveJoint):
          controlled_values[str(base_joint.l_wheel_vel)].weight_id = 0.002
          controlled_values[str(base_joint.r_wheel_vel)].weight_id = 0.002

        # CAMERA STUFF
        cam_to_obj = gm.pos_of(obj_pose) - cam_pos
        look_goal  = 1 - (gm.dot_product(cam_to_obj, cam_forward) / gm.norm(cam_to_obj))


        # GOAL CONSTAINT GENERATION
        goal_constraints = {'reach_point': PIDC(geom_distance, geom_distance, 1, k_i=0.01),
                            'look_at_obj':   SC(   -look_goal,    -look_goal, 1, look_goal),
                            'avoid_collisions': SC.from_constraint(closest_distance_constraint_world(eef_pose, eef_path[:-1], 0.03), 100)}

        # EXERT SUFICIENT FORCE
        if False and robot in torque_limits:
            torque_relevant_joints = gm.free_symbols(p_internals.contact_a).intersection(joint_symbols)
            relevant_torque_limits = {j_sym: torque_limits[robot][Path(j_sym)[1]] for j_sym in torque_relevant_joints if Path(j_sym)[1] in torque_limits[robot]}
            torque_ordered_joints, ordered_torque_limits = zip(*relevant_torque_limits.items())

            torque_projection = cart_force_to_q_forces(p_internals.contact_a,
                                                       -20 * p_internals.normal_b_to_a,
                                                       torque_ordered_joints)
            torque_bounds_delta = [t_limit - gm.abs(c) for c, t_limit in zip(torque_projection.elements(), ordered_torque_limits)]
            for symbol, bound_delta in zip(torque_ordered_joints, torque_bounds_delta):
                goal_constraints[f'{symbol} torque_bound'] = SC(-bound_delta, 1e9, 1, bound_delta)
                printed_exprs[f'{symbol} torque_bound'] = bound_delta

        # goal_constraints.update(c_avoidance_constraints)
        goal_constraints.update({f'open_object_{x}': PIDC(s, s, 1) for x, s in enumerate(gm.free_symbols(obj_pose))})

        in_contact = gm.less_than(geom_distance, 0.01)

        if robot in start_poses:
          start_state.update({gm.Position(Path(robot) + (k,)): v  for k, v in start_poses[robot].items()})

        if args.vis_plan == 'during':
            qpb = GQPB(coll_world, constraints, goal_constraints, controlled_values, visualizer=visualizer)
        else:
            qpb = GQPB(coll_world, constraints, goal_constraints, controlled_values)

        start_state.update({c.symbol: 0.0 for c in controlled_values.values()})

        qpb._cb_draw = p_internals.f_debug_draw
        integrator = CommandIntegrator(qpb,
                                       integration_rules,
                                       start_state=start_state,
                                       recorded_terms={'distance': geom_distance,
                                                       'gaze_align': look_goal,
                                                       'in contact': in_contact,
                                                       'goal': next(iter(goal_constraints.values())).expr,
                                                       'location_x': base_joint.x_pos,
                                                       'location_y': base_joint.y_pos,
                                                       'rotation_a': base_joint.a_pos},
                                       printed_exprs=printed_exprs)


        # RUN
        int_factor = 0.02
        integrator.restart('{} Cartesian Goal Example'.format(robot))
        # print('\n'.join('{}: {}'.format(s, r) for s, r in integrator.integration_rules.items()))
        try:
            start = Time.now()
            integrator.run(int_factor, 500, logging=False, real_time=False)
            total_dur.append((Time.now() - start).to_sec())
            n_iter.append(integrator.current_iteration + 1)
        except Exception as e:
            print('Exception during integration:\n{}'.format(e))



        # DRAW
        print('Drawing recorders')
        # draw_recorders([filter_contact_symbols(integrator.recorder, integrator.qp_builder), integrator.sym_recorder], 4.0/9.0, 8, 4).savefig('{}/{}_sandbox_{}_plots.png'.format(plot_dir, robot, part))
        if PANDA_LOGGING:
            rec_w, rec_b, rec_c, recs = convert_qp_builder_log(integrator.qp_builder)
            rec_c.title = 'Joint Velocity Commands'
            rec_c.data = {s.split(SPS)[1]: d for s, d in rec_c.data.items() if s.split(SPS)[1] in start_poses['pr2']}
            print(rec_c.data)
            color_gen = ColorGenerator(v=1.0, s_lb=0.75)
            rec_c.colors = {s: color_gen.get_color_hex() for s in rec_c.data.keys()}
            rec_c.data_lim = {}
            rec_c.compute_limits()
            rec_c.set_xtitle('Iteration')
            rec_c.set_grid(True)
            rec_c.set_xspace(-5, integrator.current_iteration + 5)
            rec_c.set_legend_location('upper left')
            fig = plt.figure(figsize=(4, 2.5))
            ax  = fig.add_subplot(1, 1, 1)
            rec_c.plot(ax)
            # ax.get_legend().loc = 'upper right'
            # ax.get_legend().bbox_to_anchor = (1.0, 0.0)
            fig.tight_layout()
            fig.savefig('{}/{}_sandbox_{}_plots.png'.format(plot_dir, robot, part))

            # draw_recorders([rec_c], 0.5, 4, 2.5).savefig('{}/{}_sandbox_{}_plots.png'.format(plot_dir, robot, part))
            # draw_recorders([rec_b, rec_c] + [r for _, r in sorted(recs.items())], 1, 8, 4).savefig('{}/{}_sandbox_{}_constraints.png'.format(plot_dir, robot, part))

        if args.vis_plan == 'after':
            traj_vis.visualize(integrator.recorder.data, hz=50)
            pass

        if rospy.is_shutdown():
          break

    traj_vis.shutdown()

    print('Mean Iter: {}\nMean Iter D: {:>2.6f}\nIter D SD: {:>2.6f}\nMean planning duration: {:>2.6f}s'.format(np.mean(n_iter),
                                                                 np.mean([d / float(i) for d, i in zip(total_dur, n_iter)]),
                                                                 np.std([d / float(i) for d, i in zip(total_dur, n_iter)]),
                                                                 np.mean(total_dur)))
