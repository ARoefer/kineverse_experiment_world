#!/usr/bin/env python
import rospy
import numpy as np
import pandas as pd
import argparse

from kineverse_experiment_world.tracking_node import TrackerNode

from kineverse_experiment_world.msg    import PoseStampedArray as PSAMsg
from kineverse_msgs.msg                import ValueMap         as ValueMapMsg

from kineverse.model.paths                  import Path
from kineverse.operations.urdf_operations   import load_urdf
from kineverse.motion.min_qp_builder        import QPSolverException
from kineverse.gradients.gradient_math      import Symbol, subs, cm, dot
from kineverse.visualization.plotting       import convert_qp_builder_log, draw_recorders
from kineverse.time_wrapper                 import Time
from kineverse.type_sets                    import is_symbolic
from kineverse.utils                        import res_pkg_path, real_quat_from_matrix
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer
from kineverse.urdf_fix                     import urdf_filler

from kineverse_experiment_world.tracking_node import TrackerNode
from kineverse_experiment_world.utils         import random_normal_translation, random_rot_normal

from geometry_msgs.msg import PoseStamped as PoseStampedMsg

from urdf_parser_py.urdf import URDF

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark for the simple object tracker.')
    parser.add_argument('--step', '-d', type=float, default=1, help='Size of the integration step. 0 < s < 1')
    parser.add_argument('--max-iter', '-mi', type=int, default=10, help='Maximum number of iterations per observation.')
    parser.add_argument('--samples', '-s', type=int, default=200, help='Number of observations to generate per configuration.')
    parser.add_argument('--noise-lin', '-nl', type=float, default=0.15, help='Maximum linear noise.')
    parser.add_argument('--noise-ang', '-na', type=float, default=10.0, help='Maximum angular noise in degrees.')
    parser.add_argument('--noise-steps', '-ns', type=int, default=5, help='Number of steps from lowest to highest noise.')
    parser.add_argument('--out', '-o', type=str, default='tracker_results_n_dof.csv', help='Name of the resulting csv file.')
    args = parser.parse_args()

    rospy.init_node('kineverse_tracking_node')
    tracker = TrackerNode('/tracked/state', '/pose_obs', args.step, args.max_iter, use_timer=False)

    with open(res_pkg_path('package://iai_kitchen/urdf_obj/IAI_kitchen.urdf'), 'r') as urdf_file:
        urdf_kitchen_str = urdf_file.read()

    kitchen_model = urdf_filler(URDF.from_xml_string(urdf_kitchen_str))
    load_urdf(tracker.km_client, Path('iai_oven_area'), kitchen_model)

    tracker.km_client.clean_structure()
    tracker.km_client.dispatch_events()

    groups = [[('iai_oven_area/links/room_link', 'iai_oven_area/room_link'),
               ('iai_oven_area/links/fridge_area', 'iai_oven_area/fridge_area'),
               ('iai_oven_area/links/fridge_area_footprint', 'iai_oven_area/fridge_area_footprint'),
               ('iai_oven_area/links/fridge_area_lower_drawer_handle', 'iai_oven_area/fridge_area_lower_drawer_handle'),
               ('iai_oven_area/links/fridge_area_lower_drawer_main', 'iai_oven_area/fridge_area_lower_drawer_main')]
             ,[('iai_oven_area/links/iai_fridge_door', 'iai_oven_area/iai_fridge_door'),
               ('iai_oven_area/links/iai_fridge_door_handle', 'iai_oven_area/iai_fridge_door_handle'),
               ('iai_oven_area/links/iai_fridge_main', 'iai_oven_area/iai_fridge_main')]
             ,[('iai_oven_area/links/kitchen_island', 'iai_oven_area/kitchen_island'),
               ('iai_oven_area/links/kitchen_island_footprint', 'iai_oven_area/kitchen_island_footprint'),
               ('iai_oven_area/links/kitchen_island_left_panel', 'iai_oven_area/kitchen_island_left_panel'),
               ('iai_oven_area/links/kitchen_island_middle_panel', 'iai_oven_area/kitchen_island_middle_panel'),
               ('iai_oven_area/links/kitchen_island_right_panel', 'iai_oven_area/kitchen_island_right_panel'),
               ('iai_oven_area/links/kitchen_island_stove', 'iai_oven_area/kitchen_island_stove')]
             ,[('iai_oven_area/links/kitchen_island_left_lower_drawer_handle', 'iai_oven_area/kitchen_island_left_lower_drawer_handle'),
               ('iai_oven_area/links/kitchen_island_left_lower_drawer_main', 'iai_oven_area/kitchen_island_left_lower_drawer_main')]
             ,[('iai_oven_area/links/kitchen_island_left_upper_drawer_handle', 'iai_oven_area/kitchen_island_left_upper_drawer_handle'),
               ('iai_oven_area/links/kitchen_island_left_upper_drawer_main', 'iai_oven_area/kitchen_island_left_upper_drawer_main')]
             ,[('iai_oven_area/links/kitchen_island_middle_lower_drawer_handle', 'iai_oven_area/kitchen_island_middle_lower_drawer_handle'),
               ('iai_oven_area/links/kitchen_island_middle_lower_drawer_main', 'iai_oven_area/kitchen_island_middle_lower_drawer_main')]
             ,[('iai_oven_area/links/kitchen_island_middle_upper_drawer_handle', 'iai_oven_area/kitchen_island_middle_upper_drawer_handle'),
               ('iai_oven_area/links/kitchen_island_middle_upper_drawer_main', 'iai_oven_area/kitchen_island_middle_upper_drawer_main')]
             ,[('iai_oven_area/links/kitchen_island_right_lower_drawer_handle', 'iai_oven_area/kitchen_island_right_lower_drawer_handle'),
               ('iai_oven_area/links/kitchen_island_right_lower_drawer_main', 'iai_oven_area/kitchen_island_right_lower_drawer_main')]
             ,[('iai_oven_area/links/kitchen_island_right_upper_drawer_handle', 'iai_oven_area/kitchen_island_right_upper_drawer_handle'),
               ('iai_oven_area/links/kitchen_island_right_upper_drawer_main', 'iai_oven_area/kitchen_island_right_upper_drawer_main')]
             ,[('iai_oven_area/links/oven_area_area', 'iai_oven_area/oven_area_area'),
               ('iai_oven_area/links/oven_area_area_footprint', 'iai_oven_area/oven_area_area_footprint')]
             ,[('iai_oven_area/links/oven_area_area_left_drawer_handle', 'iai_oven_area/oven_area_area_left_drawer_handle'),
               ('iai_oven_area/links/oven_area_area_left_drawer_main', 'iai_oven_area/oven_area_area_left_drawer_main')]
             ,[('iai_oven_area/links/oven_area_area_middle_lower_drawer_handle', 'iai_oven_area/oven_area_area_middle_lower_drawer_handle'),
               ('iai_oven_area/links/oven_area_area_middle_lower_drawer_main', 'iai_oven_area/oven_area_area_middle_lower_drawer_main')]
             ,[('iai_oven_area/links/oven_area_area_middle_upper_drawer_handle', 'iai_oven_area/oven_area_area_middle_upper_drawer_handle'),
               ('iai_oven_area/links/oven_area_area_middle_upper_drawer_main', 'iai_oven_area/oven_area_area_middle_upper_drawer_main')]
             ,[('iai_oven_area/links/oven_area_area_right_drawer_handle', 'iai_oven_area/oven_area_area_right_drawer_handle'),
               ('iai_oven_area/links/oven_area_area_right_drawer_main', 'iai_oven_area/oven_area_area_right_drawer_main')]
             ,[('iai_oven_area/links/oven_area_oven_door', 'iai_oven_area/oven_area_oven_door'),
               ('iai_oven_area/links/oven_area_oven_door_handle', 'iai_oven_area/oven_area_oven_door_handle')]
             ,[('iai_oven_area/links/oven_area_oven_knob_oven', 'iai_oven_area/oven_area_oven_knob_oven'),
               ('iai_oven_area/links/oven_area_oven_main', 'iai_oven_area/oven_area_oven_main'),
               ('iai_oven_area/links/oven_area_oven_panel', 'iai_oven_area/oven_area_oven_panel'),
               ('iai_oven_area/links/oven_area_oven_knob_stove_1', 'iai_oven_area/oven_area_oven_knob_stove_1')]
             ,[('iai_oven_area/links/oven_area_oven_knob_stove_2', 'iai_oven_area/oven_area_oven_knob_stove_2')]
             ,[('iai_oven_area/links/oven_area_oven_knob_stove_3', 'iai_oven_area/oven_area_oven_knob_stove_3')]
             ,[('iai_oven_area/links/oven_area_oven_knob_stove_4', 'iai_oven_area/oven_area_oven_knob_stove_4')]
             ,[('iai_oven_area/links/sink_area', 'iai_oven_area/sink_area'),
               ('iai_oven_area/links/sink_area_right_panel', 'iai_oven_area/sink_area_right_panel'),
               ('iai_oven_area/links/sink_area_sink', 'iai_oven_area/sink_area_sink')]
             ,[('iai_oven_area/links/sink_area_dish_washer_door', 'iai_oven_area/sink_area_dish_washer_door'),
               ('iai_oven_area/links/sink_area_dish_washer_door_handle', 'iai_oven_area/sink_area_dish_washer_door_handle')]
             ,[('iai_oven_area/links/sink_area_dish_washer_main', 'iai_oven_area/sink_area_dish_washer_main'),
               ('iai_oven_area/links/sink_area_footprint', 'iai_oven_area/sink_area_footprint')]
             ,[('iai_oven_area/links/sink_area_left_bottom_drawer_handle', 'iai_oven_area/sink_area_left_bottom_drawer_handle'),
               ('iai_oven_area/links/sink_area_left_bottom_drawer_main', 'iai_oven_area/sink_area_left_bottom_drawer_main')]
             ,[('iai_oven_area/links/sink_area_left_middle_drawer_handle', 'iai_oven_area/sink_area_left_middle_drawer_handle'),
               ('iai_oven_area/links/sink_area_left_middle_drawer_main', 'iai_oven_area/sink_area_left_middle_drawer_main')]
             ,[('iai_oven_area/links/sink_area_left_upper_drawer_handle', 'iai_oven_area/sink_area_left_upper_drawer_handle'),
               ('iai_oven_area/links/sink_area_left_upper_drawer_main', 'iai_oven_area/sink_area_left_upper_drawer_main')]
             ,[('iai_oven_area/links/sink_area_trash_drawer_handle', 'iai_oven_area/sink_area_trash_drawer_handle'),
               ('iai_oven_area/links/sink_area_trash_drawer_main', 'iai_oven_area/sink_area_trash_drawer_main')]]    


    columns = ['DoF', 'Poses', 'Linear SD', 'Angular SD', 'Mean Error', 'SD Error', 'Min Error', 'Max Error', 'Mean Iterations', 'Iteration Duration']
    result_rows = []
    visualizer  = ROSBPBVisualizer('/tracker_vis', 'world')

    last_n_dof = 0

    total_start = Time.now()

    for group in groups:
        for path, alias in group:
            tracker.track(path, alias)

        constraints = tracker.km_client.get_constraints_by_symbols(tracker.joints)

        constraints = {c.expr: [float(subs(c.lower, {c.expr: 0})), float(subs(c.upper, {c.expr: 0}))] for k, c in constraints.items() if cm.is_symbol(c.expr) and not is_symbolic(subs(c.lower, {c.expr: 0})) and not is_symbolic(subs(c.upper, {c.expr: 0}))}

        joint_array = [s for _, s in sorted([(str(s), s) for s in tracker.joints])]
        if len(joint_array) == last_n_dof:
            continue

        last_n_dof = len(joint_array)

        world  = tracker.km_client.get_active_geometry(set(joint_array))


        bounds = np.array([constraints[s] if s in constraints else 
                           [-np.pi, np.pi] for s in joint_array])
        # print('\n'.join(['{}: {} {}'.format(c, t[0], t[1]) for c, t in constraints.items()]))

        offset = bounds.T[0]
        scale  = bounds.T[1] - bounds.T[0]

        r_alias     = {p: a for a, p in tracker.aliases.items()}
        frames      = {a: tracker.km_client.get_data('{}/pose'.format(p)) for a, p in tracker.aliases.items()}
        true_frames = {p: tracker.km_client.get_data('{}/pose'.format(p)) for a, p in tracker.aliases.items()}
        print('n frames', len(frames))
        print('n joints', len(joint_array))

        update_msg = PSAMsg()
        update_msg.poses = [PoseStampedMsg() for x in range(len(frames))]

        iter_times   = []
        n_iter       = []
        config_delta = []
        n_crashes    = 0


        for linear_std, angular_std in zip(np.linspace(0, args.noise_lin, args.noise_steps), 
                                           np.linspace(0, args.noise_ang * (np.pi / 180.0), args.noise_steps)):
            for x in range(args.samples):
                # Uniformly sample a joint state
                joint_state = dict(zip(joint_array, np.random.rand(len(joint_array)) * scale + offset))

                # Generate n noise transforms
                noise = [cm.to_numpy(dot(t, r)) for t, r in zip(random_normal_translation(len(frames), 0, linear_std),
                                                            random_rot_normal(len(frames), 0, angular_std))]

                # Calculate forward kinematics of frames
                obs_frames = {k: subs(f, joint_state).dot(n) for (k, f), n in zip(frames.items(), noise)}

                for x, (k, f) in enumerate(obs_frames.items()):
                    update_msg.poses[x].header.frame_id = k
                    update_msg.poses[x].pose.position.x = float(f[0, 3])
                    update_msg.poses[x].pose.position.y = float(f[1, 3])
                    update_msg.poses[x].pose.position.z = float(f[2, 3])

                    qx, qy, qz, qw = real_quat_from_matrix(f)
                    update_msg.poses[x].pose.orientation.x = qx
                    update_msg.poses[x].pose.orientation.y = qy
                    update_msg.poses[x].pose.orientation.z = qz
                    update_msg.poses[x].pose.orientation.w = qw

                try:
                    tracker.cb_process_obs(update_msg)

                    time_start = Time.now()
                    tracker.cb_tick(None)

                    iter_times.append((Time.now() - time_start).to_sec() / (tracker.integrator.current_iteration + 1))
                    n_iter.append(tracker.integrator.current_iteration + 1)
                    config_delta.append([float(np.abs(joint_state[s] - tracker.integrator.state[s])) for s in joint_array])
                except QPSolverException as e:
                    n_crashes += 1

                if rospy.is_shutdown():
                    break
            if rospy.is_shutdown():
                    break


            result_rows.append([len(joint_array),
                                len(frames),
                                linear_std, 
                                angular_std, 
                                np.mean(np.mean(config_delta, 1)), 
                                np.std(np.mean(config_delta, 1)), 
                                np.min(np.mean(config_delta, 1)), 
                                np.max(np.mean(config_delta, 1)),
                                np.mean(n_iter),
                                np.mean(iter_times)])

            print('Mean s/iteration: {}\n'
                  'Mean n iter: {}\n'
                  'Mean final delta: {}\n'
                  'Std final delta: {}\n'
                  'Min final delta: {}\n'
                  'Max final delta: {}\n'
                  'Number of solver crashes: {}'.format(
                    np.mean(iter_times),
                    np.mean(n_iter),
                    np.mean(np.mean(config_delta, 1)),
                    np.std(np.mean(config_delta, 1)),
                    np.min(np.mean(config_delta, 1)),
                    np.max(np.mean(config_delta, 1)),
                    n_crashes))

    print('Total time taken: {} s'.format((Time.now() - total_start).to_sec()))

    df_results = pd.DataFrame(columns=columns, data=result_rows)
    df_results.to_csv(args.out, float_format='%.5f', index=False)
