#!/usr/bin/env python
import rospy
import numpy as np
import pandas as pd


from kineverse_experiment_world.tracking_node import TrackerNode

from kineverse_experiment_world.msg    import PoseStampedArray as PSAMsg
from kineverse.msg                     import ValueMap         as ValueMapMsg

from kineverse.bpb_wrapper import pb, create_object, create_cube_shape, create_sphere_shape, create_cylinder_shape, create_compound_shape, load_convex_mesh_shape, matrix_to_transform
from kineverse.motion.min_qp_builder   import QPSolverException
from kineverse.gradients.gradient_math import Symbol, subs
from kineverse.visualization.plotting  import convert_qp_builder_log, draw_recorders
from kineverse.time_wrapper            import Time
from kineverse.type_sets               import is_symbolic
from kineverse.utils                   import res_pkg_path, real_quat_from_matrix
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer

from kineverse_experiment_world.tracking_node import TrackerNode
from kineverse_experiment_world.utils         import random_normal_translation, random_rot_normal

from geometry_msgs.msg import PoseStamped as PoseStampedMsg


if __name__ == '__main__':
    rospy.init_node('kineverse_tracking_node')
    tracker = TrackerNode('/tracked/state', '/pose_obs', 1, 10, use_timer=False)

    # tracker.track('/iai_oven_area/links/sink_area_dish_washer_door', 'iai_kitchen/sink_area_dish_washer_door')
    # tracker.track('/iai_oven_area/links/sink_area_left_upper_drawer_main', 'iai_kitchen/sink_area_left_upper_drawer_main')
    # tracker.track('/iai_oven_area/links/sink_area_left_middle_drawer_main', 'iai_kitchen/sink_area_left_middle_drawer_main')
    # tracker.track('/iai_oven_area/links/sink_area_left_bottom_drawer_main', 'iai_kitchen/sink_area_left_bottom_drawer_main')
    # tracker.track('/iai_oven_area/links/sink_area_trash_drawer_main', 'iai_kitchen/sink_area_trash_drawer_main')
    # tracker.track('/iai_oven_area/links/iai_fridge_door', 'iai_kitchen/iai_fridge_door')
    # tracker.track('/iai_oven_area/links/fridge_area_lower_drawer_main', 'iai_kitchen/fridge_area_lower_drawer_main')
    # tracker.track('/iai_oven_area/links/oven_area_oven_door', 'iai_kitchen/oven_area_oven_door')
    # tracker.track('/iai_oven_area/links/oven_area_area_right_drawer_main', 'iai_kitchen/oven_area_right_drawer_main')
    # tracker.track('/iai_oven_area/links/oven_area_area_left_drawer_main', 'iai_kitchen/oven_area_left_drawer_main')
    # tracker.track('/iai_oven_area/links/oven_area_area_middle_upper_drawer_main', 'iai_kitchen/oven_area_area_middle_upper_drawer_main')
    # tracker.track('/iai_oven_area/links/oven_area_area_middle_lower_drawer_main', 'iai_kitchen/oven_area_area_middle_lower_drawer_main')

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
    df_results = pd.DataFrame(columns=columns)
    visualizer = ROSBPBVisualizer('/tracker_vis', 'map')

    last_n_dof = 0

    for group in groups:
        for path, alias in group:
            tracker.track(path, alias)

        constraints = tracker.km_client.get_constraints_by_symbols(tracker.joints)

        constraints = {c.expr: [float(subs(c.lower, {c.expr: 0})), float(subs(c.upper, {c.expr: 0}))] for k, c in constraints.items() if type(c.expr) == Symbol and not is_symbolic(subs(c.lower, {c.expr: 0})) and not is_symbolic(subs(c.upper, {c.expr: 0}))}

        joint_array = [s for _, s in sorted([(str(s), s) for s in tracker.joints])]
        if len(joint_array) == last_n_dof:
            continue

        last_n_dof = len(joint_array)

        world  = tracker.km_client.km.get_active_geometry(set(joint_array))


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


        for linear_std, angular_std in zip(np.linspace(0, 0.15, 5), np.linspace(0, 10 * (np.pi / 180.0), 5)):
            n_samples   = 200
            for x in range(n_samples):
                # Uniformly sample a joint state
                joint_state = dict(zip(joint_array, np.random.rand(len(joint_array)) * scale + offset))
                # print('\n'.join(['{:>65}: {} <= {} <= {}'.format(k, l, joint_state[k], u) for k, (l, u) in zip(joint_array, bounds)]))
                # exit(0)
                # visualizer.begin_draw_cycle('objects')

                # for k, o in world.named_objects.items():
                #     if k in true_frames:
                #         tf = matrix_to_transform(subs(true_frames[k], joint_state))
                #         o.transform = tf
                #         visualizer.draw_collision_object('objects', o, r=0.2, b=0)

                # visualizer.render('objects')

                # rospy.sleep(2)
                # _ = raw_input('Press enter to continue')

                # Generate n noise transforms
                noise = [t * r for t, r in zip(random_normal_translation(len(frames), 0, linear_std), 
                                               random_rot_normal(len(frames), 0, angular_std))]

                # Calculate forward kinematics of frames
                obs_frames = {k: f.subs(joint_state) * n for (k, f), n in zip(frames.items(), noise)}

                # visualizer.begin_draw_cycle('noise')

                # for k, o in world.named_objects.items():
                #     if r_alias[k] in obs_frames:
                #         tf = matrix_to_transform(obs_frames[r_alias[k]])
                #         o.transform = tf
                #         visualizer.draw_collision_object('noise', o, r=0.9, g=0, b=0)

                # visualizer.render('noise')
                # _ = raw_input('Press enter to continue')

                for x, (k, f) in enumerate(obs_frames.items()):
                    update_msg.poses[x].header.frame_id = k
                    update_msg.poses[x].pose.position.x = f[0, 3]
                    update_msg.poses[x].pose.position.y = f[1, 3]
                    update_msg.poses[x].pose.position.z = f[2, 3]

                    qx, qy, qz, qw = real_quat_from_matrix(f)
                    update_msg.poses[x].pose.orientation.x = qx
                    update_msg.poses[x].pose.orientation.y = qy
                    update_msg.poses[x].pose.orientation.z = qz
                    update_msg.poses[x].pose.orientation.w = qw

                try:
                    tracker.cb_process_obs(update_msg)

                    time_start = Time.now()
                    tracker.cb_tick(None)

                    # visualizer.begin_draw_cycle('noise')            
                    # visualizer.begin_draw_cycle('solved')

                    # for k, o in world.named_objects.items():
                    #     if k in true_frames:
                    #         tf = matrix_to_transform(subs(true_frames[k], tracker.integrator.state))
                    #         o.transform = tf
                    #         visualizer.draw_collision_object('solved', o, r=0, g=0.2, b=1)

                    # visualizer.render()
                    # _ = raw_input('Press enter to continue')
                    # exit(0)

                    iter_times.append((Time.now() - time_start).to_sec() / (tracker.integrator.current_iteration + 1))
                    n_iter.append(tracker.integrator.current_iteration + 1)
                    config_delta.append([float(np.abs(joint_state[s] - tracker.integrator.state[s])) for s in joint_array])
                except QPSolverException as e:
                    n_crashes += 1

                if rospy.is_shutdown():
                    break
            if rospy.is_shutdown():
                    break


            df_results = df_results.append(pd.DataFrame(data=[[len(joint_array),
                                                               len(frames),
                                                               linear_std, 
                                                               angular_std, 
                                                               np.mean(np.mean(config_delta, 1)), 
                                                               np.std(np.mean(config_delta, 1)), 
                                                               np.min(np.mean(config_delta, 1)), 
                                                               np.max(np.mean(config_delta, 1)),
                                                               np.mean(n_iter),
                                                               np.mean(iter_times)]], 
                                                           columns=columns), ignore_index=True)

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

    df_results.to_csv('tracker_results_n_dof.csv', float_format='%.5f', index=False)
