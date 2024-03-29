#!/usr/bin/env python
import rospy
import argparse
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

import kineverse.gradients.gradient_math as gm 
from kineverse.model.geometry_model         import GeometryModel, Path
from kineverse.operations.urdf_operations   import load_urdf
from kineverse.urdf_fix                     import hacky_urdf_parser_fix, \
                                                   urdf_filler
from kineverse.utils                        import res_pkg_path, union
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer

from kineverse_experiment_world.utils     import random_normal_translation, random_rot_normal, str2bool
from kineverse_experiment_world.ekf_model import EKFModel

from urdf_parser_py.urdf import URDF

from time import time
from tqdm import tqdm

    
@profile
def main(create_figure=False, 
         vis_mode=False, 
         log_csv=True, 
         min_n_dof=1,
         samples=300,
         n_observations=25,
         noise_lin=0.2,
         noise_ang=30,
         noise_steps=5):

    wait_duration = rospy.Duration(0.1)

    vis = ROSBPBVisualizer('ekf_vis', 'world') if vis_mode != 'none' else None
    km  = GeometryModel()

    with open(res_pkg_path('package://iai_kitchen/urdf_obj/IAI_kitchen.urdf'), 'r') as urdf_file:
        urdf_kitchen_str = urdf_file.read()
        kitchen_model = urdf_filler(URDF.from_xml_string(hacky_urdf_parser_fix(urdf_kitchen_str)))
        load_urdf(km, Path('kitchen'), kitchen_model)

    km.clean_structure()
    km.dispatch_events()

    kitchen = km.get_data('kitchen')

    tracking_pools = []
    for name, link in kitchen.links.items():
        symbols = gm.free_symbols(link.pose)
        if len(symbols) == 0:
            continue

        for x in range(len(tracking_pools)):
            syms, l = tracking_pools[x]
            if len(symbols.intersection(syms)) != 0: # BAD ALGORITHM, DOES NOT CORRECTLY IDENTIFY SETS
                tracking_pools[x] = (syms.union(symbols), l + [(name, link.pose)])
                break
        else:
            tracking_pools.append((symbols, [(name, link.pose)]))

    # tracking_pools = [tracking_pools[7]]
    # print('Identified {} tracking pools:\n{}'.format(len(tracking_pools), tracking_pools))

    all_ekfs = [EKFModel(dict(poses), km.get_constraints_by_symbols(symbols))
                                         for symbols, poses in tracking_pools] # np.eye(len(symbols)) * 0.001
    print('Created {} EKF models'.format(len(all_ekfs)))
    print('\n'.join(str(e) for e in all_ekfs))

    # Sanity constraint
    min_n_dof = min(min_n_dof, len(all_ekfs))

    iteration_times = []    

    for u in range(min_n_dof, len(all_ekfs) + 1):
        if rospy.is_shutdown():
            break;

        ekfs = all_ekfs[:u]
    
        observed_poses = {}
        for ekf in ekfs:
            for link_name, _ in ekf.takers:
                observed_poses[link_name] = kitchen.links[link_name].pose
        names, poses = zip(*sorted(observed_poses.items()))

        state_symbols = union([gm.free_symbols(p) for p in poses])
        ordered_state_vars = [s for _, s in sorted((str(s), s) for s in state_symbols)]

        state_constraints = {}
        for n, c in km.get_constraints_by_symbols(state_symbols).items():
            if gm.is_symbol(c.expr):
                s  = gm.free_symbols(c.expr).pop()
                fs = gm.free_symbols(c.lower).union(gm.free_symbols(c.upper))
                if len(fs.difference({s})) == 0:
                    state_constraints[s] = (float(gm.subs(c.lower, {s: 0})), float(gm.subs(c.upper, {s: 0})))

        state_bounds = np.array([state_constraints[s] if  s in state_constraints else
                                [-np.pi * 0.5, np.pi * 0.5] for s in ordered_state_vars])

        state_fn = gm.speed_up(gm.vstack(*poses), ordered_state_vars)
        subworld = km.get_active_geometry(state_symbols)

        # Generate observation noise
        print('Generating R matrices...')
        n_cov_obs = 400
        full_log  = []

        dof_iters = []

        # EXPERIMENT
        for lin_std, ang_std in [(noise_lin, noise_ang * (np.pi / 180.0))]: 
                                # zip(np.linspace(0, noise_lin, noise_steps), 
                                #     np.linspace(0, noise_ang * (np.pi / 180.0), noise_steps)):
            if rospy.is_shutdown():
                break;
            # INITIALIZE SENSOR MODEL
            training_obs = []
            state = np.random.uniform(state_bounds.T[0], state_bounds.T[1])
            observations = state_fn.call2(state)

            for _ in range(n_cov_obs):
                noisy_obs = {}
                for x, noise in enumerate([t.dot(r) for t, r in zip(random_normal_translation(len(poses), 0, lin_std),
                                                                    random_rot_normal(len(poses), 0, ang_std))]):
                    noisy_obs[names[x]] = observations[x*4:x*4 + 4,:4].dot(noise)
                training_obs.append(noisy_obs)

            for ekf in ekfs:
                ekf.generate_R(training_obs)
                # ekf.set_R(np.eye(len(ekf.ordered_vars)) * 0.1)


            # Generate figure
            gridsize  = (4, samples)
            plot_size = (4, 4)
            fig = plt.figure(figsize=(gridsize[1] * plot_size[0], 
                                      gridsize[0] * plot_size[1])) if create_figure else None

            gt_states = []
            states    = [[] for x in range(samples)]
            variances = [[] for x in range(samples)]
            e_obs     = [[] for x in range(samples)]

            print('Starting iterations')
            for k in tqdm(range(samples)):
                if rospy.is_shutdown():
                    break;

                state = np.random.uniform(state_bounds.T[0], state_bounds.T[1])
                gt_states.append(state)
                observations = state_fn.call2(state).copy()
                gt_obs_d     = {n: observations[x*4:x*4 + 4,:4] for x, n in enumerate(names)}
                subworld.update_world(dict(zip(ordered_state_vars, state)))

                if vis_mode == 'iter' or vis_mode == 'io':
                    vis.begin_draw_cycle('gt', 'noise', 'estimate', 't_n', 't0')
                    vis.draw_world('gt', subworld, g=0, b=0)
                    vis.render('gt')

                estimates = []
                for ekf in ekfs:
                    particle = ekf.spawn_particle()
                    estimates.append(particle)

                initial_state = dict(sum([[(s, v) for s, v in zip(ekf.ordered_vars, e.state)]
                                                  for e, ekf in zip(estimates, ekfs)], []))
                initial_state = np.array([initial_state[s] for s in ordered_state_vars])
                if initial_state.min() < state_bounds.T[0].min() or initial_state.max() > state_bounds.T[1].max():
                    raise Exception('Estimate initialization is out of bounds: {}'.format(
                                    np.vstack([initial_state, state_bounds.T]).T))
                initial_delta = state - initial_state


                for y in range(n_observations):
                    # Add noise to observations
                    noisy_obs = {}
                    for x, noise in enumerate([t.dot(r) for t, r in zip(random_normal_translation(len(poses), 0, lin_std),
                                                                        random_rot_normal(len(poses), 0, ang_std))]):
                        noisy_obs[names[x]] = observations[x*4:x*4 + 4,:4].dot(noise)

                    if vis_mode in {'iter', 'iter-trail'} or (vis_mode == 'io' and y == 0):
                        for n, t in noisy_obs.items():
                            subworld.named_objects[Path(('kitchen', 'links', n))].np_transform = t
                        if vis_mode != 'iter-trail':
                            vis.begin_draw_cycle('noise')
                        vis.draw_world('noise', subworld, r=0, g=0, a=0.1)
                        vis.render('noise')

                    start_time = time()
                    for estimate, ekf in zip(estimates, ekfs):
                        if y > 0:
                            control = np.zeros(len(ekf.ordered_controls))
                            estimate.state, estimate.cov = ekf.predict(estimate.state.flatten(), estimate.cov, control)
                            obs_vector = ekf.gen_obs_vector(noisy_obs)
                            estimate.state, estimate.cov = ekf.update(estimate.state, estimate.cov, ekf.gen_obs_vector(noisy_obs))

                            if vis_mode in {'iter', 'iter-trail'}:
                                subworld.update_world({s: v for s, v in zip(ekf.ordered_vars, estimate.state)})
                        else:
                            obs_vector = ekf.gen_obs_vector(noisy_obs)

                            for _ in range(1):
                                h_prime   = ekf.h_prime_fn.call2(estimate.state)
                                obs_delta = obs_vector.reshape((len(obs_vector), 1)) - ekf.h_fn.call2(estimate.state)
                                estimate.state += (h_prime.T.dot(obs_delta) * 0.1).reshape(estimate.state.shape)

                            if vis_mode in {'iter', 'io'}:
                                subworld.update_world({s: v for s, v in zip(ekf.ordered_vars, estimate.state)})
                    
                    if vis_mode != 'none' and y == 0:
                        vis.draw_world('t0', subworld, b=0, a=1)
                        vis.render('t0')
                    elif vis_mode in {'iter', 'iter-trail'}:
                        if vis_mode != 'iter-trail':
                            vis.begin_draw_cycle('t_n')
                        vis.draw_world('t_n', subworld, b=0, a=1)
                        vis.render('t_n')


                    if log_csv or fig is not None:
                        e_state_d = dict(sum([[(s, v) for s, v   in zip(ekf.ordered_vars, e.state)]
                                                      for e, ekf in zip(estimates, ekfs)], []))
                        covs      = dict(sum([[(s, v) for s, v   in zip(ekf.ordered_vars, np.sqrt(np.trace(e.cov)))]
                                                      for e, ekf in zip(estimates, ekfs)], []))
                        e_state = np.hstack([e_state_d[s] for s in ordered_state_vars]).reshape((len(e_state_d),))
                        
                        if log_csv:
                            full_log.append(np.hstack(([lin_std, ang_std], state, 
                                                       e_state.flatten(), np.array([covs[s] for s in ordered_state_vars]))))

                        if fig is not None:
                            e_obs[k].append(np.array([np.abs(ekf.gen_obs_vector(gt_obs_d) - ekf.h_fn.call2(e.state)).max() for e, ekf in zip(estimates, ekfs)]))
                            states[k].append(e_state)
                            variances[k].append(np.array([covs[s] for s in ordered_state_vars]))
                else:
                    if vis_mode == 'io':
                        for estimate, ekf in zip(estimates, ekfs):
                            subworld.update_world({s: v for s, v in zip(ekf.ordered_vars, estimate.state)})

                        vis.draw_world('t_n', subworld, r=0, b=0, a=1)
                        vis.render('t_n')

                    
                    dof_iters.append(time() - start_time)

            

            if fig is not None:
                axes = [plt.subplot2grid(gridsize, (y, 0), colspan=1, rowspan=1) for y in range(gridsize[0])]
                axes = np.array(sum([[plt.subplot2grid(gridsize, (y, x), colspan=1, rowspan=1, sharey=axes[y])
                                                                            for y in range(gridsize[0])]
                                                                            for x in range(1, gridsize[1])], axes)).reshape((gridsize[1], gridsize[0]))

                for x, (gt_s, state, variance, obs_delta, (ax_s, ax_d, ax_o, ax_v)) in enumerate(zip(gt_states, states, variances, e_obs, axes)):

                    for y in gt_s:
                        ax_s.axhline(y, xmin=0.97, xmax=1.02)

                    ax_s.set_title('State; Sample: {}'.format(x))
                    ax_d.set_title('Delta from GT; Sample: {}'.format(x))
                    ax_o.set_title('Max Delta in Obs; Sample: {}'.format(x))
                    ax_v.set_title('Standard Deviation; Sample: {}'.format(x))
                    ax_s.plot(state)
                    ax_d.plot(gt_s - np.vstack(state))
                    ax_o.plot(obs_delta)
                    ax_v.plot(variance)
                    ax_s.grid(True)
                    ax_d.grid(True)
                    ax_o.grid(True)
                    ax_v.grid(True)

                fig.tight_layout()
                plt.savefig(res_pkg_path('package://kineverse_experiment_world/test/ekf_object_tracker_{}_{}.png'.format(lin_std, ang_std)))

        iteration_times.append(dof_iters)

        if log_csv:
            df = pd.DataFrame(columns=['lin_std', 'ang_std'] + 
                                      ['gt_{}'.format(x) for x in range(len(state_symbols))] +
                                      ['ft_{}'.format(x) for x in range(len(state_symbols))] + 
                                      ['var_{}'.format(x) for x in range(len(state_symbols))],
                              data=full_log)
            df.to_csv(res_pkg_path('package://kineverse_experiment_world/test/ekf_object_tracker.csv'), index=False)
    

    df = pd.DataFrame(columns=[str(x) for x in range(1, len(iteration_times) + 1)],
                      data=np.vstack(iteration_times).T)
    df.to_csv(res_pkg_path('package://kineverse_experiment_world/test/ekf_object_tracker_performance.csv'), index=False)


if __name__ == '__main__':
    rospy.init_node('ekf_node')

    parser = argparse.ArgumentParser(description='Implements articulated object state estimation via EKF.')
    parser.add_argument('--noise-lin', type=float, default=0.2, help='SD of linear noise in observations (m).')
    parser.add_argument('--noise-ang', type=float, default=30, help='SD of angular noise in observations (deg).')
    parser.add_argument('--noise-steps', type=int, default=5, help='Number interpolation steps between 0 and max noise.')
    parser.add_argument('--vis-mode', type=str, default='none', help='Visualize state estimation. [ none | iter | iter-trail | io ]')
    parser.add_argument('--n-obs', type=int, default=25, help='Number of total observations per sample.')
    parser.add_argument('--figure', type=str2bool, default=False, help='Create stats plot for samples. USE WITH LOW SAMPLE COUNT ONLY.')
    parser.add_argument('--log-csv', type=str2bool, default=False, help='Capture performance stats and save them to a csv.')
    parser.add_argument('--min-n-dof', type=int, default=1, help='Minimum number of DoF to track.')
    parser.add_argument('--n-samples', type=int, default=300, help='Number of scenarios to run.')

    args = parser.parse_args()

    main(create_figure=args.figure, 
         vis_mode=args.vis_mode, 
         log_csv=args.log_csv, 
         min_n_dof=max(1, args.min_n_dof),
         samples=abs(args.n_samples),
         n_observations=abs(args.n_obs),
         noise_lin=abs(args.noise_lin),
         noise_ang=abs(args.noise_ang),
         noise_steps=abs(args.noise_steps))
