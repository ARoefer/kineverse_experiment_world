#!/usr/bin/env python
import numpy as np
import pandas as pd
from collections import namedtuple

from kineverse.gradients.gradient_math import *
from kineverse.visualization.plotting  import *
from kineverse.motion.min_qp_builder   import TypedQPBuilder  as TQPB, \
                                              SoftConstraint  as SC,   \
                                              ControlledValue as CV,   \
                                              QPSolverException
from kineverse.motion.integrator       import CommandIntegrator
from kineverse.utils                   import res_pkg_path
from kineverse.time_wrapper            import Time

from kineverse_experiment_world.utils  import np_sphere_sampling, \
                                              sphere_sampling,    \
                                              random_rot_uniform  \

OptResult = namedtuple('OptResult', ['error_9d', 'error_axa', 'x', 'y', 'z', 'n_steps', 'sec_per_it'])
TotalResults = namedtuple('TotalResults', ['errors_9d', 'errors_axa', 'iterations'])

def rot_goal_non_hack(r_ctrl, r_goal):
    axis, angle   = axis_angle_from_matrix(rot_of(r_ctrl).T * rot_of(r_goal))
    r_rot_control = axis * angle

    r_dist = norm(r_rot_control)

    return {'Align rotation no-hack':  SC(-r_dist, -r_dist, 1, r_dist)}


def rot_goal_hack(r_ctrl, r_goal):
    axis, angle   = axis_angle_from_matrix(rot_of(r_ctrl).T * rot_of(r_goal))
    r_rot_control = axis * angle

    hack = rotation3_axis_angle([0, 0, 1], 0.0001)

    axis, angle = axis_angle_from_matrix((rot_of(r_ctrl).T * (rot_of(r_ctrl) * hack)).T)
    c_aa = (axis * angle)

    r_dist = norm(r_rot_control - c_aa)

    return {'Align rotation hack':  SC(-r_dist, -r_dist, 1, r_dist)}

def rot_goal_9D(r_ctrl, r_goal):
    r_dist = norm(list(r_ctrl[:3, :3] - r_goal[:3, :3]))
    return {'Align rotation 9d': SC(-r_dist, -r_dist, 1, r_dist)}

if __name__ == '__main__':
    
    r_points = 1000
    step     = 0.5


    methods = {f.func_name: f for f in [rot_goal_hack, rot_goal_non_hack, rot_goal_9D]}
    rs = {}

    s_x, s_y, s_z = [Position(x) for x in 'xyz']
    for k in methods.keys():
        a = vector3(s_x, s_y, s_z)
        rs[k] = rotation3_axis_angle(a / (norm(a) + 1e-4), norm(a))

    results = []

    for x, r_goal in enumerate(random_rot_uniform(r_points)):
        goal_ax, goal_a = axis_angle_from_matrix(r_goal)
        goal_axa = goal_ax * goal_a
        t_results = {}
        for k, m in methods.items():
            r_ctrl = rs[k]
            constraints = m(r_ctrl, r_goal)

            integrator = CommandIntegrator(TQPB({}, 
                                                constraints, 
                                                {str(s): CV(-1, 1, DiffSymbol(s), 1e-3) for s in sum([list(c.expr.free_symbols) 
                                                                                            for c in constraints.values()], [])}),
                                                start_state={s_x: 1, s_y: 0, s_z: 0}, equilibrium=0.001)#,
                                                # recorded_terms=recorded_terms)
            integrator.restart('Convergence Test {} {}'.format(x, k))
            try:
                t_start = Time.now()
                integrator.run(step, 100)
                t_per_it = (Time.now() - t_start).to_sec() / integrator.current_iteration
                final_r  = r_ctrl.subs(integrator.state)
                final_ax, final_a = axis_angle_from_matrix(final_r)
                final_axa = final_ax * final_a
                error_9d  = sum([np.abs(float(x)) for x in list(final_r - r_goal)])
                error_axa_sym = norm(final_axa - goal_axa)
                # print(final_ax, final_a, type(final_a))
                error_axa = float(error_axa_sym)
                t_results[k] = OptResult(error_9d,
                                         error_axa, 
                                         integrator.recorder.data['x_p'], 
                                         integrator.recorder.data['y_p'], 
                                         integrator.recorder.data['z_p'], 
                                         integrator.current_iteration,
                                         t_per_it)
            except QPSolverException:
                print('Solver Exception')
                t_results[k] = None
        
        # print('\n'.join([str(d) for k, d in sorted(integrator.sym_recorder.data.items())]))
        # final_r1 = r1.subs(integrator.state)
        # final_r2 = r2.subs(integrator.state)
        # final_errors.append((sum([np.abs(float(x)) for x in list(final_r1 - r_goal)]), sum([np.abs(float(x)) 
        #                                                  for x in list(final_r2 - r_goal)])))

        results.append(t_results)

        # data_array = np.array([[np.abs(float(x)) for x in integrator.sym_recorder.data[k]] 
        #                                                for k in sorted(recorded_terms.keys())])
        # means.append(np.mean(data_array, axis=1))
        # stds.append(np.std(data_array, axis=1))
        

    sub_headings = ['Mean', 'SD', 'Min', 'Max']
    np_functions = [np.mean, np.std, np.min, np.max]
    def apply_np_transform(d, c):
        for s, f in zip(sub_headings, np_functions):
            if s in c:
                return f(d)
        raise Exception('Could not figure out which np function to apply for "{}"'.format(c))


    columns = ['Method'] + ['Error 9d {}'.format(x)  for x in sub_headings] + \
                           ['Error AxA {}'.format(x) for x in sub_headings] + \
                           ['Iterations {}'.format(x) for x in sub_headings[:2]] + \
                           ['s/Iteration {}'.format(x) for x in sub_headings] + \
                           ['Failures in %']
    rows = []

    for k in sorted(methods.keys()):
        errors_9d  = [] 
        errors_axa = []
        iterations = []
        avg_iter   = []
        failures   = 0

        for rd in results:
            result = rd[k]
            if result is not None:
                errors_9d.append(float(result.error_9d))
                errors_axa.append(float(result.error_axa))
                iterations.append(result.n_steps)
                avg_iter.append(result.sec_per_it)
            else:
                failures += 1

        rows.append([k] + [apply_np_transform(d, c) for c, d in zip(columns[1:-1], [errors_9d]*4 + [errors_axa]*4 + [iterations]*2 + [avg_iter]*4)] + [float(failures) / len(results)])
        print('Results for {}:\n'
              '  Error 9d:\n    Mean: {}\n    StdD: {}\n    Min: {}\n    Max: {}\n'
              '  Error AxA:\n    Mean: {}\n    StdD: {}\n    Min: {}\n    Max: {}\n'
              '  Iterations:\n    Mean: {}\n    StdD: {}\n'
              '  s/Iteration:\n    Mean: {}\n    StdD: {}\n    Min: {}\n    Max: {}\n'
              '  Failures:  {}  -> {}%'.format(
                k, 
                np.mean(errors_9d),
                np.std(errors_9d),
                np.min(errors_9d),
                np.max(errors_9d),
                np.mean(errors_axa),
                np.std(errors_axa),
                np.min(error_axa),
                np.max(errors_axa),
                np.mean(iterations),
                np.std(iterations),
                np.mean(avg_iter),
                np.std(avg_iter),
                np.min(avg_iter),
                np.max(avg_iter),
                failures,
                float(failures) / len(results)))

        # means = np.array(means)
        # stds = np.array(stds)
        # final_errors = np.array(final_errors)
        # print('\n'.join(['{} Mean: {} Std: {}'.format(l, np.mean(means[:,x]), np.std(stds[:,x])) for x, l in enumerate('xyz')]))
        # for x in range(final_errors.shape[1]):
        #     print('Final error {}:\n  Mean: {}\n  StdD: {}\n  Min: {}\n  Max: {}'.format(x, 
        #                                                                                  np.mean(final_errors[:, x]), 
        #                                                                                  np.std(final_errors[:, x]), 
        #                                                                                  np.min(final_errors[:, x]), 
        #                                                                                  np.max(final_errors[:, x])))

    df = pd.DataFrame(columns=columns, data=rows)
    print(df)
    df.to_csv('rotation_comparison.csv', float_format='%.4f', index=False)

    # draw_recorders([rpy_integrator.recorder, ax_integrator.recorder, 
    #                 rpy_integrator.sym_recorder, ax_integrator.sym_recorder], 1, 4, 4).savefig('{}/rpy_vs_axis_angle.png'.format(res_pkg_path('package://kineverse_experiment_world/test'))
