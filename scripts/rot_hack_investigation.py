#!/usr/bin/env python
import numpy as np
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

OptResult = namedtuple('OptResult', ['error_9d', 'error_axa', 'x', 'y', 'z', 'n_steps', 'sec_per_it'])
TotalResults = namedtuple('TotalResults', ['errors_9d', 'errors_axa', 'iterations'])

# Uniform sampling of points on a sphere according to:
#  https://demonstrations.wolfram.com/RandomPointsOnASphere/
def np_sphere_sampling(n_points):
    r_theta = np.random.rand(n_points, 1) * np.pi
    r_u     = np.random.rand(n_points, 1)
    factor  = np.sqrt(1.0 - r_u**2)
    coords  = np.hstack((np.cos(r_theta) * factor, np.sin(r_theta) * factor, r_u))
    return coords # 

def sphere_sampling(n_points):
    return [vector3(*row) for row in np_sphere_sampling(n_rots)]

def random_rot_uniform(n_rots):
    # Random rotation angles about the z axis
    r_theta = np.random.rand(n_rots, 1)

    r_z_points = np_sphere_sampling(n_rots)
    x_angles   = np.arccos(r_z_points[:, 2]).reshape((n_rots, 1))
    z_angles   = np.arctan2(r_z_points[:, 1], r_z_points[:, 0]).reshape((n_rots, 1))
    print(r_theta.shape, x_angles.shape, z_angles.shape)
    return [rotation3_axis_angle([0,0,1], r_z) * 
            rotation3_axis_angle([1,0,0], r_x) * 
            rotation3_axis_angle([0,0,1], r_t) for r_t, r_x, r_z in np.hstack((r_theta, x_angles, z_angles))]

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
    
    r_points = 50
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
                                                start_state={s_x: 1, s_y: 0, s_z: 0})#,
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
                error_axa = float(norm(final_axa - goal_axa))
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
                np.mean(error_axa),
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

    # draw_recorders([rpy_integrator.recorder, ax_integrator.recorder, 
    #                 rpy_integrator.sym_recorder, ax_integrator.sym_recorder], 1, 4, 4).savefig('{}/rpy_vs_axis_angle.png'.format(res_pkg_path('package://kineverse_experiment_world/test'))
