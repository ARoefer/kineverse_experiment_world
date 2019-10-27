#!/usr/bin/env python

from kineverse.gradients.gradient_math import *
from kineverse.visualization.plotting  import *
from kineverse.motion.min_qp_builder   import TypedQPBuilder  as TQPB, \
                                              SoftConstraint  as SC,   \
                                              ControlledValue as CV
from kineverse.motion.integrator       import CommandIntegrator
from kineverse.utils                   import res_pkg_path

if __name__ == '__main__':
    

    eef  = point3(1,0,0)
    goal = rotation3_rpy(0, -0.8, 2.56) * point3(1, 0, 0)

    rpy_model = rotation3_rpy(Position('roll'), Position('pitch'), Position('yaw')) * eef

    axis = vector3(Position('x'), Position('y'), 0) #Position('z'))
    ax_model  = rotation3_axis_angle(axis / (norm(axis) + 1e-4), norm(axis)) * eef

    goal_rpy = SC(-norm(goal - rpy_model), -norm(goal - rpy_model), 1, norm(goal - rpy_model))
    goal_ax  = SC(-norm(goal -  ax_model), -norm(goal -  ax_model), 1, norm(goal -  ax_model))

    rpy_integrator = CommandIntegrator(TQPB({}, {'rpy': goal_rpy}, 
                                            {str(s): CV(-0.6, 0.6, get_diff_symbol(s), 0.001) for s in rpy_model.free_symbols}),
                                            recorded_terms={'dist rpy': norm(goal - rpy_model)})

    ax_integrator  = CommandIntegrator(TQPB({}, {'ax': goal_ax}, 
                                            {str(s): CV(-0.6, 0.6, get_diff_symbol(s), 0.001) for s in ax_model.free_symbols}),
                                            recorded_terms={'dist ax': norm(goal - ax_model)})
    rpy_integrator.restart('Convergence Test: RPY vs AxAngle')
    rpy_integrator.run()
    ax_integrator.restart('Convergence Test: RPY vs AxAngle')
    ax_integrator.run()

    draw_recorders([rpy_integrator.recorder, ax_integrator.recorder, 
                    rpy_integrator.sym_recorder, ax_integrator.sym_recorder], 1, 4, 4).savefig('{}/rpy_vs_axis_angle.png'.format(res_pkg_path('package://kineverse_experiment_world/test')))