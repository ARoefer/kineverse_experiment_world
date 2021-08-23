import kineverse.gradients.gradient_math as gm

from kineverse.motion.min_qp_builder import TypedQPBuilder as TQPB, \
                                            SoftConstraint as SC, \
                                            generate_controlled_values, \
                                            depth_weight_controlled_values, \


class SixDPoseController(object):
    def __init__(self, km,
                       actuated_pose,
                       goal_pose,
                       controlled_symbols,
                       weight_override=None):
        
        eef = actuated_pose
        goal_pos_error = gm.norm(gm.pos_of(eef.pose) - gm.pos_of(goal_pose))
        goal_rot_error = gm.norm(eef.pose[:3, :3] - goal_pose[:3, :3])

        goal_constraints = {'align position': SC(-goal_pos_error,
                                                 -goal_pos_error, 10, goal_pos_error),
                            'align rotation': SC(-goal_rot_error, -goal_rot_error, 1, goal_rot_error)}

        constraints = km.get_constraints_by_symbols(gm.free_symbols(eef).union(controlled_symbols))
        cvs, constraints = generate_controlled_values(constraints, controlled_symbols, weights=weight_override)
        cvs = depth_weight_controlled_values(km, cvs, exp_factor=1.02)

        self.qp = TQPB(constraints,
                       goal_constraints,
                       cvs)

    def get_cmd(self, state, deltaT):
        return self.qp.get_cmd(state, deltaT=deltaT)

    def current_error(self):
        return self.qp.latest_error

    def equilibrium_reached(self, low_eq=1e-3, up_eq=-1e-3):
        return self.qp.equilibrium_reached(low_eq, up_eq)