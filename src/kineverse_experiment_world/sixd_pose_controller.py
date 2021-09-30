import kineverse.gradients.gradient_math as gm

from kineverse.motion.min_qp_builder import TypedQPBuilder as TQPB, \
                                            SoftConstraint as SC, \
                                            generate_controlled_values, \
                                            depth_weight_controlled_values


class SixDPoseController(object):
    def __init__(self, km,
                       actuated_pose,
                       goal_pose,
                       controlled_symbols,
                       weight_override=None,
                       draw_fn=None):
        
        eef = actuated_pose
        goal_pos_error = gm.norm(gm.pos_of(eef) - gm.pos_of(goal_pose))
        self.eef_rot = eef[:3, :3]
        self.goal_rot = goal_pose[:3, :3]
        self.rot_error_matrix = self.eef_rot - self.goal_rot
        goal_rot_error = gm.norm(self.rot_error_matrix)

        rot_align_scale = gm.exp(-3 * goal_rot_error) 

        goal_constraints = {'align position': SC(-goal_pos_error * rot_align_scale,
                                                 -goal_pos_error * rot_align_scale, 10, goal_pos_error),
                            'align rotation': SC(-goal_rot_error, -goal_rot_error, 1, goal_rot_error)}

        self.goal_rot_error = goal_rot_error

        constraints = km.get_constraints_by_symbols(gm.free_symbols(eef).union(controlled_symbols))
        cvs, constraints = generate_controlled_values(constraints, controlled_symbols, weights=weight_override)
        cvs = depth_weight_controlled_values(km, cvs, exp_factor=1.02)

        self.draw_fn = draw_fn

        self.qp = TQPB(constraints,
                       goal_constraints,
                       cvs)

    def get_cmd(self, state, deltaT):
        if self.draw_fn is not None:
            self.draw_fn(state)
            print(f'EEF Rot:\n{gm.subs(self.eef_rot, state)}\nGoal Rot:\n{gm.subs(self.goal_rot, state)}\nRot Delta:\n{gm.subs(self.rot_error_matrix, state)}\nError: {gm.subs(self.goal_rot_error, state)}')
        return self.qp.get_cmd(state, deltaT=deltaT)

    def current_error(self):
        return self.qp.latest_error

    def equilibrium_reached(self, low_eq=1e-3, up_eq=-1e-3):
        return self.qp.equilibrium_reached(low_eq, up_eq)