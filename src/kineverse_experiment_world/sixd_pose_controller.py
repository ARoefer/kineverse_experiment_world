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
                       pos_approach_margin=0.05,
                       draw_fn=None):

        # goal_pose = gm.dot(goal_pose, gm.translation3(0, 0, -0.1))

        eef = actuated_pose
        self.eef_rot = eef[:3, :3]
        self.goal_rot = goal_pose[:3, :3]
        self.rot_error_matrix = self.eef_rot - self.goal_rot
        goal_rot_error = gm.norm(self.rot_error_matrix)
        goal_pos_delta = gm.pos_of(eef) - gm.pos_of(goal_pose)
        goal_pos_error = gm.norm(goal_pos_delta)

        # proj_tangent = gm.dot(gm.z_of(goal_pose).T, gm.z_of(goal_pose))

        # pos_tangent  = gm.dot(proj_tangent, goal_pos_delta)
        # pos_tangent_error = gm.norm(pos_tangent)
        # pos_normal_error  = gm.abs(gm.dot_product(-gm.z_of(goal_pose), goal_pos_delta))

        # rot_align_scale    = gm.exp(-3 * goal_rot_error) 
        # approach_ready     = (gm.exp(-pos_tangent_error) ** 2) * rot_align_scale
        # approach_not_ready = 1 - approach_ready

        rot_x_dot = gm.dot_product(gm.x_of(actuated_pose), gm.x_of(goal_pose))
        rot_y_dot = gm.dot_product(gm.y_of(actuated_pose), gm.y_of(goal_pose))
        rot_z_dot = gm.dot_product(gm.z_of(actuated_pose), gm.z_of(goal_pose))

        goal_constraints = {
                            'align position': SC(-goal_pos_error,
                                                 -goal_pos_error, 1, goal_pos_error),
                            'align rot x': SC(1 - rot_x_dot, 1 - rot_x_dot, 1, rot_x_dot),
                            'align rot y': SC(1 - rot_y_dot, 1 - rot_y_dot, 1, rot_y_dot),
                            'align rot z': SC(1 - rot_z_dot, 1 - rot_z_dot, 1, rot_z_dot),
                            # 'align position approach': SC(-pos_normal_error + pos_approach_margin * approach_not_ready,
                            #                               -pos_normal_error + approach_not_ready, 10, pos_normal_error),
                            # 'align position tangent': SC(-pos_tangent_error,
                            #                              -pos_tangent_error, 10, pos_tangent_error),
                            # 'align rotation': SC(-goal_rot_error, -goal_rot_error, 1, goal_rot_error)
                            }

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