from collections import namedtuple

import kineverse.gradients.gradient_math as gm

from kineverse.model.geometry_model    import contact_geometry, \
                                              generate_contact_model, \
                                              closest_distance_constraint_world, \
                                              closest_distance_constraint
from kineverse.motion.min_qp_builder   import PID_Constraint as PIDC, \
                                              SoftConstraint as SC,   \
                                              Constraint,             \
                                              generate_controlled_values, \
                                              depth_weight_controlled_values, \
                                              GeomQPBuilder as GQPB

PushingInternals = namedtuple('PushingInternals', ['contact_a', 'normal_b_to_a', 'relative_pos', 'relative_vel', 'f_debug_draw'])


def sign(x):
    return -1 if x < 0 else (1 if x > 0 else 0)

def generate_push_closing(km, grounding_state, controlled_symbols, 
                          eef_pose, obj_pose, eef_path, obj_path, 
                          nav_method='cross', cp_offset=0, static_symbols=set()):
    # CONTACT GEOMETRY
    robot_cp, object_cp, contact_normal = contact_geometry(eef_pose, obj_pose, eef_path, obj_path)
    object_cp     = object_cp - contact_normal * cp_offset
    geom_distance = gm.dot_product(contact_normal, robot_cp - object_cp)
    coll_world    = km.get_active_geometry(gm.free_symbols(geom_distance))

    # GEOMETRY NAVIGATION LOGIC
    # This is exploiting task knowledge which makes this function inflexible.
    contact_grad    = sum([sign(-grounding_state[s]) * gm.vector3(gm.diff(object_cp[0], s), 
                                                                  gm.diff(object_cp[1], s), 
                                                                  gm.diff(object_cp[2], s)) for s in gm.free_symbols(obj_pose) 
                                                                                             if s not in static_symbols], gm.vector3(0,0,0))
    neutral_tangent = gm.cross(contact_grad, contact_normal)
    active_tangent  = gm.cross(neutral_tangent, contact_normal)

    contact_constraints, in_contact = generate_contact_model(robot_cp,
                                                             controlled_symbols,
                                                             object_cp,
                                                             contact_normal,
                                                             gm.free_symbols(obj_pose),
                                                             dist_threshold=0.03)
    
    target_pos = None
    if nav_method == 'linear':
        geom_distance = gm.norm(object_cp + active_tangent * geom_distance + contact_grad * 0.05 - robot_cp)
    elif nav_method == 'cubic':
        dist_scaling  = 2 ** (-0.5*((geom_distance - 0.2) / (0.2 * 0.2))**2)
        geom_distance = gm.norm(object_cp + active_tangent * dist_scaling - robot_cp)
    elif nav_method == 'cross':
        geom_distance = gm.norm(object_cp + active_tangent * gm.norm(neutral_tangent) + contact_grad * 0.05 - robot_cp)
    elif nav_method == 'cross_deep':
        geom_distance = gm.norm(object_cp + active_tangent * gm.norm(neutral_tangent) + contact_grad * -gm.dot_product(contact_normal, contact_grad) - robot_cp)
    elif nav_method == 'none' or nav_method is None:
        pass
    elif nav_method == 'proj':
        obj_cp_dist = gm.dot_product(contact_normal, object_cp - gm.pos_of(obj_pose))
        target_pos  = gm.pos_of(obj_pose) + contact_normal * obj_cp_dist - contact_normal * 0.02 # Drive into the surface
        geom_distance = gm.norm(robot_cp - target_pos)

    contact_relative_pos = gm.dot(gm.rot_of(obj_pose), robot_cp - gm.pos_of(obj_pose))
    contact_relative_vel = gm.vector3(sum([gm.diff(contact_relative_pos[0], s) for s in controlled_symbols], 0),
                                      sum([gm.diff(contact_relative_pos[1], s) for s in controlled_symbols], 0),
                                      sum([gm.diff(contact_relative_pos[2], s) for s in controlled_symbols], 0))

    # PUSH CONSTRAINT GENERATION
    constraints = km.get_constraints_by_symbols(gm.free_symbols(geom_distance).union(controlled_symbols))
    constraints.update(contact_constraints)
    # for x, n in enumerate('xyz'):
    #     constraints[f'zero tangent vel_{n}'] = Constraint((1 -  in_contact) * -1e3, 
    #                                                       (1 -  in_contact) *  1e3, contact_relative_pos[x] * (1.0 + 0.1 * x))

    def debug_draw(vis, state, cmd):
        vis.begin_draw_cycle('debug_vecs')
        s_object_cp       = gm.subs(object_cp, state)
        s_neutral_tangent = gm.subs(neutral_tangent, state)
        vis.draw_vector('debug_vecs', s_object_cp, gm.subs(contact_grad, state), r=0, b=0)
        vis.draw_vector('debug_vecs', s_object_cp, gm.subs(active_tangent, state), r=0, b=1)
        vis.draw_vector('debug_vecs', s_object_cp, gm.subs(neutral_tangent, state), r=1, g=1, b=0)
        if target_pos is not None:
            vis.draw_sphere('debug_vecs', gm.subs(target_pos, state), 0.01, r=0, b=1)
        # print(f'{gm.norm(gm.subs(contact_normal, state))}')
        # vis.draw_vector('debug_vecs', s_object_cp, s_ortho_vel_vec, r=1, b=0)
        vis.render('debug_vecs')

    return constraints, geom_distance, coll_world, PushingInternals(robot_cp,
                                                                    contact_normal,
                                                                    contact_relative_pos,
                                                                    contact_relative_vel,
                                                                    debug_draw)


class PushingController(object):
    def __init__(self, km, 
                       actuated_object_path, target_object_path, 
                       controlled_symbols, start_state, 
                       camera_path=None,
                       navigation_method='cross',
                       visualizer=None,
                       weight_override=None,
                       collision_avoidance_paths=None):
        print(f'{actuated_object_path}\n{target_object_path}')

        actuated_object = km.get_data(actuated_object_path)
        target_object   = km.get_data(target_object_path)

        all_controlled_symbols = controlled_symbols.union({gm.DiffSymbol(j) for j in gm.free_symbols(target_object.pose)
                                                                            if 'location' not in str(j)})
        static_symbols = {s for s in gm.free_symbols(target_object.pose) if 'location' in str(s)}

        # Generate push problem
        constraints, \
        geom_distance, \
        coll_world, \
        self.p_internals = generate_push_closing(km,
                                                 start_state,
                                                 all_controlled_symbols,
                                                 actuated_object.pose,
                                                 target_object.pose,
                                                 actuated_object_path,
                                                 target_object_path,
                                                 navigation_method,
                                                 static_symbols=static_symbols,
                                                 cp_offset=-0.02)

        start_state.update({s: 0.0 for s in gm.free_symbols(coll_world)})

        weight_override = {} if weight_override is None else weight_override

        controlled_values, \
        constraints = generate_controlled_values(constraints, 
                                                 all_controlled_symbols)
        controlled_values = depth_weight_controlled_values(km, 
                                                           controlled_values, 
                                                           exp_factor=1.1)

        goal_constraints = {'reach_point': PIDC(geom_distance, geom_distance, 1, k_i=0.00)}

        for s, w in weight_override.items():
            for cv in controlled_values.values():
                if cv.symbol is s:
                    cv.weight_id = w
                    break

        # CAMERA STUFF
        if camera_path is not None:
            camera      = km.get_data(camera_path)
            cam_pos     = gm.pos_of(camera.pose)
            cam_to_obj  = gm.pos_of(target_object.pose) - cam_pos
            cam_forward = gm.dot(camera.pose, gm.vector3(1, 0, 0))
            look_goal   = 1 - (gm.dot_product(cam_to_obj, cam_forward) / gm.norm(cam_to_obj))
            goal_constraints['look_at_obj'] = SC(-look_goal, -look_goal, 1, look_goal)


        # GOAL CONSTAINT GENERATION
        # 'avoid_collisions': SC.from_constraint(closest_distance_constraint_world(eef_pose, eef_path[:-1], 0.03), 100)
        # }

        if collision_avoidance_paths is not None:
            for p in collision_avoidance_paths:
                obj = km.get_data(p)
                goal_constraints[f'avoid_collision {p}'] = SC.from_constraint(closest_distance_constraint(actuated_object.pose,
                                                                                                          obj.pose,
                                                                                                          actuated_object_path, p, 0.01), 100)


        goal_constraints.update({f'open_object_{x}': PIDC(s, s, 1) for x, s in enumerate(gm.free_symbols(target_object.pose))})

        self.look_goal         = look_goal if camera_path is not None else None
        self.in_contact        = gm.less_than(geom_distance, 0.01)
        self.controlled_values = controlled_values
        self.geom_distance     = geom_distance

        self.qpb = GQPB(coll_world, 
                        constraints, 
                        goal_constraints, 
                        controlled_values, visualizer=visualizer)
        self.qpb._cb_draw = self.p_internals.f_debug_draw


    def get_qp(self):
        return self.qpb

    def get_cmd(self, state, deltaT):
        return self.qpb.get_cmd(state, deltaT=deltaT)
