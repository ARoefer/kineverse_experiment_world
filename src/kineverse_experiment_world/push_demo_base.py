from kineverse.gradients.gradient_math import *
from kineverse.model.geometry_model    import contact_geometry, \
                                              generate_contact_model


def sign(x):
    return -1 if x < 0 else (1 if x > 0 else 0)

def generate_push_closing(km, grounding_state, controlled_symbols, eef_pose, obj_pose, eef_path, obj_path, nav_method='cross', cp_offset=0):
    # CONTACT GEOMETRY
    robot_cp, object_cp, contact_normal = contact_geometry(eef_pose, obj_pose, eef_path, obj_path)
    object_cp     = object_cp - contact_normal * cp_offset
    geom_distance = dot(contact_normal, robot_cp - object_cp)
    coll_world    = km.get_active_geometry(geom_distance.free_symbols)

    # GEOMETRY NAVIGATION LOGIC
    # This is exploiting task knowledge which makes this function inflexible.
    contact_grad    = sum([sign(-grounding_state[s]) * vector3(*[x.diff(s) for x in object_cp[:3]]) for s in obj_pose.free_symbols], vector3(0,0,0))
    neutral_tangent = cross(contact_grad, contact_normal)
    active_tangent  = cross(neutral_tangent, contact_normal)

    if nav_method == 'linear':
        geom_distance = norm(object_cp + active_tangent * geom_distance - robot_cp)
    elif nav_method == 'cubic':
        dist_scaling  = 2 ** (-0.5*((geom_distance - 0.2) / (0.2 * 0.2))**2)
        geom_distance = norm(object_cp + active_tangent * dist_scaling - robot_cp)
    elif nav_method == 'cross':
        geom_distance = norm(object_cp + active_tangent * norm(neutral_tangent) - robot_cp)

    # PUSH CONSTRAINT GENERATION
    constraints = km.get_constraints_by_symbols(geom_distance.free_symbols.union(controlled_symbols))
    constraints.update(generate_contact_model(robot_cp, controlled_symbols, object_cp, contact_normal, obj_pose.free_symbols))

    def debug_draw(vis, state, cmd):
        vis.begin_draw_cycle('debug_vecs')
        s_object_cp       = subs(object_cp, state)
        s_neutral_tangent = subs(neutral_tangent, state)
        vis.draw_vector('debug_vecs', s_object_cp, subs(contact_grad, state), r=0, b=0)
        vis.draw_vector('debug_vecs', s_object_cp, subs(active_tangent, state), r=0, b=1)
        # vis.draw_vector('debug_vecs', s_object_cp, s_ortho_vel_vec, r=1, b=0)
        vis.render('debug_vecs')

    return constraints, geom_distance, coll_world, debug_draw
