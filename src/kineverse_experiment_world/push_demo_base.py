import kineverse.gradients.gradient_math as gm

from kineverse.model.geometry_model    import contact_geometry, \
                                              generate_contact_model


def sign(x):
    return -1 if x < 0 else (1 if x > 0 else 0)

def generate_push_closing(km, grounding_state, controlled_symbols, eef_pose, obj_pose, eef_path, obj_path, nav_method='cross', cp_offset=0):
    # CONTACT GEOMETRY
    robot_cp, object_cp, contact_normal = contact_geometry(eef_pose, obj_pose, eef_path, obj_path)
    object_cp     = object_cp - contact_normal * cp_offset
    geom_distance = gm.dot_product(contact_normal, robot_cp - object_cp)
    coll_world    = km.get_active_geometry(gm.free_symbols(geom_distance))

    # GEOMETRY NAVIGATION LOGIC
    # This is exploiting task knowledge which makes this function inflexible.
    contact_grad    = sum([sign(-grounding_state[s]) * gm.vector3(*[gm.diff(object_cp[x], s) for x in range(3)]) for s in gm.free_symbols(obj_pose)], gm.vector3(0,0,0))
    neutral_tangent = gm.cross(contact_grad, contact_normal)
    active_tangent  = gm.cross(neutral_tangent, contact_normal)

    if nav_method == 'linear':
        geom_distance = gm.norm(object_cp + active_tangent * geom_distance - robot_cp)
    elif nav_method == 'cubic':
        dist_scaling  = 2 ** (-0.5*((geom_distance - 0.2) / (0.2 * 0.2))**2)
        geom_distance = gm.norm(object_cp + active_tangent * dist_scaling - robot_cp)
    elif nav_method == 'cross':
        geom_distance = gm.norm(object_cp + active_tangent * gm.norm(neutral_tangent) - robot_cp)

    # PUSH CONSTRAINT GENERATION
    constraints = km.get_constraints_by_symbols(gm.free_symbols(geom_distance)
                                                  .union(controlled_symbols))
    constraints.update(generate_contact_model(robot_cp, 
                                              controlled_symbols, 
                                              object_cp, 
                                              contact_normal, 
                                              gm.free_symbols(obj_pose)))

    def debug_draw(vis, state, cmd):
        vis.begin_draw_cycle('debug_vecs')
        s_object_cp       = gm.subs(object_cp, state)
        s_neutral_tangent = gm.subs(neutral_tangent, state)
        vis.draw_vector('debug_vecs', s_object_cp, gm.subs(contact_grad, state), r=0, b=0)
        vis.draw_vector('debug_vecs', s_object_cp, gm.subs(active_tangent, state), r=0, b=1)
        # vis.draw_vector('debug_vecs', s_object_cp, s_ortho_vel_vec, r=1, b=0)
        vis.render('debug_vecs')

    return constraints, geom_distance, coll_world, debug_draw
