import math

import kineverse.gradients.gradient_math as gm

from collections import namedtuple
from kineverse.model.paths          import Path, CPath
from kineverse.model.geometry_model import ArticulatedObject, \
                                           Box, Mesh, Cylinder, Sphere, \
                                           RigidBody
from kineverse.operations.basic_operations import CreateValue, \
                                                  ExecFunction
from kineverse.operations.urdf_operations  import RevoluteJoint


NobiliaDebug = namedtuple('NobiliaDebug', ['poses', 'vectors', 'expressions'])

def inner_triangle_angle(a, b, c):
  """Given the lengths of a triangle's sides, calculates the angle opposite of side c
  
  Args:
      a (TYPE): Length of side a
      b (TYPE): Length of side a
      c (TYPE): Length of side a
  
  Returns:
      TYPE: Angle opposite of side c
  """
  return gm.acos((a**2 + b**2 - c**2) / (2 * a * b))


def create_nobilia_shelf(km, prefix, origin_pose=gm.eye(4), parent_path=Path('world')):
    km.apply_operation(f'create {prefix}', ExecFunction(prefix, ArticulatedObject, str(prefix)))

    shelf_height = 0.72
    shelf_width  = 0.6
    shelf_body_depth = 0.35

    wall_width = 0.016

    l_prefix = prefix + ('links',)
    geom_body_wall_l = Box(l_prefix + ('body',), gm.translation3(0,  0.5 * (shelf_width - wall_width), 0),
                                                 gm.vector3(shelf_body_depth, wall_width, shelf_height))
    geom_body_wall_r = Box(l_prefix + ('body',), gm.translation3(0, -0.5 * (shelf_width - wall_width), 0),
                                                 gm.vector3(shelf_body_depth, wall_width, shelf_height))
    
    geom_body_ceiling = Box(l_prefix + ('body',), gm.translation3(0, 0, 0.5 * (shelf_height - wall_width)),
                                                  gm.vector3(shelf_body_depth, shelf_width - wall_width, wall_width))
    geom_body_floor   = Box(l_prefix + ('body',), gm.translation3(0, 0, -0.5 * (shelf_height - wall_width)),
                                                  gm.vector3(shelf_body_depth, shelf_width - wall_width, wall_width))

    geom_body_shelf_1 = Box(l_prefix + ('body',), gm.translation3(0.02, 0, -0.2 * (shelf_height - wall_width)),
                                                  gm.vector3(shelf_body_depth - 0.04, shelf_width - wall_width, wall_width))

    geom_body_shelf_2 = Box(l_prefix + ('body',), gm.translation3(0.02, 0, 0.2 * (shelf_height - wall_width)),
                                                  gm.vector3(shelf_body_depth - 0.04, shelf_width - wall_width, wall_width))

    geom_body_back    = Box(l_prefix + ('body',), gm.translation3(0.5 * (shelf_body_depth - 0.005), 0, 0),
                                                  gm.vector3(0.005, shelf_width - 2 * wall_width, shelf_height - 2 * wall_width))

    shelf_geom = [geom_body_wall_l,
                  geom_body_wall_r,
                  geom_body_ceiling,
                  geom_body_floor,
                  geom_body_back,
                  geom_body_shelf_1,
                  geom_body_shelf_2]

    rb_body = RigidBody(parent_path, origin_pose, geometry=dict(enumerate(shelf_geom)), 
                                                  collision=dict(enumerate(shelf_geom)))

    geom_panel_top = Box(l_prefix + ('panel_top',), gm.eye(4), 
                                                    gm.vector3(0.357, 0.595, wall_width))
    geom_panel_bottom = Box(l_prefix + ('panel_bottom',), gm.eye(4), 
                                                          gm.vector3(0.357, 0.595, wall_width))
    
    handle_width    = 0.16
    handle_depth    = 0.05
    handle_diameter = 0.012

    geom_handle_r = Box(l_prefix + ('handle',), gm.translation3(0.5 * handle_depth,  0.5 * (handle_width - handle_diameter), 0), 
                                                gm.vector3(handle_depth, handle_diameter, handle_diameter))
    geom_handle_l = Box(l_prefix + ('handle',), gm.translation3(0.5 * handle_depth, -0.5 * (handle_width - handle_diameter), 0), 
                                                gm.vector3(handle_depth, handle_diameter, handle_diameter))
    geom_handle_bar = Box(l_prefix + ('handle',), gm.translation3(handle_depth - 0.5 * handle_diameter, 0, 0), 
                                                  gm.vector3(handle_diameter, handle_width - handle_diameter, handle_diameter))

    handle_geom = [geom_handle_l, geom_handle_r, geom_handle_bar]
 
    # Sketch of mechanism
    #                 
    #           T ---- a
    #         ----      \  Z
    #       b ..... V    \
    #       |      ...... d
    #    B  |       ------
    #       c ------
    #                L
    #
    # Diagonal V is virtual  
    #
    #
    # Angles:
    #   a -> alpha (given)
    #   b -> gamma_1 + gamma_2 = gamma
    #   c -> don't care
    #   d -> delta_1 + delta_2 = delta
    #

    opening_position = gm.Position(prefix + ('door',))

    # Calibration results
    #
    # Solution top hinge: cost = 0.03709980624159568 [ 0.08762252 -0.01433833  0.2858676   0.00871125]
    # Solution bottom hinge: cost = 0.025004236048128934 [ 0.1072496  -0.01232362  0.27271013  0.00489996]

    # Added 180 deg rotation due to -x being the forward facing side in this model
    top_hinge_in_body_marker           = gm.translation3(0.08762252 - 0.015,                   0, -0.01433833)
    top_panel_marker_in_top_hinge      = gm.translation3(0.2858676 - 0.003, -wall_width + 0.0025,  0.00871125 - 0.003)
    front_hinge_in_top_panel_maker     = gm.translation3(0.1072496 - 0.02,                     0, -0.01232362 + 0.007)
    bottom_panel_marker_in_front_hinge = gm.translation3(0.27271013,                           0,  0.00489996)

    # Top hinge - Data taken from observation
    body_marker_in_body = gm.dot(gm.rotation3_axis_angle(gm.vector3(0, 0, 1), math.pi), 
                                 gm.translation3(0.5 * shelf_body_depth - 0.062, -0.5 * shelf_width + 0.078, 0.5 * shelf_height))
    top_panel_marker_in_top_panel = gm.translation3( geom_panel_top.scale[0] * 0.5 - 0.062,
                                                    -geom_panel_top.scale[1] * 0.5 + 0.062,
                                                     geom_panel_top.scale[2] * 0.5)
    bottom_panel_marker_in_bottom_panel = gm.translation3( geom_panel_bottom.scale[0] * 0.5 - 0.062,
                                                          -geom_panel_bottom.scale[1] * 0.5 + 0.062,
                                                           geom_panel_bottom.scale[2] * 0.5)
    
    top_hinge_in_body        = gm.dot(body_marker_in_body, top_hinge_in_body_marker)
    top_panel_in_top_hinge   = gm.dot(top_panel_marker_in_top_hinge, gm.inverse_frame(top_panel_marker_in_top_panel))
    front_hinge_in_top_panel = gm.dot(top_panel_marker_in_top_panel, front_hinge_in_top_panel_maker)
    bottom_panel_in_front_hinge = gm.dot(bottom_panel_marker_in_front_hinge, gm.inverse_frame(bottom_panel_marker_in_bottom_panel))


    # Point a in body reference frame
    point_a  = gm.dot(gm.diag(1, 0, 1, 1), gm.pos_of(top_hinge_in_body))
    point_d  = gm.point3(-shelf_body_depth * 0.5 + 0.045, 0, shelf_height * 0.5 - 0.182)
    # Zero alpha along the vertical axis
    vec_a_to_d = gm.dot(point_d - point_a)
    alpha      = math.atan2(vec_a_to_d[0], -vec_a_to_d[2]) + opening_position

    top_panel_in_body   = gm.dot(top_hinge_in_body,  # Translation hinge to body frame
                                 gm.rotation3_axis_angle(gm.vector3(0, 1, 0), -opening_position + 0.5 * math.pi), # Hinge around y
                                 top_panel_in_top_hinge)
    front_hinge_in_body = gm.dot(top_panel_in_body, front_hinge_in_top_panel)

    # Point b in top panel reference frame
    point_b_in_top_hinge = gm.pos_of(gm.dot(gm.diag(1, 0, 1, 1), 
                                            front_hinge_in_top_panel, 
                                            top_panel_in_top_hinge))
    point_b  = gm.dot(gm.diag(1, 0, 1, 1), gm.pos_of(front_hinge_in_body))
    # Hinge lift arm in body reference frame
    point_c_in_bottom_panel = gm.dot(gm.diag(1, 0, 1, 1), 
                                     bottom_panel_marker_in_bottom_panel,
                                     gm.point3(-0.095, -0.034, -0.072))
    point_c_in_front_hinge  = gm.dot(gm.diag(1, 0, 1, 1), 
                                     gm.dot(bottom_panel_in_front_hinge, 
                                            point_c_in_bottom_panel))
    length_z = gm.norm(point_a - point_d)
    

    vec_a_to_b = point_b - point_a
    length_t = gm.norm(vec_a_to_b)
    length_b = gm.norm(point_c_in_front_hinge[:3])
    length_l = 0.38

    vec_b_to_d = point_d - point_b
    length_v = gm.norm(vec_b_to_d)
    gamma_1  = inner_triangle_angle(length_t, length_v, length_z)
    gamma_2  = inner_triangle_angle(length_b, length_v, length_l)

    top_panel_offset_angle = math.atan2(point_b_in_top_hinge[2], point_b_in_top_hinge[0]) 
    bottom_offset_angle    = math.atan2(point_c_in_front_hinge[2], point_c_in_front_hinge[0]) 

    gamma = gamma_1 + gamma_2

    rb_panel_top = RigidBody(l_prefix + ('body',), gm.dot(rb_body.pose, top_panel_in_body),
                                                   top_panel_in_body,
                                                   geometry={0: geom_panel_top},
                                                   collision={0: geom_panel_top})

    # old offset: 0.5 * geom_panel_top.scale[2] + 0.03
    tf_bottom_panel = gm.dot(front_hinge_in_top_panel,
                             gm.rotation3_axis_angle(gm.vector3(0,  1, 0), math.pi + bottom_offset_angle - top_panel_offset_angle),
                             gm.rotation3_axis_angle(gm.vector3(0, -1, 0), gamma),
                             bottom_panel_in_front_hinge)

    rb_panel_bottom = RigidBody(l_prefix + ('panel_top',), gm.dot(rb_panel_top.pose, tf_bottom_panel),
                                                           tf_bottom_panel,
                                                           geometry={0: geom_panel_bottom},
                                                           collision={0: geom_panel_bottom})

    handle_transform = gm.dot(gm.translation3(geom_panel_bottom.scale[0] * 0.5 - 0.05, 0, 0.5 * wall_width),
                              gm.rotation3_axis_angle(gm.vector3(0, 1, 0), -math.pi * 0.5))
    rb_handle       = RigidBody(l_prefix + ('panel_bottom',), gm.dot(rb_panel_bottom.pose, handle_transform),
                                                              handle_transform,
                                                              geometry={x: g for x, g in enumerate(handle_geom)},
                                                              collision={x: g for x, g in enumerate(handle_geom)})
    # Only debugging
    point_c = gm.dot(rb_panel_bottom.pose, point_c_in_bottom_panel)
    vec_b_to_c = point_c - point_b

    km.apply_operation(f'create {prefix}/links/body', CreateValue(rb_panel_top.parent, rb_body))
    km.apply_operation(f'create {prefix}/links/panel_top', CreateValue(rb_panel_bottom.parent, rb_panel_top))
    km.apply_operation(f'create {prefix}/links/panel_bottom', CreateValue(l_prefix + ('panel_bottom',) , rb_panel_bottom))
    km.apply_operation(f'create {prefix}/links/handle', CreateValue(l_prefix + ('handle',) , rb_handle))
    km.apply_operation(f'create {prefix}/joints/hinge', ExecFunction(prefix + Path('joints/hinge'), 
                                                                     RevoluteJoint,
                                                                     CPath(rb_panel_top.parent),
                                                                     CPath(rb_panel_bottom.parent),
                                                                     opening_position,
                                                                     gm.vector3(0, 1, 0),
                                                                     gm.eye(4),
                                                                     0,
                                                                     2.4))

    return NobiliaDebug([top_hinge_in_body, 
                         gm.dot(top_hinge_in_body, 
                               gm.rotation3_axis_angle(gm.vector3(0, 1, 0), -opening_position + 0.5 * math.pi), 
                               top_panel_in_top_hinge,
                               front_hinge_in_top_panel),
                        body_marker_in_body,
                        gm.dot(rb_panel_top.pose, top_panel_marker_in_top_panel),
                        gm.dot(rb_panel_bottom.pose, bottom_panel_marker_in_bottom_panel)],
                        [(point_a, vec_a_to_d),
                         (point_a, vec_a_to_b),
                         (point_b, vec_b_to_d),
                         (point_b, vec_b_to_c)],
                        {'gamma_1': gamma_1, 
                         'gamma_1 check_dot': gamma_1 - gm.acos(gm.dot_product(-vec_a_to_b / gm.norm(vec_a_to_b), vec_b_to_d / gm.norm(vec_b_to_d))),
                         'gamma_1 check_cos': gamma_1 - inner_triangle_angle(gm.norm(vec_a_to_b), gm.norm(vec_b_to_d), gm.norm(vec_a_to_d)),
                         'gamma_2': gamma_2,
                         'gamma_2 check_dot': gamma_2 - gm.acos(gm.dot_product(vec_b_to_c / gm.norm(vec_b_to_c), vec_b_to_d / gm.norm(vec_b_to_d))),
                         'length_v': length_v,
                         'length_b': length_b,
                         'length_l': length_l,
                         'position': opening_position,
                         'alpha': alpha,
                         'dist c d': gm.norm(point_d - point_c)
                         })
