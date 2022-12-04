import math
import kineverse as kv

from kineverse import gm

from collections import namedtuple
from kineverse.model.frames         import Frame
from kineverse.model.paths          import Path
from kineverse.operations.basic_operations import CreateValue, \
                                                  ExecFunction
from kineverse.operations.urdf_operations  import RevoluteJoint


NobiliaDebug = namedtuple('NobiliaDebug', ['poses', 'vectors', 'expressions', 'tuning_params'])

class MarkedArticulatedObject(kv.ArticulatedObject):
    def __init__(self, name):
        super(MarkedArticulatedObject, self).__init__(name)
        self.markers = {}

    @classmethod
    def json_factory(cls, name, links, joints, markers):
        out = cls(name)
        out.links   = links
        out.joints  = joints
        out.markers = markers
        return out

    def __deepcopy__(self, memo):
        out = type(self)(self.name)
        memo[id(self)] = out
        out.links  = {k: deepcopy(v) for k, v in self.links.items()}
        out.joints = {k: deepcopy(v) for k, v in self.joints.items()}
        out.markers = {k: deepcopy(v) for k, v in self.markers.items()}
        return out

    def __eq__(self, other):
        if isinstance(other, MarkedArticulatedObject):
            return self.name == other.name and self.links == other.links and self.joints == other.joints and self.markers == markers
        return False



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


def create_nobilia_shelf(km, prefix, origin_pose=gm.eye(4), parent_path=kv.Path('world')):
    km.apply_operation(f'create {prefix}', kv.ExecFunction(prefix, MarkedArticulatedObject, str(prefix)))

    shelf_height = 0.72
    shelf_width  = 0.6
    shelf_body_depth = 0.35

    wall_width = 0.016

    l_prefix = prefix + ('links',)
    geom_body_wall_l = kv.Box(l_prefix + ('body',), gm.translation3(0,  0.5 * (shelf_width - wall_width), 0),
                                                    gm.vector3(shelf_body_depth, wall_width, shelf_height))
    geom_body_wall_r = kv.Box(l_prefix + ('body',), gm.translation3(0, -0.5 * (shelf_width - wall_width), 0),
                                                    gm.vector3(shelf_body_depth, wall_width, shelf_height))
    
    geom_body_ceiling = kv.Box(l_prefix + ('body',), gm.translation3(0, 0, 0.5 * (shelf_height - wall_width)),
                                                     gm.vector3(shelf_body_depth, shelf_width - wall_width, wall_width))
    geom_body_floor   = kv.Box(l_prefix + ('body',), gm.translation3(0, 0, -0.5 * (shelf_height - wall_width)),
                                                     gm.vector3(shelf_body_depth, shelf_width - wall_width, wall_width))

    geom_body_shelf_1 = kv.Box(l_prefix + ('body',), gm.translation3(0.02, 0, -0.2 * (shelf_height - wall_width)),
                                                     gm.vector3(shelf_body_depth - 0.04, shelf_width - wall_width, wall_width))

    geom_body_shelf_2 = kv.Box(l_prefix + ('body',), gm.translation3(0.02, 0, 0.2 * (shelf_height - wall_width)),
                                                     gm.vector3(shelf_body_depth - 0.04, shelf_width - wall_width, wall_width))

    geom_body_back    = kv.Box(l_prefix + ('body',), gm.translation3(0.5 * (shelf_body_depth - 0.005), 0, 0),
                                                     gm.vector3(0.005, shelf_width - 2 * wall_width, shelf_height - 2 * wall_width))

    shelf_geom = [geom_body_wall_l,
                  geom_body_wall_r,
                  geom_body_ceiling,
                  geom_body_floor,
                  geom_body_back,
                  geom_body_shelf_1,
                  geom_body_shelf_2]

    rb_body = kv.RigidBody(parent_path, origin_pose, geometry=dict(enumerate(shelf_geom)), 
                                                     collision=dict(enumerate(shelf_geom)))

    geom_panel_top = kv.Box(l_prefix + ('panel_top',), gm.eye(4), 
                                                       gm.vector3(0.357, 0.595, wall_width))
    geom_panel_bottom = kv.Box(l_prefix + ('panel_bottom',), gm.eye(4), 
                                                             gm.vector3(0.357, 0.595, wall_width))
    
    handle_width    = 0.16
    handle_depth    = 0.05
    handle_diameter = 0.012

    geom_handle_r = kv.Box(l_prefix + ('handle',), gm.translation3(0.5 * handle_depth,  0.5 * (handle_width - handle_diameter), 0), 
                                                   gm.vector3(handle_depth, handle_diameter, handle_diameter))
    geom_handle_l = kv.Box(l_prefix + ('handle',), gm.translation3(0.5 * handle_depth, -0.5 * (handle_width - handle_diameter), 0), 
                                                   gm.vector3(handle_depth, handle_diameter, handle_diameter))
    geom_handle_bar = kv.Box(l_prefix + ('handle',), gm.translation3(handle_depth - 0.5 * handle_diameter, 0, 0), 
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
    body_marker_in_body      = gm.dot(gm.rotation3_axis_angle(gm.vector3(0, 0, 1), math.pi), 
                                      gm.translation3( 0.5 * shelf_body_depth - 0.062, -0.5 * shelf_width + 0.078, 0.5 * shelf_height))
    body_side_marker_in_body = gm.dot(gm.translation3(-0.5 * shelf_body_depth + 0.062,  0.5 * shelf_width, 0.5 * shelf_height - 0.062),
                                      gm.rotation3_rpy(math.pi * 0.5, 0, math.pi * 1.0))
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
    point_d  = gm.point3(-shelf_body_depth * 0.5 + 0.09, 0, shelf_height * 0.5 - 0.192)
    # point_d  = gm.point3(-shelf_body_depth * 0.5 + gm.Symbol('point_d_x'), 0, shelf_height * 0.5 - gm.Symbol('point_d_z'))
    # Zero alpha along the vertical axis
    vec_a_to_d = gm.dot(point_d - point_a)
    alpha      = gm.atan2(vec_a_to_d[0], -vec_a_to_d[2]) + opening_position

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
                                     gm.point3(-0.094, -0.034, -0.072),
                                     # gm.point3(-gm.Symbol('point_c_x'), -0.034, -gm.Symbol('point_c_z'))
                                     )
    point_c_in_front_hinge  = gm.dot(gm.diag(1, 0, 1, 1), 
                                     gm.dot(bottom_panel_in_front_hinge, 
                                            point_c_in_bottom_panel))
    length_z = gm.norm(point_a - point_d)
    

    vec_a_to_b = point_b - point_a
    length_t = gm.norm(vec_a_to_b)
    length_b = gm.norm(point_c_in_front_hinge[:3])
    # length_l = gm.Symbol('length_l') # 0.34
    length_l = 0.372

    vec_b_to_d = point_d - point_b
    length_v = gm.norm(vec_b_to_d)
    gamma_1  = inner_triangle_angle(length_t, length_v, length_z)
    gamma_2  = inner_triangle_angle(length_b, length_v, length_l)

    top_panel_offset_angle = gm.atan2(point_b_in_top_hinge[2], point_b_in_top_hinge[0]) 
    bottom_offset_angle    = gm.atan2(point_c_in_front_hinge[2], point_c_in_front_hinge[0]) 

    gamma = gamma_1 + gamma_2

    rb_panel_top = kv.RigidBody(l_prefix + ('body',), gm.dot(rb_body.pose, top_panel_in_body),
                                                      top_panel_in_body,
                                                      geometry={0: geom_panel_top},
                                                      collision={0: geom_panel_top})

    # old offset: 0.5 * geom_panel_top.scale[2] + 0.03
    tf_bottom_panel = gm.dot(front_hinge_in_top_panel,
                             gm.rotation3_axis_angle(gm.vector3(0,  1, 0), math.pi + bottom_offset_angle - top_panel_offset_angle),
                             gm.rotation3_axis_angle(gm.vector3(0, -1, 0), gamma),
                             bottom_panel_in_front_hinge)

    rb_panel_bottom = kv.RigidBody(l_prefix + ('panel_top',), gm.dot(rb_panel_top.pose, tf_bottom_panel),
                                                              tf_bottom_panel,
                                                              geometry={0: geom_panel_bottom},
                                                              collision={0: geom_panel_bottom})

    handle_transform = gm.dot(gm.translation3(geom_panel_bottom.scale[0] * 0.5 - 0.08, 0, 0.5 * wall_width),
                              gm.rotation3_axis_angle(gm.vector3(0, 1, 0), -math.pi * 0.5))
    rb_handle       = kv.RigidBody(l_prefix + ('panel_bottom',), gm.dot(rb_panel_bottom.pose, handle_transform),
                                                                 handle_transform,
                                                                 geometry={x: g for x, g in enumerate(handle_geom)},
                                                                 collision={x: g for x, g in enumerate(handle_geom)})
    # Only debugging
    point_c = gm.dot(rb_panel_bottom.pose, point_c_in_bottom_panel)
    vec_b_to_c = point_c - point_b

    km.apply_operation(f'create {prefix}/links/body', kv.CreateValue(rb_panel_top.parent, rb_body))
    km.apply_operation(f'create {prefix}/links/panel_top', kv.CreateValue(rb_panel_bottom.parent, rb_panel_top))
    km.apply_operation(f'create {prefix}/links/panel_bottom', kv.CreateValue(l_prefix + ('panel_bottom',) , rb_panel_bottom))
    km.apply_operation(f'create {prefix}/links/handle', kv.CreateValue(l_prefix + ('handle',) , rb_handle))
    km.apply_operation(f'create {prefix}/joints/hinge', kv.ExecFunction(prefix + kv.Path('joints/hinge'), 
                                                                     RevoluteJoint,
                                                                     kv.CPath(rb_panel_top.parent),
                                                                     kv.CPath(rb_panel_bottom.parent),
                                                                     opening_position,
                                                                     gm.vector3(0, 1, 0),
                                                                     gm.eye(4),
                                                                     0,
                                                                     1.84,
                                                                     **{f'{opening_position}': kv.Constraint(0 - opening_position,
                                                                                                          1.84 - opening_position,
                                                                                                          opening_position),
                                                                        f'{gm.DiffSymbol(opening_position)}': kv.Constraint(-0.25, 0.25, gm.DiffSymbol(opening_position))}))
    m_prefix = prefix + ('markers',)
    km.apply_operation(f'create {prefix}/markers/body',         kv.ExecFunction(m_prefix + ('body',), kv.Frame,
                                                                                                   kv.CPath(l_prefix + ('body', )),
                                                                                                   gm.dot(rb_body.pose, body_marker_in_body), 
                                                                                                   body_marker_in_body))
    km.apply_operation(f'create {prefix}/markers/body_side',    kv.ExecFunction(m_prefix + ('body_side',), kv.Frame,
                                                                                                   kv.CPath(l_prefix + ('body', )),
                                                                                                   gm.dot(rb_body.pose, body_side_marker_in_body), 
                                                                                                   body_side_marker_in_body))                                                                                                   
    km.apply_operation(f'create {prefix}/markers/top_panel',    kv.ExecFunction(m_prefix + ('top_panel',), kv.Frame, 
                                                                                                        kv.CPath(l_prefix + ('panel_top', )),
                                                                                                        gm.dot(rb_panel_top.pose, top_panel_marker_in_top_panel), 
                                                                                                        top_panel_marker_in_top_panel))
    km.apply_operation(f'create {prefix}/markers/bottom_panel', kv.ExecFunction(m_prefix + ('bottom_panel',), kv.Frame, 
                                                                                                           kv.CPath(l_prefix + ('panel_bottom', )),
                                                                                                           gm.dot(rb_panel_bottom.pose, bottom_panel_marker_in_bottom_panel),
                                                                                                           bottom_panel_marker_in_bottom_panel))

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
                         }, {gm.Symbol('point_c_x'): 0.094,
                             gm.Symbol('point_c_z'): 0.072,
                             gm.Symbol('point_d_x'): 0.09,
                             gm.Symbol('point_d_z'): 0.192,
                             gm.Symbol('length_l'): 0.372})
