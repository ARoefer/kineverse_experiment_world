import math

import kineverse.gradients.gradient_math as gm

from kineverse.model.paths          import Path, CPath
from kineverse.model.geometry_model import ArticulatedObject, \
                                           Box, Mesh, Cylinder, Sphere, \
                                           RigidBody
from kineverse.operations.basic_operations import CreateValue, \
                                                  ExecFunction
from kineverse.operations.urdf_operations  import RevoluteJoint


def create_nobilia_shelf(km, prefix, origin_pose=gm.eye(4), parent_path=Path('world')):
    km.apply_operation(f'create {prefix}', ExecFunction(prefix, ArticulatedObject, str(prefix)))

    shelf_height = 0.72
    shelf_width  = 0.6
    shelf_body_depth = 0.35

    wall_width = 0.016

    l_prefix = prefix + ('links',)
    geom_body_wall_l = Box(l_prefix + ('body',), gm.translation3(0,  0.5 * (shelf_width - 2 * wall_width), 0),
                                                 gm.vector3(shelf_body_depth, wall_width, shelf_height))
    geom_body_wall_r = Box(l_prefix + ('body',), gm.translation3(0, -0.5 * (shelf_width - 2 * wall_width), 0),
                                                 gm.vector3(shelf_body_depth, wall_width, shelf_height))
    
    geom_body_ceiling = Box(l_prefix + ('body',), gm.translation3(0, 0, 0.5 * (shelf_height - wall_width)),
                                                  gm.vector3(shelf_body_depth, shelf_width - 2 * wall_width, wall_width))
    geom_body_floor   = Box(l_prefix + ('body',), gm.translation3(0, 0, -0.5 * (shelf_height - wall_width)),
                                                  gm.vector3(shelf_body_depth, shelf_width - 2 * wall_width, wall_width))

    geom_body_shelf_1 = Box(l_prefix + ('body',), gm.translation3(0.02, 0, -0.2 * (shelf_height - wall_width)),
                                                  gm.vector3(shelf_body_depth - 0.04, shelf_width - 2 * wall_width, wall_width))

    geom_body_shelf_2 = Box(l_prefix + ('body',), gm.translation3(0.02, 0, 0.2 * (shelf_height - wall_width)),
                                                  gm.vector3(shelf_body_depth - 0.04, shelf_width - 2 * wall_width, wall_width))

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
                                                    gm.vector3(wall_width, 
                                                               shelf_width - 0.005,
                                                               shelf_height * 0.5 - 0.005))
    geom_panel_bottom = Box(l_prefix + ('panel_bottom',), gm.eye(4), 
                                                          gm.vector3(wall_width, 
                                                                     shelf_width - 0.005,
                                                                     shelf_height * 0.5 - 0.005))
    

    opening_position = gm.Position(prefix + ('door',))
    tf_top_panel = gm.dot(gm.translation3(-shelf_body_depth * 0.5 - 0.5 * wall_width, 0, 0.5 * shelf_height - wall_width),
                          gm.rotation3_axis_angle(gm.vector3(0, 1, 0), opening_position + math.pi * 0.5),
                          gm.translation3(geom_panel_top.scale[2] * 0.5, 0, 0),
                          gm.rotation3_axis_angle(gm.vector3(0, 1, 0), math.pi * 0.5))

    rb_panel_top = RigidBody(l_prefix + ('body',), gm.dot(rb_body.pose, tf_top_panel),
                                                   tf_top_panel,
                                                   geometry={0: geom_panel_top},
                                                   collision={0: geom_panel_top})

    tf_bottom_panel = gm.dot(gm.translation3(0, 0, 0.5 * shelf_height + 0.03),
                             gm.rotation3_axis_angle(gm.vector3(0, 1, 0), -opening_position),
                             gm.translation3(0, 0, 0.5 * shelf_height - 0.03))

    rb_panel_bottom = RigidBody(l_prefix + ('panel_top',), gm.dot(rb_panel_top.pose, tf_bottom_panel),
                                                           tf_bottom_panel,
                                                           geometry={0: geom_panel_bottom},
                                                           collision={0: geom_panel_bottom})

    km.apply_operation(f'create {prefix}/links/body', CreateValue(rb_panel_top.parent, rb_body))
    km.apply_operation(f'create {prefix}/links/panel_top', CreateValue(rb_panel_bottom.parent, rb_panel_top))
    km.apply_operation(f'create {prefix}/links/panel_bottom', CreateValue(l_prefix + ('panel_top',) , rb_panel_top))
    km.apply_operation(f'create {prefix}/joints/hinge', ExecFunction(prefix + Path('joints/hinge'), 
                                                                     RevoluteJoint,
                                                                     CPath(rb_panel_top.parent),
                                                                     CPath(rb_panel_bottom.parent),
                                                                     opening_position,
                                                                     gm.vector3(0, 1, 0),
                                                                     gm.eye(4),
                                                                     0,
                                                                     math.pi * 0.8))


