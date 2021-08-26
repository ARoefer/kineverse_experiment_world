import rospy

import math
import numpy as np
import kineverse.gradients.gradient_math as gm

from copy import copy

from kineverse.model.paths                import Path, CPath
from kineverse.model.geometry_model       import RigidBody,     \
                                                 Box, \
                                                 Cylinder, \
                                                 Constraint, \
                                                 ArticulatedObject
from kineverse.operations.basic_operations import Operation
from kineverse.operations.basic_operations import CreateValue, ExecFunction
from kineverse.operations.urdf_operations  import RevoluteJoint, \
                                                  CreateURDFFrameConnection


class ConditionalDoorHandleConstraints(Operation):
    def __init__(self, door_position, handle_position, locking_tolerance, handle_release_angle):
        super(ConditionalDoorHandleConstraints, self).__init__({}, door_position=door_position,
                                                                   handle_position=handle_position,
                                                                   locking_tolerance=locking_tolerance,
                                                                   handle_release_angle=handle_release_angle)
    def _execute_impl(self, door_position, handle_position, locking_tolerance, handle_release_angle):
        door_vel = gm.DiffSymbol(door_position)
        is_unlocked = gm.alg_not(gm.alg_and(gm.less_than(door_position, locking_tolerance), gm.less_than(handle_position, handle_release_angle)))
        self.constraints = {f'lock {door_position}': Constraint(-1e9, is_unlocked * 1e9, door_vel)}

def create_door(km, prefix, height, width, frame_width=0.05, to_world_tf=gm.eye(4)):
    km.apply_operation(f'create {prefix}', ExecFunction(prefix, ArticulatedObject, 'door'))

    prefix = prefix + ('links',)

    base_plate_geom = Box(prefix + ('frame',), gm.translation3(0, 0, 0.015), gm.vector3(0.2, width + 0.2, 0.03))
    frame_pillar_l_geom = Box(prefix + ('frame',), gm.translation3(0, 0.5 * (width + frame_width), 0.5 * height + 0.03),
                                                   gm.vector3(frame_width, frame_width, height))
    frame_pillar_r_geom = Box(prefix + ('frame',), gm.translation3(0, -0.5 * (width + frame_width), 0.5 * height + 0.03),
                                                   gm.vector3(frame_width, frame_width, height)) 
    frame_bar_geom      = Box(prefix + ('frame',), gm.translation3(0, 0, height + 0.5 * frame_width + 0.03),
                                                   gm.vector3(frame_width, width + 2 * frame_width, frame_width)) 
    frame_rb   = RigidBody(Path('world'), to_world_tf, geometry={1: base_plate_geom, 
                                                                 2: frame_pillar_l_geom, 
                                                                 3: frame_pillar_r_geom,
                                                                 4: frame_bar_geom}, 
                                                       collision={1: base_plate_geom,
                                                                  2: frame_pillar_l_geom, 
                                                                  3: frame_pillar_r_geom,
                                                                  4: frame_bar_geom})
    door_geom1 = Box(prefix + ('door',), gm.translation3( 0.015, 0, 0), gm.vector3(0.03, width, height))
    door_geom2 = Box(prefix + ('door',), gm.translation3(-0.005, 0, 0.01), gm.vector3(0.01, width + 0.02, height + 0.01))

    handle_bar_geom = Box(prefix + ('handle',), gm.translation3(-0.08, 0.06, 0), gm.vector3(0.02, 0.12, 0.02))
    handle_cylinder_geom = Cylinder(prefix + ('handle',), 
                                    gm.dot(gm.translation3(-0.04, 0, 0), 
                                           gm.rotation3_axis_angle(gm.vector3(0, 1, 0), 0.5 * math.pi)), 
                                    0.02, 0.08)
    
    door_rb = RigidBody(Path(f'{prefix}/frame'), gm.translation3(0.0, 0.5 * -width - 0.01, 0), 
                                                 geometry={1: door_geom1, 
                                                           2: door_geom2},
                                                collision={1: door_geom1, 
                                                           2: door_geom2})
    
    handle_rb = RigidBody(Path(f'{prefix}/door'), gm.eye(4), geometry={1: handle_bar_geom,
                                                                       2: handle_cylinder_geom},
                                                            collision={1: handle_bar_geom,
                                                                       2: handle_cylinder_geom})

    km.apply_operation(f'create {prefix}/frame',  CreateValue(prefix + ('frame',),
                                                              frame_rb))
    km.apply_operation(f'create {prefix}/door',   CreateValue(prefix + ('door',),
                                                              door_rb))
    km.apply_operation(f'create {prefix}/handle', CreateValue(prefix + ('handle',),
                                                              handle_rb))

    door_position   = gm.Position('door')
    handle_position = gm.Position('handle')

    prefix = prefix[:-1] + ('joints',)
    km.apply_operation(f'create {prefix}',
                       ExecFunction(prefix + ('hinge',),
                                    RevoluteJoint,
                                    CPath(door_rb.parent),
                                    CPath(handle_rb.parent),
                                    door_position,
                                    gm.vector3(0, 0, -1),
                                    gm.translation3(0.5 * -frame_width - 0.005, 
                                                    0.5 * width + 0.01,
                                                    0.5 * height + 0.03),
                                    0,
                                    0.75 * math.pi,
                                    100,
                                    1,
                                    0))

    km.apply_operation(f'create {prefix}',
                       ExecFunction(prefix + ('handle',),
                                    RevoluteJoint,
                                    CPath(handle_rb.parent),
                                    CPath(f'{prefix}/handle'),
                                    handle_position,
                                    gm.vector3(-1, 0, 0),
                                    gm.translation3(0, 
                                                    -0.5 * width - 0.02 + 0.06,
                                                    0),
                                    0,
                                    0.25 * math.pi,
                                    100,
                                    1,
                                    0))

    prefix = prefix[:-1]
    km.apply_operation(f'connect {prefix}/links/frame {prefix}/links/door',
                CreateURDFFrameConnection(prefix + ('joints', 'hinge'), 
                                          Path(door_rb.parent),
                                          Path(handle_rb.parent)))
    km.apply_operation(f'connect {prefix}/links/door {prefix}/links/handle',
                CreateURDFFrameConnection(prefix + ('joints', 'handle'), 
                                          Path(handle_rb.parent),
                                          Path(f'{prefix}/links/handle')))
    km.apply_operation(f'add lock {door_position}', 
                ConditionalDoorHandleConstraints(door_position, handle_position, math.pi * 0.01, math.pi * 0.15))
