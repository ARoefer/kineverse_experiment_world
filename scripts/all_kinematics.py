#!/usr/bin/env python
import rospy

from kineverse.gradients.gradient_math      import *
from kineverse.model.geometry_model         import GeometryModel, RigidBody, Geometry
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer

from math import pi


if __name__ == '__main__':


    # PRISMATIC
    a = Position('a')

    parent_frame = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,0,0))
    child_frame  = parent_frame * translation3(a, 0, 0)


    # HINGE
    parent_frame = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-1,0))
    child_frame  = parent_frame * rotation3_axis_angle(vector3(0,1,0), a)

    # CYLINDRICAL 
    b = Position('b')
    parent_frame = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-2,0))
    child_frame  = parent_frame * translation3(a, 0, 0) * rotation3_axis_angle(vector3(1,0,0), b)

    # SCREW
    thread_pitch = 2  # Millimeters per revolution
    parent_frame = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-3,0))
    child_frame  = parent_frame * translation3((thread_pitch * 1e-3) * (a / pi), 0, 0) * rotation3_axis_angle(vector3(1,0,0), a)

    # BALL
    ax, ay, az      = [Position('a{}'.format(x)) for x in 'xyz']
    rotation_vector = vector3(ax, ay, az)
    neutral_axis    = vector3(1, 0, 0)
    parent_frame    = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-4,0))
    child_frame     = parent_frame * rotation3_axis_angle(rotation_vector / (norm(rotation_vector) + 1e-5), norm(rotation_vector))

    constraint      = Constraint(-limi_ang, limi_ang, asin(norm(cross(rotation_vector, neutral_axis))))

    # Planar
    c = Position('c')
    parent_frame    = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-5,0))
    child_frame     = parent_frame * translation3(a, b, 0) * rotation3_axis_angle(vector3(0, 0, 1), c)

    # Remote Control
    rc_pos          = GC(a, {b: 1})
    parent_frame    = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-8,0))
    child_frame     = parent_frame * translation3(rc_pos)

    # Mimic - Trashcan
    parent_frame    = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-9,0))
    child1_frame    = parent_frame * rotation3_axis_angle(vector3(0, 1, 0),      a)
    child2_frame    = parent_frame * rotation3_axis_angle(vector3(0, 1, 0), 5 * -a)

    # Roomba
    lx, ly, la = [Position('l{}') for x in 'xya']
    jx, jy, ja = [Velocity('j{}') for x in 'xya']
    bx = GC(lx, {jx: cos(la), jy: sin(la)})
    by = GC(ly, {jx: sin(la), jy: cos(la)})
    ba = GC(la, {ja: 1})
    parent_frame = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-10,0))
    child_frame  = parent_frame * translation3(bx, by, 0) * rotation3_axis_angle(vector3(0, 0, 1), ba)

    # COMPARTMENT
    parent_frame = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-10,0))
    door_frame   = parent_frame * translation3(0.2, 0.2, 0) * rotation3_axis_angle(vector3(0, 0, 1), a) * translation3(0, -0.2, 0)
    handle_frame = translation3(0, 0.02, 0) * rotation3_axis_angle(vector3(1, 0, 0), b)
    unlock_position    = 1.0
    locking_constraint = Constraint(-1e9, greater_than(b, unlock_position) * 1e9, DiffSymbol(a))
