#!/usr/bin/env python
import rospy

from kineverse.gradients.gradient_math      import *
from kineverse.model.geometry_model         import GeometryModel, RigidBody, Geometry
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer

from math import pi
from time import time

NULL_SYMBOL = Position('null_var')

def simple_kinematics(km, vis):
    # PRISMATIC
    a = Position('a')

    parent_frame = frame3_axis_angle(vector3(NULL_SYMBOL, 0, 1), 0.56, point3(0,0,0))
    child_frame  = parent_frame * translation3(a, 0, 0)

    box_parent   = Geometry('prismatic_parent', spw.eye(4), 'box', vector3(1.00, 0.08, 0.08))
    box_child    = Geometry('prismatic_child', spw.eye(4), 'box', vector3(0.12, 0.12, 0.12))
    parent_obj   = RigidBody('map', parent_frame, geometry={1: box_parent}, collision={1: box_parent})
    child_obj    = RigidBody('prismatic_parent', child_frame, geometry={1: box_child}, collision={1: box_child})

    km.set_data('prismatic_parent', parent_obj)
    km.set_data('prismatic_child',  child_obj)

    # HINGE
    parent_frame = frame3_axis_angle(vector3(NULL_SYMBOL,0,1), 0.56 + pi, point3(0,-1,0))
    child_frame  = parent_frame * translation3(0,0,0.08) * rotation3_axis_angle(vector3(0,0,1), a)

    hinge_parent = Geometry('hinge_parent', spw.eye(4), 'mesh', mesh='package://kineverse_experiment_world/meshes/hinge_part_b.obj')
    hinge_child  = Geometry('hinge_child', spw.eye(4), 'mesh', mesh='package://kineverse_experiment_world/meshes/hinge_part_a.obj')
    parent_obj   = RigidBody('map', parent_frame, geometry={1: hinge_parent}, collision={1: hinge_parent})
    child_obj    = RigidBody('hinge_parent', child_frame, geometry={1: hinge_child}, collision={1: hinge_child})

    km.set_data('hinge_parent', parent_obj)
    km.set_data('hinge_child',  child_obj)

    # # CYLINDRICAL 
    b = Position('b')
    parent_frame = frame3_axis_angle(vector3(NULL_SYMBOL,0,1), 0.56, point3(0,-2,0))
    child_frame  = parent_frame * translation3(a, 0, 0) * rotation3_axis_angle(vector3(1,0,0), b)

    cylin_parent = Geometry('cylin_parent', rotation3_rpy(0, 0.5*pi, 0), 'cylinder', scale=vector3(0.1, 0.1, 1.0))
    cylin_child  = Geometry('cylin_child', rotation3_rpy(0, 0.5*pi, 0), 'mesh', mesh='package://kineverse_experiment_world/meshes/weird_nut.obj')
    parent_obj   = RigidBody('map', parent_frame, geometry={1: cylin_parent}, collision={1: cylin_parent})
    child_obj    = RigidBody('cylin_parent', child_frame, geometry={1: cylin_child}, collision={1: cylin_child})    

    km.set_data('cylin_parent', parent_obj)
    km.set_data('cylin_child',  child_obj)

    # # SCREW
    thread_pitch = 2  # Millimeters per revolution
    parent_frame = frame3_axis_angle(vector3(NULL_SYMBOL,0,1), 0.56, point3(0,-3,0))
    child_frame  = parent_frame * translation3((thread_pitch * 1e-3) * (a *100 / pi), 0, 0) * rotation3_axis_angle(vector3(1,0,0), a *100)

    screw_parent = Geometry('screw_parent', translation3(-0.5,0,0) * rotation3_rpy(0, 0.5*pi, 0), 'mesh', mesh='package://kineverse_experiment_world/meshes/screw_thread.obj')
    screw_child  = Geometry('screw_child', rotation3_rpy(0, 0.5*pi, 0), 'mesh', mesh='package://kineverse_experiment_world/meshes/regular_nut.obj')
    parent_obj   = RigidBody('map', parent_frame, geometry={1: screw_parent}, collision={1: screw_parent})
    child_obj    = RigidBody('screw_parent', child_frame, geometry={1: screw_child}, collision={1: screw_child})    

    km.set_data('screw_parent', parent_obj)
    km.set_data('screw_child',  child_obj)

    # BALL
    ax, ay, az      = [Position('a{}'.format(x)) for x in 'xyz']
    rotation_vector = vector3(ax, ay, az)
    neutral_axis    = vector3(1, 0, 0)
    parent_frame    = frame3_axis_angle(vector3(NULL_SYMBOL,0,1), 0.56, point3(0,-4,0))
    child_frame     = parent_frame * rotation3_axis_angle(rotation_vector / (norm(rotation_vector) + 1e-5), norm(rotation_vector))

    ball_parent = Geometry('ball_parent', rotation3_rpy(0, 0.5*pi, 0), 'mesh', mesh='package://kineverse_experiment_world/meshes/ball_joint_socket.obj')
    ball_child  = Geometry('ball_child', rotation3_rpy(0, 0.5*pi, 0), 'mesh', mesh='package://kineverse_experiment_world/meshes/ball_joint.obj')
    parent_obj   = RigidBody('map', parent_frame, geometry={1: ball_parent}, collision={1: ball_parent})
    child_obj    = RigidBody('ball_parent', child_frame, geometry={1: ball_child}, collision={1: ball_child})    

    km.set_data('ball_parent', parent_obj)
    km.set_data('ball_child',  child_obj)


    # constraint      = Constraint(-limi_ang, limi_ang, asin(norm(cross(rotation_vector, neutral_axis))))

    # # Planar
    # c = Position('c')
    # parent_frame    = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-5,0))
    # child_frame     = parent_frame * translation3(a, b, 0) * rotation3_axis_angle(vector3(0, 0, 1), c)

    parent_frame = frame3_axis_angle(vector3(NULL_SYMBOL,0,1), 0.56, point3(0,-2,0))
    wheel_frame  = parent_frame * rotation3_axis_angle(vector3(0,0,1), a)
    pulley_frame = parent_frame * rotation3_axis_angle(vector3(0,0,1), a)

    wheel_geom   = Geometry('cylin_parent', rotation3_rpy(0, 0.5*pi, 0), 'cylinder', scale=vector3(0.1, 0.1, 1.0))
    cylin_child  = Geometry('cylin_child', rotation3_rpy(0, 0.5*pi, 0), 'mesh', mesh='package://kineverse_experiment_world/meshes/weird_nut.obj')
    parent_obj   = RigidBody('map', parent_frame, geometry={1: cylin_parent}, collision={1: cylin_parent})
    child_obj    = RigidBody('cylin_parent', child_frame, geometry={1: cylin_child}, collision={1: cylin_child})    

    km.set_data('cylin_parent', parent_obj)
    km.set_data('cylin_child',  child_obj)


    km.clean_structure()
    km.dispatch_events()


if __name__ == '__main__':
    rospy.init_node('kinematics_test')

    vis = ROSBPBVisualizer('kinematics_test', 'map')

    km = GeometryModel()
    simple_kinematics(km, vis)

    symbols = set(km._symbol_co_map.keys())

    coll_world = km.get_active_geometry(symbols)

    last_update = time()
    while not rospy.is_shutdown():
        now = time()
        if now - last_update >= (1.0 / 50.0):
            state = {s: sin(time()) for s in symbols}
            state[NULL_SYMBOL] = 0

            coll_world.update_world(state)

            vis.begin_draw_cycle('prismatic')
            vis.draw_world('prismatic', coll_world)
            vis.render('prismatic')
            last_update = time()

    # # PRISMATIC
    # a = Position('a')

    # parent_frame = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,0,0))
    # child_frame  = parent_frame * translation3(a, 0, 0)


    # # HINGE
    # parent_frame = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-1,0))
    # child_frame  = parent_frame * rotation3_axis_angle(vector3(0,1,0), a)

    # # CYLINDRICAL 
    # b = Position('b')
    # parent_frame = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-2,0))
    # child_frame  = parent_frame * translation3(a, 0, 0) * rotation3_axis_angle(vector3(1,0,0), b)

    # # SCREW
    # thread_pitch = 2  # Millimeters per revolution
    # parent_frame = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-3,0))
    # child_frame  = parent_frame * translation3((thread_pitch * 1e-3) * (a / pi), 0, 0) * rotation3_axis_angle(vector3(1,0,0), a)

    # # BALL
    # ax, ay, az      = [Position('a{}'.format(x)) for x in 'xyz']
    # rotation_vector = vector3(ax, ay, az)
    # neutral_axis    = vector3(1, 0, 0)
    # parent_frame    = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-4,0))
    # child_frame     = parent_frame * rotation3_axis_angle(rotation_vector / (norm(rotation_vector) + 1e-5), norm(rotation_vector))

    # constraint      = Constraint(-limi_ang, limi_ang, asin(norm(cross(rotation_vector, neutral_axis))))

    # # Planar
    # c = Position('c')
    # parent_frame    = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-5,0))
    # child_frame     = parent_frame * translation3(a, b, 0) * rotation3_axis_angle(vector3(0, 0, 1), c)

    # # Remote Control
    # rc_pos          = GC(a, {b: 1})
    # parent_frame    = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-8,0))
    # child_frame     = parent_frame * translation3(rc_pos)

    # # Mimic - Trashcan
    # parent_frame    = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-9,0))
    # child1_frame    = parent_frame * rotation3_axis_angle(vector3(0, 1, 0),      a)
    # child2_frame    = parent_frame * rotation3_axis_angle(vector3(0, 1, 0), 5 * -a)

    # # Roomba
    # lx, ly, la = [Position('l{}') for x in 'xya']
    # jx, jy, ja = [Velocity('j{}') for x in 'xya']
    # bx = GC(lx, {jx: cos(la), jy: sin(la)})
    # by = GC(ly, {jx: sin(la), jy: cos(la)})
    # ba = GC(la, {ja: 1})
    # parent_frame = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-10,0))
    # child_frame  = parent_frame * translation3(bx, by, 0) * rotation3_axis_angle(vector3(0, 0, 1), ba)

    # # COMPARTMENT
    # parent_frame       = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-10,0))
    # door_frame         = parent_frame * translation3(0.2, 0.2, 0) 
    #                                   * rotation3_axis_angle(vector3(0, 0, 1), a) 
    #                                   * translation3(0, -0.2, 0)
    # handle_frame       = door_frame * translation3(0, 0.02, 0) 
    #                                 * rotation3_axis_angle(vector3(1, 0, 0), b)
    # unlock_position    = 1.0
    # locking_constraint = Constraint(-1e9, greater_than(b, unlock_position) * 1e9, DiffSymbol(a))

    # # Microwave
    # parent_frame       = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-11,0))
    # door_frame         = parent_frame * translation3(0.2, 0.2, 0) 
    #                                   * rotation3_axis_angle(vector3(0, 0, 1), a) 
    #                                   * translation3(0, -0.2, 0)
    # button_frame       = parent_frame * translation3(0, 0.02, 0) * rotation3_axis_angle(vector3(1, 0, 0), b)
    # unlock_position    = 1.0
    # locking_constraint = Constraint(-1e9, greater_than(b, unlock_position) * 1e9, DiffSymbol(a))
