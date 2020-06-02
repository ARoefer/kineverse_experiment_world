#!/usr/bin/env python
import rospy
import kineverse.gradients.common_math as cm

from kineverse.gradients.gradient_math      import *
from kineverse.model.geometry_model         import GeometryModel, RigidBody, Geometry, Frame
from kineverse.ros.tf_publisher             import ModelTFBroadcaster_URDF
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer

from math import pi
from time import time

NULL_SYMBOL = Position('null_var')
SYM_ROTATION = Position('rotation')


def simple_kinematics(km, vis):
    km.set_data('world', Frame('none'))

    # PRISMATIC
    a = Position('a')

    parent_frame    = frame3_axis_angle(vector3(NULL_SYMBOL, 0, 1), 0.56, point3(0,0,0))
    child_to_parent = translation3(a, 0, 0)
    child_frame     = parent_frame * child_to_parent

    box_parent   = Geometry('prismatic_parent', cm.eye(4), 'box', vector3(1.00, 0.08, 0.08))
    box_child    = Geometry('prismatic_child', cm.eye(4), 'box', vector3(0.12, 0.12, 0.12))
    parent_obj   = RigidBody('world', parent_frame, geometry={1: box_parent}, collision={1: box_parent})
    child_obj    = RigidBody('prismatic_parent', child_frame, to_parent=child_to_parent, geometry={1: box_child}, collision={1: box_child})

    km.set_data('prismatic_parent', parent_obj)
    km.set_data('prismatic_child',  child_obj)

    # HINGE
    parent_frame    = frame3_axis_angle(vector3(NULL_SYMBOL,0,1), 0.56 + pi, point3(0,-1,0))
    child_to_parent = translation3(0,0,0.08) * rotation3_axis_angle(vector3(0,0,1), a)
    child_frame     = parent_frame * child_to_parent

    hinge_parent = Geometry('hinge_parent', cm.eye(4), 'mesh', mesh='package://kineverse_experiment_world/meshes/hinge_part_b.obj')
    hinge_child  = Geometry('hinge_child', cm.eye(4), 'mesh', mesh='package://kineverse_experiment_world/meshes/hinge_part_a.obj')
    parent_obj   = RigidBody('world', parent_frame, geometry={1: hinge_parent}, collision={1: hinge_parent})
    child_obj    = RigidBody('hinge_parent', child_frame, to_parent=child_to_parent, geometry={1: hinge_child}, collision={1: hinge_child})

    km.set_data('hinge_parent', parent_obj)
    km.set_data('hinge_child',  child_obj)

    # # CYLINDRICAL 
    b = Position('b')
    parent_frame    = frame3_axis_angle(vector3(NULL_SYMBOL,0,1), 0.56, point3(0,-2,0))
    child_to_parent = translation3(a, 0, 0) * rotation3_axis_angle(vector3(1,0,0), b)
    child_frame     = parent_frame * child_to_parent

    cylin_parent = Geometry('cylin_parent', rotation3_rpy(0, 0.5*pi, 0), 'cylinder', scale=vector3(0.1, 0.1, 1.0))
    cylin_child  = Geometry('cylin_child', rotation3_rpy(0, 0.5*pi, 0), 'mesh', mesh='package://kineverse_experiment_world/meshes/weird_nut.obj')
    parent_obj   = RigidBody('world', parent_frame, geometry={1: cylin_parent}, collision={1: cylin_parent})
    child_obj    = RigidBody('cylin_parent', child_frame, to_parent=child_to_parent, geometry={1: cylin_child}, collision={1: cylin_child})    

    km.set_data('cylin_parent', parent_obj)
    km.set_data('cylin_child',  child_obj)

    # # SCREW
    thread_pitch    = 2  # Millimeters per revolution
    parent_frame    = frame3_axis_angle(vector3(NULL_SYMBOL,0,1), 0.56, point3(0,-3,0))
    child_to_parent = translation3((thread_pitch * 1e-3) * (a *100 / pi), 0, 0) * rotation3_axis_angle(vector3(1,0,0), a *100)
    child_frame     = parent_frame * child_to_parent

    screw_parent = Geometry('screw_parent', translation3(-0.5,0,0) * rotation3_rpy(0, 0.5*pi, 0), 'mesh', mesh='package://kineverse_experiment_world/meshes/screw_thread.obj')
    screw_child  = Geometry('screw_child', rotation3_rpy(0, 0.5*pi, 0), 'mesh', mesh='package://kineverse_experiment_world/meshes/regular_nut.obj')
    parent_obj   = RigidBody('world', parent_frame, geometry={1: screw_parent}, collision={1: screw_parent})
    child_obj    = RigidBody('screw_parent', child_frame, to_parent=child_to_parent, geometry={1: screw_child}, collision={1: screw_child})    

    km.set_data('screw_parent', parent_obj)
    km.set_data('screw_child',  child_obj)

    # BALL
    ax, ay, az      = [Position('a{}'.format(x)) for x in 'xyz']
    rotation_vector = vector3(ax, ay, az)
    neutral_axis    = vector3(1, 0, 0)
    parent_frame    = frame3_axis_angle(vector3(NULL_SYMBOL,0,1), 0.56, point3(0,-4,0))
    child_to_parent = rotation3_axis_angle(rotation_vector / (norm(rotation_vector) + 1e-5), norm(rotation_vector))
    child_frame     = parent_frame * child_to_parent

    ball_parent = Geometry('ball_parent', rotation3_rpy(0, 0.5*pi, 0), 'mesh', mesh='package://kineverse_experiment_world/meshes/ball_joint_socket.obj')
    ball_child  = Geometry('ball_child', rotation3_rpy(0, 0.5*pi, 0), 'mesh', mesh='package://kineverse_experiment_world/meshes/ball_joint.obj')
    parent_obj   = RigidBody('world', parent_frame, geometry={1: ball_parent}, collision={1: ball_parent})
    child_obj    = RigidBody('ball_parent', child_frame, to_parent=child_to_parent, geometry={1: ball_child}, collision={1: ball_child})    

    km.set_data('ball_parent', parent_obj)
    km.set_data('ball_child',  child_obj)


    # constraint      = Constraint(-limi_ang, limi_ang, asin(norm(cross(rotation_vector, neutral_axis))))

    # # Planar
    # c = Position('c')
    # parent_frame    = frame3_axis_angle(vector3(0,0,1), 0.56, point3(0,-5,0))
    # child_frame     = parent_frame * translation3(a, b, 0) * rotation3_axis_angle(vector3(0, 0, 1), c)

    
    parent_frame    = frame3_axis_angle(vector3(NULL_SYMBOL,0,1), 0.56, point3(0,-5,0))
    rotator_to_parent = rotation3_axis_angle(vector3(1,0,0), SYM_ROTATION)
    rotator_frame   = parent_frame  * rotator_to_parent
    rod_to_rotator  = frame3_axis_angle(vector3(1,0,0), -asin(sin(SYM_ROTATION) * 0.05 / 0.15) - SYM_ROTATION, point3(0, 0, 0.05))
    rod_frame       = rotator_frame * rod_to_rotator
    # frame3_axis_angle(vector3(1,0,0), asin((-sin(SYM_ROTATION) * 0.05 / 0.15)), rotator_frame * point3(0, 0, 0.05))
    piston_to_rod   = frame3_axis_angle(vector3(1,0,0), asin(sin(SYM_ROTATION) * 0.05 / 0.15), point3(0,0,0.15))
    piston_frame    = rod_frame * piston_to_rod # translation3(*(rod_frame * point3(0,0,0.15)))

    rotator_geom = Geometry('rotator', cm.eye(4), 'mesh', mesh='package://kineverse_experiment_world/meshes/rotator.obj')
    rod_geom     = Geometry('rod', cm.eye(4), 'mesh', mesh='package://kineverse_experiment_world/meshes/connecting_rod.obj')
    piston_geom  = Geometry('piston', cm.eye(4), 'mesh', mesh='package://kineverse_experiment_world/meshes/piston.obj')
    rotator_obj  = RigidBody('world', rotator_frame, to_parent=rotator_frame, geometry={1: rotator_geom}, collision={1: rotator_geom})
    rod_obj      = RigidBody('rotator', rod_frame, to_parent=rod_to_rotator, geometry={1: rod_geom}, collision={1: rod_geom})
    piston_obj   = RigidBody('rod', piston_frame, to_parent=piston_to_rod, geometry={1: piston_geom}, collision={1: piston_geom})

    km.set_data('rotator', rotator_obj)
    km.set_data('rod', rod_obj)
    km.set_data('piston', piston_obj)


    # Faucet
    base_frame   = frame3_axis_angle(vector3(NULL_SYMBOL,0,1), 0.56, point3(0,-6,0))
    head_to_base = translation3(0, -0.006, 0.118) * rotation3_rpy(0.3 * (ay**2) - 0.3, 0, ay)
    head_frame   = base_frame * head_to_base

    geom_head = Geometry('faucet_head', cm.eye(4), 'mesh', mesh='package://kineverse_experiment_world/meshes/faucet_handle.obj')
    geom_base = Geometry('faucet_base', cm.eye(4), 'mesh', mesh='package://kineverse_experiment_world/meshes/faucet_base.obj')

    rb_base   = RigidBody('world', base_frame, geometry={0: geom_base}, collision={0: geom_base})
    rb_head   = RigidBody('faucet_base', head_frame, to_parent=head_to_base, geometry={0: geom_head}, collision={0: geom_head})

    water_flow = dot(cm.y_of(head_frame), cm.y_of(base_frame))
    water_temperature = dot(cm.y_of(head_frame), cm.y_of(base_frame))

    km.set_data('faucet_base', rb_base)
    km.set_data('faucet_head', rb_head)

    # Compartment
    parent_frame    = frame3_axis_angle(vector3(NULL_SYMBOL,0,1), 0.56, point3(0,-7,0))
    door_to_parent  = frame3_axis_angle(vector3(0,0,1), a * 0.5 + 0.5, point3(-0.25,-0.2,0))
    door_frame      = parent_frame  * door_to_parent
    handle_to_door  = frame3_axis_angle(vector3(1,0,0), a * 0.5 + 0.5 - pi * 0.5, point3(0, 0.37, 0))
    handle_frame    = door_frame * handle_to_door

    compartment_geom = Geometry('compartment', cm.eye(4), 'mesh', mesh='package://kineverse_experiment_world/meshes/compartment.obj')
    door_geom    = Geometry('door', cm.eye(4), 'mesh', mesh='package://kineverse_experiment_world/meshes/compartment_door.obj')
    handle_geom  = Geometry('handle', cm.eye(4), 'mesh', mesh='package://kineverse_experiment_world/meshes/compartment_handle.obj')
    compartment_obj = RigidBody('world', parent_frame, geometry={1: compartment_geom}, collision={1: compartment_geom})
    door_obj        = RigidBody('compartment', door_frame, to_parent=door_to_parent, geometry={1: door_geom}, collision={1: door_geom})
    handle_obj      = RigidBody('door', handle_frame, to_parent=handle_to_door, geometry={1: handle_geom}, collision={1: handle_geom})

    km.set_data('compartment', compartment_obj)
    km.set_data('door', door_obj)
    km.set_data('handle', handle_obj)

    # Multidependent kinematics 

    parent_frame = frame3_axis_angle(vector3(NULL_SYMBOL,0,1), 0.56, point3(0,-8,0))
    mount_offset = 0.1
    null_angle = pi / 5
    mount_l = rotation3_axis_angle(unitY, null_angle) * translation3(mount_offset,  0.05, 0)
    mount_r = rotation3_axis_angle(unitY, null_angle) * translation3(mount_offset, -0.05, 0)
    anchor  = translation3(0, 0, 0)
    rotation_base = rotation3_axis_angle(unitY, null_angle)
    bridge_length = 0.14

    pos_l   = 0.125 + a * 0.025
    angle_l = pi + acos((pos_l**2 + mount_offset ** 2 - bridge_length ** 2) / (2 * pos_l * mount_offset))
    rot_piston_l = rotation3_axis_angle(unitY, angle_l)
    pb_l_to_parent = mount_l * rot_piston_l
    ph_l_to_pb = translation3(pos_l, 0, 0)
    pep_l      = cm.pos_of(pb_l_to_parent * ph_l_to_pb)

    pos_r   = 0.125 + b * 0.025
    angle_r = pi + acos((pos_r**2 + mount_offset ** 2 - bridge_length ** 2) / (2 * pos_r * mount_offset))
    rot_piston_r = rotation3_axis_angle(unitY, angle_r)
    pb_r_to_parent = mount_r * rot_piston_r
    ph_r_to_pb = translation3(pos_r, 0, 0)
    pep_r      = cm.pos_of(pb_r_to_parent * ph_r_to_pb)

    bar_center = (cm.diag(1,0,1,1) * (pep_l + pep_r)) * 0.5
    bar_offset = bar_center - cm.pos_of(anchor)
    bar_pitch  = rotation3_axis_angle(-unitY, cm.atan2(bar_offset[2], bar_offset[0]))
    bar_diag   = pep_r - pep_l
    bar_roll   = rotation3_axis_angle(unitX, 
                                      cm.atan2(dot(cm.z_of(bar_pitch), bar_diag), dot(cm.y_of(bar_pitch), bar_diag)))


    sphere     = Geometry('world', cm.eye(4), 'sphere', scale=vector3(0.01, 0.01, 0.01))
    thick_box  = Geometry('world', translation3(0.025,0,0), 'box', scale=vector3(0.05, 0.02, 0.02))
    thin_box   = Geometry('world', translation3(-0.05,0,0),  'box', scale=vector3(0.1, 0.01, 0.01))
    bar_box    = Geometry('world', translation3(bridge_length * 0.5, 0, 0),  
                                   'box', scale=vector3(bridge_length, 0.02, 0.01))
    parbar_box = Geometry('world', translation3(bridge_length, 0, 0),  
                                  'box', scale=vector3(0.02, 0.1, 0.02))

    mdj_base = Frame('world', parent_frame, to_parent=parent_frame)

    pb_l_obj = RigidBody('mdj_base', parent_frame * pb_l_to_parent, 
                                  to_parent=pb_l_to_parent,
                                  geometry={ 1: thick_box},
                                  collision={1: thick_box})
    ph_l_obj = RigidBody('pb_l', parent_frame * pb_l_to_parent * ph_l_to_pb, 
                                  to_parent=ph_l_to_pb,
                                  geometry={ 1: thin_box},
                                  collision={1: thin_box})
    pb_r_obj = RigidBody('mdj_base', parent_frame * pb_r_to_parent, 
                                  to_parent=pb_r_to_parent,
                                  geometry={ 1: thick_box},
                                  collision={1: thick_box})
    ph_r_obj = RigidBody('pb_r', parent_frame * pb_r_to_parent * ph_r_to_pb, 
                                  to_parent=ph_r_to_pb,
                                  geometry={ 1: thin_box},
                                  collision={1: thin_box})
    bar_obj  = RigidBody('mdj_base', parent_frame * bar_pitch * bar_roll,
                                  to_parent=bar_pitch * bar_roll, #
                                  geometry={ 1: bar_box, 2: parbar_box},
                                  collision={1: bar_box, 2: parbar_box})


    km.set_data('mdj_base', mdj_base)
    km.set_data('pb_l', pb_l_obj)
    km.set_data('pb_r', pb_r_obj)
    km.set_data('ph_l', ph_l_obj)
    km.set_data('ph_r', ph_r_obj)
    km.set_data('bar_obj', bar_obj)

    km.clean_structure()
    km.dispatch_events()


if __name__ == '__main__':
    rospy.init_node('kinematics_test')

    vis = ROSBPBVisualizer('kinematics_test', 'world')

    km = GeometryModel()
    simple_kinematics(km, vis)

    symbols = set(km._symbol_co_map.keys())

    coll_world = km.get_active_geometry(symbols)

    broadcaster = ModelTFBroadcaster_URDF('robot_description', '', km.get_data(''))
    print('\n  '.join([str(x) for x in sorted(broadcaster.frame_info)]))

    last_update = time()
    while not rospy.is_shutdown():
        now = time()
        if now - last_update >= (1.0 / 50.0):
            state = {s: sin(time() + x * 0.1) for x, s in enumerate(symbols)}
            # print('----\n{}'.format('\n'.join(['{}: {}'.format(s, v) for s, v in state.items()])))
            state[NULL_SYMBOL] = 0
            state[SYM_ROTATION] = now

            broadcaster.update_state(state)
            broadcaster.publish_state()
            coll_world.update_world(state)

            # vis.begin_draw_cycle('prismatic')
            # vis.draw_world('prismatic', coll_world)
            # vis.render('prismatic')
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
