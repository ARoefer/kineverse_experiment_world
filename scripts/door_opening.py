import rospy

import math
import numpy as np
import kineverse.gradients.gradient_math as gm

from copy import copy

from kineverse.model.paths                import Path, CPath
from kineverse.model.geometry_model       import GeometryModel, \
                                                 RigidBody,     \
                                                 Box, \
                                                 Cylinder, \
                                                 Constraint, \
                                                 ArticulatedObject
from kineverse.operations.basic_operations import Operation
from kineverse.operations.basic_operations import CreateValue, ExecFunction
from kineverse.operations.urdf_operations  import load_urdf, \
                                                  RevoluteJoint, \
                                                  CreateURDFFrameConnection
from kineverse.motion.min_qp_builder       import GeomQPBuilder as GQPB, \
                                                  SoftConstraint, \
                                                  generate_controlled_values, \
                                                  PANDA_LOGGING
from kineverse.motion.integrator           import CommandIntegrator
from kineverse.urdf_fix import hacky_urdf_parser_fix, urdf_filler
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer
from kineverse.visualization.plotting       import draw_recorders,  \
                                                   split_recorders, \
                                                   convert_qp_builder_log, \
                                                   filter_contact_symbols
from kineverse.utils import res_pkg_path

from kineverse_tools.ik_solver import ik_solve_one_shot


from urdf_parser_py.urdf import URDF

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



if __name__ == '__main__':
    rospy.init_node('kineverse_door_opening')

    visualizer = ROSBPBVisualizer('~visualization', base_frame='world')

    km = GeometryModel()

    with open(res_pkg_path('package://kineverse_experiment_world/urdf/iiwa_wsg_50.urdf'), 'r') as f:
        iiwa_urdf = urdf_filler(URDF.from_xml_string(hacky_urdf_parser_fix(f.read())))
        load_urdf(km, Path('iiwa'), iiwa_urdf)

    km.clean_structure()
    km.dispatch_events()

    door_x, door_y = [gm.Position(f'localization_{x}') for x in 'xy']

    create_door(km, Path('door'), 0.5, 0.35, to_world_tf=gm.translation3(door_x, door_y, 0.1))

    km.clean_structure()
    km.dispatch_events()

    door    = km.get_data('door')
    handle  = door.links['handle']
    iiwa    = km.get_data('iiwa')
    symbols = gm.free_symbols(door.links['handle'].pose).union(
              gm.free_symbols(iiwa.links['link_7'].pose))

    door_position   = door.joints['hinge'].position
    handle_position = door.joints['handle'].position

    # Fun for visuals
    world   = km.get_active_geometry(symbols)
    symbols = world.free_symbols
    
    eef = iiwa.links['wsg_50_tool_frame']

    goal_pose = gm.dot(handle.pose, 
                       gm.translation3(-0.08, 0.1, 0),
                       gm.rotation3_rpy(0, math.pi * 0.5,0))

    for x_coord in np.linspace(0.5, 1.0, 4):
        for y_coord in np.linspace(0.2, -0.8, 4):
            ik_goal_start = {s: 0 for s in world.free_symbols}
            ik_goal_start[door_x] = x_coord
            ik_goal_start[door_y] = y_coord
            err_ik, q_ik_goal = ik_solve_one_shot(km, eef.pose, ik_goal_start, goal_pose)

            world.update_world(q_ik_goal)
            visualizer.begin_draw_cycle('pre_open_world')
            visualizer.draw_poses('pre_open_world', gm.eye(4), 0.2, 0.01, [gm.subs(goal_pose, {s: 0 for s in gm.free_symbols(goal_pose)}),
                                                                  gm.subs(eef.pose, q_ik_goal)])
            visualizer.draw_world('pre_open_world', world, g=0.6, b=0.6)
            visualizer.render('pre_open_world')

            # Build Door opening problem

            grasp_err_rot = gm.norm(gm.rot_of(goal_pose - eef.pose).elements())
            grasp_err_lin = gm.norm(gm.pos_of(goal_pose - eef.pose))

            active_symbols = {s for s in gm.free_symbols(eef.pose) if gm.get_symbol_type(s) == gm.TYPE_POSITION}\
                             .union({door_position, handle_position})
            controlled_symbols = {gm.DiffSymbol(s) for s in active_symbols}

            controlled_values, constraints = generate_controlled_values(km.get_constraints_by_symbols(controlled_symbols.union(active_symbols)),
                                                                        controlled_symbols)

            goal_grasp_lin = SoftConstraint(-grasp_err_lin, -grasp_err_lin, 100.0, grasp_err_lin)
            goal_grasp_ang = SoftConstraint(-grasp_err_rot, -grasp_err_rot, 10.0, grasp_err_rot)

            goal_door_angle = SoftConstraint(math.pi * 0.45 - door_position, 
                                             math.pi * 0.45 - door_position, 1.0, door_position)
            goal_handle_angle = SoftConstraint(math.pi * 0.25 - handle_position, 
                                               math.pi * 0.25 - handle_position, 1.0, handle_position)

            qp = GQPB(world, 
                      constraints,
                      {'grasp_constraint_lin': goal_grasp_lin,
                       'grasp_constraint_ang': goal_grasp_ang,
                       'goal_door_angle':   goal_door_angle,
                       'goal_handle_angle': goal_handle_angle},
                      controlled_values,
                      visualizer=visualizer)


            is_unlocked = gm.alg_and(gm.greater_than(door_position, 0.4), gm.less_than(handle_position, 0.15))

            # integrator = CommandIntegrator(qp, start_state=q_ik_goal, 
            #                                    recorded_terms={'is_unlocked': is_unlocked,
            #                                                    'handle_position': handle_position,
            #                                                    'door_position': door_position})

            # try:
            #     integrator.restart(title='Door opening generator')
            #     integrator.run(dt=0.05, max_iterations=500, logging=True, show_progress=True, real_time=False)

            #     draw_recorders([integrator.sym_recorder], 2, 8, 4).savefig(res_pkg_path('package://kineverse_experiment_world/plots/door_opening_terms.png'))

            #     if PANDA_LOGGING:
            #         rec_w, rec_b, rec_c, recs = convert_qp_builder_log(integrator.qp_builder)
            #         draw_recorders([rec_b, rec_c] + [r for _, r in sorted(recs.items())], 1, 8, 4).savefig(res_pkg_path('package://kineverse_experiment_world/plots/door_opening.png'))
            # except Exception as e:
            #     traceback.print_exception(type(e), e, e.__traceback__)
            #     print(f'Motion generation crashed:\n{e}')

            open_start = copy(q_ik_goal)
            open_start[door_position] = math.pi * 0.45
            open_start[handle_position] = math.pi * 0.25
            err_open, q_goal_open = ik_solve_one_shot(km, eef.pose, open_start, goal_pose)

            world.update_world(q_goal_open)

            visualizer.begin_draw_cycle('open_world')
            visualizer.draw_poses('open_world', gm.eye(4), 0.2, 0.01, [gm.subs(goal_pose, open_start),
                                                                       gm.subs(eef.pose, q_goal_open)])
            visualizer.draw_world('open_world', world, r=0.6, b=0.6)
            visualizer.render('open_world')    

            start = rospy.Time.now()
            # while not rospy.is_shutdown():
            rospy.sleep(0.3)

            # print(f'IK error: {err_ik}\nOpen error: {err_open}')

            bla = input('Hit enter to continue to the next config')
            if rospy.is_shutdown():
                break

        if rospy.is_shutdown():
                break

    #     delta = (rospy.Time.now() - start).to_sec()
    #     state = {s: math.pi * 0.5 * 0.5 * (math.cos(delta) + 1) for s in symbols}
    #     world.update_world(state)
    #     visualizer.begin_draw_cycle('world')
    #     visualizer.draw_world('world', world)
    #     visualizer.render('world')

    #     # print('lol')

    #     rospy.sleep(0.02)
    
