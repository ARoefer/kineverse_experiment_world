import rospy
import kineverse.gradients.gradient_math as gm
import math

from tqdm import tqdm

from kineverse.model.geometry_model         import GeometryModel, Path
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer
from kineverse.motion.min_qp_builder        import GeomQPBuilder as GQPB, \
                                                   TypedQPBuilder as TQPB, \
                                                   SoftConstraint as SC,  \
                                                   generate_controlled_values, \
                                                   depth_weight_controlled_values
from kineverse.operations.urdf_operations   import load_urdf
from kineverse.urdf_fix                     import load_urdf_file
from kineverse.utils                        import generate_transition_function

from kineverse_tools.ik_solver              import ik_solve_one_shot

from kineverse_experiment_world.nobilia_shelf import create_nobilia_shelf
from kineverse_experiment_world.cascading_qp  import CascadingQP


if __name__ == '__main__':
    rospy.init_node('synchronized_motion')

    vis = ROSBPBVisualizer('~vis', base_frame='world')

    km = GeometryModel()
    pr2_urdf = load_urdf_file('package://iai_pr2_description/robots/pr2_calibrated_with_ft2.xml')
    load_urdf(km, Path('pr2'), pr2_urdf, Path('world'))

    km.clean_structure()

    create_nobilia_shelf(km,
                         Path('nobilia'),
                         gm.frame3_rpy(0, 0, 0.57, gm.point3(1.0, 0, 0.7)),
                         Path('world'))
    km.clean_structure()    
    km.dispatch_events()

    print('\n'.join(km.timeline_tags.keys()))

    if rospy.get_param('~use_base', False):
        insert_omni_base(km, Path('pr2'), pr2_urdf.get_root(), 'world')
        km.clean_structure()
        km.dispatch_events()

    pr2     = km.get_data('pr2')
    nobilia = km.get_data('nobilia')

    handle  = nobilia.links['handle']
    eef     = pr2.links['r_gripper_tool_frame']

    collision_world = km.get_active_geometry(gm.free_symbols(handle.pose).union(gm.free_symbols(eef.pose)))

    # Joint stuff
    joint_symbols = [j.position for j in km.get_data(f'pr2/joints').values() 
                                if hasattr(j, 'position') and gm.is_symbol(j.position)]
    robot_controlled_symbols = {gm.DiffSymbol(j) for j in joint_symbols} # if 'torso' not in str(j)}
    

    # Init step
    grasp_in_handle = gm.dot(gm.translation3(0.04, 0, 0), gm.rotation3_rpy(math.pi * 0.5, 0, math.pi))
    goal_pose       = gm.dot(handle.pose, grasp_in_handle)
    goal_0_pose     = gm.subs(goal_pose, {s: 0 for s in gm.free_symbols(goal_pose)})

    start_pose = {'l_elbow_flex_joint' : -2.1213,
                    'l_shoulder_lift_joint': 1.2963,
                    'l_wrist_flex_joint' : -1.16,
                    'r_shoulder_pan_joint': -1.0,
                    'r_shoulder_lift_joint': 0.9,
                    'r_upper_arm_roll_joint': -1.2,
                    'r_elbow_flex_joint' : -2.1213,
                    'r_wrist_flex_joint' : -1.05,
                    'r_forearm_roll_joint': 3.14,
                    'r_wrist_roll_joint': 0,
                    'torso_lift_joint'   : 0.16825}
    start_pose = {gm.Position(Path(f'pr2/{n}')): v for n, v in start_pose.items()}
    start_state = {s: 0 for s in gm.free_symbols(collision_world)}
    start_state.update(start_pose)
    ik_err, robot_start_state = ik_solve_one_shot(km, eef.pose, start_state, goal_0_pose)

    start_state.update({s: 0 for s in gm.free_symbols(handle.pose)})
    start_state.update(robot_start_state)

    collision_world.update_world(start_state)

    vis.begin_draw_cycle('ik_solution')
    vis.draw_world('ik_solution', collision_world)
    vis.render('ik_solution')

    print(f'IK error: {ik_err}')
    
    def gen_dv_cvs(km, constraints, controlled_symbols):
        cvs, constraints = generate_controlled_values(constraints, controlled_symbols)
        cvs = depth_weight_controlled_values(km, cvs, exp_factor=1.02)
        print('\n'.join(f'{cv.symbol}: {cv.weight_id}' for _, cv in sorted(cvs.items())))
        return cvs, constraints

    dyn_goal_pos_error = gm.norm(gm.pos_of(eef.pose) - gm.pos_of(goal_pose))
    dyn_goal_rot_error = gm.norm(eef.pose[:3, :3] - goal_pose[:3, :3])

    lead_goal_constraints = {'open_object': SC(1.84 - nobilia.joints['hinge'].position,
                                               1.84 - nobilia.joints['hinge'].position, 1, nobilia.joints['hinge'].position)}

    follower_goal_constraints = {'keep position': SC(-dyn_goal_pos_error,
                                                     -dyn_goal_pos_error, 10, dyn_goal_pos_error),
                                 'keep rotation': SC(-dyn_goal_rot_error, -dyn_goal_rot_error, 1, dyn_goal_rot_error)}

    blacklist = {gm.Velocity(Path('pr2/torso_lift_joint'))}

    solver = CascadingQP(km, 
                         lead_goal_constraints, 
                         follower_goal_constraints, 
                         f_gen_follower_cvs=gen_dv_cvs,
                         # controls_blacklist=blacklist
                         )

    # exit()
    sym_dt = gm.Symbol('dT')
    t_symbols, t_function, t_params = generate_transition_function(sym_dt, solver.state_symbols)

    start_state.update({s: 0 for s in solver.controlled_symbols})
    start_state.update({s: 0 for s in blacklist})
    start_state[sym_dt] = 0.02
    for x in tqdm(range(500), desc='Generating motion...'):
        try:
            cmd = solver.get_cmd(start_state.copy(), deltaT=start_state[sym_dt])
        except Exception as e:
            print(f'Exception during calculation of next step:\n{e}')
            break

        for s, v in zip(t_symbols, t_function.call2([start_state[s] if s not in cmd else cmd[s] for s in t_params])):
            start_state[s] = v
        collision_world.update_world(start_state)
        vis.begin_draw_cycle('cascading_qp')
        vis.draw_world('cascading_qp', collision_world)
        vis.render('cascading_qp')

        if solver.equilibrium_reached():
            break

        # integrator.run(0.02, 500, logging=False, real_time=True)


