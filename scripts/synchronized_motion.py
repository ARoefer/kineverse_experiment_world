import rospy
import kineverse.gradients.gradient_math as gm
import math
import numpy as np

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
from kineverse.utils                        import generate_transition_function, \
                                                   static_var_bounds

from kineverse_tools.ik_solver              import ik_solve_one_shot

from kineverse_experiment_world.nobilia_shelf import create_nobilia_shelf
from kineverse_experiment_world.cascading_qp  import CascadingQP
from kineverse_experiment_world.utils         import insert_omni_base, \
                                                     insert_diff_base, \
                                                     load_model

def generic_setup(km, robot_path, eef_link='gripper_link'):
    joint_symbols = [j.position for j in km.get_data(f'{robot_path}/joints').values() 
                                if hasattr(j, 'position') and gm.is_symbol(j.position)]
    robot_controlled_symbols = {gm.DiffSymbol(j) for j in joint_symbols} # if 'torso' not in str(j)}
    blacklist = set()
    return joint_symbols, \
           robot_controlled_symbols, \
           {}, \
           km.get_data(robot_path).links[eef_link], \
           blacklist


def pr2_setup(km, pr2_path):
    start_pose = {'l_elbow_flex_joint' : -2.1213,
                  'l_shoulder_lift_joint': 1.2963,
                  'l_wrist_flex_joint' : -1.16,
                  'l_shoulder_pan_joint': 0.7,
                  'r_shoulder_pan_joint': -1.0,
                  'r_shoulder_lift_joint': 0.9,
                  'r_upper_arm_roll_joint': -1.2,
                  'r_elbow_flex_joint' : -2.1213,
                  'r_wrist_flex_joint' : -1.05,
                  'r_forearm_roll_joint': 3.14,
                  'r_wrist_roll_joint': 0,
                  'torso_lift_joint'   : 0.16825}
    start_pose = {gm.Position(Path(f'{pr2_path}/{n}')): v for n, v in start_pose.items()}
    joint_symbols = [j.position for j in km.get_data(f'{pr2_path}/joints').values() 
                                if hasattr(j, 'position') and gm.is_symbol(j.position)]
    robot_controlled_symbols = {gm.DiffSymbol(j) for j in joint_symbols} # if 'torso' not in str(j)}
    blacklist = {gm.Velocity(Path(f'{pr2_path}/torso_lift_joint'))}
    return joint_symbols, \
           robot_controlled_symbols, \
           start_pose, \
           km.get_data(pr2_path).links['r_gripper_tool_frame'], \
           blacklist


if __name__ == '__main__':
    rospy.init_node('synchronized_motion')

    vis = ROSBPBVisualizer('~vis', base_frame='world')

    km = GeometryModel()
    robot_type = rospy.get_param('~robot', 'pr2')
    if robot_type.lower() == 'pr2':
        robot_urdf = load_urdf_file('package://iai_pr2_description/robots/pr2_calibrated_with_ft2.xml')
    elif robot_type.lower() == 'hsrb':
        robot_urdf = load_urdf_file('package://hsr_description/robots/hsrb4s.obj.urdf')
    elif robot_type.lower() == 'fetch':
        robot_urdf = load_urdf_file('package://fetch_description/robots/fetch.urdf')
    elif robot_type.lower() == 'fmm':
        robot_urdf = load_urdf_file('package://fmm/robots/fmm.urdf')
        rospy.set_param('~eef', rospy.get_param('~eef', 'panda_hand_tcp'))
    else:
        print(f'Unknown robot {robot_type}')
        exit(1)

    robot_name = robot_urdf.name
    robot_path = Path(robot_name)
    load_urdf(km, robot_path, robot_urdf, Path('world'))

    km.clean_structure()

    model_name = load_model(km, 
                            rospy.get_param('~model', 'nobilia'),
                            'world',
                            1.0, 0, 0.7, rospy.get_param('~yaw', 0.57))
    km.clean_structure()    
    km.dispatch_events()

    print('\n'.join(km.timeline_tags.keys()))

    if rospy.get_param('~use_base', False):
        if robot_type.lower() == 'fetch':
            insert_diff_base(km, 
                             robot_path, 
                             robot_urdf.get_root(),
                             km.get_data(robot_path + ('joints', 'r_wheel_joint', 'position')),
                             km.get_data(robot_path + ('joints', 'l_wheel_joint', 'position')),
                             world_frame='world',
                             wheel_radius=0.12 * 0.5,
                             wheel_distance=0.3748,
                             wheel_vel_limit=17.4)
        else:
            insert_omni_base(km, robot_path, robot_urdf.get_root(), 'world')
        km.clean_structure()
        km.dispatch_events()

    robot   = km.get_data(robot_path)
    nobilia = km.get_data(model_name)

    handle  = nobilia.links[rospy.get_param('~handle', 'handle')]
    integration_rules = None

    sym_dt = gm.Symbol('dT')

    if robot_name == 'pr2':
        joint_symbols, \
        robot_controlled_symbols, \
        start_pose, \
        eef, \
        blacklist = pr2_setup(km, robot_path)
    elif robot_name == 'fetch':
        joint_symbols, \
        robot_controlled_symbols, \
        start_pose, \
        eef, \
        blacklist = generic_setup(km, robot_path, rospy.get_param('~eef', 'gripper_link'))
        if rospy.get_param('~use_base', False):
            base_joint = robot.joints['to_world']
            robot_controlled_symbols |= {base_joint.l_wheel_vel, base_joint.r_wheel_vel}
            robot_controlled_symbols.difference_update(gm.DiffSymbol(s) for s in [base_joint.x_pos, base_joint.y_pos, base_joint.a_pos])
            blacklist.update(gm.DiffSymbol(s) for s in [base_joint.x_pos, base_joint.y_pos, base_joint.a_pos])
            integration_rules = {
                      base_joint.x_pos: base_joint.x_pos + sym_dt * (base_joint.r_wheel_vel * gm.cos(base_joint.a_pos) * base_joint.wheel_radius * 0.5 + base_joint.l_wheel_vel * gm.cos(base_joint.a_pos) * base_joint.wheel_radius * 0.5),
                      base_joint.y_pos: base_joint.y_pos + sym_dt * (base_joint.r_wheel_vel * gm.sin(base_joint.a_pos) * base_joint.wheel_radius * 0.5 + base_joint.l_wheel_vel * gm.sin(base_joint.a_pos) * base_joint.wheel_radius * 0.5),
                      base_joint.a_pos: base_joint.a_pos + sym_dt * (base_joint.r_wheel_vel * (base_joint.wheel_radius / base_joint.wheel_distance) + base_joint.l_wheel_vel * (- base_joint.wheel_radius / base_joint.wheel_distance))}
        start_pose = {'wrist_roll_joint'   : 0.0,
                      'shoulder_pan_joint' : 0.3,
                      'elbow_flex_joint'   : 1.72,
                      'forearm_roll_joint' : -1.2,
                      'upperarm_roll_joint': -1.57,
                      'wrist_flex_joint'   : 1.66,
                      'shoulder_lift_joint': 1.4,
                      'torso_lift_joint'   : 0.2}
        start_pose = {gm.Position(Path(f'{robot_path}/{n}')): v for n, v in start_pose.items()}
    else:
        joint_symbols, \
        robot_controlled_symbols, \
        start_pose, \
        eef, \
        blacklist = generic_setup(km, robot_path, rospy.get_param('~eef', 'gripper_link'))
        if robot_type.lower() == 'fmm':
            start_pose = {'panda_joint1': 0,
                          'panda_joint2': 0.7,
                          'panda_joint3': 0,
                          'panda_joint4': -2.2,
                          'panda_joint5': 0,
                          'panda_joint6': 2.2,
                          'panda_joint7': 0}
            start_pose = {gm.Position(Path(f'{robot_path}/{n}')): v for n, v in start_pose.items()}       

    collision_world = km.get_active_geometry(gm.free_symbols(handle.pose).union(gm.free_symbols(eef.pose)))

    # Init step
    grasp_in_handle = gm.dot(gm.translation3(0.04, 0, 0), gm.rotation3_rpy(math.pi * 0.5, 0, math.pi))
    if robot_type.lower() == 'fmm':
        grasp_in_handle = gm.dot(gm.translation3(0.05, 0, 0), gm.rotation3_rpy(math.pi * -0.5, math.pi * 0, math.pi * 0.5))
    # grasp_in_handle = gm.dot(gm.translation3(0.04, 0, 0), gm.rotation3_rpy(math.pi * 0.5, 0, 0))
    goal_pose       = gm.dot(handle.pose, grasp_in_handle)
    goal_0_pose     = gm.subs(goal_pose, {s: 0 for s in gm.free_symbols(goal_pose)})

    start_state = {s: 0 for s in gm.free_symbols(collision_world)}
    start_state.update(start_pose)
    print('\n  '.join(f'{p}: {v}' for p, v in start_state.items()))
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

    o_vars, bounds, _ = static_var_bounds(km, gm.free_symbols(handle.pose))

    lead_goal_constraints = {f'open_object {s}': SC(ub - s,
                                                    ub - s, 1, s) for s, (_, ub) in zip(o_vars, bounds)}

    follower_goal_constraints = {'keep position': SC(-dyn_goal_pos_error,
                                                     -dyn_goal_pos_error, 10, dyn_goal_pos_error),
                                 'keep rotation': SC(-dyn_goal_rot_error, -dyn_goal_rot_error, 1, dyn_goal_rot_error)}

    solver = CascadingQP(km, 
                         lead_goal_constraints, 
                         follower_goal_constraints, 
                         f_gen_follower_cvs=gen_dv_cvs,
                         controls_blacklist=blacklist,
                         transition_overrides=integration_rules,
                        #  t_follower=GQPB,
                        #  visualizer=vis,
                         )

    # exit()
    t_symbols, t_function, t_params = generate_transition_function(sym_dt, solver.state_symbols, overrides=integration_rules)

    start_state.update({s: 0 for s in solver.controlled_symbols})
    start_state.update({s: 0 for s in blacklist})
    start_state[sym_dt] = 0.02

    times = []

    visualize = rospy.get_param('~vis', False)
    
    trajectory = {s: [] for s in robot_start_state}
    print(trajectory.keys())

    for x in tqdm(range(500), desc='Generating motion...'):
        if rospy.is_shutdown():
            break

        try:
            start = rospy.Time.now()
            cmd = solver.get_cmd(start_state.copy(), deltaT=start_state[sym_dt])
            times.append((rospy.Time.now() - start).to_sec())
        except Exception as e:
            print(f'Exception during calculation of next step:\n{e}')
            break

        for s, v in zip(t_symbols, t_function.call2([start_state[s] if s not in cmd else cmd[s] for s in t_params])):
            start_state[s] = v
            if s in trajectory:
                trajectory[s].append(v[0])
        
        if visualize:
            collision_world.update_world(start_state)
            vis.begin_draw_cycle('cascading_qp')
            vis.draw_world('cascading_qp', collision_world)
            vis.render('cascading_qp')

        if solver.equilibrium_reached():
            break
        # integrator.run(0.02, 500, logging=False, real_time=True)

    for s, d in sorted([(str(s), t) for s, t in trajectory.items()]):
        if len(d) > 0:
            a = np.asarray(d)
            print(f'{s} ({len(a)}):\n  Min: {a.min()}  Max: {a.max()}\n  Mean: {a.mean()}  Median: {np.median(a)}')

    times = np.array(times)
    print(f'Timing stats:\nMean: {times.mean()} s\nSD: {times.std()} s\nMin: {times.min()} s\nMax: {times.max()} s')

