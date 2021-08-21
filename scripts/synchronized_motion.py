import rospy
import kineverse.gradients.gradient_math as gm
import math

from tqdm import tqdm

from kineverse.model.geometry_model         import GeometryModel, Path
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer
from kineverse.motion.integrator            import CommandIntegrator
from kineverse.motion.min_qp_builder        import GeomQPBuilder as GQPB, \
                                                   TypedQPBuilder as TQPB, \
                                                   SoftConstraint as SC,  \
                                                   generate_controlled_values, \
                                                   depth_weight_controlled_values
from kineverse.operations.urdf_operations   import load_urdf
from kineverse.urdf_fix                     import load_urdf_file
from kineverse.utils                        import union, \
                                                   generate_transition_function

from kineverse_tools.ik_solver              import ik_solve_one_shot

from kineverse_experiment_world.nobilia_shelf import create_nobilia_shelf


class CascadingQP(object):
    """Double layered solver which produces one step
       in a leading problem and then updates a follower problem.
       Finally it returns the true velocity commands for the leader problem
       and approximated velocities for the follower problem.
    """
    def __init__(self, km, 
                       lead_goal_constraints, 
                       follower_goal_constraints, 
                       t_leader=TQPB, t_follower=TQPB, 
                       f_gen_lead_cvs=None,
                       f_gen_follower_cvs=None,
                       visualizer=None,
                       controls_blacklist=set()):
        lead_symbols     = union(gm.free_symbols(c.expr) for c in lead_goal_constraints.values())
        follower_symbols = union(gm.free_symbols(c.expr) for c in follower_goal_constraints.values())
        self.lead_symbols     = lead_symbols
        self.follower_symbols = follower_symbols

        self.lead_controlled_symbols     = {gm.DiffSymbol(s) for s in lead_symbols 
                                                        if gm.get_symbol_type(s) != gm.TYPE_UNKNOWN 
                                                        and gm.DiffSymbol(s) not in controls_blacklist}
        # Only update the symbols that are unique to the follower
        self.follower_controlled_symbols = {gm.DiffSymbol(s) for s in follower_symbols 
                                                        if gm.get_symbol_type(s) != gm.TYPE_UNKNOWN 
                                                        and s not in lead_symbols 
                                                        and gm.DiffSymbol(s) not in controls_blacklist}
        
        f_gen_lead_cvs = self.gen_controlled_values if f_gen_lead_cvs is None else f_gen_lead_cvs
        lead_cvs, \
        lead_constraints = f_gen_lead_cvs(km, 
                                          km.get_constraints_by_symbols(lead_symbols.union(self.lead_controlled_symbols)),
                                          self.lead_controlled_symbols)
        
        f_gen_follower_cvs = self.gen_controlled_values if f_gen_follower_cvs is None else f_gen_follower_cvs
        follower_cvs, \
        follower_constraints = f_gen_follower_cvs(km, 
                                                  km.get_constraints_by_symbols(follower_symbols.union(self.follower_controlled_symbols)),
                                                  self.follower_controlled_symbols)

        if issubclass(t_leader, GQPB):
            lead_world = km.get_active_geometry(lead_symbols)
            self.lead_qp = t_leader(lead_world,
                                    lead_constraints,
                                    lead_goal_constraints,
                                    lead_cvs,
                                    visualizer=visualizer)
        else:
            self.lead_qp = t_leader(lead_constraints,
                                    lead_goal_constraints,
                                    lead_cvs)

        self.sym_dt = gm.Symbol('dT')
        self.follower_o_symbols, \
        self.follower_t_function, \
        self.follower_o_controls = generate_transition_function(self.sym_dt, follower_symbols)

        self.follower_delta_map = {gm.IntSymbol(s): s for s in self.follower_controlled_symbols}

        if issubclass(t_follower, GQPB):
            follower_world   = km.get_active_geometry(follower_symbols)
            self.follower_qp = t_follower(follower_world,
                                          follower_constraints,
                                          follower_goal_constraints,
                                          follower_cvs,
                                          visualizer=visualizer)
        else:
            self.follower_qp = t_follower(follower_constraints,
                                          follower_goal_constraints,
                                          follower_cvs)

    @property
    def state_symbols(self):
        return self.lead_symbols.union(self.follower_symbols)

    @property
    def controlled_symbols(self):
        return self.lead_controlled_symbols.union(self.follower_controlled_symbols)

    def gen_controlled_values(self, km, constraints, controlled_symbols):
        """Base implementation expected to return a tuple of 
           controlled values and constraints"""
        return generate_controlled_values(constraints, controlled_symbols)


    def get_cmd(self, state, deltaT=0.02, max_follower_iter=20):
        lead_cmd = self.lead_qp.get_cmd(state, deltaT=deltaT)
        for s, v in lead_cmd.items():
            s_i = gm.IntSymbol(s)
            if s_i in state:
                state[s_i] += v * deltaT

        ref_state = {s: state[s] for s in self.follower_delta_map.keys()}
        # Simple solution, no convergence
        for x in range(max_follower_iter):
            follower_cmd = self.follower_qp.get_cmd(state, deltaT=0.5)
            if self.follower_qp.equilibrium_reached():
                break

            for s, v in zip(self.follower_o_symbols,
                            self.follower_t_function.call2([state[s] if s not in follower_cmd else follower_cmd[s] 
                                                                     for s in self.follower_o_controls])):
                state[s] = v

        lead_cmd.update({s_c: (state[s] - ref_state[s]) / deltaT for s, s_c in self.follower_delta_map.items()})
        return lead_cmd

    def equilibrium_reached(self, low_eq=1e-3, up_eq=-1e-3):
        return self.lead_qp.equilibrium_reached(low_eq, up_eq)


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

    # Actual motion
    # all_controlled_symbols = {gm.DiffSymbol(s) for s in gm.free_symbols(handle.pose)}.union(robot_controlled_symbols)

    # controlled_values, \
    # constraints = generate_controlled_values(km.get_constraints_by_symbols(all_controlled_symbols.union(gm.free_symbols(handle.pose).union(gm.free_symbols(eef.pose)))), 
    #                                          all_controlled_symbols)
    
    def gen_dv_cvs(km, constraints, controlled_symbols):
        cvs, constraints = generate_controlled_values(constraints, controlled_symbols)
        cvs = depth_weight_controlled_values(km, cvs, exp_factor=1.02)
        print('\n'.join(f'{cv.symbol}: {cv.weight_id}' for _, cv in sorted(cvs.items())))
        return cvs, constraints

    # controlled_values = depth_weight_controlled_values(km, 
    #                                                    controlled_values, 
    #                                                    exp_factor=1.1)

    # goal_lin_vel = gm.vector3(sum([gm.diff(goal_pose[0, 3], s) for s in all_controlled_symbols], 0),
    #                           sum([gm.diff(goal_pose[1, 3], s) for s in all_controlled_symbols], 0),
    #                           sum([gm.diff(goal_pose[2, 3], s) for s in all_controlled_symbols], 0))

    # eef_lin_vel  = gm.vector3(sum([gm.diff(eef.pose[0, 3], s) for s in all_controlled_symbols], 0),
    #                           sum([gm.diff(eef.pose[1, 3], s) for s in all_controlled_symbols], 0),
    #                           sum([gm.diff(eef.pose[2, 3], s) for s in all_controlled_symbols], 0))
    # vel_errors = [goal_lin_vel[x] - eef_lin_vel[x] for x in range(3)] 

    dyn_goal_pos_error = gm.norm(gm.pos_of(eef.pose) - gm.pos_of(goal_pose))
    dyn_goal_rot_error = gm.norm(eef.pose[:3, :3] - goal_pose[:3, :3])

    lead_goal_constraints = {'open_object': SC(2.0 - nobilia.joints['hinge'].position,
                                               2.0 - nobilia.joints['hinge'].position, 1, nobilia.joints['hinge'].position)}

    follower_goal_constraints = {'keep position': SC(-dyn_goal_pos_error,
                                                     -dyn_goal_pos_error, 10, dyn_goal_pos_error),
                                 'keep roatation': SC(-dyn_goal_rot_error, -dyn_goal_rot_error, 1, dyn_goal_rot_error)}
    # goal_constraints.update({f'sync vel {x}': SC(0, 0, 1, v) for x, v in enumerate(vel_errors)})

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

    # qp = GQPB(collision_world, constraints, goal_constraints, controlled_values, visualizer=vis)

    # integrator = CommandIntegrator(qp, start_state=start_state)
    # integrator.restart(f'PR2 synchronized motion')
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


