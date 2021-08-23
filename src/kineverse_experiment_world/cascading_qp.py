import kineverse.gradients.gradient_math as gm

from kineverse.motion.min_qp_builder        import GeomQPBuilder as GQPB, \
                                                   TypedQPBuilder as TQPB, \
                                                   generate_controlled_values
from kineverse.utils                        import union, \
                                                   generate_transition_function


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
                       controls_blacklist=set(),
                       transition_overrides=None):
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
        self.lead_o_symbols, \
        self.lead_t_function, \
        self.lead_o_controls = generate_transition_function(self.sym_dt, lead_symbols, transition_overrides)

        self.follower_o_symbols, \
        self.follower_t_function, \
        self.follower_o_controls = generate_transition_function(self.sym_dt, follower_symbols, transition_overrides)

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
        local_state = state.copy()
        lead_cmd = self.lead_qp.get_cmd(local_state, deltaT=deltaT)
        local_state[self.sym_dt] = deltaT

        for s, v in zip(self.lead_o_symbols,
                        self.lead_t_function.call2([local_state[s] if s not in lead_cmd else lead_cmd[s] 
                                                                   for s in self.lead_o_controls])):
            local_state[s] = v

        ref_state = {s: local_state[s] for s in self.follower_delta_map.keys()}
        # Simple solution, no convergence
        for x in range(max_follower_iter):
            follower_cmd = self.follower_qp.get_cmd(local_state, deltaT=0.5)
            if self.follower_qp.equilibrium_reached():
                break

            for s, v in zip(self.follower_o_symbols,
                            self.follower_t_function.call2([local_state[s] if s not in follower_cmd else follower_cmd[s] 
                                                                           for s in self.follower_o_controls])):
                local_state[s] = v

        lead_cmd.update({s_c: (local_state[s] - ref_state[s]) / deltaT for s, s_c in self.follower_delta_map.items()})
        return lead_cmd

    def equilibrium_reached(self, low_eq=1e-3, up_eq=-1e-3):
        return self.lead_qp.equilibrium_reached(low_eq, up_eq)

