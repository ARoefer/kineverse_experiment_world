import numpy  as np
import pandas as pd

import kineverse.gradients.gradient_math as gm

from kineverse.model.paths import Path
from kineverse.motion.min_qp_builder import TypedQPBuilder as TQPB,    \
                                            SoftConstraint as SC,      \
                                            generate_controlled_values,\
                                            QPSolverException
from kineverse.utils                 import union, \
                                            generate_transition_function
from kineverse.time_wrapper          import Time

class Particle(object):
    def __init__(self, state, cov, probability=1.0):
        self.state = state
        self.cov   = cov
        self.probability = probability

class QPStateModel(object):
    DT_SYM = gm.Symbol('dt')

    def __init__(self, km, observations, transition_rules=None, max_iterations=20):
        """Sets up an EKF estimating the underlying state of a set of observations.
        
        Args:
            km (ArticulationModel): Articulation model to query for constraints
            observations (dict): A dict of observations. Names are mapped to 
                                 any kind of symbolic expression/matrix
            transition_rules (dict, optional): Maps symbols to their transition rule.
                                               Rules will be generated automatically, if not provided here.
        """
        state_vars = union([gm.free_symbols(o) for o in observations.values()])


        self.ordered_vars,  \
        self.transition_fn, \
        self.transition_args = generate_transition_function(QPStateModel.DT_SYM, 
                                                            state_vars, 
                                                            transition_rules)
        self.command_vars = {s for s in self.transition_args 
                                if s not in state_vars and str(s) != str(QPStateModel.DT_SYM)}

        obs_constraints = {}
        obs_switch_vars = {}

        # State as column vector n * 1
        self.switch_vars = {}
        self._obs_state  = {}
        self.obs_vars  = {}
        self.takers = {}
        flat_obs    = []
        for o_label, o in sorted(observations.items()):
            if gm.is_symbolic(o):
                obs_switch_var = gm.Symbol(f'{o_label}_observed')
                self.switch_vars[o_label] = obs_switch_var
                if o_label not in obs_constraints:
                    obs_constraints[o_label] = {}
                if o_label not in self.obs_vars:
                    self.obs_vars[o_label] = []

                if gm.is_matrix(o):
                    if type(o) == gm.GM:
                        components = zip(sum([[(y, x) for x in range(o.shape[1])] 
                                                      for y in range(o.shape[0])], []), iter(o))
                    else:
                        components = zip(sum([[(y, x) for x in range(o.shape[1])] 
                                                      for y in range(o.shape[0])], []), o.T.elements()) # Casadi iterates vertically
                    indices = []
                    for coords, c in components:
                        if gm.is_symbolic(c):
                            obs_symbol = gm.Symbol('{}_{}_{}'.format(o_label, *coords))
                            obs_error  = gm.abs(obs_symbol - c)
                            constraint = SC(-obs_error - (1 - obs_switch_var) * 1e3,
                                            -obs_error + (1 - obs_switch_var) * 1e3, 1, obs_error)
                            obs_constraints[o_label][f'{o_label}:{Path(obs_symbol)}'] = constraint
                            self.obs_vars[o_label].append(obs_symbol)
                            indices.append(coords[0] * o.shape[1] + coords[1])

                    if len(indices) > 0:
                        self.takers[o_label] = indices
                else:
                    obs_symbol = gm.Symbol(f'{o_label}_value')
                    obs_error  = gm.abs(obs_symbol - c)
                    constraint = SC(-obs_error - obs_switch_var * 1e9, 
                                    -obs_error + obs_switch_var * 1e9, 1, obs_error)
                    obs_constraints[o_label][f'{o_label}:{Path(obs_symbol)}'] = constraint

                    self.obs_vars[o_label].append(obs_symbol)
                    self.takers[o_label] = [0]

        state_constraints = km.get_constraints_by_symbols(state_vars)

        cvs, hard_constraints = generate_controlled_values(state_constraints, 
                                                           {gm.DiffSymbol(s) for s in state_vars 
                                                                             if gm.get_symbol_type(s) != gm.TYPE_UNKNOWN})
        flat_obs_constraints = dict(sum([list(oc.items()) for oc in obs_constraints.values()], []))

        self.qp = TQPB(hard_constraints, flat_obs_constraints, cvs)

        self._state = {s: 0 for s in state_vars}
        self._state_buffer = []
        self._state.update({s: 0 for s in self.transition_args})
        self._obs_state = {s: 0 for s in sum(self.obs_vars.values(), [])}
        self._obs_count = 0
        self._stamp_last_integration = None
        self._max_iterations = 10
        self._current_error  = 1e9

    def _integrate_state(self):
        now = Time.now()
        if self._stamp_last_integration is not None:
            dt = (now - self._stamp_last_integration).to_sec()
            self._state[QPStateModel.DT_SYM] = dt
            new_state = self.transition_fn.call2([self._state[x] for x in self.transition_args]).flatten()
            for x, (s, v) in enumerate(zip(self.ordered_vars, new_state)):
                delta = v - self._state[s]
                self._state[s] = v
                for state in self._state_buffer:
                    state[x] += delta 

        self._stamp_last_integration = now        

    def set_command(self, command):
        self._integrate_state()

        for s in self.command_vars:
            if s in command:
                self._state[s] = command[s]

    @profile
    def update(self, observation):
        
        self._integrate_state()

        for o_label, o_vars in self.obs_vars.items():
            if o_label in observation:
                self._obs_state[self.switch_vars[o_label]] = 1
                sub_obs = np.take(observation[o_label], self.takers[o_label])
                self._obs_state.update({s: v for s, v in zip(o_vars, sub_obs)})
            else:
                self._obs_state[self.switch_vars[o_label]] = 0

        self._obs_state.update(self._state)
        self._obs_state[QPStateModel.DT_SYM] = 0.5

        for x in range(self._max_iterations):
            cmd = self.qp.get_cmd(self._obs_state, deltaT=0.5)
            self._obs_state.update(cmd)
            new_state = self.transition_fn.call2([self._obs_state[s] for s in self.transition_args])
            self._obs_state.update({s: v for s, v in zip(self.ordered_vars, new_state)})
            if self.qp.equilibrium_reached():
                break

        # CMA update
        self._obs_count += 1
        # cma_n   = np.array([self._state[s] for s in self.ordered_vars])
        x_n_1   = np.array([self._obs_state[s] for s in self.ordered_vars]).flatten()

        self._state_buffer.append(x_n_1)
        if len(self._state_buffer) > 7:
            self._state_buffer = self._state_buffer[-7:]

        state_mean = np.mean(self._state_buffer, axis=0)

        # print(f'cma_n: {cma_n}\nx_n_1: {x_n_1}')
        # cma_n_1 = cma_n + (x_n_1 - cma_n) / self._obs_count if self._obs_count > 1 else x_n_1
        self._state.update({s: v for s, v in zip(self.ordered_vars, state_mean)})
        return self.qp.latest_error

    def state(self):
        self._integrate_state()
        return {s: self._state[s] for s in self.ordered_vars}

    @property
    def latest_error(self):
        return self.qp.latest_error

    def __str__(self):
        return 'QP estimating:\n  {}\nFrom:\n  {}\nWith controls:\n  {}'.format(
                        '\n  '.join(str(s) for s in self.ordered_vars), 
                        '\n  '.join(l for l in sorted(str(s) for s in sum(self.obs_vars.values(), []))),
                        '\n  '.join(c for c in sorted(str(c) for c in self.command_vars)))