import rospy
import numpy  as np
import pandas as pd

import kineverse.gradients.gradient_math  as gm 
from kineverse.model.geometry_model       import GeometryModel, Path
from kineverse.operations.urdf_operations import load_urdf
from kineverse.urdf_fix                   import hacky_urdf_parser_fix, \
                                                 urdf_filler
from kineverse.utils                      import res_pkg_path, union

from kineverse_experiment_world.utils     import random_normal_translation, random_rot_normal

from urdf_parser_py.urdf import URDF

from tqdm import tqdm


class Particle(object):
    def __init__(self, state, cov, probability=1.0):
        self.state = state
        self.cov   = cov
        self.probability = probability

class EKFModel(object):
    DT_SYM = gm.Symbol('dt')

    def __init__(self, observations, constraints, Q=None, transition_rules=None):
        state_vars = union([gm.free_symbols(o) for o in observations.values()])

        self.ordered_vars = [s for _, s in sorted((str(s), s) for s in state_vars)]
        self.Q = Q if Q is not None else np.zeros((len(self.ordered_vars), len(self.ordered_vars)))

        st_fn = {}
        for s in self.ordered_vars:
            st_fn[s] = gm.wrap_expr(s + gm.DiffSymbol(s) * EKFModel.DT_SYM)
        
        if transition_rules is not None:
            varset = set(self.ordered_vars).union({gm.DiffSymbol(s) for s in self.ordered_vars}).union({EKFModel.DT_SYM})
            for s, r in transition_rules.items():
                if s in st_fn:
                    if len(gm.free_symbols(r).difference(varset)) == 0:
                        st_fn[s] = gm.wrap_expr(r)
                    else:
                        print('Dropping rule "{}: {}". Symbols missing from state: {}'.format(s, r, gm.free_symbols(r).difference(varset)))
        control_vars = union([gm.free_symbols(r) for r in st_fn.values()]) \
                        .difference(self.ordered_vars)            \
                        .difference({EKFModel.DT_SYM})
        self.ordered_controls = [s for _, s in sorted((str(s), s) for s in control_vars)]
        
        # State as column vector n * 1
        temp_g_fn = gm.Matrix([gm.extract_expr(st_fn[s]) for s in self.ordered_vars])
        self.g_fn = gm.speed_up(temp_g_fn, [EKFModel.DT_SYM] + self.ordered_vars + self.ordered_controls)
        temp_g_prime_fn = gm.Matrix([[gm.extract_expr(st_fn[s][d]) if d in st_fn[s] else 0
                                                                   for d in self.ordered_controls] 
                                                                   for s in self.ordered_vars])
        self.g_prime_fn = gm.speed_up(temp_g_prime_fn, [EKFModel.DT_SYM] + self.ordered_vars + self.ordered_controls)

        self.obs_labels = []
        self.takers = []
        flat_obs    = []
        for o_label, o in sorted(observations.items()):
            if gm.is_symbolic(o):
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
                            self.obs_labels.append('{}_{}_{}'.format(o_label, *coords))
                            flat_obs.append(gm.wrap_expr(c))
                            indices.append(coords[0] * o.shape[1] + coords[1])
                    if len(indices) > 0:
                        self.takers.append((o_label, indices))
                else:
                    self.obs_labels.append(o_label)
                    flat_obs.append(gm.wrap_expr(o))
                    self.takers.append((o_label, [0]))

        temp_h_fn = gm.Matrix([gm.extract_expr(o) for o in flat_obs])
        self.h_fn = gm.speed_up(temp_h_fn, self.ordered_vars)
        temp_h_prime_fn = gm.Matrix([[gm.extract_expr(o[d]) if d in o else 0 
                                                            for d in self.ordered_controls]
                                                            for o in flat_obs])
        self.h_prime_fn = gm.speed_up(temp_h_prime_fn, self.ordered_vars)

        constraints = {c.expr: [float(subs(c.lower, {c.expr: 0})), float(subs(c.upper, {c.expr: 0}))] 
                        for k, c in constraints.items() if gm.is_symbol(c.expr) and 
                        len(gm.free_symbols(c.lower).union(gm.free_symbols(c.upper))) == 0 or
                        gm.free_symbols(c.lower).union(gm.free_symbols(c.upper)) == {c.expr}}
        self.state_bounds = np.array([constraints[s] if  s in constraints else 
                                     [-np.pi, np.pi] for s in self.ordered_vars])
        self.obs_mean  = np.zeros(len(self.obs_labels))
        self.obs_count = 0
        self.R = np.zeros((len(self.obs_labels), len(self.obs_labels)))

    def add_observation(self, obs):
        self.obs_mean   = (self.obs_mean * self.obs_count + obs) / (self.obs_count + 1)
        delta           = (obs - self.obs_mean).reshape((self.R.shape[0], 1))
        self.R          = (self.R * self.obs_count + delta.dot(delta.T)) / (self.obs_count + 1)
        self.obs_count += 1

    def reset_observation(self):
        self.obs_mean  = np.zeros(len(self.obs_labels))
        self.obs_count = 0
        self.R = np.zeros((len(self.obs_labels), len(self.obs_labels)))        

    def gen_obs_vector(self, obs_dict):
        return np.hstack([np.take(obs_dict[l], i) for l, i in self.takers])

    def pandas_R(self):
        if self.R is None:
            return None
        return pd.DataFrame(data=self.R, index=self.obs_labels, columns=self.obs_labels)

    def predict(self, state_t, Sigma_t, control, dt=0.05):
        params = np.hstack(([dt], state_t, control))
        F_t    = self.g_prime_fn.call2(params)
        return self.g_fn.call2(params), F_t.dot(Sigma_t).dot(F_t.T) + self.Q

    def update(self, state_t, Sigma_t, obs_t):
        H_t = self.h_prime_fn.call2(state_t)
        PHT = np.dot(Sigma_t, H_t.T)
        S_t = np.dot(H_t, PHT) + self.R
        if np.linalg.det(S_t) != 0.0:
            K_t = np.dot(PHT, np.linalg.inv(S_t)) # + self.Q))
            y_t = obs_t - self.h_fn.call2(state_t).flatten()
            state_t = (state_t + np.dot(K_t, y_t)).flatten()
            I_KH    = np.eye(Sigma_t.shape[0]) - np.dot(K_t, H_t)
            Sigma_t = np.dot(I_KH, Sigma_t).dot(I_KH.T) + np.dot(K_t, self.R).dot(K_t.T)
            return state_t, Sigma_t
        else:
            return state_t, Sigma_t

    def spawn_particle(self):
        return Particle(np.random.uniform(self.state_bounds.T[0], self.state_bounds.T[1]),
                        np.diag(self.state_bounds.T[1] - self.state_bounds.T[0]))

    def __str__(self):
        return 'EKF estimating:\n  {}\nFrom:\n  {}\nWith controls:\n  {}'.format(
                        '\n  '.join(str(s) for s in self.ordered_vars), 
                        '\n  '.join(l for l in self.obs_labels),
                        '\n  '.join(str(c) for c in self.ordered_controls))


if __name__ == '__main__':

    km = GeometryModel()

    with open(res_pkg_path('package://iai_kitchen/urdf_obj/IAI_kitchen.urdf'), 'r') as urdf_file:
        urdf_kitchen_str = urdf_file.read()
        kitchen_model = urdf_filler(URDF.from_xml_string(hacky_urdf_parser_fix(urdf_kitchen_str)))
        load_urdf(km, Path('kitchen'), kitchen_model)

    km.clean_structure()
    km.dispatch_events()

    kitchen = km.get_data('kitchen')
    
    tracking_pools = []
    for name, link in kitchen.links.items():
        symbols = gm.free_symbols(link.pose)
        if len(symbols) == 0:
            continue

        for x in range(len(tracking_pools)):
            syms, l = tracking_pools[x]
            if len(symbols.intersection(syms)) != 0: # BAD ALGORITHM, DOES NOT CORRECTLY IDENTIFY SETS
                tracking_pools[x] = (syms.union(symbols), l + [(name, link.pose)])
                break
        else:
            tracking_pools.append((symbols, [(name, link.pose)]))

    tracking_pools = [tracking_pools[7]]
    # print('Identified {} tracking pools:\n{}'.format(len(tracking_pools), tracking_pools))

    ekfs = [EKFModel(dict(poses), km.get_constraints_by_symbols(symbols)) for symbols, poses in tracking_pools]
    print('Created {} EKF models'.format(len(ekfs)))
    print('\n'.join(str(e) for e in ekfs))

    observed_poses = {}
    for ekf in ekfs:
        for link_name, _ in ekf.takers:
            observed_poses[link_name] = kitchen.links[link_name].pose
    names, poses = zip(*sorted(observed_poses.items()))

    state_symbols = union([gm.free_symbols(p) for p in poses])
    ordered_state_vars = [s for _, s in sorted((str(s), s) for s in state_symbols)]
    state_constraints = km.get_constraints_by_symbols(state_symbols)
    state_constraints = {c.expr: [float(subs(c.lower, {c.expr: 0})), float(subs(c.upper, {c.expr: 0}))] 
                         for k, c in state_constraints.items() if gm.is_symbol(c.expr) and 
                         len(gm.free_symbols(c.lower).union(gm.free_symbols(c.upper))) == 0 or
                         gm.free_symbols(c.lower).union(gm.free_symbols(c.upper)) == {c.expr}}
    state_bounds = np.array([state_constraints[s] if  s in state_constraints else 
                                  [-np.pi, np.pi] for s in ordered_state_vars])

    state_fn = gm.speed_up(gm.vstack(*poses), ordered_state_vars)

    samples = 1
    lin_std = 0.2
    ang_std = 0.2
    n_observations = 10

    improvements = []
    final_deltas = []

    for x in tqdm(range(samples)):
        state = np.random.uniform(state_bounds.T[0], state_bounds.T[1])
        observations = state_fn.call2(state)

        estimates = []
        for ekf in ekfs:
            ekf.reset_observation()
            particle = ekf.spawn_particle()
            estimates.append(particle)

        initial_state = dict(sum([[(s, v) for s, v in zip(ekf.ordered_vars, e.state)]
                                          for e, ekf in zip(estimates, ekfs)], []))
        initial_state = np.array([initial_state[s] for s in ordered_state_vars])
        initial_delta = state - initial_state

        for y in range(n_observations):
            # Add noise to observations
            noisy_obs = {}
            for x, noise in enumerate([t.dot(r) for t, r in zip(random_normal_translation(len(poses), 0, lin_std), 
                                                                random_rot_normal(len(poses), 0, ang_std))]):
                noisy_obs[names[x]] = observations[x*4:x*4 + 4,:4].dot(noise)

            for estimate, ekf in zip(estimates, ekfs):
                control = np.zeros(len(ekf.ordered_controls))
                estimate.state, estimate.cov = ekf.predict(estimate.state.flatten(), estimate.cov, control)
                obs_vector = ekf.gen_obs_vector(noisy_obs)
                ekf.add_observation(obs_vector)
                estimate.state, estimate.cov = ekf.update(estimate.state, estimate.cov, ekf.gen_obs_vector(noisy_obs))

        final_estimate = dict(sum([[(s, v) for s, v in zip(ekf.ordered_vars, e.state)]
                                           for e, ekf in zip(estimates, ekfs)], []))

        final_state = np.array([final_estimate[s] for s in ordered_state_vars])
        final_delta = np.abs(state - final_state)
        final_deltas.append(final_delta)
        improvements.append((np.abs(initial_delta) - final_delta) / np.abs(initial_delta))

    pd.options.display.float_format = '{:9.4f}'.format

    print(pd.DataFrame(index=[str(s) for s in ordered_state_vars], columns=['final delta', 'improvements'], data=np.vstack((np.mean(final_deltas, axis=0),
                                                                                                                            np.mean(improvements, axis=0))).T))

