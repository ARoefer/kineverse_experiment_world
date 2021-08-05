import numpy  as np
import pandas as pd

import kineverse.gradients.gradient_math as gm

from kineverse.utils import union


class Particle(object):
    def __init__(self, state, cov, probability=1.0):
        self.state = state
        self.cov   = cov
        self.probability = probability

class EKFModel(object):
    DT_SYM = gm.Symbol('dt')

    def __init__(self, observations, constraints, Q=None, transition_rules=None):
        """Sets up an EKF estimating the underlying state of a set of observations.
        
        Args:
            observations (dict): A dict of observations. Names are mapped to 
                                 any kind of symbolic expression/matrix
            constraints (dict): A dict of named constraints that govern the 
                                configuration space of the estimated quantities
            Q (matrix, optional): Process noise of the estimated quantities. 
                                  Note: Quantities are expected to be ordered alphabetically
            transition_rules (dict, optional): Maps symbols to their transition rule.
                                               Rules will be generated automatically, if not provided here.
        """
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
                        print(f'Dropping rule "{s}: {r}". Symbols missing from state: {gm.free_symbols(r).difference(varset)}')
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

        state_constraints = {}
        for n, c in constraints.items():
            if gm.is_symbol(c.expr):
                s  = gm.free_symbols(c.expr).pop()
                fs = gm.free_symbols(c.lower).union(gm.free_symbols(c.upper))
                if len(fs.difference({s})) == 0:
                    state_constraints[s] = (float(gm.subs(c.lower, {s: 0})), float(gm.subs(c.upper, {s: 0})))

        self.state_bounds = np.array([state_constraints[s] if  s in state_constraints else
                                     [-np.pi, np.pi] for s in self.ordered_vars])
        self.R = None # np.zeros((len(self.obs_labels), len(self.obs_labels)))

    def gen_control_vector(self, control_dict):
        """Generates a control vector from a given dict mapping symbols to values
        
        Args:
            control_dict (dict): Map symbol -> float

        Returns:
            np.ndarray: Vectorized constrol
        """
        return np.array([control_dict[s] for s in self.ordered_controls])

    def gen_obs_vector(self, obs_dict):
        """Generates a flat vector of observations from a dictionary of them.
        
        Args:
            obs_dict (dict): Dict of observations. Names need to match those 
                             given to __init__
        Returns:
            np.ndarray: Flat vector of observations
        """
        return np.hstack([np.take(obs_dict[l], i) for l, i in self.takers])

    def pandas_R(self):
        """Returns the measurement covariance matrix as pandas frame.
        
        Returns:
            pd.DataFrame: Measurement covariance as DataFrame if matrix is stored.
                          None otherwise.
        """
        if self.R is None:
            return None
        return pd.DataFrame(data=self.R, index=self.obs_labels, columns=self.obs_labels)

    @profile
    def predict(self, state_t, Sigma_t, control, dt=0.05):
        """Predicts the next state and covariance from current state, covariance, and
           control signal.
        
        Args:
            state_t (np.ndarray): Current state vector
            Sigma_t (np.ndarray): Current covariance
            control (nd.ndarray): Current control vector
            dt (float, optional): Size of the prediction step
        
        Returns:
            (np.ndarray, np.ndarray): predicted_state, predicted_covariance
        """
        params = np.hstack(([dt], state_t, control))
        F_t    = self.g_prime_fn.call2(params)
        return self.g_fn.call2(params), F_t.dot(Sigma_t.dot(F_t.T)) + self.Q

    @profile
    def update(self, state_t, Sigma_t, obs_t):
        """Performs the kalman update.
        
        Args:
            state_t (np.ndarray): Current state
            Sigma_t (np.ndarray): Current covariance
            obs_t (np.ndarray): Current observation
        
        Returns:
            (np.ndarray, np.ndarray): Updated state, updated covariance
        
        Raises:
            Exception: Will raise an exception if the measurement covariance R is not set
        """
        if self.R is None:
            raise Exception('No noise model set for EKF model.')

        H_t = self.h_prime_fn.call2(state_t)
        S_t = np.dot(H_t, np.dot(Sigma_t, H_t.T)) + self.R
        if np.linalg.det(S_t) != 0.0:
            K_t = Sigma_t.dot(H_t.T.dot(np.linalg.inv(S_t)))
            y_t = obs_t - self.h_fn.call2(state_t).flatten()
            state_t = (state_t + K_t.dot(y_t)).flatten()
            Sigma_t = (np.eye(Sigma_t.shape[0]) - K_t.dot(H_t)).dot(Sigma_t)
            return state_t, Sigma_t
        else:
            return state_t, Sigma_t

    def set_R(self, R):
        """Set the measurement covariance matrix R
        
        Args:
            R (np.ndarray): Measurement covariance matrix
        """
        self.R = R

    @profile
    def generate_R(self, noisy_observations):
        """Generates the covariance matrix from a set of noisy observations.
           Once generated, the matrix will be stored in the model.
        
        Args:
            noisy_observations ([dict]): List of observation dictionaries.
        """
        obs = np.vstack([self.gen_obs_vector(obs) for obs in noisy_observations])
        cov = np.cov(obs.T)
        self.set_R(cov)

    def spawn_particle(self):
        """Spawns a particle initialized to be in the center of the configuration space.
           The covariance is initialized as variance of the observed quantities.
        
        Returns:
            Particle: A particle modeling an estimate of the state of this model
        """
        return Particle((self.state_bounds.T[0] + self.state_bounds.T[1]) * 0.5,
                        np.diag(self.state_bounds.T[1] - self.state_bounds.T[0]) ** 2)

    def __str__(self):
        return 'EKF estimating:\n  {}\nFrom:\n  {}\nWith controls:\n  {}'.format(
                        '\n  '.join(str(s) for s in self.ordered_vars), 
                        '\n  '.join(l for l in self.obs_labels),
                        '\n  '.join(str(c) for c in self.ordered_controls))