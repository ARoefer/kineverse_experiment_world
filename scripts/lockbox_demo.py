#!/usr/bin/env python
from kineverse_experiment_world.scenario_base import Scenario,       \
                                                     Constraint,     \
                                                     SoftConstraint, \
                                                     ControlledValue
from kineverse.gradients.diff_logic           import Position, DiffSymbol
from kineverse.gradients.gradient_math        import subs,          \
                                                     alg_and,       \
                                                     greater_than,  \
                                                     less_than,     \
                                                     subs,          \
                                                     GC
from kineverse.type_sets               import symengine_types, symbolic_types
from kineverse.visualization.plotting  import split_recorders, draw_recorders, ValueRecorder, ColorGenerator
from kineverse.utils                   import res_pkg_path

from pprint import pprint

def sign(x):
    return -1 if x < 0 else (1 if x > 0 else 0)

class Lockbox(Scenario):
    def __init__(self, name):
        super(Lockbox, self).__init__('Lockbox - {}'.format(name))

        self.lock_a_p = Position('lock_a')
        self.lock_b_p = Position('lock_b')
        self.lock_c_p = Position('lock_c')
        self.lock_d_p = Position('lock_d')
        self.lock_e_p = Position('lock_e')
        self.lock_f_p = Position('lock_f')
        pos_symbols = {self.lock_a_p,
                       self.lock_b_p,
                       self.lock_c_p,
                       self.lock_d_p,
                       self.lock_e_p,
                       self.lock_f_p}
        self.pos_symbols_str = {str(s) for s in pos_symbols}

        self.lock_a_v = DiffSymbol(self.lock_a_p)
        self.lock_b_v = DiffSymbol(self.lock_b_p)
        self.lock_c_v = DiffSymbol(self.lock_c_p)
        self.lock_d_v = DiffSymbol(self.lock_d_p)
        self.lock_e_v = DiffSymbol(self.lock_e_p)
        self.lock_f_v = DiffSymbol(self.lock_f_p)
        vel_symbols = {self.lock_a_v, 
                       self.lock_b_v, 
                       self.lock_c_v, 
                       self.lock_d_v, 
                       self.lock_e_v, 
                       self.lock_f_v}
        self.vel_symbols_str = {str(s) for s in vel_symbols} 
        self.int_rules = {s: s for s in vel_symbols}

        b_open_threshold = 0.4
        c_open_threshold = 0.6
        d_open_threshold = 0.8
        e_open_threshold = 1.0
        f_open_threshold = 1.2

        # Locking rules
        # b and c lock a
        # d locks b and c
        # e locks c and d
        # f locks e
        a_open_condition  = alg_and(greater_than(self.lock_b_p, b_open_threshold), greater_than(self.lock_c_p, c_open_threshold))
        b_open_condition  = greater_than(self.lock_d_p, d_open_threshold)
        c_open_condition  = alg_and(greater_than(self.lock_d_p, b_open_threshold + 0.1),greater_than(self.lock_e_p, e_open_threshold))
        d_open_condition  = greater_than(self.lock_e_p, e_open_threshold - 0.1)
        e_open_condition  = greater_than(self.lock_f_p, f_open_threshold)

        self.lock_str_labels = [
            'a open = $b \\succ {} \\curlywedge c \\succ {}$'.format(b_open_threshold, c_open_threshold),
            'b open = $d \\succ {}$'.format(d_open_threshold),
            'c open = $d \\succ {} \\curlywedge e \\succ {}$'.format(b_open_threshold + 0.1, e_open_threshold),
            'd open = $e \\succ {}$'.format(e_open_threshold - 0.1),
            'e open = $f \\succ {}$'.format(f_open_threshold)]

        self.recorded_terms = dict(zip(self.lock_str_labels, [a_open_condition.expr,
                                                              b_open_condition.expr,
                                                              c_open_condition.expr,
                                                              d_open_condition.expr,
                                                              e_open_condition.expr]))

        # Velocity constraints
        self.km.add_constraint('lock_a_velocity', 
                                Constraint(-0.4 * a_open_condition, 0.4 * a_open_condition, self.lock_a_v))
        self.km.add_constraint('lock_b_velocity', 
                                Constraint(-0.1, 0.1 * b_open_condition, self.lock_b_v))
        self.km.add_constraint('lock_c_velocity', 
                                Constraint(-0.1, 0.1 * c_open_condition, self.lock_c_v))
        self.km.add_constraint('lock_d_velocity', 
                                Constraint(-0.1, 0.1 * d_open_condition, self.lock_d_v))
        self.km.add_constraint('lock_e_velocity', 
                                Constraint(-0.1, 0.1 * e_open_condition, self.lock_e_v))
        self.km.add_constraint('lock_f_velocity', 
                                Constraint(-0.4, 0.4, self.lock_f_v))

        # Configuration space
        self.km.add_constraint('lock_b_position', Constraint(-self.lock_b_p, 0.7  - self.lock_b_p, self.lock_b_p))
        self.km.add_constraint('lock_c_position', Constraint(-self.lock_c_p, 0.8  - self.lock_c_p, self.lock_c_p))
        self.km.add_constraint('lock_d_position', Constraint(-self.lock_d_p, 0.9 - self.lock_d_p, self.lock_d_p))
        self.km.add_constraint('lock_e_position', Constraint(-self.lock_e_p, 1.1  - self.lock_e_p, self.lock_e_p))


class LockboxOpeningGenerator(Lockbox):
    def __init__(self):
        super(LockboxOpeningGenerator, self).__init__('Generated Opening')

        self.start_state = {self.lock_a_p: 0,
                            self.lock_b_p: 0,
                            self.lock_c_p: 0,
                            self.lock_d_p: 0,
                            self.lock_e_p: 0,
                            self.lock_f_p: 0, #} # ,
                            self.lock_a_v: 0,
                            self.lock_b_v: 0,
                            self.lock_c_v: 0,
                            self.lock_d_v: 0,
                            self.lock_e_v: 0,
                            self.lock_f_v: 0}

        self.soft_constraints = lock_explorer(self.km, self.start_state, 
                                    {'open_a': SoftConstraint(1.2 - self.lock_a_p, 
                                                              1.2 - self.lock_a_p, 
                                                              1, 
                                                              self.lock_a_p)}, set())
        print('\n'.join(['{}:\n {}'.format(k, str(c)) for k, c in self.soft_constraints.items()]))
        total_symbols = set()
        for c in self.soft_constraints.values():
            total_symbols.update(c.expr.free_symbols)
        control_symbols = {DiffSymbol(s) for s in total_symbols}
        total_symbols.update(control_symbols)
        constraints = self.km.get_constraints_by_symbols(total_symbols)
        for n, c in constraints.items():
            if c.expr in control_symbols:
                self.controlled_values[str(c.expr)] = ControlledValue(c.lower, c.upper, c.expr, 0.001)
            else:
                self.hard_constraints[n] = c


    def run(self, integration_step=0.02, max_iterations=200):
        super(LockboxOpeningGenerator, self).run(integration_step, max_iterations)
        col_gen  = ColorGenerator()
        colors   = {x: col_gen.get_color_hex() for x in 'abcdef'}
        pos_data = {k[5]: d for k, d in self.value_recorder.data.items() if k in self.pos_symbols_str}
        vel_data = {k[5]: d for k, d in self.value_recorder.data.items() if k in self.vel_symbols_str}

        self.vel_recorder = ValueRecorder(None, *[('$\\dot{}{}{}$'.format('{', k, '}'), c) for k, c in colors.items()])
        self.vel_recorder.data = {'$\\dot{}{}{}$'.format('{', k, '}'): d for k, d in vel_data.items()}
        self.vel_recorder.set_grid(True)
        self.vel_recorder.compute_limits()
        self.vel_recorder.set_legend_location('center right')
        self.vel_recorder.set_xspace(-10, 220)
        self.vel_recorder.set_xlabels([])
        self.vel_recorder.set_ytitle('Velocity $\\dot{q}$')

        self.value_recorder.title = 'Lockbox - Generated Opening'
        self.value_recorder.set_grid(True)
        self.value_recorder.data   = {'${}$'.format(k): d for k, d in pos_data.items()}
        self.value_recorder.colors = {'${}$'.format(k): c for k, c in colors.items()}
        self.value_recorder.set_legend_location('center right')
        self.value_recorder.set_xspace(-10, 220)
        self.value_recorder.set_xlabels([])
        self.value_recorder.set_ytitle('Position $q$')

        self.symbol_recorder.title  = None
        self.symbol_recorder.colors = {k: self.value_recorder.colors[x] for k, x in zip(self.lock_str_labels, ['${}$'.format(x) for x in 'abcde'])}
        self.symbol_recorder.set_grid(True)
        self.symbol_recorder.set_ylabels(['locked', 'open'])
        self.symbol_recorder.set_legend_location('center right')
        self.symbol_recorder.set_xtitle('Iterations')
        self.symbol_recorder.set_xspace(-10, 220)


def lock_explorer(km, state, goals, generated_constraints):

    final_goals = goals.copy()
    for n, goal in goals.items():
        symbols = goal.expr.free_symbols
        goal_sign = sign(subs(goal.lower, state)) + sign(subs(goal.upper, state))
        if goal_sign == 0:  # Constraint is satisfied
            continue 

        goal_expr = goal.expr
        if type(goal_expr) != GC:
            goal_expr = GC(goal_expr)
        
        goal_expr.do_full_diff()

        diff_value = {s: subs(g, state) for s, g in goal_expr.gradients.items()}
        diff_sign  = {s: sign(g) * goal_sign for s, g in diff_value.items()}
        
        diff_symbols = set(diff_sign.keys())
        diff_constraints = km.get_constraints_by_symbols(diff_symbols)

        # Constraints constraining the DoF listed by symbol
        blocking_constraints = {s: {} for s in diff_symbols}

        # Iterate over constraints which directly constrain a symbol -> type(c.expr) == Symbol
        for n, c in {n: c for n, c in diff_constraints.items() if c.expr in diff_symbols}.items():
            s = c.expr
            c_upper = subs(c.upper, state)
            c_lower = subs(c.lower, state)
            sign_u  = sign(c_upper)
            sign_l  = sign(c_lower)
            
            # Check if constraint is blocking the DoF from moving in the desired direction
            if diff_sign[s] > 0 and sign_u <= 0:
                blocking_constraints[s][n] = c
            elif diff_sign[s] < 0 and sign_l >= 0:
                blocking_constraints[s][n] = c


        new_goals = {}
        # If all symbols are blocked from going in the desired direction
        if min([len(cd) for cd in blocking_constraints.values()]) > 0:
            for s, cd in blocking_constraints.items():
                for n, c in cd.items():
                    u_const_name = 'unlock {} upper bound'.format(n)
                    l_const_name = 'unlock {} lower bound'.format(n)
                    if diff_sign[s] > 0 and type(c.upper) in symbolic_types and u_const_name not in generated_constraints:
                        new_goals[u_const_name] = SoftConstraint(less_than(c.upper, 0), 1000, goal.weight, c.upper)
                        generated_constraints.add(u_const_name)
                    elif diff_sign[s] < 0 and type(c.lower) in symbolic_types and l_const_name not in generated_constraints:
                        new_goals[l_const_name] = SoftConstraint(-1000, -greater_than(c.lower, 0), goal.weight, c.lower)
                        generated_constraints.add(l_const_name)
        
        final_goals.update(lock_explorer(km, state, new_goals, generated_constraints))

    return final_goals


if __name__ == '__main__':
    scenario = LockboxOpeningGenerator()

    scenario.run(0.19)

    draw_recorders([scenario.value_recorder, scenario.vel_recorder, scenario.symbol_recorder], 4.0/9.0, 10, 3).savefig(res_pkg_path('package://kineverse_experiment_world/test/plots/lockbox_opening.png'))
