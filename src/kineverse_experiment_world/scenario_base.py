from kineverse.model.geometry_model  import GeometryModel, \
                                            Constraint
from kineverse.motion.integrator     import CommandIntegrator
from kineverse.motion.min_qp_builder import TypedQPBuilder  as TQPB,  \
                                            SoftConstraint,           \
                                            ControlledValue


class Scenario(object):
    def __init__(self, name):
        self.name = name
        self.km   = GeometryModel()
        self.qp_type           = TQPB
        self.int_rules         = {}
        self.recorded_terms    = {}
        self.hard_constraints  = {}
        self.soft_constraints  = {}
        self.controlled_values = {}
        self.start_state       = {}
        self.value_recorder    = None
        self.symbol_recorder   = None

    def run(self, integration_step=0.02, max_iterations=200):
        integrator = CommandIntegrator(self.qp_type(self.hard_constraints, self.soft_constraints, self.controlled_values), self.int_rules, self.start_state, self.recorded_terms)

        integrator.restart(self.name)
        integrator.run(integration_step, max_iterations)
        self.value_recorder  = integrator.recorder
        self.symbol_recorder = integrator.sym_recorder