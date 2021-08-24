import kineverse.gradients.gradient_math as gm
import math

from kineverse.time_wrapper import Time

from kineverse.motion.min_qp_builder import TypedQPBuilder as TQPB,        \
                                            SoftConstraint as SC,          \
                                            generate_controlled_values,    \
                                            depth_weight_controlled_values

class IdleController(object):
    def __init__(self, km, controlled_symbols, resting_pose, camera_path=None):
        tucking_constraints = {}
        if resting_pose is not None:
            tucking_constraints = {f'tuck {s}': SC(p - s, p - s, 1, s) for s, p in resting_pose.items()}
            # print('Tuck state:\n  {}\nTucking constraints:\n  {}'.format('\n  '.join(['{}: {}'.format(k, v) for k, v in self._resting_pose.items()]), '\n  '.join(tucking_constraints.keys())))

        # tucking_constraints.update(self.taxi_constraints)

        self.use_camera = camera_path is not None
        if camera_path is not None:
            self._poi_pos = gm.Symbol('poi')
            poi = gm.point3(1.5, 0.5, 0.0) + gm.vector3(0, self._poi_pos * 2.0, 0)

            camera = km.get_data(camera_path)
            cam_to_poi = poi - gm.pos_of(camera.pose)
            lookat_dot = 1 - gm.dot_product(gm.x_of(camera.pose), cam_to_poi) / gm.norm(cam_to_poi)
            tucking_constraints['sweeping gaze'] = SC(-lookat_dot * 5, -lookat_dot * 5, 1, lookat_dot)

        symbols = set()
        for c in tucking_constraints.values():
            symbols |= gm.free_symbols(c.expr)

        joint_symbols      = {s for s in symbols if gm.get_symbol_type(s) != gm.TYPE_UNKNOWN}
        controlled_symbols = {gm.DiffSymbol(s) for s in joint_symbols}
        
        hard_constraints = km.get_constraints_by_symbols(symbols.union(controlled_symbols))

        controlled_values, hard_constraints = generate_controlled_values(hard_constraints, controlled_symbols)
        controlled_values = depth_weight_controlled_values(km, controlled_values)

        self.qp = TQPB(hard_constraints, tucking_constraints, controlled_values)
        self._start = Time.now()

    def get_cmd(self, state, deltaT):
        _state = state.copy()
        if self.use_camera:
            _state[self._poi_pos] = math.sin((Time.now() - self._start).to_sec())

        return self.qp.get_cmd(_state)

    def current_error(self):
        return self.qp.latest_error

    def equilibrium_reached(self, low_eq=1e-3, up_eq=-1e-3):
        return self.qp.equilibrium_reached(low_eq, up_eq)