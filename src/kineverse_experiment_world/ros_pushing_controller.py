import rospy
import kineverse.gradients.gradient_math as gm
import math

from multiprocessing import RLock

from kineverse.motion.min_qp_builder import TypedQPBuilder as TQPB, \
                                            SoftConstraint as SC, \
                                            generate_controlled_values, \
                                            depth_weight_controlled_values
from kineverse.model.geometry_model  import RigidBody

from sensor_msgs.msg    import JointState as JointStateMsg
from kineverse_msgs.msg import ValueMap   as ValueMapMsg

from kineverse_experiment_world.push_demo_base  import PushingController
from kineverse_experiment_world.idle_controller import IdleController


class ROSPushingBehavior(object):
    def __init__(self, km, 
                       robot_prefix,
                       eef_path, ext_paths, 
                       controlled_symbols, control_alias, cam_path=None,
                       weights=None, resting_pose=None,
                       navigation_method='linear', visualizer=None):
        self.km = km
        self.robot_prefix = robot_prefix
        self.eef_path = eef_path
        self.cam_path = cam_path

        self.visualizer    = visualizer
        self.weights       = weights
        self.control_alias = control_alias
        self.controlled_symbols = controlled_symbols
        self.navigation_method  = navigation_method

        self.controller = None

        self._state_lock = RLock()
        self._state = {}
        self._target_map = {}
        self._target_body_map  = {}
        self._last_controller_update = None

        self._current_target = None
        self._robot_cmd_msg  = JointStateMsg()
        self._robot_cmd_msg.name = list(control_alias.keys())
        self._last_external_cmd_msg = None

        self._idle_controller = IdleController(km, 
                                               self.controlled_symbols,
                                               resting_pose,
                                               cam_path)

        self._build_ext_symbol_map(km, ext_paths)

        self.pub_robot_command    = rospy.Publisher('~robot_command', JointStateMsg, queue_size=1, tcp_nodelay=True)
        self.pub_external_command = rospy.Publisher('~external_command', ValueMapMsg, queue_size=1, tcp_nodelay=True)

        self.sub_robot_js    = rospy.Subscriber('/joint_states', JointStateMsg, callback=self.cb_robot_js, queue_size=1)
        self.sub_external_js = rospy.Subscriber('~external_js',  ValueMapMsg, callback=self.cb_external_js, queue_size=1)


    def _build_ext_symbol_map(self, km, ext_paths):
        # total_ext_symbols = set()
        # Map of path to set of symbols
        self._target_body_map = {}

        for path in ext_paths:
            data = km.get_data(path)
            if not isinstance(data, RigidBody):
                print(f'Given external path "{path}" is not a rigid body.')
                continue

            # total_ext_symbols.update(gm.free_symbols(data.pose))
            self._target_body_map[path] = {s for s in gm.free_symbols(data.pose) if 'location' not in str(s)}

        list_sets = list(self._target_body_map.values())
        for x, s in enumerate(list_sets):
            # Only keep symbols unique to one target
            s.difference_update(list_sets[:x] + list_sets[x + 1:])

        for p, symbols in self._target_body_map.items():
            for s in symbols:
                self._target_map[s] = p


    def behavior_update(self):
        if self._current_target is None:
            with self._state_lock:
                for s, p in self._target_map.items():
                    if s in self._state and self._state[s] > 2e-2: # Some thing in the scene is open
                        self._current_target = p
                        self.controller = PushingController(self.km,
                                                            self.eef_path,
                                                            self._current_target,
                                                            self.controlled_symbols,
                                                            self._state,
                                                            self.cam_path,
                                                            self.navigation_method,
                                                            self.visualizer,
                                                            self.weights)
                        print(f'New target is {self._current_target}')
                        break
                else: # Nothing to be done
                    if self.controller is None: # Create an idle controller if none exists
                        self.controller = self._idle_controller
        else:
            # Reset target to None if goal is satisfied
            with self._state_lock:
                current_external_error = {s: self._state[s] for s in self._target_body_map[self._current_target]}
            print('Current error state:\n  {}'.format('\n  '.join([f'{s}: {v}' for s, v in current_external_error.items()])))
            if min([v <= 2e-2 for v in current_external_error.values()]):
                print('Target fulfilled. Setting it to None')
                self._current_target = None
                self.controller = None
                return

        if self.controller is None:
            return

        now = rospy.Time.now()
        with self._state_lock:
            deltaT  = 0.05 if self._last_controller_update is None else (now - self._last_controller_update).to_sec()
            command = self.controller.get_cmd(self._state, deltaT=deltaT)

        self._robot_cmd_msg.header.stamp = now
        self._robot_cmd_msg.name, self._robot_cmd_msg.velocity = zip(*[(self.control_alias[s], v) for s, v in command.items() 
                                                                                                   if s in self.control_alias])
        self.pub_robot_command.publish(self._robot_cmd_msg)

        if self._current_target is not None: # Implies there is some external command
            external_command = {s: v for s, v in command.items() if s not in self.controlled_symbols}
            ext_msg = ValueMapMsg()
            ext_msg.header.stamp = now
            ext_msg.symbol, ext_msg.value = zip(*[(str(s), v) for s, v in external_command.items()])
            self.pub_external_command.publish(ext_msg)
            self._last_external_cmd_msg = ext_msg
        elif self._last_external_cmd_msg is not None: 
            # Ensure that velocities are assumed to be 0 when not operating anything
            self._last_external_cmd_msg.value = [0]*len(self._last_external_cmd_msg.value)
            self.pub_external_command.publish(self._last_external_cmd_msg)
            self._last_external_cmd_msg = None


    def cb_robot_js(self, js_msg):
        state_update = {}
        for x, name in enumerate(js_msg.name):
            if len(js_msg.position) > x:
                state_update[gm.Position(self.robot_prefix + (name,))] = js_msg.position[x]
            if len(js_msg.velocity) > x:
                state_update[gm.Velocity(self.robot_prefix + (name,))] = js_msg.velocity[x]

        with self._state_lock:
            self._state.update(state_update)

        self.behavior_update()

    def cb_external_js(self, value_msg):
        state_update = {gm.Symbol(s): v for s, v in zip(value_msg.symbol, value_msg.value)}

        with self._state_lock:
            self._state.update(state_update)


