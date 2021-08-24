import rospy
import kineverse.gradients.gradient_math as gm
import math
import numpy as np
import traceback

from multiprocessing import RLock

from kineverse.motion.min_qp_builder import GeomQPBuilder  as GQPB, \
                                            TypedQPBuilder as TQPB, \
                                            SoftConstraint as SC, \
                                            generate_controlled_values, \
                                            depth_weight_controlled_values
from kineverse.model.frames          import Frame
from kineverse.model.paths           import Path
from kineverse.utils                 import union, \
                                            static_var_bounds

from sensor_msgs.msg    import JointState             as JointStateMsg
from kineverse_msgs.msg import ValueMap               as ValueMapMsg
from control_msgs.msg   import GripperCommand         as GripperCommandMsg, \
                               GripperCommandFeedback as GripperCommandFeedbackMsg

from kineverse_experiment_world.cascading_qp         import CascadingQP
from kineverse_experiment_world.push_demo_base       import PushingController
from kineverse_experiment_world.idle_controller      import IdleController
from kineverse_experiment_world.sixd_pose_controller import SixDPoseController
from kineverse_experiment_world.utils                import np_inverse_frame

class GripperWrapper(object):
    def sync_set_gripper_position(self, position, effort=50):
        raise NotImplementedError

def gen_dv_cvs(km, constraints, controlled_symbols):
    cvs, constraints = generate_controlled_values(constraints, controlled_symbols)
    cvs = depth_weight_controlled_values(km, cvs, exp_factor=1.02)
    print('\n'.join(f'{cv.symbol}: {cv.weight_id}' for _, cv in sorted(cvs.items())))
    return cvs, constraints


class ROSOpeningBehavior(object):
    def __init__(self, km, 
                       gripper_wrapper,
                       robot_prefix,
                       eef_path, ext_paths_and_poses, 
                       controlled_symbols, control_alias, cam_path=None,
                       weights=None, resting_pose=None, visualizer=None):
        self.km = km
        self.robot_prefix = robot_prefix
        self.gripper_wrapper = gripper_wrapper
        self.eef_path = eef_path
        self.cam_path = cam_path

        self.visualizer    = visualizer
        self.weights       = weights
        self.control_alias = control_alias
        self.controlled_symbols = controlled_symbols

        self.controller = None

        self._phase = 'homing'

        self._gripper_pos = None
        self._gripper_done_or_stalled = False

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
                                               None)

        self._build_ext_symbol_map(km, ext_paths_and_poses)

        self.pub_robot_command    = rospy.Publisher('~robot_command', JointStateMsg, queue_size=1, tcp_nodelay=True)
        self.pub_external_command = rospy.Publisher('~external_command', ValueMapMsg, queue_size=1, tcp_nodelay=True)

        self.sub_robot_js    = rospy.Subscriber('/joint_states',  JointStateMsg, callback=self.cb_robot_js, queue_size=1)
        self.sub_external_js = rospy.Subscriber('~external_js',   ValueMapMsg, callback=self.cb_external_js, queue_size=1)


    def _build_ext_symbol_map(self, km, ext_paths):
        # total_ext_symbols = set()
        # Map of path to set of symbols
        self._target_body_map = {}
        self._grasp_poses = {}
        self._var_upper_bound = {}

        for path, grasp_pose in ext_paths.items():
            data = km.get_data(path)
            if not isinstance(data, Frame):
                print(f'Given external path "{path}" is not a frame.')
                continue
            self._grasp_poses[path] = gm.dot(data.pose, grasp_pose)

            symbols = {s for s in gm.free_symbols(data.pose) if 'location' not in str(s)}

            static_bounded_vars, \
            static_bounds, \
            _ = static_var_bounds(km, symbols)

            print(static_bounds.shape)
            if len(static_bounds) == 0:
                constraints = km.get_constraints_by_symbols(symbols)
                raise Exception('None of {} seem to have static bounds. Constraints are:\n  '.format(
                                ', '.join(str(s) for s in symbols),
                                '\n  '.join(f'{n} : {c}' for n, c in constraints.items())))

            self._var_upper_bound.update(zip(static_bounded_vars, static_bounds[:, 1]))

            # total_ext_symbols.update(gm.free_symbols(data.pose))
            self._target_body_map[path] = set(static_bounded_vars)

        list_sets = list(self._target_body_map.values())
        for x, s in enumerate(list_sets):
            # Only keep symbols unique to one target
            s.difference_update(list_sets[:x] + list_sets[x + 1:])

        for p, symbols in self._target_body_map.items():
            for s in symbols:
                self._target_map[s] = p

        print('Monitored symbols are:\n  {}'.format('\n  '.join(str(s) for s in self._target_map.keys())))


    def behavior_update(self):
        if self.controller is not None:
            now = rospy.Time.now()
            with self._state_lock:
                deltaT = 0.05 if self._last_controller_update is None else (now - self._last_controller_update).to_sec()
                try:
                    command = self.controller.get_cmd(self._state, deltaT=deltaT)
                except Exception as e:
                    print(traceback.format_exc())
                    rospy.signal_shutdown('die lol')

            self._robot_cmd_msg.header.stamp = now
            self._robot_cmd_msg.name, self._robot_cmd_msg.velocity = zip(*[(self.control_alias[s], v) for s, v in command.items() 
                                                                                                       if s in self.control_alias])
            self.pub_robot_command.publish(self._robot_cmd_msg)
            self._last_controller_update = now

        # Lets not confuse the tracker
        if self._phase != 'opening' and self._last_external_cmd_msg is not None: 
            # Ensure that velocities are assumed to be 0 when not operating anything
            self._last_external_cmd_msg.value = [0]*len(self._last_external_cmd_msg.value)
            self.pub_external_command.publish(self._last_external_cmd_msg)
            self._last_external_cmd_msg = None

        if   self._phase == 'idle':
            # if there is no controller, instantiate idle

            # check for new target
            # -> open gripper
            #    instantiate 6d controller
            #    switch to "grasping"
            if self.controller is None:
                self.gripper_wrapper.sync_set_gripper_position(0.07)
                self.controller = self._idle_controller

            with self._state_lock:
                for s, p in self._target_map.items():
                    if s in self._state and self._state[s] < self._var_upper_bound[s] * 0.7: # Some thing in the scene is closed
                        self._current_target = p
                        print(f'New target is {self._current_target}')
                        self.gripper_wrapper.sync_set_gripper_position(0.07)
                        print('Gripper is open. Proceeding to grasp...')
                        self.controller = SixDPoseController(self.km,
                                                             self.km.get_data(self.eef_path).pose,
                                                             self._grasp_poses[p],
                                                             self.controlled_symbols,
                                                             self.weights)
                        self._phase = 'grasping'
                        print(f'Now entering {self._phase} state')
                        break
                else: 
                    print('None of the monitored symbols are closed:\n  {}'.format('\n  '.join(f'{self._state[s]} < {self._var_upper_bound[s]}' for s in self._target_map.keys())))
        elif self._phase == 'grasping':
            # check if grasp is acheived
            # -> close gripper
            #    instantiate cascading controller
            #    switch to "opening"
            # if there is no more command but the goal error is too great -> "homing"
            if self.controller.equilibrium_reached(0.02, -0.02):
                if self.controller.current_error() > 0.01:
                    self.controller = self._idle_controller
                    self._phase = 'homing'
                    print(f'Now entering {self._phase} state')
                else:
                    print('Closing gripper...')
                    self.gripper_wrapper.sync_set_gripper_position(0)
                    print('Generating opening controller')
                    
                    eef = self.km.get_data(self.eef_path)
                    obj = self.km.get_data(self._current_target)
                    with self._state_lock:
                        static_eef_pose    = gm.subs(eef.pose, self._state)
                        static_object_pose = gm.subs(obj.pose, self._state)

                    offset_pose = np_inverse_frame(static_object_pose).dot(static_eef_pose)
                    print(offset_pose)

                    goal_pose   = gm.dot(obj.pose, gm.Matrix(offset_pose))

                    goal_pos_error = gm.norm(gm.pos_of(eef.pose) - gm.pos_of(goal_pose))
                    goal_rot_error = gm.norm(eef.pose[:3, :3] - goal_pose[:3, :3])

                    target_symbols = self._target_body_map[self._current_target]
                    lead_goal_constraints = {f'open_{s}': SC(self._var_upper_bound[s] - s,
                                                             1000, 1, s) for s in target_symbols if s in self._var_upper_bound}
                    for n, c in lead_goal_constraints.items():
                        print(f'{n}: {c}')

                    follower_goal_constraints = {'keep position': SC(-goal_pos_error,
                                                                     -goal_pos_error, 10, goal_pos_error),
                                                 'keep rotation': SC(-goal_rot_error, -goal_rot_error, 1, goal_rot_error)}

                    blacklist = {gm.Velocity(Path('pr2/torso_lift_joint'))}.union({gm.DiffSymbol(s) for s in gm.free_symbols(obj.pose) if 'location' in str(s)})

                    self.controller = CascadingQP(self.km, 
                                                  lead_goal_constraints, 
                                                  follower_goal_constraints, 
                                                  f_gen_follower_cvs=gen_dv_cvs,
                                                  controls_blacklist=blacklist,
                                                  t_follower=GQPB
                                                  )
                    print(self.controller)
                    # self.gripper_wrapper.sync_set_gripper_position(0.07)
                    # rospy.signal_shutdown('die lol')
                    # return
                    self._phase = 'opening'
                    print(f'Now entering {self._phase} state')
        elif self._phase == 'opening':
            # Wait for monitored symbols to be in open position
            # -> open gripper
            #    generate 6d retraction goal: -10cm in tool frame
            #    spawn 6d controller
            #    switch to "retracting"
            external_command = {s: v for s, v in command.items() if s not in self.controlled_symbols}
            ext_msg = ValueMapMsg()
            ext_msg.header.stamp = now
            ext_msg.symbol, ext_msg.value = zip(*[(str(s), v) for s, v in external_command.items()])
            self.pub_external_command.publish(ext_msg)
            self._last_external_cmd_msg = ext_msg

            with self._state_lock:
                current_external_error = {s: self._state[s] for s in self._target_body_map[self._current_target]}
            print('Current error state:\n  {}'.format('\n  '.join([f'{s}: {v}' for s, v in current_external_error.items()])))
            if min([v >= self._var_upper_bound[s] * 0.8 for s, v in current_external_error.items()]):
                print('Target fulfilled. Setting it to None')
                self._current_target = None
                
                eef = self.km.get_data(self.eef_path)
                with self._state_lock:
                    static_eef_pose = gm.subs(eef.pose, self._state)
                goal_pose = static_eef_pose.dot(np.array([[1, 0, 0, -0.07],
                                                          [0, 1, 0, 0],
                                                          [0, 0, 1, 0],
                                                          [0, 0, 0, 1]]))

                self.controller = SixDPoseController(self.km,
                                                     eef.pose,
                                                     goal_pose,
                                                     self.controlled_symbols,
                                                     self.weights)

                self.gripper_wrapper.sync_set_gripper_position(0.07)
                self._phase = 'retracting'
                print(f'Now entering {self._phase} state')
        elif self._phase == 'retracting':
            # Wait for retraction to complete
            # -> spawn idle controller
            #    switch to "homing"
            if self.controller.equilibrium_reached(0.02, -0.02):
                self.controller = self._idle_controller
                self._phase = 'homing'
                print(f'Now entering {self._phase} state')
        elif self._phase == 'homing':
            # Wait for idle controller to have somewhat low error
            # -> switch to "idle"
            if self.controller is None:
                self.gripper_wrapper.sync_set_gripper_position(0.07)
                self.controller = self._idle_controller
                return

            if self.controller.equilibrium_reached(0.1, -0.1):
                self._phase = 'idle'
                print(f'Now entering {self._phase} state')
        else:
            raise Exception(f'Unknown state "{self._phase}')


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
