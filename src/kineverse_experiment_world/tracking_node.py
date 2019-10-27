import rospy
from collections import namedtuple

from multiprocessing import RLock

from kineverse.gradients.gradient_math import spw,\
                                              rotation3_axis_angle,\
                                              frame3_axis_angle, \
                                              frame3_quaternion, \
                                              norm, \
                                              rot_of, \
                                              pos_of, \
                                              vector3
from kineverse.gradients.diff_logic    import get_diff_symbol
from kineverse.model.paths             import Path
from kineverse.motion.min_qp_builder   import SoftConstraint as SC,\
                                              TypedQPBuilder as TQPB
from kineverse.motion.integrator       import CommandIntegrator
from kineverse.network.model_client    import ModelClient

from kineverse_experiment_world.msg import PoseStampedArray as PSAMsg
from sensor_msgs.msg   import JointState  as JointStateMsg

TrackerEntry = namedtuple('TrackerEntry', ['pose', 'update_state', 'model_cb'])

class TrackerNode(object):
    def __init__(self, js_topic, obs_topic, integration_factor=0.05, iterations=30):
        self.km_client = ModelClient(None)

        self._js_msg = JointStateMsg()
        self._integration_factor = integration_factor
        self._iterations = iterations

        self.aliases = {}
        self.tracked_poses = {}
        self.integrator    = None
        self.lock          = RLock()
        
        self.pub_js  = rospy.Publisher(js_topic,    JointStateMsg, queue_size=1)
        self.sub_obs = rospy.Subscriber(obs_topic, PSAMsg, self.cb_process_obs, queue_size=5)


    def track(self, model_path, alias):
        with self.lock:
            if type(model_path) is not str:
                model_path = str(model_path)

            start_tick = len(self.tracked_poses) == 0

            if model_path not in self.tracked_poses:
                syms = [(Path(model_path) + (x,)).to_symbol() for x in ['ax', 'ay', 'az', 'x', 'y', 'z']]
                def rf(msg, state):
                    matrix = frame3_quaternion(msg.position.x, msg.position.y, msg.position.z, 
                                               msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
                    axis, angle = spw.axis_angle_from_matrix(matrix)
                    state[syms[0]] = axis[0] * angle
                    state[syms[1]] = axis[1] * angle
                    state[syms[2]] = axis[2] * angle
                    state[syms[3]] = o[0, 3]
                    state[syms[4]] = o[1, 3]
                    state[syms[5]] = o[2, 3]

                def process_model_update(data):
                    self._generate_pose_constraints(model_path, data)

                axis = vector3(*syms[:3])
                self.aliases[alias]  = model_path 
                self.tracked_poses[model_path] = TrackerEntry(frame3_axis_angle(axis / (norm(axis) + 1e-7), norm(axis), syms[3:]), rf, process_model_update)

                self.km_client.register_on_model_changed(Path(model_path), process_model_update)

            if start_tick:
                self.timer = rospy.Timer(rospy.Duration(1.0 / 50), self.cb_tick)


    def stop_tracking(self, model_path):
        with self.lock:
            if type(model_path) is not str:
                model_path = str(model_path)

            if model_path in self.tracked_poses:
                te = self.tracked_poses[model_path]
                self.km_client.deregister_on_model_changed(te.model_cb)
                del self.tracked_poses[model_path]
                del self.soft_constraints['{} align rotation 0'.format(model_path)]
                del self.soft_constraints['{} align rotation 1'.format(model_path)]
                del self.soft_constraints['{} align rotation 2'.format(model_path)]
                del self.soft_constraints['{} align position'.format(model_path)]
                self.generate_opt_problem()

                if len(self.tracked_poses) == 0:
                    self.timer.shutdown()


    def cb_tick(self, timer_event):
        with self.lock:
            if self.integrator is not None:
                self.integrator.run(self._integration_factor, self._iterations)
                self._js_msg.header.stamp = rospy.Time.now()
                self._js_msg.name, self._js_msg.position = zip(*[(str(Path(s)), v) for s, v in self.integrator.state.items()])
                self.pub_js.publish(self._js_msg)
            else:
                print('Integrator does not exist')


    def cb_process_obs(self, poses_msg):
        #print('Got new observation')
        with self.lock:
            for pose_msg in poses_msg.poses:
                if pose_msg.header.frame_id in self.aliases and self.integrator is not None:
                    self.tracked_poses[self.aliases[pose_msg.header.frame_id]].update_state(pose_msg.pose, self.integrator.state)


    def _generate_pose_constraints(self, str_path, model):
        if str_path in self.tracked_poses:
            te = self.tracked_poses[str_path]

            axis, angle = axis_angle_from_matrix((rot_of(model).T * te.pose))
            r_rot_control = axis * angle

            hack = rotation3_axis_angle([0, 0, 1], 0.0001)

            axis, angle = axis_angle_from_matrix(rot_of(model) * hack)
            c_aa = (axis * angle)

            self.soft_constraints['{} align rotation 0'.format(str_path)] =                                                     SC(r_rot_control[0],
                                                       r_rot_control[0],
                                                       1,
                                                       c_aa[0])
            self.soft_constraints['{} align rotation 1'.format(str_path)] =                                                     SC(r_rot_control[1],
                                                       r_rot_control[1],
                                                       1,
                                                       c_aa[1])
            self.soft_constraints['{} align rotation 2'.format(str_path)] =                                                     SC(r_rot_control[2],
                                                       r_rot_control[2],
                                                       1,
                                                       c_aa[2])
        
            dist = norm(pos_of(model) - pos_of(te.pose))
            self.soft_constraints['{} align position'.format(str_path)] = SC(-dist, -dist, 1, dist)

            self.generate_opt_problem()


    def generate_opt_problem(self):
        joint_symbols = set(sum([list(p.free_symbols) for p in self.tracked_poses.values()]))
        opt_symbols      = {get_diff_symbol(j) for j in joint_symbols}
        hard_constraints = self.km_client.get_constraints_by_symbols(joint_symbols | opt_symbols)

        controlled_values = {}
        to_remove = set()
        for k, c in hard_constraints.items():
            if type(c.expr) is spw.Symbol and c.expr in controlled_symbols:
                controlled_values[str(c.expr)] = ControlledValue(c.lower, c.upper, c.expr, 0.01)
                to_remove.add(k)

        hard_constraints = {k: c for k, c in hard_constraints.items() if k not in to_remove}
        for s in controlled_symbols:
            if str(s) not in controlled_values:
                controlled_values[str(s)] = ControlledValue(-1e9, 1e9, s, 0.01)

        self.integrator = CommandIntegrator(TQPB(hard_constraints, self.soft_constraints, controlled_values))
        self.integrator.restart('Pose Tracking')