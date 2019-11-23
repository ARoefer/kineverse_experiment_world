import rospy
from collections import namedtuple

from multiprocessing import RLock

from kineverse.gradients.gradient_math      import spw,\
                                                   rotation3_axis_angle,\
                                                   frame3_axis_angle, \
                                                   frame3_quaternion, \
                                                   norm, \
                                                   rot_of, \
                                                   pos_of, \
                                                   vector3, \
                                                   Position
from kineverse.gradients.diff_logic         import get_diff_symbol, erase_type
from kineverse.model.paths                  import Path
from kineverse.motion.min_qp_builder        import SoftConstraint as SC,\
                                                   TypedQPBuilder as TQPB, \
                                                   generate_controlled_values
from kineverse.motion.integrator            import CommandIntegrator
from kineverse.network.model_client         import ModelClient
from kineverse.visualization.ros_visualizer import ROSVisualizer
from kineverse.time_wrapper                 import Time

from kineverse_experiment_world.msg import PoseStampedArray as PSAMsg
from sensor_msgs.msg                import JointState       as JointStateMsg

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
        self.soft_constraints = {}
        self.joints        = set()
        self.joint_aliases = {}
        
        self.visualizer = ROSVisualizer('/tracker_vis', 'map')

        self.pub_js  = rospy.Publisher(js_topic,    JointStateMsg, queue_size=1)
        self.sub_obs = rospy.Subscriber(obs_topic, PSAMsg, self.cb_process_obs, queue_size=5)


    def track(self, model_path, alias):
        with self.lock:
            if type(model_path) is not str:
                model_path = str(model_path)

            start_tick = len(self.tracked_poses) == 0

            if model_path not in self.tracked_poses:
                syms = [Position(Path(model_path) + (x,)) for x in ['ax', 'ay', 'az', 'x', 'y', 'z']]
                def rf(msg, state):
                    matrix = frame3_quaternion(msg.position.x, msg.position.y, msg.position.z, 
                                               msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
                    axis, angle = spw.axis_angle_from_matrix(matrix)
                    state[syms[0]] = axis[0] * angle
                    state[syms[1]] = axis[1] * angle
                    state[syms[2]] = axis[2] * angle
                    state[syms[3]] = msg.position.x
                    state[syms[4]] = msg.position.y
                    state[syms[5]] = msg.position.z

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

    @profile
    def cb_tick(self, timer_event):
        with self.lock:
            if self.integrator is not None:
                self.integrator.run(self._integration_factor, self._iterations)
                self._js_msg.header.stamp = Time.now()
                self._js_msg.name, self._js_msg.position = zip(*[(n, self.integrator.state[s]) for s, n in self.joint_aliases.items()])
                self.pub_js.publish(self._js_msg)

                if self.visualizer is not None:
                    poses = [t.pose.subs(self.integrator.state) for t in self.tracked_poses.values()]
                    self.visualizer.begin_draw_cycle('obs_points')
                    self.visualizer.draw_poses('obs_points', spw.eye(4), 0.1, 0.02, [p for p in poses if len(p.free_symbols) == 0])
                    self.visualizer.render()

            else:
                print('Integrator does not exist')

    @profile
    def cb_process_obs(self, poses_msg):
        #print('Got new observation')
        #with self.lock:
        for pose_msg in poses_msg.poses:
            if pose_msg.header.frame_id in self.aliases and self.integrator is not None:
                self.tracked_poses[self.aliases[pose_msg.header.frame_id]].update_state(pose_msg.pose, self.integrator.state)


    def _generate_pose_constraints(self, str_path, model):
        with self.lock:
            if str_path in self.tracked_poses:
                align_rotation = '{} align rotation'.format(str_path)
                align_position = '{} align position'.format(str_path)
                if model is not None:
                    te = self.tracked_poses[str_path]

                    self.joints |= model.pose.free_symbols
                    self.joint_aliases = {s: Path(erase_type(s))[-1] for s in self.joints}

                    axis, angle = spw.axis_angle_from_matrix(rot_of(model.pose).T * rot_of(te.pose))
                    r_rot_control = axis * angle

                    hack = rotation3_axis_angle([0, 0, 1], 0.0001)

                    axis, angle = spw.axis_angle_from_matrix((rot_of(model.pose).T * (rot_of(model.pose) * hack)).T)
                    c_aa = (axis * angle)

                    r_dist = norm(r_rot_control - c_aa)

                    self.soft_constraints[align_rotation] = SC(-r_dist, -r_dist, 1, r_dist)
                
                    dist = norm(pos_of(model.pose) - pos_of(te.pose))
                    self.soft_constraints[align_position] = SC(-dist, -dist, 1, dist)
                    self.generate_opt_problem()
                else:
                    regenerate_problem = False
                    if align_position in self.soft_constraints:
                        del self.soft_constraints[align_position]
                        regenerate_problem = True

                    if align_rotation in self.soft_constraints:
                        del self.soft_constraints[align_rotation]
                        regenerate_problem = True
                
                    if regenerate_problem:
                        self.generate_opt_problem()


    def generate_opt_problem(self):
        joint_symbols    = self.joints
        opt_symbols      = {get_diff_symbol(j) for j in joint_symbols}
        hard_constraints = self.km_client.get_constraints_by_symbols(joint_symbols | opt_symbols)

        controlled_values, hard_constraints = generate_controlled_values(hard_constraints, opt_symbols)

        self.integrator = CommandIntegrator(TQPB(hard_constraints, self.soft_constraints, controlled_values))
        self.integrator.restart('Pose Tracking')