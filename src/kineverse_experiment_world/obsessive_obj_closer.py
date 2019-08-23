import rospy

from kineverse.gradients.gradient_math       import *
from kineverse.gradients.diff_logic          import erase_type, get_symbol_type, TYPE_POSITION
from kineverse.model.paths                   import Path
from kineverse.model.geometry_model          import GeometryModel, closest_distance, closest_distance_constraint
from kineverse.motion.min_qp_builder         import GeomQPBuilder as GQPB,\
                                                    SoftConstraint as SC,\
                                                    ControlledValue
from kineverse.network.model_client          import ModelClient
from kineverse.operations.urdf_operations    import URDFRobot
from kineverse.operations.special_kinematics import RoombaJoint
from kineverse.visualization.bpb_visualizer  import ROSBPBVisualizer

from nav_msgs.msg    import Odometry   as OdometryMsg
from sensor_msgs.msg import JointState as JointStateMsg

goal_joints = {'prismatic', 'revolute'}


class ObsessiveObjectCloser(object):
    def __init__(self, urdf_path, robot_eef, robot_camera, obj_js_topic, robot_topic_prefix, waiting_location=(0,0,0,0)):
        self.km = ModelClient(GeometryModel)
        self.robot_eef_path     = robot_eef
        self.robot_camera_path  = robot_camera
        self.robot              = None
        self.robot_eef          = None
        self.robot_camera       = None
        self.state = {}
        self.robot_js_aliases   = None
        self.inverse_js_aliases = {}
        self.obj_js_aliases     = None
        self._current_target    = None
        self.waiting_location   = point3(*waiting_location[:3])
        self.waiting_direction  = vector3(cos(waiting_location[3]), sin(waiting_location[3]), 0)
        self.idle_controller    = None
        self.pushing_controller = None
        self._needs_odom        = False
        self._has_odom          = False

        self.debug_visualizer = ROSBPBVisualizer('/debug_vis', 'map')

        self.urdf_path = urdf_path
        self.obj = None

        self.km.register_on_model_changed(self.robot_eef_path[:-2], self.cb_robot_model_changed)
        self.km.register_on_model_changed(self.urdf_path, self.cb_obj_model_changed)

        self.pub_cmd = rospy.Publisher('{}/commands/joint_velocities'.format(robot_topic_prefix), JointStateMsg, queue_size=1)

        self.sub_obj_js   = rospy.Subscriber(obj_js_topic, JointStateMsg, self.cb_obj_joint_state,  queue_size=1)
        self.sub_odom     = rospy.Subscriber('{}/odometry'.format(robot_topic_prefix), OdometryMsg, self.cb_robot_odometry, queue_size=1)
        self.sub_robot_js = rospy.Subscriber('{}/joint_states'.format(robot_topic_prefix), JointStateMsg, self.cb_robot_joint_state, queue_size=1)

    @profile
    def cb_obj_model_changed(self, model):
        print('URDF model changed')
        if type(model) is not URDFRobot:
            raise Exception('Path "{}" does not refer to a urdf_robot. Type is "{}"'.format(self.urdf_path, type(model)))

        self.obj = model
        self.possible_targets = [(Path(joint.child), joint.position) for jname, joint in self.obj.joints.items() if joint.type in goal_joints]
        self.target_symbol_map = dict(self.possible_targets)
        new_state = {s: 0.0 for _, s in self.possible_targets if type(s) is spw.Symbol}
        self.obj_js_aliases = {str(Path(erase_type(s))[len(self.urdf_path):]): s for s in new_state.keys()}
        print('Object aliases:\n{}'.format('\n'.join(['{:>20}: {}'.format(k, v) for k, v in self.obj_js_aliases.items()])))
        self.inverse_js_aliases.update({erase_type(v): k for k, v in self.obj_js_aliases.items()})
        self.state.update(new_state)

    @profile
    def cb_robot_model_changed(self, model):
        print('Robot model changed')
        self.robot = model
        self.base_joint = [j for j in self.robot.joints.values() if j.parent == 'map'][0]
        self.base_link = Path(self.base_joint.child)[len(self.robot_eef_path) - 2:].get_data(self.robot)
        self.robot_camera = self.robot_camera_path[len(self.robot_camera_path) - 2:].get_data(self.robot)
        self.robot_eef    = self.robot_eef_path[len(self.robot_eef_path) - 2:].get_data(self.robot)

        new_state = {s: 0.0 for s in self.robot_camera.pose.free_symbols.union(self.robot_eef.pose.free_symbols) if get_symbol_type(s) == TYPE_POSITION}

        self.robot_joint_symbols = {j.position for j in self.robot.joints.values() if hasattr(j, 'position') and type(j.position) is spw.Symbol}
        new_state.update({s: 0.0 for s in self.robot_joint_symbols})
        self.robot_controlled_symbols = {get_diff_symbol(j) for j in self.robot_joint_symbols}
        self.state.update(new_state)

        if type(self.base_joint) is RoombaJoint:
            self.robot_controlled_symbols.update({self.base_joint.lin_vel, self.base_joint.ang_vel})
            self.robot_joint_symbols.update({self.base_joint.x_pos, self.base_joint.y_pos, self.base_joint.z_pos, self.base_joint.a_pos})
            self._needs_odom = True

        common_prefix = self.robot_camera_path[:-2]
        self.robot_js_aliases = {str(Path(erase_type(s))[len(common_prefix):]): s for s in self.robot_joint_symbols.union(self.robot_controlled_symbols)}
        self.inverse_js_aliases.update({erase_type(v): k for k, v in self.robot_js_aliases.items()})
        print('Robot aliases:\n{}'.format('\n'.join(['{:>20}: {}'.format(k, v) for k, v in self.robot_js_aliases.items()])))

        # CREATE TAXI CONSTRAINTS
        camera_o_z = x_of(self.robot_camera.pose)[2]
        dist_start_location = norm(self.waiting_location - pos_of(self.base_link.pose))
        angular_alignment   = dot(self.waiting_direction, self.base_link.pose * vector3(1,0,0))

        self.taxi_constraints = {'to_position': SC(-dist_start_location, -dist_start_location, 1, dist_start_location),
                                 'to_orientation': SC(1 - angular_alignment - 2 * less_than(dist_start_location, 0.2), 1 - angular_alignment, 1, angular_alignment),
                                 'camera_orientation': SC(-0.2 - camera_o_z, -0.2 - camera_o_z, 1, camera_o_z)}
        self.generate_idle_controller()


    @profile
    def cb_obj_joint_state(self, state_msg):
        #print('Got object js')
        if self.obj_js_aliases is None:
            return

        for name, p in zip(state_msg.name, state_msg.position):
            if name in self.obj_js_aliases:
                self.state[self.obj_js_aliases[name]] = p

        if self.obj is not None:
            if self._current_target is None:
                for p, s in self.possible_targets:
                    position = self.state[s]
                    if position > 0.01:
                        print('ARGH, {} is open!'.format(p))
                        self._current_target = p
                        self.generate_push_controller()

                # If there is still no new target
                if self._current_target is None and self.idle_controller is None:
                    print('There is nothing to do for me...')
                    self.generate_idle_controller()

            else:
                if self.state[self.target_symbol_map[self._current_target]] <= 0.01:
                    print('{} is finally closed again.'.format(self._current_target))
                    self._current_target = None

    @profile
    def cb_robot_joint_state(self, state_msg):
        if self.robot_js_aliases is None:
            return

        for name, p in zip(state_msg.name, state_msg.position):
            if name in self.robot_js_aliases:
                self.state[self.robot_js_aliases[name]] = p

        cmd = {}
        if self._current_target is None:
            if self.idle_controller is not None:

                str_state = {str(s): v for s, v in self.state.items() if s in self.idle_controller.free_symbols}

                print('Idling around...')
                cmd = self.idle_controller.get_cmd(self.state)
        else:
            if self.pushing_controller is not None:
                if not self._needs_odom or self._has_odom:
                    str_state = {str(s): v for s, v in self.state.items() if s in self.idle_controller.free_symbols}

                    print('Need to close {}'.format(self._current_target))
                    cmd = self.pushing_controller.get_cmd(self.state)
                else:
                    print('Waiting for odom...')

        if len(cmd) > 0:
            cmd_msg = JointStateMsg()
            cmd_msg.header.stamp = rospy.Time.now()
            cmd_msg.name, cmd_msg.velocity = zip(*[(self.inverse_js_aliases[erase_type(s)], v) for s, v in cmd.items()])
            cmd_msg.position = [0]*len(cmd_msg.name)
            cmd_msg.effort   = cmd_msg.position
            self.pub_cmd.publish(cmd_msg)

    @profile
    def cb_robot_odometry(self, odom_msg):
        if type(self.base_joint) is RoombaJoint:
            self.state[self.base_joint.x_pos] = odom_msg.twist.twist.linear.x
            self.state[self.base_joint.y_pos] = odom_msg.twist.twist.linear.y
            self.state[self.base_joint.z_pos] = odom_msg.twist.twist.linear.z 
            self.state[self.base_joint.a_pos] = odom_msg.twist.twist.angular.z
            self._has_odom = True

    @profile
    def generate_idle_controller(self):
        if self.robot is None:
            return

        symbols = set()
        for c in self.taxi_constraints.values():
            symbols |= c.expr.free_symbols

        joint_symbols = self.robot_joint_symbols.intersection(symbols)
        controlled_symbols = self.robot_controlled_symbols
        
        hard_constraints = self.km.get_constraints_by_symbols(symbols.union(controlled_symbols))
        self.geom_world = self.km.get_active_geometry(joint_symbols)

        controlled_values, hard_constraints = self._extract_controlled_values(hard_constraints, controlled_symbols)

        for name in self.geom_world.names:
            path = Path(name)
            for x in range(3):
                hard_constraints['{} collision_avoidance {}'.format(path, x)] = closest_distance_constraint(self.km.get_data(path + ('pose',)), spw.eye(4), path, Path('anon/{}'.format(x)))

        self.idle_controller = GQPB(self.geom_world, hard_constraints, self.taxi_constraints, controlled_values, visualizer=self.debug_visualizer)


    @profile
    def generate_push_controller(self):
        if self.robot is None:
            return

        if self._current_target is None:
            return

        target_symbol = self.target_symbol_map[self._current_target]

        pose_path = self._current_target[len(self.urdf_path):] + ('pose',)

        geom_distance = closest_distance(self.robot_eef.pose, pose_path.get_data(self.obj), self.robot_eef_path, self._current_target)
        condition = less_than(geom_distance, 0.01)

        soft_constraints = {'close {}'.format(self._current_target): SC(-condition, -condition, 1, target_symbol),
                            'keep_contact': SC(-geom_distance, -geom_distance, 1, geom_distance)}
        symbols = set()
        for c in soft_constraints.values():
            symbols |= c.expr.free_symbols

        print('Generating push controller for symbols:\n {}'.format('\n '.join([str(s) for s in symbols])))
            
        joint_symbols = self.robot_joint_symbols.intersection(symbols).union({target_symbol})
        controlled_symbols = self.robot_controlled_symbols.union({get_diff_symbol(target_symbol)})

        hard_constraints = self.km.get_constraints_by_symbols(symbols.union(controlled_symbols))
        self.geom_world = self.km.get_active_geometry(joint_symbols)
        controlled_values, hard_constraints = self._extract_controlled_values(hard_constraints, controlled_symbols)
        # for name in self.geom_world.names:
        #     path = Path(name)
        #     if path != self._current_target:
        #         for x in range(3):
        #             hard_constraints['{} collision_avoidance {}'.format(path, x)] = closest_distance_constraint(self.km.get_data(path + ('pose',)), spw.eye(4), path, Path('anon/{}'.format(x)))

        self.pushing_controller = GQPB(self.geom_world, hard_constraints, soft_constraints, controlled_values, visualizer=self.debug_visualizer)


    @profile
    def _extract_controlled_values(self, constraints, controlled_symbols):
        controlled_values = {}
        to_remove = set()
        for k, c in constraints.items():
            if type(c.expr) is spw.Symbol and c.expr in controlled_symbols:
                weight = 0.01 # if c.expr != roomba_joint.lin_vel and c.expr != roomba_joint.ang_vel else 0.2
                controlled_values[str(c.expr)] = ControlledValue(c.lower, c.upper, c.expr, weight)
                to_remove.add(k)

        constraints = {k: c for k, c in constraints.items() if k not in to_remove}
        for s in controlled_symbols:
            if str(s) not in controlled_values:
                controlled_values[str(s)] = ControlledValue(-1e9, 1e9, s, 0.01)

        return controlled_values, constraints
