import rospy
import numpy as np

import kineverse.gradients.common_math as cm

from kineverse.gradients.gradient_math       import *
from kineverse.gradients.gradient_container  import collect_free_symbols
from kineverse.gradients.diff_logic          import erase_type, get_symbol_type, TYPE_POSITION
from kineverse.model.paths                   import Path
from kineverse.model.articulation_model         import Constraint
from kineverse.model.geometry_model          import GeometryModel, contact_geometry, generate_contact_model
from kineverse.motion.min_qp_builder         import GeomQPBuilder  as GQPB,\
                                                    TypedQPBuilder as TQPB,\
                                                    SoftConstraint as SC,\
                                                    PID_Constraint as PIDC, \
                                                    ControlledValue, \
                                                    generate_controlled_values, \
                                                    depth_weight_controlled_values
from kineverse.network.model_client          import ModelClient
from kineverse.operations.urdf_operations    import URDFRobot
from kineverse.operations.special_kinematics import DiffDriveJoint, OmnibaseJoint
from kineverse.type_sets                     import is_symbolic
from kineverse.time_wrapper                  import Time
from kineverse.visualization.bpb_visualizer  import ROSBPBVisualizer

from kineverse_experiment_world.push_demo_base import generate_push_closing

from nav_msgs.msg       import Odometry   as OdometryMsg
from sensor_msgs.msg    import JointState as JointStateMsg
from kineverse_msgs.msg import ValueMap   as ValueMapMsg

goal_joints = {'prismatic', 'revolute'}

# Bullet's convex to convex distance calc is off.
BULLET_FIXED_OFFSET = 0.08


class ObsessiveObjectCloser(object):
    def __init__(self, urdf_path, robot_eef, robot_camera, obj_js_topic, robot_topic_prefix, waiting_location=(0,0,0,0), resting_pose=None):
        self.km = ModelClient(GeometryModel)
        self.robot_eef_path     = robot_eef
        self.robot_path         = self.robot_eef_path[:-2]
        self.robot_camera_path  = robot_camera
        self.robot              = None
        self.robot_eef          = None
        self.robot_camera       = None
        self.base_joint         = None
        self.state              = {}
        self.robot_js_aliases   = None
        self.inverse_js_aliases = {}
        self.obj_js_aliases     = None
        self.obj_world          = None
        self._current_target    = None
        self.waiting_location   = point3(*waiting_location[:3])
        self.waiting_direction  = vector3(cos(waiting_location[3]), sin(waiting_location[3]), 0)
        self.idle_controller    = None
        self.pushing_controller = None
        self._needs_odom        = False
        self._has_odom          = False
        self._t_last_update     = None
        self._resting_pose      = {Position(self.robot_path + (n,)): v for n, v in resting_pose.items()} if resting_pose is not None else None

        self.debug_visualizer = ROSBPBVisualizer('/debug_vis', 'world')

        self.urdf_path = urdf_path
        self.obj = None

        self._poi_pos = Position('poi')
        self.poi = point3(1.5, 0.5, 0.0) + vector3(0, self._poi_pos * 2.0, 0)

        self.km.register_on_model_changed(self.robot_path, self.cb_robot_model_changed)
        self.km.register_on_model_changed(self.urdf_path,  self.cb_obj_model_changed)

        self.pub_cmd = rospy.Publisher('{}/commands/joint_velocities'.format(robot_topic_prefix), JointStateMsg, queue_size=1)

        self.sub_obj_js   = rospy.Subscriber(obj_js_topic, ValueMapMsg, self.cb_obj_joint_state,  queue_size=1)
        self.sub_odom     = rospy.Subscriber('{}/odometry'.format(robot_topic_prefix), OdometryMsg, self.cb_robot_odometry, queue_size=1)
        self.sub_robot_js = rospy.Subscriber('{}/joint_states'.format(robot_topic_prefix), JointStateMsg, self.cb_robot_joint_state, queue_size=1)

    @profile
    def cb_obj_model_changed(self, model):
        print('URDF model changed')
        if type(model) is not URDFRobot:
            raise Exception('Path "{}" does not refer to a urdf_robot. Type is "{}"'.format(self.urdf_path, type(model)))

        self.obj = model
        self.possible_targets  = [(Path(joint.child), joint.position) for jname, joint in self.obj.joints.items() if joint.type in goal_joints]
        self.target_symbol_map = dict(self.possible_targets)
        self.obj_world         = self.km.get_active_geometry({s for _, s in self.possible_targets if cm.is_symbol(s)})
        new_state = {s: 0.0 for _, s in self.possible_targets if cm.is_symbol(s)}
        self.obj_js_aliases = {str(Path(erase_type(s))[len(self.urdf_path):]): s for s in new_state.keys()}
        print('Object aliases:\n{}'.format('\n'.join(['{:>20}: {}'.format(k, v) for k, v in self.obj_js_aliases.items()])))
        self.inverse_js_aliases.update({erase_type(v): k for k, v in self.obj_js_aliases.items()})
        self.state.update(new_state)

    @profile
    def cb_robot_model_changed(self, model):
        print('Robot model changed')
        self.robot = model
        temp = [j for j in self.robot.joints.values() if j.parent == 'world']
        if len(temp) > 0:
            self.base_joint = temp[0]
            self.base_link = Path(self.base_joint.child)[len(self.robot_eef_path) - 2:].get_data(self.robot)
        else:
            self.base_link = [l for l in self.robot.links.values() if l.parent == 'world'][0]
        self.robot_camera = self.robot_camera_path[len(self.robot_camera_path) - 2:].get_data(self.robot)
        self.robot_eef    = self.robot_eef_path[len(self.robot_eef_path) - 2:].get_data(self.robot)

        self.robot_joint_symbols = {j.position for j in self.robot.joints.values() if hasattr(j, 'position') and cm.is_symbol(j.position)}
        self.robot_controlled_symbols = {DiffSymbol(j) for j in self.robot_joint_symbols}

        if type(self.base_joint) is DiffDriveJoint:
            self.robot_controlled_symbols.update({self.base_joint.l_wheel_vel, self.base_joint.r_wheel_vel})
            self.robot_joint_symbols.update({self.base_joint.x_pos, self.base_joint.y_pos, self.base_joint.z_pos, self.base_joint.a_pos})
            self._needs_odom = True
        elif type(self.base_joint) is OmnibaseJoint:
            self.robot_controlled_symbols.update({DiffSymbol(x) for x in [self.base_joint.x_pos, self.base_joint.y_pos, self.base_joint.a_pos]})
            self.robot_joint_symbols.update({self.base_joint.x_pos, self.base_joint.y_pos, self.base_joint.a_pos})
            self._needs_odom = True

        new_state = {s: 0.0 for s in self.robot_camera.pose.free_symbols.union(self.robot_eef.pose.free_symbols) if get_symbol_type(s) == TYPE_POSITION}
        new_state.update({s: 0.0 for s in self.robot_joint_symbols})
        self.state.update(new_state)
        
        common_prefix = self.robot_camera_path[:-2]
        self.robot_js_aliases = {str(Path(erase_type(s))[len(common_prefix):]): s for s in self.robot_joint_symbols}
        self.inverse_js_aliases.update({erase_type(v): k for k, v in self.robot_js_aliases.items()})
        self.inverse_js_aliases.update({erase_type(s): str(Path(erase_type(s))[len(common_prefix):]) for s in self.robot_controlled_symbols})
        print('Robot aliases:\n{}'.format('\n'.join(['{:>20}: {}'.format(k, v) for k, v in self.robot_js_aliases.items()])))

        # CREATE TAXI CONSTRAINTS
        camera_o_z = x_of(self.robot_camera.pose)[2]
        dist_start_location = norm(self.waiting_location - pos_of(self.base_link.pose))
        angular_alignment   = dot_product(self.waiting_direction, self.base_link.pose * vector3(1,0,0))


        self.taxi_constraints = {'to_position': SC(-dist_start_location, -dist_start_location, 1, dist_start_location),
                                 'to_orientation': SC(1 - angular_alignment - 2 * less_than(dist_start_location, 0.2), 1 - angular_alignment, 1, angular_alignment),
                                 'camera_orientation': SC(-0.2 - camera_o_z, -0.2 - camera_o_z, 1, camera_o_z)}
        self.generate_idle_controller()


    @profile
    def cb_obj_joint_state(self, state_msg):
        #print('Got object js')
        if self.obj_js_aliases is None:
            return

        if type(state_msg) == JointStateMsg:
            for name, p in zip(state_msg.name, state_msg.position):
                if name in self.obj_js_aliases:
                    self.state[self.obj_js_aliases[name]] = p
        else:
            for name, p in zip(state_msg.symbol, state_msg.value):
                self.state[Symbol(name)] = p

        self.obj_world.update_world(self.state)

        if self.obj is not None:
            if self._current_target is None:
                for p, s in self.possible_targets:
                    position = self.state[s]
                    if position > 0.02:
                        print('ARGH, {} is open!'.format(p))
                        self._current_target = p
                        self.generate_push_controller()

                # If there is still no new target
                if self._current_target is None and self.idle_controller is None:
                    print('There is nothing to do for me...')
                    self.generate_idle_controller()

            else:
                if self.state[self.target_symbol_map[self._current_target]] < 0.02:
                    print('{} is finally closed again.'.format(self._current_target))
                    self._current_target = None

    @profile
    def cb_robot_joint_state(self, state_msg):
        if self._t_last_update is None:
            self._t_last_update = Time.now()
            return

        now = Time.now()
        dt  = (now - self._t_last_update).to_sec()
        self._t_last_update = now
        self.state[self._poi_pos] = cos(now.to_sec())

        if self.robot_js_aliases is None:
            return

        for name, p in zip(state_msg.name, state_msg.position):
            if name in self.robot_js_aliases:
                self.state[self.robot_js_aliases[name]] = p

        cmd = {}
        if self._current_target is None:
            if self.idle_controller is not None:
                print('Idling around...')
                cmd = self.idle_controller.get_cmd(self.state, deltaT=dt)
        else:
            if self.pushing_controller is not None:
                if not self._needs_odom or self._has_odom:
                    try:
                        now = Time.now()
                        cmd = self.pushing_controller.get_cmd(self.state, deltaT=dt)
                        time_taken = Time.now() - now
                        print('Command generated. Time taken: {} Rate: {} hz'.format(time_taken.to_sec(), 1.0 / time_taken.to_sec()))
                    except Exception as e:    
                        time_taken = Time.now() - now
                        print('Command generation failed. Time taken: {} Rate: {} hz\nError: {}'.format(time_taken.to_sec(), 1.0 / time_taken.to_sec(), e))
                    #print(self.pushing_controller.last_matrix_str())
                else:
                    print('Waiting for odom...')

        if len(cmd) > 0:
            #print('commands:\n  {}'.format('\n  '.join(['{}: {}'.format(s, v) for s, v in cmd.items()])))
            cmd_msg = JointStateMsg()
            cmd_msg.header.stamp = Time.now()
            cmd_msg.name, cmd_msg.velocity = zip(*[(self.inverse_js_aliases[erase_type(s)], v) for s, v in cmd.items()])
            cmd_msg.position = [0]*len(cmd_msg.name)
            cmd_msg.effort   = cmd_msg.position
            self.pub_cmd.publish(cmd_msg)

    @profile
    def cb_robot_odometry(self, odom_msg):
        if type(self.base_joint) is DiffDriveJoint:
            self.state[self.base_joint.x_pos] = odom_msg.pose.pose.position.x
            self.state[self.base_joint.y_pos] = odom_msg.pose.pose.position.y
            self.state[self.base_joint.z_pos] = odom_msg.pose.pose.position.z 
            self.state[self.base_joint.a_pos] = np.arccos(odom_msg.pose.pose.orientation.w) * 2 * np.sign(odom_msg.pose.pose.orientation.z)
            self._has_odom = True
        elif type(self.base_joint) is OmnibaseJoint:
            self.state[DiffSymbol(self.base_joint.x_pos)] = odom_msg.twist.twist.linear.x
            self.state[DiffSymbol(self.base_joint.y_pos)] = odom_msg.twist.twist.linear.y
            self.state[self.base_joint.x_pos] = odom_msg.pose.pose.position.x
            self.state[self.base_joint.y_pos] = odom_msg.pose.pose.position.y
            self.state[self.base_joint.a_pos] = np.arccos(odom_msg.pose.pose.orientation.w) * 2 * np.sign(odom_msg.pose.pose.orientation.z)
            self._has_odom = True

    @profile
    def generate_idle_controller(self):
        if self.robot is None:
            return
        
        tucking_constraints = {}
        if self._resting_pose is not None:
            tucking_constraints = {'tuck {}'.format(s): SC(p - s, p - s, 1, s) for s, p in self._resting_pose.items() if s in self.robot_joint_symbols}
            # print('Tuck state:\n  {}\nTucking constraints:\n  {}'.format('\n  '.join(['{}: {}'.format(k, v) for k, v in self._resting_pose.items()]), '\n  '.join(tucking_constraints.keys())))

        tucking_constraints.update(self.taxi_constraints)

        cam_to_poi = self.poi - pos_of(self.robot_camera.pose)
        lookat_dot = 1 - dot_product(self.robot_camera.pose * vector3(1,0,0), cam_to_poi) / norm(cam_to_poi)
        tucking_constraints['sweeping gaze'] = SC(-lookat_dot * 5, -lookat_dot * 5, 1, lookat_dot)

        symbols = set()
        for c in tucking_constraints.values():
            symbols |= c.expr.free_symbols

        joint_symbols = self.robot_joint_symbols.intersection(symbols)
        controlled_symbols = self.robot_controlled_symbols
        
        hard_constraints = self.km.get_constraints_by_symbols(symbols.union(controlled_symbols))
        self.geom_world  = self.km.get_active_geometry(joint_symbols, include_static=True)

        controlled_values, hard_constraints = generate_controlled_values(hard_constraints, controlled_symbols)
        controlled_values = depth_weight_controlled_values(self.km, controlled_values)

        self.idle_controller = GQPB(self.geom_world, hard_constraints, tucking_constraints, controlled_values, visualizer=self.debug_visualizer)

    def get_idle_controller_log(self):
        if self.idle_controller is not None:
            return self.idle_controller.H_dfs, self.idle_controller.A_dfs, self.idle_controller.cmd_df
        return None, None, None

    @profile
    def generate_push_controller(self):
        if self.robot is None:
            return

        if self._current_target is None:
            return

        target_symbol = self.target_symbol_map[self._current_target]
        pose_path     = self._current_target[len(self.urdf_path):] + ('pose',)
        obj_pose      = pose_path.get_data(self.obj)

        joint_symbols = self.robot_joint_symbols.union({target_symbol})
        controlled_symbols = self.robot_controlled_symbols.union({DiffSymbol(target_symbol)})

        # print('Generating push controller for symbols:\n {}'.format('\n '.join([str(s) for s in symbols])))
        
        hard_constraints, geom_distance, self.geom_world, debug_draw = generate_push_closing(self.km, 
                                                                                        self.state, 
                                                                                        controlled_symbols,
                                                                                        self.robot_eef.pose,
                                                                                        obj_pose,
                                                                                        self.robot_eef_path,
                                                                                        self._current_target,
                                                                                        'linear', # 'cross',
                                                                                        BULLET_FIXED_OFFSET)
        controlled_values, hard_constraints = generate_controlled_values(hard_constraints, controlled_symbols)
        controlled_values = depth_weight_controlled_values(self.km, controlled_values, exp_factor=1.1)
        if isinstance(self.base_joint, DiffDriveJoint):
          controlled_values[str(self.base_joint.l_wheel_vel)].weight = 0.001 
          controlled_values[str(self.base_joint.r_wheel_vel)].weight = 0.001 

        print('\n'.join(sorted([str(s) for s in geom_distance.free_symbols])))

        cam_to_obj = pos_of(obj_pose) - pos_of(self.robot_camera.pose)
        lookat_dot = 1 - dot_product(self.robot_camera.pose * vector3(1,0,0), cam_to_obj) / norm(cam_to_obj)

        soft_constraints = {'reach {}'.format(self._current_target): PIDC(geom_distance, geom_distance, 1, k_i=0.02),
                            'close {}'.format(self._current_target): PIDC(target_symbol, target_symbol, 1, k_i=0.06),
                            'lookat {}'.format(self._current_target): SC(-lookat_dot, -lookat_dot, 1, lookat_dot)}
        self.soft_constraints = soft_constraints

        print('Controlled Values:\n  {}'.format('\n  '.join([str(c) for c in controlled_values.values()])))

        self.pushing_controller = GQPB(self.geom_world, hard_constraints, soft_constraints, controlled_values , visualizer=self.debug_visualizer)
        #self.pushing_controller = TQPB(filtered_hard_constraints, soft_constraints, controlled_values)

    def get_push_controller_logs(self):
        if self.pushing_controller is not None:
            return self.pushing_controller.H_dfs, self.pushing_controller.A_dfs, self.pushing_controller.cmd_df
        return None, None, None
