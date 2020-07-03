import rospy
import numpy as np

from iai_bullet_sim.basic_simulator         import SimulatorPlugin
from iai_bullet_sim.multibody               import MultiBody

import kineverse.gradients.gradient_math    as gm

from kineverse.visualization.ros_visualizer import ROSVisualizer
from kineverse.bpb_wrapper                  import pb, transform_to_matrix
from kineverse_experiment_world.utils       import np_random_normal_offset, np_random_quat_normal

from kineverse_experiment_world.msg import PoseStampedArray as PSAMsg
from geometry_msgs.msg              import PoseStamped as PoseStampedMsg


class PoseObservationPublisher(SimulatorPlugin):
    def __init__(self, multibody, camera_link, fov, near, far, noise_exp, topic_prefix='', frequency=30, debug=False):
        super(PoseObservationPublisher, self).__init__('PoseObservationPublisher')
        self.topic_prefix = topic_prefix
        self.publisher = rospy.Publisher('{}/pose_obs'.format(topic_prefix), PSAMsg, queue_size=1, tcp_nodelay=True)
        self.message_templates = {}
        self.multibody   = multibody
        self.camera_link = camera_link
        self.fov         = fov
        self.near        = near
        self.far         = far
        self.noise_exp   = noise_exp
        self._enabled    = True
        self.visualizer  = ROSVisualizer('pose_obs_viz', 'world') if debug else None
        self._last_update = 1000
        self._update_wait = 1.0 / frequency


    def post_physics_update(self, simulator, deltaT):
        """Implements post physics step behavior.
        :type simulator: BasicSimulator
        :type deltaT: float
        """
        if not self._enabled:
            return

        self._last_update += deltaT
        if self._last_update >= self._update_wait:
            self._last_update = 0
            cf_tuple = self.multibody.get_link_state(self.camera_link).worldFrame
            camera_frame = gm.frame3_quaternion(cf_tuple.position.x, cf_tuple.position.y, cf_tuple.position.z, *cf_tuple.quaternion)
            cov_proj = gm.rot_of(camera_frame)[:3, :3]
            inv_cov_proj = cov_proj.T

            out = PSAMsg()

            if self.visualizer is not None:
                self.visualizer.begin_draw_cycle()
                poses = []

            for name, body in simulator.bodies.items():
                if body == self.multibody:
                    continue

                if isinstance(body, MultiBody):
                    poses_to_process = [('{}/{}'.format(name, l), body.get_link_state(l).worldFrame) for l in body.links]
                else:
                    poses_to_process = [(name, body.pose())]

                for pname, pose in poses_to_process:
                    if not pname in self.message_templates:
                        msg = PoseStampedMsg()
                        msg.header.frame_id = pname
                        self.message_templates[pname] = msg
                    else:
                        msg = self.message_templates[pname]

                    obj_pos = gm.point3(*pose.position)
                    c2o  = obj_pos - gm.pos_of(camera_frame)
                    dist = gm.norm(c2o)
                    if dist < self.far and dist > self.near and gm.dot_product(c2o, gm.x_of(camera_frame)) > gm.cos(self.fov * 0.5) * dist:


                        noise = 2 ** (self.noise_exp * dist) - 1
                        (n_quat, )  = np_random_quat_normal(1, 0, noise)
                        (n_trans, ) = np_random_normal_offset(1, 0, noise)

                        n_pose = pb.Transform(pb.Quaternion(*pose.quaternion), pb.Vector3(*pose.position)) *\
                                     pb.Transform(pb.Quaternion(*n_quat), pb.Vector3(*n_trans[:3]))

                        if self.visualizer is not None:
                            poses.append(transform_to_matrix(n_pose))
                        msg.pose.position.x = n_pose.origin.x
                        msg.pose.position.y = n_pose.origin.y
                        msg.pose.position.z = n_pose.origin.z
                        msg.pose.orientation.x = n_pose.rotation.x
                        msg.pose.orientation.y = n_pose.rotation.y
                        msg.pose.orientation.z = n_pose.rotation.z
                        msg.pose.orientation.w = n_pose.rotation.w
                        out.poses.append(msg)


                self.publisher.publish(out)

            if self.visualizer is not None:
                self.visualizer.draw_poses('debug', gm.se.eye(4), 0.1, 0.02, poses)
                self.visualizer.render()



    def disable(self, simulator):
        """Stops the execution of this plugin.
        :type simulator: BasicSimulator
        """
        self._enabled = False
        self.publisher.unregister()


    def to_dict(self, simulator):
        """Serializes this plugin to a dictionary.
        :type simulator: BasicSimulator
        :rtype: dict
        """
        return {'body': simulator.get_body_id(self.body.bId()),
                'camera_link':  self.camera_link,
                'fov':          self.fov,
                'near':         self.near,
                'far':          self.far,
                'noise_exp':    self.noise_exp,
                'topic_prefix': self.topic_prefix}

    @classmethod
    def factory(cls, simulator, init_dict):
        body = simulator.get_body(init_dict['body'])
        if body is None:
            raise Exception('Body "{}" does not exist in the context of the given simulation.'.format(init_dict['body']))
        return cls(body,
                   init_dict['camera_link'],
                   init_dict['fov'],
                   init_dict['near'],
                   init_dict['far'],
                   init_dict['noise_exp'],
                   init_dict['topic_prefix'])


    def reset(self, simulator):
        pass
