import rospy
import numpy as np
import giskardpy.symengine_wrappers as spw

from iai_bullet_sim.basic_simulator import SimulatorPlugin
from iai_bullet_sim.multibody       import MultiBody

from kineverse_experiment_world.msg import PoseStampedArray as PSAMsg
from kineverse.visualization.ros_visualizer import ROSVisualizer
from geometry_msgs.msg import PoseStamped as PoseStampedMsg

class PoseObservationPublisher(SimulatorPlugin):
    def __init__(self, multibody, camera_link, fov, near, far, noise_exp, topic_prefix=''):
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
        self.visualizer  = ROSVisualizer('pose_obs_viz', 'map')


    def post_physics_update(self, simulator, deltaT):
        """Implements post physics step behavior.
        :type simulator: BasicSimulator
        :type deltaT: float
        """
        if not self._enabled:
            return

        cf_tuple = self.multibody.get_link_state(self.camera_link).worldFrame
        camera_frame = spw.frame3_quaternion(cf_tuple.position.x, cf_tuple.position.y, cf_tuple.position.z, *cf_tuple.quaternion)
        cov_proj = spw.rot_of(camera_frame)[:3, :3]
        inv_cov_proj = cov_proj.T

        out = PSAMsg()

        self.visualizer.begin_draw_cycle()
        points = []

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

                obj_pos = spw.point3(*pose.position)
                c2o  = obj_pos - spw.pos_of(camera_frame)
                dist = spw.norm(c2o)
                if dist < self.far and dist > self.near and spw.dot(c2o, spw.x_of(camera_frame)) > spw.cos(self.fov * 0.5) * dist:


                    noise = 2 ** (self.noise_exp * dist) - 1

                    noisy_pos = spw.point3(np.random.normal(pose.position[0], noise),
                                           np.random.normal(pose.position[1], noise),
                                           np.random.normal(pose.position[2], noise))

                    points.append(noisy_pos)
                    msg.pose.position.x = noisy_pos[0]
                    msg.pose.position.y = noisy_pos[1]
                    msg.pose.position.z = noisy_pos[2]
                    msg.pose.orientation.x = pose.quaternion[0]
                    msg.pose.orientation.y = pose.quaternion[1]
                    msg.pose.orientation.z = pose.quaternion[2]
                    msg.pose.orientation.w = pose.quaternion[3]
                    out.poses.append(msg)


            self.publisher.publish(out)
        self.visualizer.draw_points('debug', spw.eye(4), 0.1, points)
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