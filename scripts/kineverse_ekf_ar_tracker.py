import rospy
import kineverse.gradients.gradient_math as gm
import tf2_ros

from geometry_msgs.msg import PoseStamped as PoseStampedMsg

from kineverse.model.geometry_model import GeometryModel, Path
from kineverse_msgs.msg             import ValueMap as ValueMapMsg

from kineverse_experiment_world.ekf_model import EKFModel
from kineverse_experiment_world.nobilia_shelf import create_nobilia_shelf


def np_frame3_quaternion(x, y, z, qx, qy, qz, qw):
    a  = [qx, qy, qz, qw]
    mf = GM if max([type(v) == GC for v in a]) else cm.Matrix
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    qw2 = qw * qw
    return mf([[qw2 + qx2 - qy2 - qz2, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy, x],
               [2 * qx * qy + 2 * qw * qz, qw2 - qx2 + qy2 - qz2, 2 * qy * qz - 2 * qw * qx, y],
               [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, qw2 - qx2 - qy2 + qz2, z],
               [0, 0, 0, 1]])


class Kineverse6DEKFTracker(object):
    def __init__(self, km, paths_observables,
                       transition_rules=None, 
                       noise_estimation_observations=100, estimate_init_steps=10):
        """Initializes the tracker to estimate variables based 
           on the given paths. The paths are assumed to point to
           6D poses.
        
        Args:
            km (ArticulationModel): Articulation model to used
            paths_observables (list): List of paths to poses that will be observed
            noise_estimation_observations (int): Number of observations to collect before initializing the EKFs
            estimate_init_steps (int): Number of steps used to initialize the estimate of an ekf
        """
        poses = {p: km.get_data(p) for p in paths_observables}


        # Identify tracking pools. All poses within one pool share at least one DoF
        tracking_pools = [(gm.free_symbols(pose), [(p, pose)]) for p, pose in poses.items() 
                                                                if len(gm.free_symbols(pose)) != 0]

        final_pools = []
        while len(tracking_pools) > 0:
            syms, pose_list = tracking_pools[0]
            initial_syms_size = len(syms)
            y = 1
            while y < len(tracking_pools):
                o_syms, o_pose_list = tracking_pools[y]
                if len(syms.intersection(o_syms)) != 0:
                    syms = syms.union(o_syms)
                    pose_list += o_pose_list
                    del tracking_pools[y]
                else:
                    y += 1

            # If the symbol set has not been enlarged, 
            # there is no more overlap with other pools
            if len(syms) == initial_syms_size:
                final_pools.append((syms, pose_list))
                tracking_pools = tracking_pools[1:]
            else:
                tracking_pools[0] = (syms, pose_list)

        tracking_pools = final_pools
        self.ekfs = [EKFModel(dict(poses), 
                              km.get_constraints_by_symbols(symbols), 
                              transition_rules=transition_rules) for symbols, poses in tracking_pools]

        print(f'Generated {len(self.ekfs)} EKFs:\n  '
               '\n  '.join(str(e) for e in self.ekfs))

        self.observation_names = [str(p) for p in paths_observables]
        self.controls          = set(sum([e.ordered_controls for e in self.ekfs], []))

        self.estimates = [None] * len(self.ekfs)

        self._n_estimation_observations = noise_estimation_observations
        self._estimation_observations   = []
        self._ekfs_initialized = False
        self._estimate_init_steps = estimate_init_steps


    def compute_observation_covariance(self, observations):
        """Computes the observational covariance from a set of observations
        
        Args:
            observations ([dict]): Dictionary of noisy observations
        """
        for ekf in self.ekfs:
            ekf.generate_R(observations)
        self._ekfs_initialized = True

    def process_update(self, observation, control, dt=0.05):
        """Processes a single observation
        
        Args:
            observation (dict): Dictionary of the observations
        """
        if not self._ekfs_initialized and len(self._estimation_observations) < self._n_estimation_observations:
            self._estimation_observations.append(observation)
        else:
            if not self._ekfs_initialized:
                self.compute_observation_covariance(self._estimation_observations)

            for x, ekf in enumerate(self.ekfs):
                # Currently only using a single estimate
                # Proper particle filtering with a clever initialization
                # would be preferable.
                # For now: initialize the estimate with gradient descent 
                # on the observation
                if self.estimates[x] is None:
                    self.estimates[x] = ekf.spawn_particle()
                    estimate = self.estimates[x]

                    for _ in range(self.estimate_init_steps):
                        h_prime   = ekf.h_prime_fn.call2(estimate.state)
                        obs_delta = obs_vector.reshape((len(obs_vector), 1)) - ekf.h_fn.call2(estimate.state)
                        estimate.state += (h_prime.T.dot(obs_delta) * 0.5).reshape(estimate.state.shape)
                else:
                    estimate    = self.estimates[x]
                    estimate.state, estimate.cov = ekf.predict(estimate.state.flatten(), 
                                                               estimate.cov, 
                                                               ekf.gen_control_vector(control), dt=0.05)
                    obs_vector  = ekf.gen_obs_vector(observation)
                    estimate.state, estimate.cov = ekf.update(estimate.state, 
                                                              estimate.cov, 
                                                              obs_vector)

    def get_estimated_state(self):
        if self._ekfs_initialized is None:
            return None
        return dict(sum([zip(ekf.ordered_vars, estimate.state.flatten()) for ekf, estimate 
                                                                          in zip(self.ekfs, self.estimates) if estimate is not None], []))

class ROSEKFManager(object):
    def __init__(self, tracker : Kineverse6DEKFTracker):
        self.tracker = tracker 
        self.last_observation = {}
        self.last_control     = None
        self.last_update      = None

        self.pub_state        = rospy.Publisher('~state_estimate', ValueMapMsg, queue_size=1, tcp_nodelay=True)
        self.sub_observations = rospy.Subscriber('~observations', PoseStampedMsg, callback=self.cb_obs, queue_size=1)
        self.sub_controls     = rospy.Subscriber('~controls',        ValueMapMsg, callback=self.cb_controls, queue_size=1)

    def cb_obs(self, pose_stamped_msg):
        matrix = np_frame3_quaternion(pose_stamped_msg.pose.position.x, 
                                      pose_stamped_msg.pose.position.y, 
                                      pose_stamped_msg.pose.position.z,
                                      pose_stamped_msg.pose.orientation.x,
                                      pose_stamped_msg.pose.orientation.y,
                                      pose_stamped_msg.pose.orientation.z,
                                      pose_stamped_msg.pose.orientation.w)
        # Yes, I am abusing the message type. Sue me
        self.last_observation[pose_stamped_msg.header.frame_id] = matrix
        try_update()

    def cb_controls(self, map_message):
        self.last_control = {gm.Symbol(str_symbol): v in zip(map_message.symbol, map_message.value)}
        try_update()

    def try_update(self):
        if min(p in self.last_observation for p in self.tracker.observation_names) is False:
            return

        if min(s in self.last_control for s in self.tracker.controls) is False:
            return

        now = rospy.Time.now()
        dt  = 0.05 if self.last_update is None else (now - self.last_update).to_sec()

        self.tracker.process_update(self.last_observation, self.last_control, dt)

        state_msg = ValueMapMsg()
        state_msg.header.stamp = now
        state_msg.symbol, state_msg.value = zip(*self.tracker.get_estimated_state().items())
        state_msg.symbol = [str(s) for s in state_msg.symbol]
        self.pub_state.publish(state_msg)

        self.last_control = {}
        self.last_observation = {}


if __name__ == '__main__':
    rospy.init_node('kineverse_ar_tracker')

    km = GeometryModel()

    shelf_location = gm.point3(*[gm.Position(f'l_{x}') for x in 'xyz'])

    origin_pose = gm.frame3_rpy(0, 0, gm.Position('l_ry'), shelf_location)

    create_nobilia_shelf(km, Path('nobilia'), origin_pose)

    km.clean_structure()
    km.dispatch_events()

    shelf   = km.get_data('nobilia')
    tracker = Kineverse6DEKFTracker(km, 
                                    [Path(f'nobilia/markers/{name}') for name in shelf.markers.keys()], 
                                    {x: x for x in gm.free_symbols(origin_pose)})

    node = ROSEKFManager(tracker)

    reference_frame = rospy.get_param('~reference_frame', 'world')
    grab_from_tf    = rospy.get_param('~grab_from_tf', False)
    grab_rate       = rospy.get_param('~grab_rate', 20.0)

    tf_buffer = tf2_ros.Buffer()
    listener  = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Duration(1 / grab_rate )

    while not rospy.is_shutdown():
        if grab_from_tf:
            start = rospy.Time.now()

            for obs in tracker.observation_names:
                try:
                    trans = tf_buffer.lookup_transform(reference_frame, obs, rospy.Time(0))
                    msg = PoseStampedMsg()
                    msg.header.frame_id  = obs
                    msg.pose.position    = trans.transform.translation
                    msg.pose.orientation = trans.transform.rotation
                    node.cb_obs(msg)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    print(f'Exception raised while looking up {obs} -> {reference_frame}:\n{e}')
                    continue
            
            time_remaining = rate - (rospy.Time.now() - start)
            if time_remaining > rospy.Duration(0):
                rospy.sleep(time_remaining)
        else:
            rospy.sleep(1.0)



