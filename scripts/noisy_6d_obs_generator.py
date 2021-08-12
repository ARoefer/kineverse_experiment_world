#!/usr/bin/env python3
import rospy
import tf2_ros
import numpy as np

from kineverse.utils import real_quat_from_matrix
from kineverse_experiment_world.msg import TransformStampedArray as TransformStampedArrayMsg

from kineverse_experiment_world.utils import random_normal_translation, \
                                             random_rot_normal

def np_frame3_quaternion(x, y, z, qx, qy, qz, qw):
    a  = [qx, qy, qz, qw]
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    qw2 = qw * qw
    return np.array([[qw2 + qx2 - qy2 - qz2, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy, x],
                     [2 * qx * qy + 2 * qw * qz, qw2 - qx2 + qy2 - qz2, 2 * qy * qz - 2 * qw * qx, y],
                     [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, qw2 - qx2 - qy2 + qz2, z],
                     [0, 0, 0, 1]])

if __name__ == '__main__':
    rospy.init_node('noisy_6d_obs_generator')

    reference_frame = rospy.get_param('~reference_frame', 'world')
    pose_names      = rospy.get_param('~poses', [])
    publish_rate    = rospy.get_param('~rate', 30.0)
    noise_lin_sd    = rospy.get_param('~noise_lin_sd', 0.05)
    noise_ang_sd    = rospy.get_param('~noise_ang_sd', 0.05)

    pub_obs = rospy.Publisher('~observations', TransformStampedArrayMsg, queue_size=1, tcp_nodelay=True)

    tf_buffer = tf2_ros.Buffer()
    listener  = tf2_ros.TransformListener(tf_buffer)


    msg = TransformStampedArrayMsg()
    msg.transforms = [None] * len(pose_names)

    print(f'Generating noisy observations for:\n  '
           '\n  '.join(pose_names))

    rate = rospy.Duration(1 / publish_rate)

    while not rospy.is_shutdown():
        start = rospy.Time.now()
        noise_poses = [t.dot(r) for t, r in zip(random_normal_translation(len(pose_names), 0, noise_lin_sd),
                                                random_rot_normal(len(pose_names), 0, noise_ang_sd))]

        for x, (pose_name, noise_pose) in enumerate(zip(pose_names, noise_poses)):
            try:
                trans = tf_buffer.lookup_transform(reference_frame, pose_name, rospy.Time(0))
                stamp = trans.header.stamp
                translation = trans.transform.translation
                rotation    = trans.transform.rotation
                clean_pose  = np_frame3_quaternion(translation.x, 
                                                   translation.y, 
                                                   translation.z,
                                                   rotation.x,
                                                   rotation.y,
                                                   rotation.z,
                                                   rotation.w)
                noisy_obs = clean_pose.dot(noise_pose)
                msg.transforms[x] = trans
                quat = real_quat_from_matrix(noisy_obs)
                trans.header.stamp = rospy.Time.now()
                trans.transform.translation.x = noisy_obs[0, 3]
                trans.transform.translation.y = noisy_obs[1, 3]
                trans.transform.translation.z = noisy_obs[2, 3]
                trans.transform.rotation.x = quat[0]
                trans.transform.rotation.y = quat[1]
                trans.transform.rotation.z = quat[2]
                trans.transform.rotation.w = quat[3]
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                print(f'Exception raised while looking up {pose_name} -> {reference_frame}:\n{e}')
                break
        else:
            pub_obs.publish(msg)
        
        time_remaining = rate - (rospy.Time.now() - start)
        if time_remaining > rospy.Duration(0):
            rospy.sleep(time_remaining)
