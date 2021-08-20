import rospy
import kineverse.gradients.gradient_math as gm
import numpy as np
import tf2_ros

from kineverse.model.geometry_model import GeometryModel, Path, Frame
from kineverse_tools.ik_solver import ik_solve_one_shot
from tqdm import tqdm

from kineverse_experiment_world.nobilia_shelf import create_nobilia_shelf
from kineverse_experiment_world.msg import TransformStampedArray as TransformStampedArrayMsg

def np_6d_pose_feature(x, y, z, qx, qy, qz, qw):
    axis_norm = np.sqrt(qx**2 + qy**2 + qz**2)
    angle = np.arctan2(axis_norm, qw) * 2
    # print(f'Angle of quat: qw -> {math.acos(qw) * 2}, {math.asin(axis_norm) * 2}. From arctan2: {angle}')
    if np.abs(angle) < 1e-4:
        return [x, y, z, 0, 0, 0]
    rotation_vector = (np.array([qx, qy, qz]) / axis_norm) * angle
    # print(f'Rotation vector: {rotation_vector} Norm: {np.sqrt(np.sum(rotation_vector**2))}')
    return np.hstack(([x, y, z], rotation_vector))


if __name__ == '__main__':
    rospy.init_node('nobilia_location_estimator')

    km = GeometryModel()

    shelf_location = gm.point3(*[gm.Position(f'l_{x}') for x in 'xyz'])
    # gm.Position('l_ry')

    yaw_rotation = gm.Position('l_ry')
    origin_pose  = gm.frame3_rpy(0, 0, yaw_rotation, shelf_location)
    # origin_pose = gm.frame3_rpy(0, 0, 0, gm.point3(2, 0, 0))

    create_nobilia_shelf(km, Path('nobilia'), origin_pose)

    km.clean_structure()
    km.dispatch_events()

    obs = []
    initialized = False
    shelf   = km.get_data('nobilia')

    reference_frame  = rospy.get_param('~reference_frame', 'world')
    grab_rate        = rospy.get_param('~grab_rate', 20.0)
    body_marker_name = rospy.get_param('~body_marker_name', 'nobilia/markers/body')
    num_obs          = rospy.get_param('~num_obs', 100)
    t = tqdm(total=num_obs, desc='Waiting for enough observations...')

    def cb_observation(transform_stamped):
        if len(obs) >= num_obs:
            return False

        obs.append(np_6d_pose_feature(trans.transform.translation.x,
                                      trans.transform.translation.y,
                                      trans.transform.translation.z,
                                      trans.transform.rotation.x,
                                      trans.transform.rotation.y,
                                      trans.transform.rotation.z,
                                      trans.transform.rotation.w))
        t.update(len(obs))

        if len(obs) >= num_obs:
            t.close()
            mp        = np.mean(obs, axis=0)
            goal_pose = gm.frame3_axis_angle(mp[3:] / np.sqrt(np.sum(mp[3:]**2)), np.sqrt(np.sum(mp[3:]**2)), mp[:3])

            error, state = ik_solve_one_shot(km,
                                             shelf.markers['body'].pose,
                                             {s: 0 for s in gm.free_symbols(shelf.markers['body'].pose)},
                                             goal_pose, solver='TQPB')
            rospy.set_param(f'/nobilia_x', float(state[shelf_location[0]]))
            rospy.set_param(f'/nobilia_y', float(state[shelf_location[1]]))
            rospy.set_param(f'/nobilia_z', float(state[shelf_location[2]]))
            rospy.set_param(f'/nobilia_yaw', float(state[yaw_rotation]))

            print(f'Localized nobilia shelf at:'
                  f'\n    x: {state[shelf_location[0]]}'
                  f'\n    y: {state[shelf_location[1]]}'
                  f'\n    z: {state[shelf_location[2]]}'
                  f'\n  yaw: {state[yaw_rotation]}'
                  f'  Error: {error}')
            return True
        return False


    tf_buffer = tf2_ros.Buffer()
    listener  = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Duration(1 / grab_rate )

    print(f'Attempting to gather observations of {body_marker_name} in {reference_frame}')

    while not rospy.is_shutdown():
        start = rospy.Time.now()

        try:
            trans = tf_buffer.lookup_transform(reference_frame, body_marker_name, rospy.Time(0))
            if cb_observation(trans):
                break
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(f'Exception raised while looking up {body_marker_name} -> {reference_frame}:\n{e}')
            continue

        time_remaining = rate - (rospy.Time.now() - start)
        if time_remaining > rospy.Duration(0):
            rospy.sleep(time_remaining)
