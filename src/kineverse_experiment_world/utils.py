import numpy as np

from kineverse.gradients.gradient_math import *
from kineverse.model.paths                   import Path, CPath
from kineverse.operations.basic_operations   import ExecFunction
from kineverse.operations.special_kinematics import create_diff_drive_joint_with_symbols, \
                                                    create_omnibase_joint_with_symbols, \
                                                    CreateAdvancedFrameConnection


def np_rotation3_axis_angle(axis, angle):
    """ Conversion of unit axis and angle to 4x4 rotation matrix according to:
        http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
    """
    ct = np.cos(angle)
    st = np.sin(angle)
    vt = 1 - ct
    m_vt_0, m_vt_1, m_vt_2  = axis[:3] * vt
    m_st_0, m_st_1, m_st_2 = axis[:3] * st
    m_vt_0_1 = m_vt_0 * axis[1]
    m_vt_0_2 = m_vt_0 * axis[2]
    m_vt_1_2 = m_vt_1 * axis[2]
    return np.array([[ct + m_vt_0 * axis[0], -m_st_2 + m_vt_0_1, m_st_1 + m_vt_0_2, 0],
                     [m_st_2 + m_vt_0_1, ct + m_vt_1 * axis[1], -m_st_0 + m_vt_1_2, 0],
                     [-m_st_1 + m_vt_0_2, m_st_0 + m_vt_1_2, ct + m_vt_2 * axis[2], 0],
                     [0, 0, 0, 1]])

def np_translation3(x, y, z, w=1):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, w]])

def np_vector3(x, y, z):
    return np.array([x, y, z, 0]).reshape((4, 1))

# Uniform sampling of points on a sphere according to:
#  https://demonstrations.wolfram.com/RandomPointsOnASphere/
def np_sphere_sampling(n_points):
    r_theta = np.random.rand(n_points, 1) * np.pi
    r_u     = np.random.rand(n_points, 1)
    factor  = np.sqrt(1.0 - r_u**2)
    coords  = np.hstack((np.cos(r_theta) * factor, np.sin(r_theta) * factor, r_u))
    return coords # 

def sphere_sampling(n_points):
    return [np_vector3(*row) for row in np_sphere_sampling(n_rots)]

def random_rot_uniform(n_rots):
    # Random rotation angles about the z axis
    r_theta = np.random.rand(n_rots, 1)

    r_z_points = np_sphere_sampling(n_rots)
    x_angles   = np.arccos(r_z_points[:, 2]).reshape((n_rots, 1))
    z_angles   = np.arctan2(r_z_points[:, 1], r_z_points[:, 0]).reshape((n_rots, 1))
    return [dot(rotation3_axis_angle([0,0,1], r_z), 
                rotation3_axis_angle([1,0,0], r_x), 
                rotation3_axis_angle([0,0,1], r_t)) for r_t, r_x, r_z 
                                                    in np.hstack((r_theta, x_angles, z_angles))]


# Generates rotations about uniformly sampled axes
def random_rot_normal(n_rots, mean, std):
    return [np_rotation3_axis_angle(ax, r) for ax, r in 
                                           zip(np_sphere_sampling(n_rots), 
                                               np.random.normal(mean, std, n_rots))]

def np_random_quat_normal(n_rots, mean, std):
    return np.vstack([np.hstack((np.sin(r, ax), [np.cos(r)])) for ax, r in 
                                        zip(np_sphere_sampling(n_rots), np.random.normal(mean, std, n_rots))])

def np_random_normal_offset(n_points, mean, std):
    return np_sphere_sampling(n_points) * np.random.normal(mean, std, (n_points, 1))

def random_normal_offset(n_points, mean, std):
    return [np_vector3(*row) for row in np_random_normal_offset(n_points, mean, std)]

def random_normal_translation(n_points, mean, std):
    return [np_translation3(*row) for row in np_random_normal_offset(n_points, mean, std)]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def insert_omni_base(km, robot_path, root_link, world_frame='world', lin_vel=1.0, ang_vel=0.6):
    base_joint_path = robot_path + Path(f'joints/to_{world_frame}')
    base_op = ExecFunction(base_joint_path,
                           create_omnibase_joint_with_symbols,
                           CPath(f'{world_frame}/pose'),
                           CPath(robot_path + Path(f'links/{root_link}/pose')),
                           gm.vector3(0,0,1),
                           1.0, 0.6, CPath(robot_path))
    km.apply_operation_after(f'create {base_joint_path}',
                             f'create {robot_path}/links/{root_link}', base_op)
    km.apply_operation_after(f'connect {world_frame} {robot_path}/links/{root_link}',
                             f'create {base_joint_path}',
                             CreateAdvancedFrameConnection(base_joint_path,
                                                           Path(world_frame),
                                                           robot_path + Path(f'links/{root_link}')))
    km.clean_structure()


def insert_diff_base(km,
                     robot_path,
                     root_link,
                     world_frame='world',
                     wheel_radius=0.06,
                     wheel_distance=0.4,
                     wheel_vel_limit=0.6):
    base_joint_path = robot_path + Path(f'joints/to_{world_frame}')
    base_op = ExecFunction(base_joint_path,
                           create_diff_drive_joint_with_symbols,
                           CPath(f'{world_frame}/pose'),
                           CPath(robot_path + CPath(f'links/{root_link}/pose')),
                           wheel_radius,
                           wheel_distance,
                           wheel_vel_limit, CPath(robot_path))
    km.apply_operation_after(f'create {base_joint_path}',
                             f'create {robot_path}/links/{root_link}', base_op)
    km.apply_operation_after(f'connect {world_frame} {robot_path}/links/{root_link}',
                             f'create {base_joint_path}',
                             CreateAdvancedFrameConnection(base_joint_path,
                                                           Path(world_frame),
                                                           robot_path + Path(f'links/{root_link}')))
    km.clean_structure()
