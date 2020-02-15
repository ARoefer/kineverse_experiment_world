import numpy as np

from kineverse.gradients.gradient_math import *

# Uniform sampling of points on a sphere according to:
#  https://demonstrations.wolfram.com/RandomPointsOnASphere/
def np_sphere_sampling(n_points):
    r_theta = np.random.rand(n_points, 1) * np.pi
    r_u     = np.random.rand(n_points, 1)
    factor  = np.sqrt(1.0 - r_u**2)
    coords  = np.hstack((np.cos(r_theta) * factor, np.sin(r_theta) * factor, r_u))
    return coords # 

def sphere_sampling(n_points):
    return [vector3(*row) for row in np_sphere_sampling(n_rots)]

def random_rot_uniform(n_rots):
    # Random rotation angles about the z axis
    r_theta = np.random.rand(n_rots, 1)

    r_z_points = np_sphere_sampling(n_rots)
    x_angles   = np.arccos(r_z_points[:, 2]).reshape((n_rots, 1))
    z_angles   = np.arctan2(r_z_points[:, 1], r_z_points[:, 0]).reshape((n_rots, 1))
    return [rotation3_axis_angle([0,0,1], r_z) * 
            rotation3_axis_angle([1,0,0], r_x) * 
            rotation3_axis_angle([0,0,1], r_t) for r_t, r_x, r_z in np.hstack((r_theta, x_angles, z_angles))]


# Generates rotations about uniformly sampled axes
def random_rot_normal(n_rots, mean, std):
    return [rotation3_axis_angle(ax, r) for ax, r in 
                                        zip(np_sphere_sampling(n_rots), np.random.normal(mean, std, n_rots))]

def np_random_quat_normal(n_rots, mean, std):
    return np.vstack([np.hstack((np.sin(r, ax), [np.cos(r)])) for ax, r in 
                                        zip(np_sphere_sampling(n_rots), np.random.normal(mean, std, n_rots))])

def np_random_normal_offset(n_points, mean, std):
    return np_sphere_sampling(n_points) * np.random.normal(mean, std, (n_points, 1))

def random_normal_offset(n_points, mean, std):
    return [vector3(*row) for row in np_random_normal_offset(n_points, mean, std)]

def random_normal_translation(n_points, mean, std):
    return [translation3(*row) for row in np_random_normal_offset(n_points, mean, std)]
