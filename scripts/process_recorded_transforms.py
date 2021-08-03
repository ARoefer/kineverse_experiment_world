import csv
import kineverse.gradients.gradient_math as gm
import math
import matplotlib.pyplot as pyplot
import numpy as np
import rospy
import scipy.optimize as optimize

from argparse import ArgumentParser
from kineverse.model.history import Timeline, StampedData


def calculate_rot_around_y(timeline):
    out = Timeline()
    x_vec = np.eye(4)[:, 0] 
    for d in timeline:
        t_vec = d.tf.dot(x_vec)
        out.add(StampedData(d.stamp, y_angle=math.atan2(t_vec[2], t_vec[0])))

    return out

# Modifies the original timelines!
def transform_to_common_origin(*timelines):
    origin = min(t[0].stamp for t in timelines if len(t) > 0)

    for t in timelines:
        for x in t:
            x.stamp = rospy.Time(0) + (x.stamp - origin)

    return timelines


def sample_timeline(stamp, timeline, value='y_angle'):
    x, data = timeline.get_ceil(stamp)

    if data is None:
        x    = len(timeline) - 1
        data = timeline[-1]

    if x == 0:
        return getattr(data, value)


    p_data = timeline[x - 1]
    segment_length = (data.stamp - p_data.stamp).to_sec()
    if segment_length < 1e-4:
        return getattr(data, value)

    fac = (stamp - p_data.stamp).to_sec() / segment_length

    return (1 - fac) * getattr(p_data, value) + fac * getattr(data, value)


angle_hinge      = gm.Symbol('angle_top')
x_hinge_in_parent, z_hinge_in_parent = [gm.Symbol(f'hinge_in_parent_{x}') for x in 'xz']
x_child_in_hinge,   z_child_in_hinge = [gm.Symbol(f'child_in_hinge_{x}')  for x in 'xz']

fwd_kinematic_hinge = gm.dot(gm.translation3(x_hinge_in_parent, 0, z_hinge_in_parent),
                             gm.rotation3_axis_angle(gm.vector3(0, -1, 0), angle_hinge),
                             gm.translation3(x_child_in_hinge, 0, z_child_in_hinge))
# we don't care about the location in y
fwd_kinematic_hinge_residual_tf = gm.speed_up(gm.dot(gm.diag(1, 0, 1, 1), fwd_kinematic_hinge), gm.free_symbols(fwd_kinematic_hinge))

def compute_y_hinge_axis_residual(x, angle_tf):
    """Computes the residual for an estimate of the hinge locations on the nobilia shelf
    
    Args:
        x (np.ndarray): [xh, zh, xc, zc] x and z locations of hinge in parent and child in hinge
        angle_1_tf (TYPE): list of tuples (a, tf) where a is the angle of the top panel relative to the parent
    """
    param_dict = {str(x_hinge_in_parent): x[0],
                  str(z_hinge_in_parent): x[1],
                  str(x_child_in_hinge): x[2],
                  str(z_child_in_hinge): x[3]}
    out = []
    for angle, obs_tf in angle_tf:
        param_dict[str(angle_hinge)] = angle
        out.append(np.sqrt(np.sum((fwd_kinematic_hinge_residual_tf(**param_dict) - obs_tf)[:3, 3]**2)))
    return out


if __name__ == '__main__':

    parser = ArgumentParser(description='Parses a file of recorded transforms from a csv.')
    parser.add_argument('csv_file', help='CSV file to process')

    args = parser.parse_args()

    with open(args.csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')

        transforms = {}

        for i, row in enumerate(reader):
            if i == 0: # skip header
                continue

            stamp  = rospy.Time(int(row[0]), int(row[1]))
            frames = (row[2], row[3])
            transform = gm.to_numpy(gm.frame3_quaternion(*[float(x) for x in row[4:]]))
            if frames not in transforms:
                transforms[frames] = Timeline()

            transforms[frames].add(StampedData(stamp, tf=transform))

    y_rots = {p: calculate_rot_around_y(t) for p, t in transforms.items()}

    transform_to_common_origin(*y_rots.values())

    colors = ['red', 'blue', 'green', 'black', 'cyan', 'purple']

    for (frames, t), c in zip(y_rots.items(), colors[:len(y_rots)]):
        pyplot.plot([d.stamp.to_sec() for d in t], [d.y_angle for d in t], 
                     color=c, label=f'{frames[1]} -> {frames[0]}')
        
    pyplot.legend()
    pyplot.savefig(f'{args.csv_file[:-4]}_timeplot.png')
    pyplot.close()

    top_frame    = ('obs_shelf_body', 'obs_shelf_top_panel')
    bottom_frame = ('obs_shelf_top_panel', 'obs_shelf_bottom_panel')

    mapping_timeline = Timeline()

    for data in y_rots[top_frame]:
        mapping_timeline.add(StampedData(stamp=data.y_angle, 
                                         value=sample_timeline(data.stamp, y_rots[bottom_frame])))

    pyplot.plot([s.stamp for s in mapping_timeline],
                [s.value for s in mapping_timeline])
    pyplot.xlabel(f'y-angle {top_frame[1]} -> {top_frame[0]}')
    pyplot.ylabel(f'y-angle {bottom_frame[1]} -> {bottom_frame[0]}')
    pyplot.savefig(f'{args.csv_file[:-4]}_funtion_plot.png')
    pyplot.close()

    top_angle_tf_map    = [(dy.y_angle, dt.tf) for dy, dt in zip(y_rots[top_frame], transforms[top_frame])]
    bottom_angle_tf_map = [(dy.y_angle, dt.tf) for dy, dt in zip(y_rots[bottom_frame], transforms[bottom_frame])]

    translations_top = optimize.least_squares(compute_y_hinge_axis_residual, [0]*4, args=(top_angle_tf_map, ))
    translations_bottom = optimize.least_squares(compute_y_hinge_axis_residual, [0]*4, args=(bottom_angle_tf_map, ))

    print(f'Solution top hinge: cost = {translations_top.cost} {translations_top.x}')
    print(f'Solution bottom hinge: cost = {translations_bottom.cost} {translations_bottom.x}')