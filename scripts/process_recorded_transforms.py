import csv
import kineverse.gradients.gradient_math as gm
import numpy as np
import rospy

from argparse import ArgumentParser
from kineverse.model.history import Timeline, StampedData


def calculate_rot_around_y(timeline: list(StampedData)):
	out = Timeline()
	x_vec = np.eye(4)[:, 0] 
	for d in timeline:
		t_vec = d.tf.dot(x_vec)
		out.append(StampedData(d.stamp, y_angle=np.atan2(t_vec[2], t_vec[0])))

	return out


if __name__ == '__main__':

	parser = ArgumentParser(description='Parses a file of recorded transforms from a csv.')
	parser.add_argument('csv_file', help='CSV file to process')

	args = parser.parse_args()

	with open(args.csv_file, 'r') as f:
		reader = csv.reader(f, delimiter=',')

		transforms = {}

		for row in reader:
			stamp  = rospy.Time(int(row[0]), int(row[1]))
			frames = (row[2], row[3])
			transform = gm.to_numpy(gm.frame3_quaternion(*row[4:]))
			if frames not in transforms:
				transforms[frames] = Timeline()

			transforms[frames].append(StampedData(stamp, tf=transform))

	y_rots = {p: calculate_rot_around_y(t) for p, t in transforms.items()}

	

