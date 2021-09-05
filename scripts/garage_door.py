import kineverse.gradients.gradient_math as gm
import numpy as np
import matplotlib.pyplot as plt

from kineverse.visualization.plotting import hsv_to_rgb, \
											 rgb_to_hex


if __name__=='__main__':

	a, b = [gm.Position(x) for x in 'ab']

	l = 2
	a_in_w = gm.dot(gm.translation3(0, 0, 2), gm.translation3(0, 0, -a))
	d_in_a = gm.rotation3_axis_angle(gm.vector3(0, 1, 0), gm.acos(a / l))
	d_in_w = gm.dot(a_in_w, d_in_a)
	A = gm.pos_of(a_in_w)
	B = gm.dot(d_in_w, gm.point3(0, 0, l))
	C = gm.dot(d_in_w, gm.point3(0, 0, l * 0.5))
	D = gm.dot(d_in_w, gm.point3(0, 0, l * 0.25))
	E = gm.dot(d_in_w, gm.point3(0, 0, l * 0.75))

	lock_bound = gm.alg_not(gm.alg_and(gm.less_than(b, 0.3), gm.less_than(1.99, a)))

	# PLOT OF MOVEMENT

	As = []
	Bs = []
	Cs = []
	Ds = []
	Es = []

	for x in np.linspace(0, l, 20):
		q = {a: x}
		As.append(np.take(gm.subs(A, q).flatten(), (0, 2)))
		Bs.append(np.take(gm.subs(B, q).flatten(), (0, 2)))
		Cs.append(np.take(gm.subs(C, q).flatten(), (0, 2)))
		Ds.append(np.take(gm.subs(D, q).flatten(), (0, 2)))
		Es.append(np.take(gm.subs(E, q).flatten(), (0, 2)))

	As = np.vstack(As)
	Bs = np.vstack(Bs)
	Cs = np.vstack(Cs)
	Ds = np.vstack(Ds)
	Es = np.vstack(Es)

	print(Bs)
	# exit()

	data = [As, Bs, Cs, Ds, Es]

	plt.figure(figsize=(3, 3))
	for data, h in zip(data, np.linspace(0, 1, len(data), endpoint=False)):
		for idx, x in zip(range(1, data.shape[0]), np.linspace(0.2, 1.0, data.shape[0])):
			plt.plot(data[idx-1:idx+1, 0], data[idx-1:idx+1, 1], color=rgb_to_hex(*hsv_to_rgb(h, x, 1)), linewidth=2)

	plt.title('Movement of Garage Door')
	plt.xlabel('Location X in $m$')
	plt.ylabel('Location Y in $m$')
	plt.tight_layout()
	plt.savefig('garage_door_plot.png')
	plt.close()

	# PLOT OF VELOCITY CONSTRAINT

	bounds = []


	plt.figure(figsize=(3, 3))

	a_steps = 2
	for x_a, h, ls in zip(np.linspace(1, 2, a_steps), 
					      np.linspace(0, 0.8, a_steps, endpoint=False),
					      ['-', '--']):
		series = np.vstack([(x_b, gm.subs(lock_bound, {a: x_a, b: x_b}).flatten()[0]) for x_b in np.linspace(0, 0.6, 50)])
		plt.plot(series.T[0], series.T[1], 
				 color=rgb_to_hex(*hsv_to_rgb(h, 1, 1)), 
				 linewidth=2,
				 label=f'$a = {x_a}$',
				 ls=ls)

	plt.title('Constraint $1-locked$')
	plt.xlabel('$1 - locked(a, b)$')
	plt.ylabel('b')
	plt.yticks(ticks=[0, 1], 
			   labels=['closed (0.0)', 'open (1.0)'], 
			   rotation='vertical', 
			   ha='center',
			   va='center')
	plt.tight_layout()
	plt.savefig('garage_door_lock_plot.png')
	plt.close()
