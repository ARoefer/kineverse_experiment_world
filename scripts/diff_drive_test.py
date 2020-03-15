#!/usr/bin/env python
import rospy
from kineverse.gradients.gradient_math import *
from kineverse.visualization.ros_visualizer import ROSVisualizer
from kineverse.motion.min_qp_builder import TypedQPBuilder  as TQPB, \
                                            SoftConstraint  as SC,   \
                                            ControlledValue as CV,   \
                                            Constraint

from math import pi

if __name__ == '__main__':
    rospy.init_node('diff_drive_node')

    DT_SYM = Position('dt')

    r = 0.05
    L = 0.4
    r_limit = 1.0 / (pi * r * 2)  # Top speed is 1m/s 
    a_limit = r_limit * (r / L) - r_limit * (-r / L)

    print(a_limit)

    rw_v, lw_v = [Velocity(x + 'w') for x in 'rl']
    l_v,   a_v = [Velocity(x) for x in 'ra']

    x1, y1, a1 = [Position(x + '1') for x in 'xya']
    x2, y2, a2 = [Position(x + '2') for x in 'xya']


    diff_drive = frame3_axis_angle(vector3(0,0,1), 
                                   GC(a1, {rw_v:   r / L,
                                           lw_v: - r / L}),
                                   point3(GC(x1, {rw_v: cos(a1) * r * 0.5,
                                                  lw_v: cos(a1) * r * 0.5}), 
                                          GC(y1, {rw_v: sin(a1) * r * 0.5,
                                                  lw_v: sin(a1) * r * 0.5}), 0))

    my_drive   = frame3_axis_angle(vector3(0,0,1),  
                                   GC(a2, {a_v: 1}), 
                                   point3(GC(x2, {l_v: cos(a2)}), 
                                          GC(y2, {l_v: sin(a2)}), 0))


    int_rules = {
        x1: x1 + DT_SYM * (rw_v * cos(a1) * r * 0.5 + lw_v * cos(a1) * r * 0.5),
        y1: y1 + DT_SYM * (rw_v * sin(a1) * r * 0.5 + lw_v * sin(a1) * r * 0.5),
        a1: a1 + DT_SYM * (rw_v * (r / L) + lw_v * (- r / L)),
        x2: x2 + DT_SYM * l_v * cos(a2),
        y2: y2 + DT_SYM * l_v * sin(a2),
        a2: a2 + DT_SYM * a_v
    }

    goal = point3(1, 1, 0)

    diff_dist = norm(diff_drive * point3(0.1, 0, 0) - goal) # 
    md_dist   = norm(my_drive * point3(0.1, 0, 0) - goal)   # 

    qpb = TQPB({},
               {'goal diff_drive': SC(-diff_dist, -diff_dist, 1, diff_dist),
                'goal my_drive':   SC(-md_dist,     -md_dist, 1,   md_dist)},
               {'rw_v': CV(-r_limit, r_limit, rw_v, 0.001),
                'lw_v': CV(-r_limit, r_limit, lw_v, 0.001),
                'l_v':  CV(-1,             1,  l_v, 0.001),
                'a_v':  CV(-a_limit, a_limit,  a_v, 0.001)})




    full_diff_points = []
    md_points   = []
    cmds = [
            {rw_v: -0.8, lw_v: -0.8},
            {rw_v: -1.0, lw_v: -0.0},
            {rw_v: -0.0, lw_v: -1.0},
            {rw_v: 0.8, lw_v: 0.8},
            {rw_v: 1.0, lw_v: 0.0},
            {rw_v: 0.0, lw_v: 1.0},
            # {rw_v: -1.0, lw_v: 1.0},
            ]
    for cmd in cmds: 
        diff_points = []
        state = {s: 0.0 for s in qpb.free_symbols}
        state[DT_SYM] = 0.1

        for x in range(100):

            state.update({s: e.subs(cmd).subs(state) for s, e in int_rules.items()})

            diff_points.append((state[x1], state[y1], state[a1]))
            md_points.append((state[x2], state[y2], state[a2]))

            # if qpb.equilibrium_reached():
            #     print('optimization ende early')
            #     break

        full_diff_points.append(diff_points)


    vis = ROSVisualizer('diff_drive_vis', 'world')


    # md_p = [point3(x, y, 0) for x, y, _ in md_points]
    # md_d = [vector3(cos(a), sin(a), 0) for _, _, a in md_points]

    vis.begin_draw_cycle('paths')
    # vis.draw_sphere('paths', goal, 0.02, r=0, b=1)
    for n, diff_points in enumerate(full_diff_points):
        diff_p = [point3(x, y, 0) + vector3((n / 3) * 0.5, (n % 3)* -0.5, 0) for x, y, _ in diff_points]
        diff_d = [vector3(cos(a), sin(a), 0) for _, _, a in diff_points]
        vis.draw_strip('paths', se.eye(4), 0.02, diff_p)
    #vis.draw_strip('paths', spw.eye(4), 0.02, md_p, r=0, g=1)
    vis.render('paths')

    rospy.sleep(0.3)
