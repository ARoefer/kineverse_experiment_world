#!/usr/bin/env python
import rospy
import pandas as pd
import matplotlib.pyplot as plt
import sys

from kineverse.motion.min_qp_builder  import PANDA_LOGGING
from kineverse.model.paths            import Path
from kineverse.visualization.plotting import ValueRecorder, draw_recorders, convert_qp_builder_log
from kineverse.utils                  import res_pkg_path
from kineverse_experiment_world.obsessive_obj_closer import ObsessiveObjectCloser

if __name__ == '__main__':
    rospy.init_node('obsessive_kitchen_closer')


    robot_str = [x for x in sys.argv[1:] if ':=' not in x][0] if len([x for x in sys.argv[1:] if ':=' not in x]) > 0 else 'fetch'
    
    if robot_str == 'fetch':
        resting_pose = {"shoulder_pan_joint"  : 1.32,
                        "shoulder_lift_joint" : 1.40,
                        "upperarm_roll_joint" : -0.2,
                        "elbow_flex_joint"    : 1.72,
                        "forearm_roll_joint"  : 0.0,
                        "wrist_flex_joint"    : 1.66,
                        "wrist_roll_joint"    : 0.0}
        closer = ObsessiveObjectCloser(Path('iai_oven_area'), Path('fetch/links/gripper_link'), 
                                       Path('fetch/links/head_camera_link'), 
                                       '/iai_kitchen/joint_states', 
                                       '/fetch', resting_pose=resting_pose)
    elif robot_str == 'pr2':
        resting_pose = {'l_elbow_flex_joint' : -2.1213,
                        'l_shoulder_lift_joint': 1.2963,
                        'l_shoulder_pan_joint': 0.0,
                        'l_upper_arm_roll_joint': 0.0,
                        'l_forearm_roll_joint': 0.0,
                        'l_wrist_flex_joint' : -1.05,
                        'r_elbow_flex_joint' : -2.1213,
                        'r_shoulder_lift_joint': 1.2963,
                        'r_shoulder_pan_joint': 0.0,
                        'r_upper_arm_roll_joint': 0.0,
                        'r_forearm_roll_joint': 0.0,
                        'r_wrist_flex_joint' : -1.05,
                        'torso_lift_joint'   : 0.16825}
        closer = ObsessiveObjectCloser(Path('iai_oven_area'), Path('pr2/links/r_gripper_r_finger_tip_link'), 
                                       Path('pr2/links/head_mount_link'), 
                                       '/iai_kitchen/joint_states', 
                                       '/pr2', resting_pose=resting_pose)


    while not rospy.is_shutdown():
        rospy.sleep(1000)

    if closer.pushing_controller is not None and PANDA_LOGGING:
        recW, recB, recC, recs = convert_qp_builder_log(closer.pushing_controller)
        
        draw_recorders([recW, recB, recC] + [r for _, r in sorted(recs.items())], 1, 12, 6).savefig(res_pkg_path('package://kineverse_experiment_world/test/keep_contact.png'))
