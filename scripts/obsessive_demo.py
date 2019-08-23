#!/usr/bin/env python
import rospy

from kineverse.model.paths import Path
from kineverse_experiment_world.obsessive_obj_closer import ObsessiveObjectCloser

if __name__ == '__main__':
    rospy.init_node('obsessive_kitchen_closer')


    closer = ObsessiveObjectCloser(Path('iai_oven_area'), Path('fetch/links/r_gripper_finger_link'), 
                                   Path('fetch/links/head_camera_link'), 
                                   '/iai_kitchen/joint_states', 
                                   '/fetch')


    while not rospy.is_shutdown():
        rospy.sleep(1000)
