#!/usr/bin/env python
import rospy
import pandas as pd
import matplotlib.pyplot as plt

from kineverse.model.paths            import Path
from kineverse.visualization.plotting import ValueRecorder, draw_recorders, convert_qp_builder_log
from kineverse.utils                  import res_pkg_path
from kineverse_experiment_world.obsessive_obj_closer import ObsessiveObjectCloser

if __name__ == '__main__':
    rospy.init_node('obsessive_kitchen_closer')


    closer = ObsessiveObjectCloser(Path('iai_oven_area'), Path('fetch/links/gripper_link'), 
                                   Path('fetch/links/head_camera_link'), 
                                   '/iai_kitchen/joint_states', 
                                   '/fetch')


    while not rospy.is_shutdown():
        rospy.sleep(1000)

    if closer.pushing_controller is not None:
        recW, recB, recC, recs = convert_qp_builder_log(closer.pushing_controller)
        
        draw_recorders([recW, recB, recC] + [r for _, r in sorted(recs.items())], 1, 12, 6).savefig(res_pkg_path('package://kineverse_experiment_world/test/keep_contact.png'))