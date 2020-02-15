#!/usr/bin/env python
import rospy

from kineverse.visualization.plotting import convert_qp_builder_log, draw_recorders
from kineverse.utils import res_pkg_path

from kineverse_experiment_world.tracking_node import TrackerNode

from geometry_msgs.msg import PoseStamped as PoseStampedMsg


if __name__ == '__main__':
    rospy.init_node('kineverse_tracking_node')
    tracker = TrackerNode('/tracked/state', '/pose_obs', 1, 3, visualize=True)

    tracker.track('/iai_oven_area/links/sink_area_dish_washer_door', 'iai_kitchen/sink_area_dish_washer_door')
    tracker.track('/iai_oven_area/links/sink_area_left_upper_drawer_main', 'iai_kitchen/sink_area_left_upper_drawer_main')
    tracker.track('/iai_oven_area/links/sink_area_left_middle_drawer_main', 'iai_kitchen/sink_area_left_middle_drawer_main')
    tracker.track('/iai_oven_area/links/sink_area_left_bottom_drawer_main', 'iai_kitchen/sink_area_left_bottom_drawer_main')
    tracker.track('/iai_oven_area/links/sink_area_trash_drawer_main', 'iai_kitchen/sink_area_trash_drawer_main')
    tracker.track('/iai_oven_area/links/iai_fridge_door', 'iai_kitchen/iai_fridge_door')
    tracker.track('/iai_oven_area/links/fridge_area_lower_drawer_main', 'iai_kitchen/fridge_area_lower_drawer_main')
    tracker.track('/iai_oven_area/links/oven_area_oven_door', 'iai_kitchen/oven_area_oven_door')
    tracker.track('/iai_oven_area/links/oven_area_right_drawer_main', 'iai_kitchen/oven_area_right_drawer_main')
    tracker.track('/iai_oven_area/links/oven_area_left_drawer_main', 'iai_kitchen/oven_area_left_drawer_main')
    tracker.track('/iai_oven_area/links/oven_area_area_middle_upper_drawer_main', 'iai_kitchen/oven_area_area_middle_upper_drawer_main')
    tracker.track('/iai_oven_area/links/oven_area_area_middle_lower_drawer_main', 'iai_kitchen/oven_area_area_middle_lower_drawer_main')

    while not rospy.is_shutdown():
        rospy.sleep(1000)


    #rec_w, rec_b, rec_c, recs = convert_qp_builder_log(tracker.integrator.qp_builder)

    #plot_dir = res_pkg_path('package://kineverse_experiment_world/test/plots')
    #draw_recorders([rec_b, rec_c] + [r for _, r in sorted(recs.items())], 1, 8, 4).savefig('{}/tracking_constraints.png'.format(plot_dir))
