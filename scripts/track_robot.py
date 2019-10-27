#!/usr/bin/env python
import rospy

from tracking_example.tracking_node import TrackerNode

from geometry_msgs.msg import PoseStamped as PoseStampedMsg


if __name__ == '__main__':
    rospy.init_node('kineverse_tracking_node')
    tracker = TrackerNode('/tracked_js', '/pose_obs')

    tracker.track('/iai_kitchen/links/sink_area_dish_washer_door', 'iai_kitchen/sink_area_dish_washer_door')
    tracker.track('/iai_kitchen/links/sink_area_left_upper_drawer_main', 'iai_kitchen/sink_area_left_upper_drawer_main')
    tracker.track('/iai_kitchen/links/sink_area_left_middle_drawer_main', 'iai_kitchen/sink_area_left_middle_drawer_main')
    tracker.track('/iai_kitchen/links/sink_area_left_bottom_drawer_main', 'iai_kitchen/sink_area_left_bottom_drawer_main')
    tracker.track('/iai_kitchen/links/sink_area_trash_drawer_main', 'iai_kitchen/sink_area_trash_drawer_main')
    tracker.track('/iai_kitchen/links/iai_fridge_door', 'iai_kitchen/iai_fridge_door')
    tracker.track('/iai_kitchen/links/fridge_area_lower_drawer_main', 'iai_kitchen/fridge_area_lower_drawer_main')
    tracker.track('/iai_kitchen/links/oven_area_oven_door', 'iai_kitchen/oven_area_oven_door')
    tracker.track('/iai_kitchen/links/oven_area_right_drawer_main', 'iai_kitchen/oven_area_right_drawer_main')
    tracker.track('/iai_kitchen/links/oven_area_left_drawer_main', 'iai_kitchen/oven_area_left_drawer_main')
    tracker.track('/iai_kitchen/links/oven_area_area_middle_upper_drawer_main', 'iai_kitchen/oven_area_area_middle_upper_drawer_main')
    tracker.track('/iai_kitchen/links/oven_area_area_middle_lower_drawer_main', 'iai_kitchen/oven_area_area_middle_lower_drawer_main')

    while not rospy.is_shutdown():
        rospy.sleep(1000)


