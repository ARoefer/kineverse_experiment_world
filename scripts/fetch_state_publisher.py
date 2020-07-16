#!/usr/bin/env python

import rospy

from kineverse.gradients.diff_logic import Position

from kineverse_msgs.msg import ValueMap as ValueMapMsg

if __name__ == '__main__':
    rospy.init_node('fetch_pose_publisher')

    pub = rospy.Publisher('fetch/state', ValueMapMsg, queue_size=1)

    poses = [{Position("fetch__torso_lift_joint")    : 0.1,
              Position("fetch__shoulder_pan_joint")  : 1.32,
              Position("fetch__shoulder_lift_joint") : 1.40,
              Position("fetch__upperarm_roll_joint") : -0.2,
              Position("fetch__elbow_flex_joint")    : 1.72,
              Position("fetch__forearm_roll_joint")  : 0.0,
              Position("fetch__wrist_flex_joint")    : 1.66,
              Position("fetch__wrist_roll_joint")    : 0.0,
              Position("fetch__head_tilt_joint")     : 0.4,
              Position("fetch__head_pan_joint")      : -0.6,
              Position("fetch__r_gripper_finger_joint") : 0.05},
             
             {Position("fetch__torso_lift_joint")    : 0.0,
              Position("fetch__shoulder_pan_joint")  : 0.0,
              Position("fetch__shoulder_lift_joint") : 0.0,
              Position("fetch__upperarm_roll_joint") : 0.0,
              Position("fetch__elbow_flex_joint")    : 0.0,
              Position("fetch__forearm_roll_joint")  : 0.0,
              Position("fetch__head_tilt_joint")     : 0.0,
              Position("fetch__head_pan_joint")      : 0.0,
              Position("fetch__r_gripper_finger_joint") : 0.0,
              Position("fetch__wrist_flex_joint")    : 1.66,
              Position("fetch__wrist_roll_joint")    : 0.0},
             
             {Position("fetch__torso_lift_joint")    : 0.3,
              Position("fetch__shoulder_pan_joint")  : 0.0,
              Position("fetch__shoulder_lift_joint") : 1.20,
              Position("fetch__upperarm_roll_joint") : 0.0,
              Position("fetch__elbow_flex_joint")    : 1.0,
              Position("fetch__forearm_roll_joint")  : 0.4,
              Position("fetch__wrist_flex_joint")    : -0.4,
              Position("fetch__wrist_roll_joint")    : 0.8,
              Position("fetch__head_tilt_joint")     : -0.4,
              Position("fetch__head_pan_joint")      : 0.6,
              Position("fetch__r_gripper_finger_joint") : 0.02},
            ]


    while not rospy.is_shutdown():
        for p in poses:
            _ = raw_input('Hit enter to publish a pose')
            msg = ValueMapMsg()
            msg.header.stamp = rospy.Time.now()
            msg.symbol, msg.value = zip(*[(str(k), v) for k, v in p.items()])

            pub.publish(msg)
            rospy.sleep(0.3)

        break
