#!/usr/bin/env python

import rospy

from kineverse.gradients.diff_logic import Position

from kineverse.msg import ValueMap as ValueMapMsg

if __name__ == '__main__':
    rospy.init_node('pr2_pose_publisher')

    pub = rospy.Publisher('pr2/state', ValueMapMsg, queue_size=1)

    poses = [{Position('pr2__l_elbow_flex_joint') : -2.1213,
              Position('pr2__l_shoulder_lift_joint'): 1.2963,
              Position('pr2__l_shoulder_pan_joint'): 0.0,
              Position('pr2__l_upper_arm_roll_joint'): 0.0,
              Position('pr2__l_forearm_roll_joint'): 0.0,
              Position('pr2__l_wrist_flex_joint') : -1.05,
              Position('pr2__r_elbow_flex_joint') : -2.1213,
              Position('pr2__r_shoulder_lift_joint'): 1.2963,
              Position('pr2__r_shoulder_pan_joint'): 0.0,
              Position('pr2__r_upper_arm_roll_joint'): 0.0,
              Position('pr2__r_forearm_roll_joint'): 0.0,
              Position('pr2__r_wrist_flex_joint') : -1.05,
              Position('pr2__torso_lift_joint')   : 0.16825,
              Position('pr2__head_pan_joint')     : 0.0,
              Position('pr2__head_tilt_joint')    : 0.0,},
             
             {Position('pr2__l_elbow_flex_joint') : -1.7,
              Position('pr2__l_shoulder_lift_joint'): 1.0,
              Position('pr2__l_shoulder_pan_joint'): 1.0,
              Position('pr2__l_upper_arm_roll_joint'): 1.2,
              Position('pr2__l_forearm_roll_joint'): 0.2,
              Position('pr2__l_wrist_flex_joint') : -1.05,
              Position('pr2__r_elbow_flex_joint') : -1.2,
              Position('pr2__r_shoulder_lift_joint'): 1.0,
              Position('pr2__r_shoulder_pan_joint'): -1.0,
              Position('pr2__r_upper_arm_roll_joint'): -1.2,
              Position('pr2__r_forearm_roll_joint'): 0.0,
              Position('pr2__r_wrist_flex_joint') : -1.05,
              Position('pr2__torso_lift_joint')   : 0.16825,
              Position('pr2__head_pan_joint')     : 0.2,
              Position('pr2__head_tilt_joint')    : 0.4,},

             {Position('pr2__l_elbow_flex_joint') : -1.4,
              Position('pr2__l_shoulder_lift_joint'): 0.4,
              Position('pr2__l_shoulder_pan_joint'): 0.6,
              Position('pr2__l_upper_arm_roll_joint'): 1.6,
              Position('pr2__l_forearm_roll_joint'): 1.5,
              Position('pr2__l_wrist_flex_joint') : -1.6,
              Position('pr2__r_elbow_flex_joint') : -1.4,
              Position('pr2__r_shoulder_lift_joint'): 0.4,
              Position('pr2__r_shoulder_pan_joint'): -0.6,
              Position('pr2__r_upper_arm_roll_joint'): -1.6,
              Position('pr2__r_forearm_roll_joint'): -1.5,
              Position('pr2__r_wrist_flex_joint') : -1.05,
              Position('pr2__torso_lift_joint')   : 0.3,
              Position('pr2__head_pan_joint')     : -0.5,
              Position('pr2__head_tilt_joint')    : 0.4}
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
