#!/usr/bin/env python
import rospy
import os
import math
import sys

from iai_bullet_sim.realtime_simulator_node import FixedTickSimulator
from iai_bullet_sim.ros_plugins             import JointVelocityController, ResetTrajectoryPositionController, JSPublisher, OdometryPublisher
from iai_bullet_sim.multibody               import OmniBaseDriver
from iai_bullet_sim.full_state_node         import FullStatePublishingNode
from iai_bullet_sim.srv                     import AddURDFRequest, AddRigidObjectRequest
from fetch_giskard.simulator_plugins        import FetchDriver

from kineverse_experiment_world.simulator_plugins import PoseObservationPublisher

from urdf_parser_py.urdf import URDF
from kineverse.urdf_fix  import urdf_filler
from kineverse.utils     import res_pkg_path


if __name__ == '__main__':
    rospy.init_node('kineverse_bullet_sim')

    filtered_args = [x for x in sys.argv[1:] if ':=' not in x]
    robot_str   = filtered_args[0] if len(filtered_args) > 0 else 'fetch'
    use_tracker = bool(filtered_args[1]) if len(filtered_args) > 1 else False

    node = FixedTickSimulator(FullStatePublishingNode)
    node.init(mode='gui')
    #node.init(mode='direct')

    req = AddRigidObjectRequest()
    req.geom_type = 'box'
    req.extents.x = req.extents.y = 10
    req.extents.z = 1
    req.pose.position.z = -0.5
    req.pose.orientation.w = 1
    req.name = 'floor'

    node.srv_add_rigid_body(req) # This is dumb!

    tmp_urdf = open('/tmp/temp_urdf.urdf', 'w') #tempfile.NamedTemporaryFile()
    filled_model = urdf_filler(URDF.from_xml_file(res_pkg_path('package://iai_kitchen/urdf_obj/IAI_kitchen.urdf')))
    tmp_urdf.write(filled_model.to_xml_string())
    tmp_urdf.close()
    #tmp_urdf.write(urdf_filler(URDF.from_xml_file(res_pkg_path('package://faculty_lounge_kitchen_description/urdfs/kitchen.urdf'))).to_xml_string())
    #snd_urdf = URDF.from_xml_file('/tmp/temp_urdf.urdf')

    req = AddURDFRequest()
    req.urdf_path  = '/tmp/temp_urdf.urdf' #'package://iai_kitchen/urdf_obj/IAI_kitchen.urdf'
    req.name       = 'iai_kitchen'
    req.fixed_base = True
    req.pose.orientation.w = 1

    node.srv_add_urdf(req) # This is reeeeeeally stupid!

    angle = math.pi * 0

    if robot_str == 'fetch':
        req.urdf_path  = 'package://fetch_description/robots/fetch.urdf'
    elif robot_str == 'pr2':
        req.urdf_path  = 'package://iai_pr2_description/robots/pr2_calibrated_with_ft2.xml'
    else:
        raise Exception('Unknown robot {}'.format(robot_str))

    req.name       = robot_str
    req.fixed_base = False
    req.pose.orientation.w = math.cos(angle * 0.5)
    req.pose.orientation.z = math.sin(angle * 0.5)

    with open('test_robot.urdf', 'w') as test:
        test.write(URDF.from_xml_file(res_pkg_path(req.urdf_path)).to_xml_string())

    node.srv_add_urdf(req) # Still reeeeeeally stupid!

    sim     = node.sim
    kitchen = sim.get_body('iai_kitchen')
    robot   = sim.get_body(robot_str)
    if robot_str == 'fetch':
        robot.joint_driver = FetchDriver(1, 0.6, 'linear_joint', z_ang_joint='angular_joint')
    else:
        robot.joint_driver = OmniBaseDriver(1, 0.6, 'localization_x', 'localization_y', 'localization_a')
        # pass

    if robot_str == 'fetch':
        camera_link = 'head_camera_link'
    elif robot_str == 'pr2':
        camera_link = 'head_mount_link'

    #sim.register_plugin(JSPublisher(kitchen, 'iai_kitchen'))
    #sim.register_plugin(JSPublisher(fetch, 'fetch'))
    sim.register_plugin(OdometryPublisher(sim, robot))
    sim.register_plugin(JointVelocityController(robot, robot_str))
    sim.register_plugin(PoseObservationPublisher(robot, camera_link, 0.942478, 0.4, 6.0, 0.0, debug=True)) # 0.065

    node.run()

    while not rospy.is_shutdown():
        rospy.sleep(1000)

    node.kill()
    os.remove('/tmp/temp_urdf.urdf')
