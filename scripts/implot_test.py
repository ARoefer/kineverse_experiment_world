import rospy

import dearpygui.dearpygui as dpg

from multiprocessing import RLock
from sensor_msgs.msg import JointState as JointStateMsg

x_coords = list(range(5000))

all_deltas = {'r_shoulder_pan_joint': [],
              'r_shoulder_lift_joint': [],
              'r_upper_arm_roll_joint': [],
              'r_elbow_flex_joint': [],
              'r_wrist_flex_joint': [],
              'r_forearm_roll_joint': [],
              'r_wrist_roll_joint': []}
last_command  = {}


if __name__ == '__main__':
    with dpg.window(label="Test"):

        for name, deltas in all_deltas.items():
            # create plot
            with dpg.plot(label=f"{name} deltas", height=200, width=900):

                # optionally create legend
                dpg.add_plot_legend()

                # REQUIRED: create x and y axes
                dpg.add_plot_axis(dpg.mvXAxis, label="x")
                dpg.set_axis_limits(dpg.last_item(), 0, len(x_coords))
                dpg.add_plot_axis(dpg.mvYAxis, label="y")
                dpg.set_axis_limits(dpg.last_item(), -1, 1)

                # series belong to a y axis
                dpg.add_line_series(x_coords[:len(deltas)], deltas, label=f'{name}', parent=dpg.last_item(), id=f'{name}_deltas')


    log_lock = RLock()

    def cb_state(state_msg):
        with log_lock:
            for name, velocity in zip(state_msg.name, state_msg.velocity):
                if name in last_command and name in all_deltas:
                    delta = last_command[name] - velocity
                    all_deltas[name].append(delta)
                    if len(all_deltas[name]) > len(x_coords):
                        all_deltas[name] = all_deltas[name][-len(x_coords):]        
                    dpg.set_value(f'{name}_deltas', [x_coords[:len(all_deltas[name])], all_deltas[name]])

    def cb_control(control_msg):
        with log_lock:
            for name, velocity in zip(control_msg.name, control_msg.velocity):
                last_command[name] = velocity

    rospy.init_node('implot_test')

    sub_state   = rospy.Subscriber('/joint_states', JointStateMsg, callback=cb_state,   queue_size=1)
    sub_control = rospy.Subscriber('/control',      JointStateMsg, callback=cb_control, queue_size=1)


dpg.start_dearpygui()
