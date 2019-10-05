#!/usr/bin/env python
import rospy
import pandas as pd
import matplotlib.pyplot as plt

from kineverse.model.paths            import Path
from kineverse.visualization.plotting import ValueRecorder,draw_recorders
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

    dfs_H, dfs_A, commands = closer.get_push_controller_logs()

    lbs     = pd.DataFrame([df.T['lb'].T for df in dfs_H]).reset_index(drop=True)
    ubs     = pd.DataFrame([df.T['ub'].T for df in dfs_H]).reset_index(drop=True)
    weights = pd.DataFrame([df.T['weight'].T for df in dfs_H]).reset_index(drop=True)
    kcs     = pd.DataFrame([df.T['keep_contact'].T for df in dfs_A]).reset_index(drop=True)

    recA = ValueRecorder('keep_contact', *[c for c in kcs.columns if (kcs[c] != 0.0).any()])
    recA.data = {c: kcs[c] for c in recA.data.keys()}
    
    recW = ValueRecorder('weights', *[c for c in weights.columns if c in recA.data])
    recW.colors = recA.colors
    recW.data = {c: weights[c]  for c in recW.data.keys()}

    recB = ValueRecorder('Bounds', *sum([['{}_lb'.format(c), '{}_ub'.format(c)] for c in lbs.columns if c in recA.data], []))
    recB.colors = dict(sum([[('{}_lb'.format(c), recA.colors[c]), ('{}_ub'.format(c), recA.colors[c])] for c in lbs.columns if c in recA.data], []))
    recB.data = dict(sum([[('{}_lb'.format(c), lbs[c]), ('{}_ub'.format(c), ubs[c])] for c in lbs.columns if c in recA.data], []))
    
    recC = ValueRecorder('commands', *[c for c in commands.columns if c in recA.data])
    recC.colors = recA.colors
    recC.data = {c: commands[c] for c in recC.data.keys()}
    draw_recorders([recA, recW, recB, recC], 1, 12, 6).savefig(res_pkg_path('package://kineverse_experiment_world/test/keep_contact.png'))