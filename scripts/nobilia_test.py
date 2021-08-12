import math
import rospy
import numpy as np
import kineverse.gradients.gradient_math as gm

from kineverse.model.geometry_model         import GeometryModel, Path
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer

from kineverse_experiment_world.nobilia_shelf import create_nobilia_shelf



if __name__=='__main__':
  rospy.init_node('nobilia_shelf_test')

  km  = GeometryModel()
  vis = ROSBPBVisualizer('kineverse/nobilia/vis', base_frame='world')

  debug = create_nobilia_shelf(km, Path('nobilia'))

  km.clean_structure()
  km.dispatch_events()

  shelf = km.get_data('nobilia')

  shelf_pos = shelf.joints['hinge'].position

  world = km.get_active_geometry({shelf_pos})

  start = rospy.Time.now()

  max_length_v = 0

  while not rospy.is_shutdown():
    state_range = shelf.joints['hinge'].limit_upper - shelf.joints['hinge'].limit_lower
    state = {shelf_pos: shelf.joints['hinge'].limit_lower + (math.cos((rospy.Time.now() - start).to_sec()) + 1) * 0.5 * state_range }
    world.update_world(state)
    vis.begin_draw_cycle('world', 'poses', 'vectors')
    vis.draw_world('world', world)
    vis.draw_poses('poses', np.eye(4), 0.1, 0.01, [gm.subs(p, state) for p in debug.poses])

    for point, vector in debug.vectors:
      vis.draw_vector('vectors', gm.subs(point, state), 
                                 gm.subs(vector, state))

    vis.render()
    
    for name, expr in debug.expressions.items():
      print(f'{name}: {gm.subs(expr, state)}')

    max_length_v = max(max_length_v, gm.subs(debug.expressions['length_v'], state))

    rospy.sleep(rospy.Duration(1 / 50))

  print(f'max length v: {max_length_v}')