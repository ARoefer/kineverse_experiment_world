import math
import rospy


from kineverse.model.geometry_model         import GeometryModel, Path
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer

from kineverse_experiment_world.nobilia_shelf import create_nobilia_shelf



if __name__=='__main__':
  rospy.init_node('nobilia_shelf_test')

  km  = GeometryModel()
  vis = ROSBPBVisualizer('kineverse/nobilia/vis', base_frame='world')

  create_nobilia_shelf(km, Path('nobilia'))

  km.clean_structure()
  km.dispatch_events()

  shelf = km.get_data('nobilia')

  shelf_pos = shelf.joints['hinge'].position

  world = km.get_active_geometry({shelf_pos})

  start = rospy.Time.now()
  while not rospy.is_shutdown():
    world.update_world({shelf_pos: (math.cos((rospy.Time.now() - start).to_sec()) + 1) * 0.5 * shelf.joints['hinge'].limit_upper})
    vis.begin_draw_cycle('world')
    vis.draw_world('world', world)
    vis.render()
    rospy.sleep(rospy.Duration(1 / 50))