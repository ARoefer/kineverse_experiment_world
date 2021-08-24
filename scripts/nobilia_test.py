import math
import rospy
import numpy as np
import kineverse.gradients.gradient_math as gm
import dearpygui.dearpygui as dpg
import signal


from kineverse.model.geometry_model         import GeometryModel, Path
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer

from kineverse_experiment_world.nobilia_shelf import create_nobilia_shelf



x_loc_sym = gm.Symbol('location_x')

def draw_shelves(shelf, params, world, visualizer, steps=4, spacing=1):
  state = params.copy()
  visualizer.begin_draw_cycle('world')
  panel_cos = gm.dot_product(gm.z_of(shelf.links['panel_top'].pose),
                             -gm.z_of(shelf.links['panel_bottom'].pose))

  for x, p in enumerate(np.linspace(0, 1.84, steps)):
    state[shelf.joints['hinge'].position] = p
    state[x_loc_sym] = (x - steps/2) * spacing
    
    cos = gm.subs(panel_cos, state)
    print('Angle at {:>8.2f}: {:> 8.2f}'.format(np.rad2deg(p), np.rad2deg(np.arccos(cos.flatten()[0]))))
    
    world.update_world(state)
    visualizer.draw_world('world', world)
  visualizer.render('world')


def handle_sigint(*args):
  dpg.stop_dearpygui()


if __name__=='__main__':
  rospy.init_node('nobilia_shelf_test')

  km  = GeometryModel()
  vis = ROSBPBVisualizer('kineverse/nobilia/vis', base_frame='world')

  origin_pose = gm.translation3(x_loc_sym, 0, 0)

  debug = create_nobilia_shelf(km, Path('nobilia'), origin_pose)

  km.clean_structure()
  km.dispatch_events()

  shelf = km.get_data('nobilia')

  shelf_pos = shelf.joints['hinge'].position

  world = km.get_active_geometry({shelf_pos}.union(gm.free_symbols(origin_pose)))

  params = debug.tuning_params

  with dpg.window(label='Parameter settings'):
    for s, p in debug.tuning_params.items():
      def update_param(sender, value, symbol):
        params[symbol] = value
        draw_shelves(shelf, params, world, vis, steps=5)

      slider = dpg.add_slider_float(label=f'{s}',
                                    id=f'{s}',
                                    callback=update_param,
                                    user_data=s,
                                    min_value=p-0.1,
                                    max_value=p+0.1,
                                    width=200,
                                    default_value=p)

  draw_shelves(shelf, params, world, vis, steps=5)

  signal.signal(signal.SIGINT, handle_sigint)

  dpg.setup_viewport()
  dpg.set_viewport_width(300)
  dpg.set_viewport_height(200)
  dpg.set_viewport_title('Nobilia Test')
  dpg.start_dearpygui()

  # start = rospy.Time.now()

  # max_length_v = 0


  # while not rospy.is_shutdown():
  #   state_range = shelf.joints['hinge'].limit_upper - shelf.joints['hinge'].limit_lower
  #   state = {shelf_pos: shelf.joints['hinge'].limit_lower + (math.cos((rospy.Time.now() - start).to_sec()) + 1) * 0.5 * state_range }
  #   world.update_world(state)
  #   vis.begin_draw_cycle('world', 'poses', 'vectors')
  #   vis.draw_world('world', world)
  #   vis.draw_poses('poses', np.eye(4), 0.1, 0.01, [gm.subs(p, state) for p in debug.poses])

  #   for point, vector in debug.vectors:
  #     vis.draw_vector('vectors', gm.subs(point, state), 
  #                                gm.subs(vector, state))

  #   vis.render()
    
  #   for name, expr in debug.expressions.items():
  #     print(f'{name}: {gm.subs(expr, state)}')

  #   max_length_v = max(max_length_v, gm.subs(debug.expressions['length_v'], state))

  #   rospy.sleep(rospy.Duration(1 / 50))

  # print(f'max length v: {max_length_v}')