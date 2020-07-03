#!/usr/bin/env python
from tqdm import tqdm

from kineverse.utils import res_pkg_path
from kineverse.time_wrapper import Time


def pybullet():
    import pybullet as pb
    
    pb.connect(pb.DIRECT)
    pb.setTimeStep(0.01)
    pb.setGravity(0, 0, -9.81)

    kitchen = pb.loadURDF(res_pkg_path('package://iai_kitchen/urdf_obj/IAI_kitchen.urdf'), 
                          [0,0,0], [0,0,0,1], 0, True, flags=pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    pr2     = pb.loadURDF(res_pkg_path('package://iai_pr2_description/robots/pr2_calibrated_with_ft2.xml'),
                          [0,0,0], [0,0,0,1], 0, True, flags=pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)

    print(kitchen, pr2)

    for x in tqdm(range(100)):    
        cps = pb.getClosestPoints(pr2, kitchen, 40.0, 1, 1)
        print('\nClosest points:\n{}'.format(cps))

@profile
def bpb():
    import numpy as np

    import kineverse.bpb_wrapper                as pb
    import kineverse.model.geometry_model       as gm
    import kineverse.operations.urdf_operations as urdf

    from kineverse.urdf_fix  import urdf_filler
    from urdf_parser_py.urdf import URDF

    km = gm.GeometryModel()

    with open(res_pkg_path('package://iai_pr2_description/robots/pr2_calibrated_with_ft2.xml'), 'r') as urdf_file:
        urdf.load_urdf(km, 'pr2', urdf_filler(URDF.from_xml_string(urdf_file.read())))

    with open(res_pkg_path('package://iai_kitchen/urdf_obj/IAI_kitchen.urdf'), 'r') as urdf_file:
        urdf.load_urdf(km, 'kitchen', urdf_filler(URDF.from_xml_string(urdf_file.read())))

    km.clean_structure()
    km.dispatch_events()


    kitchen = km.get_data('kitchen')
    pr2     = km.get_data('pr2')

    joint_symbols  = {j.position for j in kitchen.joints.values() if hasattr(j, 'position')}
    joint_symbols |= {j.position for j in pr2.joints.values()  if hasattr(j, 'position')}

    coll_world = km.get_active_geometry(joint_symbols)

    robot_parts = {n: o for n, o in coll_world.named_objects.items() if n[:4] == 'pr2/'}

    batch = {o: 2.0 for o in robot_parts.values()}

    print('Benchmarking by querying distances for {} objects. Total object count: {}.\n'.format(len(batch), len(coll_world.names)))

    dur_update    = []
    dur_distances = []

    # print('Mesh files:\n{}'.format('\n'.join(sorted({pb.pb.get_shape_filename(s) for s in sum([o.collision_shape.child_shapes for o in coll_world.collision_objects], [])}))))

    # print('Objects:\n{}'.format('\n'.join(['{}:\n  {}\n  {} {}'.format(n, 
    #                                                                    o.transform.rotation, 
    #                                                                    o.transform.origin, 
    #                                                                    pb.pb.get_shape_filename(o.collision_shape.get_child(0))) 
    #                                                                    for n, o in coll_world.named_objects.items()])))

    for x in tqdm(range(100)):
        start = Time.now()
        coll_world.update_world({s: v for s, v in zip(joint_symbols, np.random.rand(len(joint_symbols)))})
        dur_update.append(Time.now() - start)

        start = Time.now()
        distances = coll_world.closest_distances(batch)
        dur_distances.append(Time.now() - start)

    dur_update_mean    = sum([d.to_sec() for d in dur_update]) / len(dur_update)
    dur_distances_mean = sum([d.to_sec() for d in dur_distances]) / len(dur_distances)

    print('Update mean: {}\nUpdate max: {}\n'
          'Distances mean: {}\nDistances max: {}\n'.format(dur_update_mean, 
                                                           max(dur_update).to_sec(),
                                                           dur_distances_mean, 
                                                           max(dur_distances).to_sec(),))


if __name__ == '__main__':
    bpb()
    # pybullet()
