#!/usr/bin/env python

from kineverse.gradients.gradient_math     import point3, vector3, translation3, frame3_axis_angle, Position, se
from kineverse.model.geometry_model        import GeometryModel, RigidBody, Geometry, Path
from kineverse.operations.basic_operations import CreateComplexObject as OP_CCO, \
                                                  CreateSingleValue   as OP_CSV
from kineverse.operations.special_kinematics import SetBallJoint      as OP_SBJ


if __name__ == '__main__':
    km = GeometryModel()

    geom_head = Geometry(Path('head'), se.eye(4), 'mesh', mesh='package://kineverse_experiment_world/urdf/faucet_head.obj')
    rb_head   = RigidBody(Path('world'), se.eye(4), geometry={0: geom_head}, collision={0: geom_head})

    geom_base = Geometry(Path('base'), se.eye(4), 'mesh', mesh='package://kineverse_experiment_world/urdf/faucet_base.obj')
    rb_base   = RigidBody(Path('world'), se.eye(4), geometry={0: geom_base}, collision={0: geom_base})

    km.apply_operation('create base', OP_CCO(Path('base'), rb_base))
    km.apply_operation('create head', OP_CCO(Path('head'), rb_head))

    km.apply_operation('connect base head', 
        OP_SBJ(Path('base/pose'), 
               Path('head/pose'), 
               Path('ball_joint'),
               translation3(0.006, 0, 0.118),
               Position('axis_x'),
               Position('axis_y'),
               Position('axis_z'), 0.2, 1.2))

    km.clean_structure()
    km.dispatch_events()

    with open('faucet.json', 'w') as f:
        km.save_to_file(f)
