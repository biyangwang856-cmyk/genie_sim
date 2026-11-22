"""Task-specific physical manipulation phases for official FineManip scenes."""
from functools import partial
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def matrix(position, quaternion):
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
    return result


def object_matrix(record):
    return matrix(record['position'], record['quaternion_wxyz'])


def top_orientation(axis):
    axis = np.asarray(axis, dtype=float).copy()
    axis[2] = 0.
    axis /= np.linalg.norm(axis)
    down = np.array([0., 0., -1.])
    return np.column_stack([axis, np.cross(down, axis), down])


def add_pick(program, controller, name, side, long_axis=0, local_offset=None):
    program.extend([
        (f'{name}_open', partial(controller.grip_phase, side, 0.)),
        (f'{name}_approach', partial(controller.approach_phase, name, side, long_axis, local_offset, 0.15)),
        (f'{name}_descend', partial(controller.approach_phase, name, side, long_axis, local_offset, 0.)),
        (f'{name}_close', partial(controller.grip_phase, side, -0.80)),
        (f'{name}_lift', partial(controller.lift_phase, name, side, 0.12)),
        (f'{name}_verify_grasp', partial(controller.verify_grasp_phase, name, side)),
    ])


def phone_program(controller, scene):
    program = []
    base = 'benchmark_packaging_box_001'
    for name, depth in [('benchmark_phone_000', 0.008),
                        ('benchmark_packaging_box_000', 0.020),
                        ('benchmark_packaging_box_002', 0.026)]:
        add_pick(program, controller, name, 'right', long_axis=2)
        program.extend([
            (f'{name}_align', partial(controller.phone_place_phase, name, base, depth, True)),
            (f'{name}_lower', partial(controller.phone_place_phase, name, base, depth, False)),
            (f'{name}_release', partial(controller.grip_phase, 'right', 0.)),
            (f'{name}_retreat', partial(controller.retreat_phase, 'right', 0.15)),
        ])
    return program


def pen_program(controller, scene):
    instruction = controller.instruction
    targets = instruction['target']
    pen, cap = targets['id1'], targets['id2']
    pen_side = instruction['gripper']['1']['side']
    cap_side = instruction['gripper']['2']['side']
    program = []
    add_pick(program, controller, pen, pen_side, local_offset=[0.025, 0., 0.])
    add_pick(program, controller, cap, cap_side, local_offset=[-0.004, 0., 0.])
    direction = np.array([0., -1. if pen_side == 'right' else 1., 0.])
    orientation = np.column_stack([direction, [0., 0., 1.], np.cross(direction, [0., 0., 1.])])
    pen_pose = np.eye(4)
    pen_pose[:3, :3] = orientation
    pen_pose[:3, 3] = [-0.08, -0.06 if pen_side == 'right' else 0.06, 0.98]
    cap_pose = pen_pose.copy()
    cap_pose[:3, 3] += direction * (-0.07802 - 0.02127 - 0.030)
    inserted = cap_pose.copy()
    inserted[:3, 3] += direction * (0.030 + 0.03152)
    return_pose = object_matrix(scene['objects'][pen])
    return_pose[2, 3] += 0.004
    program.extend([
        ('pen_present', partial(controller.object_pose_phase, pen, pen_side, pen_pose, 55)),
        ('cap_align_30mm_gap', partial(controller.object_pose_phase, cap, cap_side, cap_pose, 55)),
        ('cap_insert', partial(controller.object_pose_phase, cap, cap_side, inserted, 80)),
        ('cap_insertion_settle', partial(controller.hold_phase, 50)),
        ('cap_release', partial(controller.grip_phase, cap_side, 0.)),
        ('cap_arm_retreat', partial(controller.retreat_phase, cap_side, 0.14)),
        ('capped_pen_return', partial(controller.object_pose_phase, pen, pen_side, return_pose, 70)),
        ('pen_release', partial(controller.grip_phase, pen_side, 0.)),
        ('pen_arm_retreat', partial(controller.retreat_phase, pen_side, 0.14)),
    ])
    return program


def block_program(controller, scene):
    path = Path(os.environ['SIM_REPO_ROOT']).parent / 'output/insertion_target.json'
    geometry = json.loads(path.read_text())
    if geometry['silhouette_outside_hole_area_fraction'] > 0.001:
        raise ValueError('The certified star silhouette does not fit the physical hole')
    name = controller.instruction['target']['id1']
    side = controller.instruction['gripper']['1']['side']
    program = []
    add_pick(program, controller, name, side)
    program.extend([
        ('star_align_to_matching_hole', partial(controller.block_entry_phase, name, side, geometry, 0.12)),
        ('star_lower_to_hole', partial(controller.block_entry_phase, name, side, geometry, 0.)),
        ('star_release_into_hole', partial(controller.grip_phase, side, 0.)),
        ('star_arm_retreat', partial(controller.retreat_phase, side, 0.15)),
        ('star_settle', partial(controller.hold_phase, 70)),
    ])
    return program


def build_program(controller, scene):
    if controller.sub_task_name == 'phone_packaging':
        return phone_program(controller, scene)
    if controller.sub_task_name == 'put_the_cap_on_the_pen_new':
        return pen_program(controller, scene)
    return block_program(controller, scene)
