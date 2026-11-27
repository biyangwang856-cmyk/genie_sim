"""Regression tests for rejecting false native rotation certificates."""

import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from rotation_probe import ACTOR, SKILL, RotationBinding, verify_rotation


def physical_pose(xyz, rotation):
    result = np.eye(4)
    result[:3, :3] = rotation.as_matrix()
    result[:3, 3] = xyz
    return result


def witness(binding, yaw_deg=15., *, object_rotates=True):
    start, movement_end, settled = 2., 2.25, 2.55
    support = np.eye(4).tolist()
    initial_object = physical_pose([.9, -.11, .765], Rotation.identity()).tolist()
    rows = []
    for index in range(67):
        clock = start + index / 120
        yaw = min(1., index / 30) * yaw_deg
        down = Rotation.from_euler("x", 180., degrees=True)
        tool = physical_pose([.9, -.11, .906], Rotation.from_euler("z", yaw, degrees=True) * down)
        obj = physical_pose([.9, -.11, .864], Rotation.from_euler("z", yaw if object_rotates else 0., degrees=True))
        rows.append({"sim_time": clock, "poses": {"object": obj.tolist(), "tcp": tool.tolist(), "support": support}})

    def observation(row):
        return {"sim_time_s": row["sim_time"], **copy.deepcopy(row["poses"])}

    evidence = {"schema": 1, "skill": SKILL, "binding": binding.to_dict(), "native_result": True,
                "initial_object": initial_object, "initial_support": support, "before": observation(rows[0]),
                "after": observation(rows[30]), "settled": observation(rows[-1])}
    joint_names = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
    commands = [{"sim_time": clock, "joint_names": joint_names, "applied_joint_positions": [step] * 7,
                 "measured_joint_positions_before": [0.9 * step] * 7}
                for clock, step in zip(np.linspace(start, movement_end, 10), np.linspace(0, .2, 10), strict=True)]
    return evidence, rows, commands


class RotationProbeTests(unittest.TestCase):
    def setUp(self):
        self.binding = RotationBinding(ACTOR, "right", "a" * 64, 15)

    def verify(self, evidence, physical, commands):
        return verify_rotation(evidence, self.binding, physics_rows=physical, command_rows=commands)

    def test_accepts_continuous_signed_motion_from_measurements(self):
        evidence, physical, commands = witness(self.binding)
        evidence["native_result"] = False
        result = self.verify(evidence, physical, commands)
        self.assertTrue(result["passed"], result)
        self.assertAlmostEqual(result["metrics"]["signed_rotation_deg"], 15.)

    def test_native_success_cannot_mask_stationary_object_or_wrong_direction(self):
        for kwargs in ({"object_rotates": False}, {"yaw_deg": -15.}):
            with self.subTest(**kwargs):
                result = self.verify(*witness(self.binding, **kwargs))
                self.assertFalse(result["passed"], result)
                self.assertFalse(result["checks"]["signed_rotation_target"])

    def test_rejects_held_transform_jump_and_support_shift_in_raw_frames(self):
        evidence, physical, commands = witness(self.binding)
        physical[13]["poses"]["object"][0][3] += .009
        physical[13]["poses"]["support"] = np.eye(4).tolist()
        physical[13]["poses"]["support"][0][3] += .004
        result = self.verify(evidence, physical, commands)
        self.assertFalse(result["passed"], result)
        self.assertFalse(result["checks"]["friction_grasp_retained"])
        self.assertFalse(result["checks"]["support_remained_still"])

    def test_rejects_stale_drive_and_discontinuous_recording(self):
        evidence, physical, commands = witness(self.binding)
        for row in commands:
            row["applied_joint_positions"] = [0.] * 7
        physical[21:26] = []
        result = self.verify(evidence, physical, commands)
        self.assertFalse(result["passed"], result)
        self.assertFalse(result["checks"]["applied_arm_motion"])
        self.assertFalse(result["checks"]["synchronized_native_trace"])

    def test_binding_is_limited_to_the_trusted_angle(self):
        for angle in (False, -15., 90.):
            with self.assertRaises(ValueError):
                RotationBinding(ACTOR, "right", "a" * 64, angle)
