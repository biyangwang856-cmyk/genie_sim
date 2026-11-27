"""Capability mapping and certification regressions, with synthetic verifier inputs."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from skill_capabilities import (HINGE_SKILL, HingeBinding, HingeCapabilityGap, RuntimeScope,
                                SkillRegistry, discover_skills, execute_hinge, grasp_hinge_handle, verify_hinge)


@contextmanager
def writable_workspace_directory():
    root = Path(__file__).resolve().parents[1]
    directory = root / (".skill-test-" + uuid.uuid4().hex)
    # Inherit the workspace permissions on Windows as well as Linux.
    directory.mkdir()
    try:
        yield directory
    finally:
        if directory.resolve().parent != root:
            raise ValueError("Test workspace moved outside the project")
        shutil.rmtree(directory)


class SimulatedAPI:
    """Supplies documented method shapes and telemetry for host tests."""

    def __init__(self, *, angle_moved=True, joint_type="revolute"):
        self.client = self
        self.angle_moved = angle_moved
        self.joint_type = joint_type
        self.angle = 0.
        self.time = 1.
        self.commands = []
        self.open_calls = []
        self.move_calls = []
        self.trace = []
        self.arc_waypoint_counts = []
        self.maximum_prefix = 12

    def get_object_prim_path(self, actor, articulation_part=None):
        path = f"/World/Objects/{actor}/entity"
        return path + "/" + articulation_part if articulation_part is not None else path

    def get_articulation_joint_info(self, root, joint_prim_path=None):
        return {"joint_path": root + "/joint", "body0_path": root + "/body",
                "body1_path": root + "/part_01", "axis": "Z", "joint_type": self.joint_type,
                "current_angle_deg": self.angle, "lower_limit_deg": -60., "upper_limit_deg": 0.}

    def physical_poses(self, angle=None):
        body = np.eye(4)
        handle = np.eye(4)
        rotate = Rotation.from_euler("z", self.angle if angle is None else angle, degrees=True)
        handle[:3, :3] = rotate.as_matrix()
        handle[:3, 3] = rotate.apply([.12, 0, 0])
        ee = handle.copy()
        ee[:3, 3] += ee[:3, :3] @ [0, .02, 0]
        return body, handle, ee

    def get_object_pose(self, actor, articulation_part=None):
        body, handle, _ = self.physical_poses()
        pose = body if articulation_part == "body" else handle
        return pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]

    def get_ee_pose(self, arm="right"):
        return self.physical_poses()[2]

    def move_actor_to_target_pose(self, actor, goals, arm="right", move_config=None):
        self.move_calls.append((actor, goals, arm, move_config.gripper_action))
        return True

    def command_any(self, command):
        if command["command"] == "ProjectSnapshot":
            return {"sim_time": self.time,
                    "joint_enabled": dict.fromkeys(command["joint_paths"], True)}
        if command["command"] == "ProjectTrackObjects":
            return {"success": True}
        if command["command"] == "GetRevoluteJointTrajectory":
            self.assert_native_towards_min(command)
            sample_count = int(np.ceil(abs(self.angle + 60))) + 1
            poses = [self.physical_poses(value)[2] for value in np.linspace(self.angle, -60, sample_count)]
            return {"trajectory_positions": [p[:3, 3].tolist() for p in poses],
                    "trajectory_orientations": [Rotation.from_matrix(p[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist()
                                                for p in poses]}
        if command["command"] == "FollowTrajectory":
            quaternions = command["trajectory_orientations"]
            self.arc_waypoint_counts.append(len(quaternions))
            if len(quaternions) > self.maximum_prefix:
                return {"success": False}
            for quat in quaternions:
                self.angle = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_euler("xyz", degrees=True)[2]
                self.tick()
                self.commands.append({"sim_time": self.time,
                                      "applied_joint_positions": [float(np.deg2rad(self.angle)), 0.]})
            return {"success": True}
        raise AssertionError(command)

    def assert_native_towards_min(self, command):
        if command["target_limit"] != "min":
            raise AssertionError("The test arc only models opening toward its minimum native limit")

    def set_frame_state(self, *args, **kwargs):
        pass

    def tick(self):
        self.time += 1 / 120
        body, handle, _ = self.physical_poses()
        self.trace.append({"sim_time": self.time,
                           "poses": {"hinge_base": body.tolist(), "hinge_link": handle.tolist(),
                                     "hinge_handle": handle.tolist()}})

    def run_step(self, steps):
        for _ in range(steps):
            self.tick()

    def open(self, actor, joint_name=None, open_config=SimpleNamespace(), arm="right"):
        self.open_calls.append((actor, joint_name, open_config.target_limit, arm))
        for desired in np.linspace(0, -60, 31):
            if self.angle_moved:
                self.angle = float(desired)
            self.tick()
            self.commands.append({"sim_time": self.time,
                                  "applied_joint_positions": [np.deg2rad(desired), 0.]})
        return True


class GraspProbeAPI(SimulatedAPI):
    """Models one local pull while retaining the native acknowledgement on an empty grasp."""

    def __init__(self, **config):
        super().__init__(**config)
        self.live_ee = super().get_ee_pose()
        self.close_calls = 0

    def open_gripper(self, arm):
        pass

    def close_gripper(self, arm):
        self.close_calls += 1

    def get_ee_pose(self, arm="right"):
        return self.live_ee.copy()

    def move_actor_to_target_pose(self, actor, goals, arm="right", move_config=None):
        self.live_ee = goals[0].copy()
        return super().move_actor_to_target_pose(actor, goals, arm, move_config)

    def command_any(self, payload):
        if payload["command"] == "ProjectGripperCollisionBounds":
            ee = self.live_ee
            return {"success": True, "shapes": [
                {"path": f"/G2/gripper_l_support_link{i + 1}/pad",
                 "corners_world_m": [(ee[:3, :3] @ [x, sign * y, z] + ee[:3, 3]).tolist()
                                     for x in (-.01, .01) for y in (.027, .033) for z in (.016, .04)]}
                for i, sign in enumerate((-1, 1))]}
        if payload["command"] == "GetRevoluteJointTrajectory":
            remaining = -60. - self.angle
            rotations = Rotation.from_euler("z", np.linspace(0, remaining, int(np.ceil(abs(remaining))) + 1),
                                            degrees=True)
            wrist = Rotation.from_matrix(self.live_ee[:3, :3])
            return {"trajectory_positions": rotations.apply(self.live_ee[:3, 3]).tolist(),
                    "trajectory_orientations": (rotations * wrist).as_quat()[:, [3, 0, 1, 2]].tolist()}
        if payload["command"] == "FollowTrajectory":
            orient = Rotation.from_quat(np.asarray(payload["trajectory_orientations"])[-1, [1, 2, 3, 0]])
            delta = orient.as_matrix() @ self.live_ee[:3, :3].T
            self.live_ee[:3, :3] = orient.as_matrix()
            self.live_ee[:3, 3] = payload["trajectory_positions"][-1]
            if self.angle_moved:
                self.angle += float(Rotation.from_matrix(delta).as_rotvec(degrees=True)[2])
            return {"success": True}
        return super().command_any(payload)


class SkillCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.adapter = SimpleNamespace(robot_id="g2crsb90d", arms=("left", "right"),
                                       capabilities=frozenset({"curobo", "friction_grasp"}))
        self.binding = HingeBinding("microwave", "joint", "part_01", "left", "min", "a" * 64, (0, .02, 0))
        self.grasp_geometry = {"contact_box_local_m": [[-.018, .005, -.01], [.018, .035, .01]],
                               "door_front_y_m": -.025}

    def _scope(self, robot):
        return RuntimeScope.from_robot_api(self.adapter, robot, image_id="sha256:" + "b" * 64,
                                           calibration_sha256="c" * 64, adapter_sha256="d" * 64)

    def test_unregistered_plan_reports_every_gap_and_candidate(self):
        robot = SimulatedAPI()
        report = discover_skills(["grasp", HINGE_SKILL, "vacuum_pick", "unsupported_assembly"], self.adapter,
                                 robot_api=robot, bindings={HINGE_SKILL: self.binding.to_dict()})
        self.assertEqual(report["code"], "SKILL_NOT_REGISTERED")
        self.assertEqual(report["message"], "Skill is not registered")
        self.assertEqual(report["missing_skills"], [HINGE_SKILL, "vacuum_pick", "unsupported_assembly"])
        self.assertEqual(report["missing_robot_capabilities"], ["vacuum_actuation"])
        self.assertEqual(report["skill_details"][HINGE_SKILL]["status"], "validating_skill")
        self.assertIn("open", report["candidate_apis"])
        self.assertEqual(report["skill_details"]["vacuum_pick"]["candidate_apis"], [])
        robot.get_articulation_joint_info = None
        missing_verifier = discover_skills([HINGE_SKILL], self.adapter, robot_api=robot)
        self.assertEqual(missing_verifier["status"], "capability_gap")
        self.assertEqual(missing_verifier["missing_verifiers"], ["get_articulation_joint_info"])

    def test_prismatic_and_native_success_with_no_hinge_motion_are_rejected(self):
        drawer = SimulatedAPI(joint_type="prismatic")
        with self.assertRaises(HingeCapabilityGap):
            execute_hinge(drawer, SimpleNamespace, self.binding)
        self.assertEqual(drawer.open_calls, [])
        stationary = SimulatedAPI(angle_moved=False)
        data = execute_hinge(stationary, SimpleNamespace, self.binding, scope=self._scope(stationary))
        self.assertTrue(data["native_result"])
        verified = verify_hinge(data, self.binding, physics_rows=stationary.trace, command_rows=stationary.commands)
        self.assertFalse(verified["passed"])
        self.assertFalse(verified["checks"]["actual_hinge_opening"])

    def test_handle_grasp_compiles_bounded_native_targets(self):
        robot = GraspProbeAPI()
        result = grasp_hinge_handle(robot, SimpleNamespace, self.binding, geometry=self.grasp_geometry)
        self.assertTrue(result["success"])
        actor, targets, arm, action = robot.move_calls[0]
        self.assertEqual((actor, arm, action), ("gripper", "left", None))
        self.assertEqual(targets.shape, (1, 4, 4))
        np.testing.assert_allclose(targets[0, :3, 1], [0., 0., 1.], atol=1e-8)
        np.testing.assert_allclose(targets[0, :3, 3], [.12, .048, 0.], atol=1e-8)
        self.assertEqual(robot.close_calls, 1)
        self.assertAlmostEqual(result["confirmation"]["progress_deg"], 3.)

    def test_handle_grasp_rejects_native_success_without_physical_motion(self):
        robot = GraspProbeAPI(angle_moved=False)
        events = []
        result = grasp_hinge_handle(
            robot, SimpleNamespace, self.binding, geometry=self.grasp_geometry, observe_attempt=events.append,
        )
        self.assertFalse(result["success"])
        self.assertEqual(len(events), 3)
        self.assertEqual(events[-1]["reason"], "grasp_contact_not_confirmed")
        self.assertTrue(events[-1]["planner_success"])
        self.assertEqual(events[-1]["progress_deg"], 0.)
        self.assertEqual(robot.open_calls, [])

    def test_raw_trace_rejects_a_handle_displaced_mid_swing(self):
        robot = SimulatedAPI()
        data = execute_hinge(robot, SimpleNamespace, self.binding, scope=self._scope(robot))
        robot.trace[15]["poses"]["hinge_handle"][0][3] += .012
        result = verify_hinge(data, self.binding, physics_rows=robot.trace, command_rows=robot.commands)
        self.assertTrue(result["checks"]["actual_hinge_opening"])
        self.assertFalse(result["checks"]["handle_remained_part_of_link"])
        self.assertFalse(result["passed"])

    def test_segmented_hinge_retries_without_motion_and_reaches_the_same_limit(self):
        robot = SimulatedAPI()
        events = []
        data = execute_hinge(robot, SimpleNamespace, self.binding, scope=self._scope(robot),
                             observe_segment=events.append)
        self.assertGreater(robot.arc_waypoint_counts[0], robot.maximum_prefix)
        self.assertEqual(events[0]["progress_deg"], 0.)
        self.assertIs(events[0]["planner_success"], False)
        self.assertAlmostEqual(data["settled"]["joint"]["current_angle_deg"], -60.)
        self.assertEqual(robot.open_calls, [])
        verification = verify_hinge(data, self.binding, physics_rows=robot.trace, command_rows=robot.commands)
        self.assertTrue(verification["passed"], verification)

    def test_promotion_is_atomic_and_scoped_to_the_runtime_and_asset(self):
        robot = SimulatedAPI()
        scope = self._scope(robot)
        data = execute_hinge(robot, SimpleNamespace, self.binding, scope=scope)
        expected_joint = "/World/Objects/microwave/entity/joint"
        self.assertEqual(robot.open_calls, [("microwave", expected_joint, "min", "left")])

        with writable_workspace_directory() as work:
            registry = SkillRegistry(work / "registry")
            evidence = work / "hinge-observations.json"
            evidence.write_text(json.dumps(data), encoding="utf-8")
            no_trace = registry.certify_hinge(scope, self.binding, evidence)
            self.assertEqual(no_trace["status"], "skill_validation_failed")
            self.assertIsNone(registry.lookup(HINGE_SKILL, scope, self.binding))
            for name, rows in (("object-physics-120hz.jsonl", robot.trace),
                               ("native-control-commands.jsonl", robot.commands)):
                (work / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with ThreadPoolExecutor(max_workers=2) as pool:
                certificates = list(pool.map(lambda _: registry.certify_hinge(scope, self.binding, evidence), (1, 2)))
            self.assertEqual([result["status"] for result in certificates], ["ready", "ready"])
            self.assertEqual(certificates[0]["certificate"], certificates[1]["certificate"])
            certified = discover_skills([HINGE_SKILL], self.adapter, scope=scope, registry=registry,
                                        bindings={HINGE_SKILL: replace(self.binding, actor="another_instance")})
            self.assertEqual(certified["status"], "ready")
            self.assertIsNone(registry.lookup(HINGE_SKILL, replace(scope, calibration_sha256="e" * 64), self.binding))
            self.assertIsNone(registry.lookup(HINGE_SKILL, scope, replace(self.binding, asset_sha256="e" * 64)))


if __name__ == "__main__":
    unittest.main()
