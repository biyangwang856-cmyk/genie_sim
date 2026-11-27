"""Certify native rotation of a friction-held phone around a fixed vertical tool."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot_adapters import ROOT, load_robot
from scene_contract import content_id, digest, read_json
from skill_capabilities import (RuntimeScope, SkillRegistry, discover_skills, _array, _distance,
                                _finite, _jsonl_path, _rigid_pose, _rotation_deg, _sha256,
                                _trace_window)


SKILL = "rotate_held_about_tool"
ACTOR = "benchmark_phone_000"
SUPPORT = "benchmark_packaging_box_001"
TASK = "probe_rotate_held_about_tool"
PROFILE_FIELD = "project_rotation_probe"
SOURCE_IDENTITY_FILES = ("robot-profile.json", "robot_adapters.py", "robot.urdf",
                         "project_runtime_hooks.py", "collection_programs.py", "rotation_probe.py",
                         "skill_capabilities.py")
_VERSION = 1
_TRACE_KEYS = ("object", "tcp", "support")
_ANGLE_DEG = 15.
_MAX_AXIS_ERROR_DEG = 4.
_MAX_TARGET_ERROR_DEG = 5.
_MAX_TCP_DRIFT_M = .012
_MAX_HOLD_DRIFT_M = .008
_MAX_HOLD_DRIFT_DEG = 5.
_MAX_SUPPORT_DRIFT_M = .002
_MAX_SUPPORT_DRIFT_DEG = 2.
_MIN_LIFT_M = .055
_VERTICAL_COSINE = float(np.cos(np.deg2rad(10.)))
_ARM_JOINTS = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))


@dataclass(frozen=True)
class RotationBinding:
    actor: str
    arm: str
    asset_sha256: str
    angle_deg: float

    def __post_init__(self):
        _sha256(self.asset_sha256)
        if self.actor != ACTOR or self.arm != "right" or _finite(self.angle_deg) != _ANGLE_DEG:
            raise ValueError("The trusted rotation probe binds the right arm and annotated phone to +15° about world Z")
        object.__setattr__(self, "angle_deg", _ANGLE_DEG)

    def to_dict(self):
        return asdict(self)

    def applicability(self):
        return self.to_dict()


def _snapshot(robot, ops, binding):
    return {
        "sim_time_s": _finite(robot.client.command_any({"command": "ProjectSnapshot"})["sim_time"]),
        "object": _rigid_pose(ops.pose(robot, binding.actor)).tolist(),
        "tcp": _rigid_pose(robot.get_ee_pose(arm=binding.arm)).tolist(),
        "support": _rigid_pose(ops.pose(robot, SUPPORT)).tolist(),
    }


def execute_probe(robot, ops, request):
    """Run only the trusted fixture and save evidence for independent host verification."""
    from client.llm_gen.robot_api import RotateConfig
    from task_geometry import task_spec

    source = Path(os.environ["GENIE_RUN_SOURCE"])
    folder = Path(os.environ["GENIE_EPISODE_EVIDENCE"])
    fixture = robot.scene_info[PROFILE_FIELD]
    binding = RotationBinding(**fixture["binding"])
    if (request["robot_id"] != "g2crsb90d" or robot.scene_info["task_template"] != "phone_packaging"
            or robot.scene_info["layout"][binding.actor]["usd"] != ACTOR
            or robot.scene_info["layout"][SUPPORT]["usd"] != SUPPORT
            or content_id(fixture["asset_dependencies"]) != binding.asset_sha256):
        raise ValueError("Rotation validation requires the frozen G2 phone and support geometry")
    adapter = load_robot(request["robot_id"])
    scope = RuntimeScope.from_robot_api(
        adapter, robot, image_id=read_json(source.parent / "launch.json")["image_id"],
        calibration_sha256=digest(source / "robot-calibration.json"),
        adapter_sha256=content_id({name: digest(source / name) for name in SOURCE_IDENTITY_FILES}),
    )
    capability = discover_skills([SKILL], adapter, robot_api=robot, scope=scope,
                                 registry=SkillRegistry(ROOT / "skill-registry"), bindings={SKILL: binding})
    ops.note(robot, "rotation_capability_discovery", **capability)
    if capability["status"] not in {"ready", "validating_skill"}:
        return False

    spec = task_spec(robot.scene_info, adapter.robot_id)
    environment = spec["environment_collision_paths"]
    robot.add_objs_to_obstacle(environment)
    obstacles = robot.client.command_any({"command": "ProjectSnapshot"})["planner_obstacles"]
    environment_ready = bool(environment) and all(
        any(path == root or path.startswith(root + "/") for path in obstacles) for root in environment
    )
    ops.note(robot, "rotation_environment_collision_gate", passed=environment_ready, paths=environment)
    if not environment_ready:
        return False

    initial_object = _rigid_pose(ops.pose(robot, binding.actor))
    initial_support = _rigid_pose(ops.pose(robot, SUPPORT))
    if not ops.pick(robot, binding.actor, binding.arm, axis=2):
        return False
    ops.require_payload_proxy(robot, binding.actor, binding.arm, "rotation_payload_attachment_gate")
    before = _snapshot(robot, ops, binding)
    before_object = _rigid_pose(before["object"])
    verified_lift = float(before_object[2, 3] - initial_object[2, 3]) >= _MIN_LIFT_M
    vertical_tcp = before["tcp"][2][2] <= -_VERTICAL_COSINE
    ops.note(robot, "rotation_grasp_gate", passed=verified_lift and vertical_tcp,
             lift_m=float(before_object[2, 3] - initial_object[2, 3]), vertical_tcp=vertical_tcp)
    if not (verified_lift and vertical_tcp and scope.matches(robot)):
        return False

    paths = {key: robot.get_object_prim_path(actor) for key, actor in
             (("object", binding.actor), ("support", SUPPORT))}
    if paths != {"object": f"/World/Objects/{ACTOR}/entity", "support": f"/World/Objects/{SUPPORT}/entity"}:
        raise ValueError("Rotation trace paths do not match the frozen fixture")
    tracking = robot.client.command_any({"command": "ProjectTrackObjects", "paths": paths,
                                        "tcp_arm": binding.arm})
    if tracking.get("success") is not True:
        raise ValueError("Native tool and object trace recording is unavailable")
    target = before_object.copy()
    target[:3, :3] = Rotation.from_euler("z", binding.angle_deg, degrees=True).as_matrix() @ target[:3, :3]
    config = RotateConfig(constraint_axises=["x", "y"], offset_and_constraint_in_goal_frame=False,
                          action_description={"zh": "在原位转动夹持的手机", "en": "rotate the held phone in place"})
    try:
        reached = robot.rotate_actor(actor=binding.actor, target_pose=target[None],
                                     arm=binding.arm, rotate_config=config)
        after = _snapshot(robot, ops, binding)
        robot.client.run_step(36)
        settled = _snapshot(robot, ops, binding)
    finally:
        robot.client.command_any({"command": "ProjectTrackObjects", "paths": {}})

    evidence = {"schema": _VERSION, "skill": SKILL, "binding": binding.to_dict(), "scope": scope.to_dict(),
                "native_result": bool(reached), "initial_object": initial_object.tolist(),
                "initial_support": initial_support.tolist(), "before": before, "after": after,
                "settled": settled}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rotation-observations.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    verification = verify_rotation(evidence, binding,
                                   physics_rows=_jsonl_path(folder / "object-physics-120hz.jsonl"),
                                   command_rows=_jsonl_path(folder / "native-control-commands.jsonl"))
    ops.note(robot, "rotation_physical_probe", **verification, native_result=bool(reached))
    return verification["passed"]


def _signed_rotation(first, second):
    return np.rad2deg(Rotation.from_matrix(second[:3, :3] @ first[:3, :3].T).as_rotvec())


def _stable_measurement(origin, poses, max_m, max_deg):
    return all(_distance(origin, pose) <= max_m and _rotation_deg(origin, pose) <= max_deg for pose in poses)


def _arm_drive_witness(commands):
    clocks, targets, measured = [], [], []
    for row in commands:
        if row["applied_joint_positions"] is None:
            continue
        names = row["joint_names"]
        if not isinstance(names, list) or len(set(names)) != len(names):
            raise ValueError("Native joint names are missing or ambiguous")
        indices = [names.index(name) for name in _ARM_JOINTS]
        clocks.append(_finite(row["sim_time"]))
        targets.append(_array(row["applied_joint_positions"], (len(names),))[indices])
        measured.append(_array(row["measured_joint_positions_before"], (len(names),))[indices])
    if len(targets) < 2:
        return 0, 0., False
    changes = int(np.count_nonzero(np.max(np.abs(np.diff(targets, axis=0)), axis=1) > 1e-5))
    motion = float(np.max(np.abs(np.asarray(measured) - measured[0])))
    return changes, motion, bool(np.all(np.diff(clocks) >= 0))


def verify_rotation(evidence, binding, *, physics_rows, command_rows):
    """Recompute the signed effect and retention from native readings and drive commands."""
    checks, metrics = {}, {}
    try:
        if (evidence["schema"] != _VERSION or evidence["skill"] != SKILL
                or evidence["binding"] != binding.to_dict()):
            raise ValueError("Evidence is for a different rotation binding")
        snapshots = [evidence[name] for name in ("before", "after", "settled")]
        t0, t1, t2 = (_finite(row["sim_time_s"]) for row in snapshots)
        checks["simulated_and_settled"] = 0 <= t0 < t1 and t2 - t1 >= .15
        initial_object = _rigid_pose(evidence["initial_object"])
        initial_support = _rigid_pose(evidence["initial_support"])
        clocks, trace = _trace_window(physics_rows, t0, t2)
        sampled = {key: [_rigid_pose(row["poses"][key]) for row in trace] for key in _TRACE_KEYS}
        observed = {key: [_rigid_pose(row[key]) for row in snapshots] for key in _TRACE_KEYS}
        nearest = [int(np.argmin(np.abs(clocks - stamp))) for stamp in (t0, t1, t2)]
        raw_object = sampled["object"]
        raw_tcp = sampled["tcp"]
        start = {key: observed[key][0] for key in _TRACE_KEYS}
        metrics["physics_samples"] = len(trace)
        checks["synchronized_native_trace"] = (
            len(trace) >= 12 and all(abs(clocks[index] - stamp) <= .03
                                     for index, stamp in zip(nearest, (t0, t1, t2), strict=True))
            and bool(np.all((np.diff(clocks) > 0) & (np.diff(clocks) <= .031)))
            and all(_stable_measurement(live, [sampled[key][index]], .004, 3.)
                    for key in _TRACE_KEYS for live, index in zip(observed[key], nearest, strict=True))
        )
        checks["continuous_object_motion"] = all(
            _distance(prior, next_pose) <= .012 and _rotation_deg(prior, next_pose) <= 8.
            for prior, next_pose in zip(raw_object, raw_object[1:])
        )
        turns = [_signed_rotation(start["object"], pose) for pose in
                 (*observed["object"][1:], raw_object[nearest[-1]])]
        metrics["signed_rotation_deg"] = float(turns[-1][2])
        metrics["settled_orientation_error_deg"] = abs(metrics["signed_rotation_deg"] - binding.angle_deg)
        checks["signed_rotation_target"] = all(
            abs(turn[2] - binding.angle_deg) <= _MAX_TARGET_ERROR_DEG
            and np.linalg.norm(turn[:2]) <= _MAX_AXIS_ERROR_DEG for turn in turns
        )
        checks["held_pose_settled"] = _stable_measurement(
            observed["object"][1], [pose for stamp, pose in zip(clocks, raw_object, strict=True)
                                    if stamp >= t1 - .015], .004, 3.
        )
        tcp_to_object = np.linalg.inv(start["tcp"]) @ start["object"]
        measures = [np.linalg.inv(tcp) @ obj for tcp, obj in zip(raw_tcp, raw_object, strict=True)]
        metrics["maximum_tcp_drift_m"] = max(_distance(start["tcp"], pose) for pose in raw_tcp)
        metrics["maximum_hold_drift_m"] = max(_distance(tcp_to_object, pose) for pose in measures)
        metrics["minimum_lift_m"] = min(float(pose[2, 3] - initial_object[2, 3]) for pose in raw_object)
        checks["fixed_vertical_tool"] = (metrics["maximum_tcp_drift_m"] <= _MAX_TCP_DRIFT_M
                                          and all(pose[2, 2] <= -_VERTICAL_COSINE for pose in raw_tcp))
        checks["friction_grasp_retained"] = (
            metrics["minimum_lift_m"] >= _MIN_LIFT_M and start["object"][2, 3] - initial_object[2, 3] >= _MIN_LIFT_M
            and _stable_measurement(tcp_to_object, measures, _MAX_HOLD_DRIFT_M, _MAX_HOLD_DRIFT_DEG)
        )
        checks["support_remained_still"] = _stable_measurement(
            initial_support, sampled["support"], _MAX_SUPPORT_DRIFT_M, _MAX_SUPPORT_DRIFT_DEG
        )
        _, commands = _trace_window(command_rows, t0, t1, tolerance=.01)
        changed, measured, ordered = _arm_drive_witness(commands)
        metrics.update(changed_arm_setpoints=changed, measured_arm_delta_rad=measured)
        checks["applied_arm_motion"] = ordered and changed >= 3 and measured >= .01
    except (KeyError, TypeError, ValueError, AttributeError, np.linalg.LinAlgError) as exc:
        return {"passed": False, "checks": checks, "metrics": metrics, "error": str(exc)}
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}
