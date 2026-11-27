"""Bind and verify containment for the shared grasp, move, and release skills."""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from task_geometry import bbox_corners, transformed_points


APPROACH = "above_container"
DROP = "container_release"
RELATION = "inside_container_and_stable"
RELEASE_CLEARANCE_M = .008
MAX_DROP_M = .085


def bind(scene, robot_id):
    target = scene["official_instruction"]["target"]
    obj, receiver = target["id1"], target["id2"]
    geometry = scene["project_container_placement"]
    basis = np.asarray(geometry["object_basis"])
    bounds = scene["layout"][obj]["entity_local_bbox"]
    extent = np.asarray(bounds["max"]) - bounds["min"]
    # Keep the grasp finger opening across the narrow tabletop dimension.
    axis = int(np.argmax(extent * np.linalg.norm(basis[:2], axis=0)))
    return {
        "role_bindings": {"operator": geometry["operator"]},
        "manipulated_objects": [obj],
        "grasp_profiles": {obj: ["topdown_box"]},
        "grasp_axes": {obj: axis},
        "targets": [APPROACH, DROP],
        "target_objects": {APPROACH: obj, DROP: obj},
        "skill_targets": {"move_held": [APPROACH, DROP]},
        "release_modes": {obj: "native"},
        "relations": [RELATION],
        "bindings": {"workpiece": obj, "receiver": receiver},
        "placement_geometry": geometry,
        "required_capabilities": ["friction_grasp", "held_object_motion", "gripper_opening_control", "curobo"],
        "reference_strategy": (
            f"Grasp {obj} using topdown_box with role operator, then move_held first to {APPROACH} "
            f"and then to {DROP} in that order. Each target is computed from the measured container pose. "
            f"Release with mode native, retreat the operator, wait_stable for {obj}, verify_relation "
            f"{RELATION} on {obj}, and finish with verify_task. The container remains on the tabletop. "
            "For recovery only repeat unfinished work from measured facts. If the item is already released, "
            "continue with empty-hand retreat, stability and containment checks."
        ),
    }


def validate_steps(plan, spec, held, achieved):
    obj = spec["bindings"]["workpiece"]
    facts = set(achieved)
    for step in plan["steps"]:
        name, params = step["skill"], step["parameters"]
        if name == "grasp":
            facts.difference_update({"released:" + obj, "retreated", "stable",
                                     "relation:" + RELATION, "target:" + APPROACH, "target:" + DROP})
        elif name == "move_held":
            if params["target"] == DROP and "target:" + APPROACH not in facts:
                raise ValueError("Approach the cavity from above before descending to the release target")
            facts.add("target:" + params["target"])
        elif name == "release":
            if "target:" + DROP not in facts:
                raise ValueError("Release requires the geometry-checked container target")
            facts.add("released:" + obj)
        elif name == "retreat":
            if "released:" + obj not in facts:
                raise ValueError("Withdraw only after releasing the workpiece")
            facts.add("retreated")
        elif name in ("wait_stable", "verify_relation"):
            if params["object"] != obj or "retreated" not in facts:
                raise ValueError("Check the workpiece after retreating the empty hand")
            if name == "wait_stable":
                facts.add("stable")
            elif "stable" not in facts or params["relation"] != RELATION:
                raise ValueError("Containment requires a settled workpiece")
            else:
                facts.add("relation:" + RELATION)
        elif name == "verify_task":
            if "relation:" + RELATION not in facts:
                raise ValueError("Verify the physical containment relation before completing the task")
        else:
            raise ValueError("Skill does not belong to the containment goal: " + name)


def grasp_options(executor, obj, arm, attempt):
    """Place pinch candidates within the measured thickness and check a short physical lift first."""
    box = executor.ops.local_box(executor.robot, obj)
    pose = executor.ops.pose(executor.robot, obj)
    world = transformed_points(pose, bbox_corners(*box))
    height = float(np.ptp(world[:, 2]))
    span = float((box[1] - box[0])[executor.spec["grasp_axes"][obj]])
    thin = height < .018
    shift = min(.010, span * .19)
    heights = (-min(.002, height * .24),) if thin else (0., -.002, .002)
    yaws = (0., np.pi)
    offsets = (0.,)
    if attempt == 1:
        heights = (0.,) if thin else (.003, -.003)
        offsets = (-shift, shift)
    elif attempt >= 2:
        heights = (-min(.002, height * .24), .001) if thin else (.004, -.004)
        yaws = (-.20, .20, np.pi - .20, np.pi + .20)
    options = {"vertical_offsets_m": heights, "yaw_offsets_rad": yaws,
               "longitudinal_offsets_m": offsets}
    executor.event("placement_grasp_candidates", object=obj, arm=arm, attempt=attempt,
                   measured_thickness_m=height, measured_long_axis_m=span, options=options)
    return {"custom_pose_options": options, "lift_probe_distance_m": .012}


def receiver_frame(pose, geometry):
    frame = np.asarray(pose).copy()
    frame[:3, :3] = frame[:3, :3] @ np.asarray(geometry["receiver_basis"]).T
    return frame


def support_heights(points, geometry):
    lower, upper = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    heights = []
    for surface_lower, surface_upper, height in geometry["support_surfaces_m"]:
        overlap = np.minimum(upper, surface_upper) - np.maximum(lower, surface_lower)
        if np.all(overlap > .001):
            heights.append(height)
    return heights


def resolve_target(geometry, reference, obj):
    spec, ops, robot = geometry.spec, geometry.ops, geometry.robot
    plan = spec["placement_geometry"]
    receiver = spec["bindings"]["receiver"]
    frame = receiver_frame(ops.pose(robot, receiver), plan)
    if frame[2, 2] < .995:
        raise ValueError("The measured container opening is no longer upright")
    current = ops.pose(robot, obj)
    arm = spec["role_bindings"]["operator"]
    hand, _ = geometry.gripper_collision_points(arm)
    corners = bbox_corners(*ops.local_box(robot, obj))
    held_shape = np.concatenate([corners, transformed_points(np.linalg.inv(current), hand)])
    goal = np.eye(4)
    goal[:3, :3] = Rotation.from_rotvec([0, 0, plan["object_yaw_rad"]]).as_matrix() @ plan["object_basis"]
    goal[:2, 3] = np.asarray(plan["center_xy_m"]) - (corners.mean(axis=0) @ goal[:3, :3].T)[:2]
    limit = plan["highest_collision_z_m"] + RELEASE_CLEARANCE_M
    goal[2, 3] = limit - transformed_points(goal, held_shape)[:, 2].min()
    if reference == APPROACH:
        goal[2, 3] += .065
    elif reference != DROP:
        raise ValueError("Unknown containment target")
    return frame @ goal, []


def check_release(executor, params):
    from skill_executor import SkillFailure
    spec, ops, robot = executor.spec, executor.ops, executor.robot
    if "target:" + DROP not in executor.facts:
        raise SkillFailure("container_release_target_not_reached", {})
    obj = params["object"]
    plan = spec["placement_geometry"]
    frame = receiver_frame(ops.pose(robot, spec["bindings"]["receiver"]), plan)
    to_local = np.linalg.inv(frame)
    corners = bbox_corners(*ops.local_box(robot, obj))
    points = transformed_points(to_local @ ops.pose(robot, obj), corners)
    hand, snapshot = executor.geometry.gripper_collision_points(executor.arm(params["role"]))
    hand = transformed_points(to_local, hand)
    lower, upper = np.asarray(plan["opening_xy_m"])
    fit = float(min(np.min(points[:, :2] - lower), np.min(upper - points[:, :2])))
    clearance = float(min(points[:, 2].min(), hand[:, 2].min()) - plan["highest_collision_z_m"])
    heights = support_heights(points, plan)
    drop = float(points[:, 2].min() - min(heights)) if heights else None
    passed = bool(frame[2, 2] >= .995 and fit >= .001 and clearance >= .003
                  and drop is not None and 0 < drop <= MAX_DROP_M)
    measurements = {"object": obj, "passed": passed, "fit_margin_m": fit,
                    "rim_clearance_m": clearance, "expected_drop_m": drop,
                    "sim_time_s": snapshot["sim_time"], "receiver_collisions_active": True}
    executor.event("container_release_gate", **measurements)
    if not passed:
        raise SkillFailure("container_release_geometry_not_ready", measurements)


def measure(samples, corners, geometry):
    lower, upper = np.asarray(geometry["opening_xy_m"])
    rim = geometry["rim_z_m"]
    check_rows = []
    for item, receiver in samples:
        frame = receiver_frame(receiver, geometry)
        points = transformed_points(np.linalg.inv(frame) @ item, corners)
        bottom = float(points[:, 2].min())
        gaps = [bottom - height for height in support_heights(points, geometry)]
        check_rows.append({
            "inside_opening": bool((points[:, :2] >= lower - .001).all()
                                   and (points[:, :2] <= upper + .001).all()),
            "below_lip": bool(points[:, 2].max() <= rim + .003),
            "floor_gap_m": min(gaps, key=abs) if gaps else None,
            "receiver_upright": bool(frame[2, 2] >= .995),
        })
    anchor_obj, anchor_receiver = samples[0]
    object_shift = max(float(np.linalg.norm(item[:3, 3] - anchor_obj[:3, 3])) for item, _ in samples)
    angle_shift = max(float(Rotation.from_matrix(anchor_obj[:3, :3].T @ item[:3, :3]).magnitude())
                      for item, _ in samples)
    receiver_shift = max(float(np.linalg.norm(receiver[:3, 3] - anchor_receiver[:3, 3]))
                         for _, receiver in samples)
    checks = {
        "entire_object_inside": all(row["inside_opening"] and row["below_lip"] for row in check_rows),
        "supported_by_interior": all(row["floor_gap_m"] is not None and abs(row["floor_gap_m"]) <= .004
                                    for row in check_rows),
        "stable_object": object_shift < .003 and angle_shift < .06,
        "stable_open_container": receiver_shift < .003 and all(row["receiver_upright"] for row in check_rows),
    }
    return checks, {"samples": len(samples), "object_translation_m": object_shift, "object_rotation_rad": angle_shift,
                    "container_translation_m": receiver_shift,
                    "floor_gap_m": [row["floor_gap_m"] for row in check_rows]}


def verify_task(executor):
    obj, receiver = (executor.spec["bindings"][key] for key in ("workpiece", "receiver"))
    if "released:" + obj not in executor.facts or executor.holding:
        return False
    samples = []
    for _ in range(4):
        executor.robot.client.run_step(20)
        samples.append(tuple(executor.ops.pose(executor.robot, item) for item in (obj, receiver)))
    checks, metrics = measure(samples, bbox_corners(*executor.ops.local_box(executor.robot, obj)),
                              executor.spec["placement_geometry"])
    passed = all(checks.values())
    executor.event("container_physics_verification", passed=passed, checks=checks, **metrics)
    return passed


def verify_relation(executor, params):
    return params["relation"] == RELATION and verify_task(executor)


def audit(run, _raw, frames):
    """Recompute the outcome from raw observations after the native arm withdrew."""
    from audit_task_geometry import pose
    from scene_contract import read_json

    if len(frames) < 20:
        return {"enough_recorded_observations": False}, {"frames": len(frames)}
    launch = read_json(run / "launch.json")
    scene = read_json(run / "tasks" / launch["task"] / str(launch["scene"]) / "scene_info.json")

    spec = bind(scene, None)
    obj, receiver = (spec["bindings"][key] for key in ("workpiece", "receiver"))
    bounds = scene["layout"][obj]["entity_local_bbox"]
    checks, metrics = measure([(pose(f, obj), pose(f, receiver)) for f in frames[-20:]],
                              bbox_corners(bounds["min"], bounds["max"]), spec["placement_geometry"])
    release_events = [event for line in (run / "evidence/skill-execution.jsonl").read_text().splitlines()
                      if (event := json.loads(line)).get("event") == "container_release_gate" and event["passed"]]
    native = json.loads((run / "evidence/gripper-contract.json").read_bytes())["grippers"][spec["role_bindings"]["operator"]]
    target_joints = dict(zip(native["joint_names"], native["opened_positions"], strict=True))
    open_in_tail = True
    for frame in frames[-20:]:
        joints = frame["robot"]["joints"]
        available = dict(zip(joints["joint_name"], joints["joint_position"], strict=True))
        if not all(name in available and abs(available[name] - target) < .03 for name, target in target_joints.items()):
            open_in_tail = False
            break
    first = frames[0]["time_stamp"]
    checks["raw_observation_and_release_evidence"] = (
        bool(release_events) and first <= release_events[-1]["sim_time_s"] < frames[-20]["time_stamp"]
    )
    checks["gripper_open_in_tail"] = open_in_tail
    container_motion = np.linalg.norm(pose(frames[-1], receiver)[:3, 3] - pose(frames[0], receiver)[:3, 3])
    checks["container_remained_in_place"] = bool(container_motion < .012)
    metrics["container_displacement_m"] = float(container_motion)
    return checks, metrics
