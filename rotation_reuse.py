"""Resume a declared packing goal after an independently certified held rotation."""

from importlib import import_module
import os
from pathlib import Path

import numpy as np

import collection_programs as ops
import rotation_probe
from goal_composition import PACKING_STAGES, goal_text, packing_goal
from robot_adapters import ROOT, load_robot
from scene_contract import content_id, digest, read_json
from skill_capabilities import RuntimeScope, SkillRegistry, _distance, _rigid_pose, _rotation_deg
from skill_executor import SkillFailure, run_structured


def _archived_modules(robot, source):
    names = ("robot_adapters", "skill_registry", "task_geometry", "skill_executor", "structured_planner",
             "qwen_backend", "goal_composition", "skill_recovery", "skill_capabilities", "rotation_probe",
             "rotation_reuse", "collection_programs")
    modules = {}
    for name in names:
        expected = source / (name + ".py")
        loaded = Path(import_module(name).__file__)
        if expected.resolve() != loaded.resolve():
            raise ValueError("The continuation loaded a module outside its immutable source snapshot")
        modules[name] = {"path": str(expected), "sha256": digest(loaded)}
    ops.note(robot, "program_loaded", **modules.pop("collection_programs"))
    ops.note(robot, "skill_modules_loaded", modules=modules)


def execute_task(robot):
    source = Path(os.environ["GENIE_RUN_SOURCE"])
    request = read_json(source / "request.json")
    fixture = robot.scene_info[rotation_probe.PROFILE_FIELD]
    if (request.get("task_id") != rotation_probe.TASK or request.get("execution_mode") != "structured"
            or request.get("robot_id") != "g2crsb90d" or fixture.get("completion") != "pack_phone"
            or (source / "prepared-plan.json").exists()):
        raise ValueError("The continuation requires a scoped rotation request and live planning")
    goal = packing_goal(robot.scene_info["project_packing_goal"])
    expected = {"effects": [dict(zip(("relation", "object", "reference"), PACKING_STAGES[0], strict=True))],
                "preserve": ["insert", "lid"]}
    if goal != expected:
        raise ValueError("The current rotation binding supports placing only the phone while preserving the insert and lid")
    binding = rotation_probe.RotationBinding(**fixture["binding"])
    _archived_modules(robot, source)
    adapter = load_robot(request["robot_id"])
    scope = RuntimeScope.from_robot_api(
        adapter, robot, image_id=read_json(source.parent / "launch.json")["image_id"],
        calibration_sha256=digest(source / "robot-calibration.json"),
        adapter_sha256=content_id({name: digest(source / name) for name in rotation_probe.SOURCE_IDENTITY_FILES}),
    )
    receipt = SkillRegistry(ROOT / "skill-registry").lookup(rotation_probe.SKILL, scope, binding)
    if receipt is None:
        ops.note(robot, "rotation_reuse_receipt_missing", skill=rotation_probe.SKILL,
                 binding=binding.to_dict(), runtime=scope.to_dict())
        return False

    snapshot = robot.client.command_any({"command": "ProjectKinematicSnapshot"})
    renderer_error = max(float(np.linalg.norm(np.asarray(snapshot["usd_links"][name])[:3, 3] - link["position"]))
                         for name, link in snapshot["links"].items())
    ops.note(robot, "render_baseline_gate", passed=renderer_error < .002, max_position_error_m=renderer_error)
    if not renderer_error < .002:
        return False
    ops.install_grasp_contact_compatibility()
    remaining = goal_text(goal)
    robot.client.set_task_task_basic_info({"task_name": request["instruction"], "english_task_name": request["instruction"],
                                           "init_scene_text": ""})
    certificate = receipt["certificate_sha256"]

    def rotate_and_measure(executor):
        if executor.arm("operator") != binding.arm or executor.spec["manipulated_objects"] != [binding.actor]:
            raise ValueError("The remaining packing plan must receive the same object in the validated arm")
        ops.note(robot, "rotation_reuse_receipt_ready", skill=rotation_probe.SKILL, certificate_sha256=certificate,
                 binding=binding.to_dict(), original_instruction=request["instruction"], remaining_instruction=remaining)
        passed = rotation_probe.execute_probe(robot, ops, request)
        if not passed:
            ops.note(robot, "rotation_reuse_skill_verdict", passed=False, certificate_sha256=certificate)
            return False
        witness = read_json(executor.folder / "rotation-observations.json")
        if witness["scope"] != scope.to_dict() or witness["binding"] != binding.to_dict():
            raise ValueError("The rotation evidence does not match its ready certificate")
        obj = _rigid_pose(ops.pose(robot, binding.actor))
        tcp = _rigid_pose(robot.get_ee_pose(arm=binding.arm))
        for key, measured in (("object", obj), ("tcp", tcp)):
            verified = _rigid_pose(witness["settled"][key])
            if _distance(measured, verified) > .004 or _rotation_deg(measured, verified) > 3.:
                raise SkillFailure("rotation_handoff_state_changed", {"measurement": key})
        ops.require_payload_proxy(robot, binding.actor, binding.arm, "rotation_reuse_payload_gate")
        executor.holding["operator"] = binding.actor
        executor.held_transforms["operator"] = np.linalg.solve(tcp, obj)
        executor.facts.add("rotated_held:" + binding.actor)
        executor.event("certified_skill_reused", skill=rotation_probe.SKILL, certificate_sha256=certificate,
                       binding=binding.to_dict(), state=executor.state())
        ops.note(robot, "rotation_reuse_skill_verdict", passed=True, certificate_sha256=certificate)
        return True

    client = robot.client
    original_playback = client.playback

    def reject_rollback(stored_state, *args, **kwargs):
        ops.note(robot, "native_rollback_request_blocked", stored_state=stored_state)
        raise SkillFailure("native_action_failed_before_rollback", {"stored_state": stored_state})

    completed = False
    client.playback = reject_rollback
    try:
        completed = run_structured(robot, ops, {**request, "instruction": remaining}, before_plan=rotate_and_measure)
        return completed
    finally:
        client.playback = original_playback
        ops.note(robot, "rotation_reuse_continuation_verdict", passed=completed is True, certificate_sha256=certificate)
