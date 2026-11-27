"""Prepare a pinned microwave hinge fixture and certify recorded native operation."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from robot_adapters import ROOT, load_robot
from scene_contract import content_id, digest, read_json, require_id, verify_manifest, write_manifest
from skill_capabilities import (HINGE_SKILL, HingeBinding, HingeCapabilityGap, RuntimeScope,
                                SkillRegistry, _jsonl_path, discover_skills, execute_hinge,
                                grasp_hinge_handle, verify_hinge)


TASK = "probe_open_hinged_lid"
ACTOR = "benchmark_microwave_oven_003"
SUPPORT = "hinge_validation_support"
UPSTREAM_SCENE = "open_microwave_door_g2/4127542583_49b042e3"
INSTRUCTION = "Open the microwave door to its native hinge limit"
PROFILE_FIELD = "project_hinge_probe"


def _asset_file(asset_root, asset_id):
    candidates = [path.resolve() for path in (asset_root / "objects").rglob("Aligned.usda")
                  if path.parent.name == asset_id]
    if len(set(candidates)) != 1:
        raise ValueError(f"Missing or ambiguous fixture asset {asset_id}")
    return candidates[0]


def _fixture_calibration(task_info):
    robot = task_info["robot"]
    if robot["robot_cfg"] != "G2_crsB_90D.json":
        raise ValueError("The hinge fixture requires the existing calibrated G2_crsB_90D profile")
    task_info["task_name"] = "g2crsb90d_finemanip_" + TASK
    task_info["task_description"] = {"task_name": INSTRUCTION, "english_task_name": INSTRUCTION,
                                     "init_scene_text": "A fixed-base microwave sits on the table"}
    task_info["backend"] = "cpu"
    return task_info


def _pin_appliance_base(stage, base_path):
    from pxr import Gf, UsdGeom, UsdPhysics

    # The upstream example anchors the appliance body. Leave its door joint intact.
    base = stage.GetPrimAtPath(base_path)
    pose = UsdGeom.XformCache().GetLocalToWorldTransform(base)
    q = pose.ExtractRotationQuat()
    pin = UsdPhysics.FixedJoint.Define(stage, str(base_path) + "/ProjectFixtureAnchor")
    pin.CreateBody1Rel().SetTargets([base_path])
    pin.CreateLocalPos0Attr(Gf.Vec3f(*pose.ExtractTranslation()))
    pin.CreateLocalRot0Attr(Gf.Quatf(float(q.GetReal()), *map(float, q.GetImaginary())))


def _handle_grasp_geometry(handle, door):
    """Measure the official handle's straight section and its clearance to the door."""
    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    to_handle = cache.GetLocalToWorldTransform(handle).GetInverse()

    def collision_vertices(root):
        meshes = []
        for prim in Usd.PrimRange(root):
            if (prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)
                    and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False):
                points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                if points is not None and len(points):
                    transform = cache.GetLocalToWorldTransform(prim) * to_handle
                    meshes.append(np.array([transform.Transform(point) for point in points]))
        if not meshes:
            raise HingeCapabilityGap("The articulated asset lacks enabled handle or door colliders")
        return np.concatenate(meshes)

    vertices = collision_vertices(handle)
    door_front = float(collision_vertices(door)[:, 1].max())
    x_low, x_high = vertices[:, 0].min(), vertices[:, 0].max()
    # The ends of the official handle curve back into the door. Work on
    # the straight offset section, away from the two mounting posts.
    x_origin = (x_low + x_high) / 2
    offset = (vertices[:, 0] - x_origin) / (x_high - x_low)
    section = vertices[(offset >= .10) & (offset <= .32) & (vertices[:, 1] > door_front + .003)]
    if len(section) < 8:
        raise HingeCapabilityGap("The official handle's graspable section is unobservable")
    low, high = section.min(axis=0), section.max(axis=0)
    return {"contact_box_local_m": [low.tolist(), high.tolist()], "door_front_y_m": door_front}


def prepare_fixture(output, *, position_x=.95, position_y=.12, template=None):
    """Use the official articulated asset in the project's calibrated workcell."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from scene_composer import asset_dependencies, save_json, set_pose, world_bounds

    root = ROOT.resolve()
    output = Path(output).resolve()
    if not output.is_relative_to(root / "jobs"):
        raise ValueError("Build the fixture under the project jobs directory")
    require_id(output.name)
    if not .7 <= position_x <= 1.15 or not -.18 <= position_y <= .18:
        raise ValueError("Fixture position is outside the calibrated G2 workcell")
    assets = (root / "GenieSimAssets").resolve()
    microwave = _asset_file(assets, ACTOR)
    try:
        table = _asset_file(assets, "table_000")
    except ValueError:
        table = _asset_file(assets, "benchmark_table_019")
    orientation = [0., 0., .999318, .036917]
    if template is not None:
        example = read_json(Path(template) / "scene_info.json")
        matching = [row for row in example["layout"].values() if row.get("usd") == ACTOR]
        if len(matching) != 1:
            raise ValueError("Example metadata does not bind the same articulated asset")
        orientation = matching[0]["xyzw"]
    orientation = np.asarray(orientation, dtype=float)
    orientation /= np.linalg.norm(orientation)

    metadata = {"scene_id": output.name, "task_template": TASK, "layout": {},
                "official_instruction": {"instruction": INSTRUCTION, "target": {"id1": ACTOR},
                                         "gripper": {"1": {"side": "left", "handover": False}}},
                "official_subtask_steps": []}
    info = _fixture_calibration(read_json(root / "tasks/g2crsb90d_finemanip_phone_packaging/0/task_info.json"))
    output.mkdir(parents=True, exist_ok=False)
    stage = Usd.Stage.CreateNew(str(output / "scene.usda"))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.Xform.Define(stage, "/World/Objects")
    for name, file, center, quaternion in [
        (SUPPORT, table, [1.02, 0., 0.], [0., 0., 0., 1.]),
        (ACTOR, microwave, [position_x, position_y, 0.], orientation),
    ]:
        prim = UsdGeom.Xform.Define(stage, "/World/Objects/" + name).GetPrim()
        prim.GetPayloads().AddPayload(str(file))
        set_pose(prim, center, quaternion)
        entity_path = str(prim.GetPath()) + "/entity"
        # Bound the physical appliance body. The asset's RectLight has a one-metre render AABB.
        bounds_path = entity_path if name == SUPPORT else entity_path + "/body"
        floor, top = world_bounds(stage, bounds_path)
        center[2] = .74 - top[2] if name == SUPPORT else .742 - floor[2]
        set_pose(prim, center, quaternion)
        if name == SUPPORT:
            support = stage.GetPrimAtPath(entity_path)
            # Preserve the original collision meshes and make the support stationary.
            if support.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(support).CreateKinematicEnabledAttr(True)
            table_bounds = world_bounds(stage, entity_path)
        else:
            work_bounds = world_bounds(stage, bounds_path)
            if np.any(work_bounds[0][:2] < table_bounds[0][:2]) or np.any(work_bounds[1][:2] > table_bounds[1][:2]):
                raise ValueError("The physical microwave base does not fit on the registered support")
        low, high = world_bounds(stage, bounds_path, local=True)
        metadata["layout"][name] = {"usd": file.parent.name, "xyz": list(map(float, center)),
                                     "xyzw": list(map(float, quaternion)), "registered": True,
                                     "asset": str(file),
                                     "entity_local_bbox": {"min": low.tolist(), "max": high.tolist()}}

    entity = stage.GetPrimAtPath(f"/World/Objects/{ACTOR}/entity")
    hinges = [UsdPhysics.RevoluteJoint(prim) for prim in Usd.PrimRange(entity) if prim.IsA(UsdPhysics.RevoluteJoint)]
    if len(hinges) != 1:
        raise HingeCapabilityGap("The official microwave fixture must expose exactly one native revolute joint")
    hinge = hinges[0]
    bodies = [relation.GetTargets() for relation in (hinge.GetBody0Rel(), hinge.GetBody1Rel())]
    prefix = str(entity.GetPath()) + "/"
    if any(len(paths) != 1 or not str(paths[0]).startswith(prefix) for paths in bodies):
        raise HingeCapabilityGap("The hinge bodies do not resolve inside the asset")
    handle = prefix + "part_01"
    link = str(bodies[1][0])
    handle_prim = stage.GetPrimAtPath(handle)
    if not handle_prim:
        raise HingeCapabilityGap("The annotated microwave handle part_01 is missing")
    fixed_handle = None
    if not (handle == link or handle.startswith(link + "/")):
        # The official asset uses a separate rigid handle linked to the moving door.
        for prim in Usd.PrimRange(handle_prim):
            if prim.IsA(UsdPhysics.FixedJoint):
                joint = UsdPhysics.FixedJoint(prim)
                parties = {str(path) for relation in (joint.GetBody0Rel(), joint.GetBody1Rel())
                           for path in relation.GetTargets()}
                if parties == {link, handle} and joint.GetJointEnabledAttr().Get() is True:
                    fixed_handle = str(prim.GetPath())
                    break
        if fixed_handle is None:
            raise HingeCapabilityGap("The handle has no enabled fixed joint to the moving revolute body")
    _pin_appliance_base(stage, bodies[0][0])
    stage.GetRootLayer().Save()
    dependencies = asset_dependencies(stage, metadata)
    asset_scope = {file.relative_to(assets).as_posix(): digest(file)
                   for file in dependencies if file.is_relative_to(microwave.parent)}
    if microwave.relative_to(assets).as_posix() not in asset_scope:
        raise ValueError("Fixture geometry missing from frozen dependencies")
    grasp_geometry = _handle_grasp_geometry(handle_prim, stage.GetPrimAtPath(link))
    contact = tuple(np.asarray(grasp_geometry["contact_box_local_m"]).mean(axis=0))
    binding = HingeBinding(ACTOR, str(hinge.GetPath()), "part_01", "left", "min",
                           content_id(asset_scope), contact, fixed_handle)
    metadata[PROFILE_FIELD] = {"binding": binding.to_dict(), "upstream_scene": UPSTREAM_SCENE,
                              "fixed_body_path": str(bodies[0][0]), "asset_dependencies": asset_scope,
                              "grasp_geometry": grasp_geometry}
    save_json(output / "scene_info.json", metadata)
    save_json(output / "task_info.json", info)
    manifest = write_manifest(output, dependencies, mode="existing",
                              source={"upstream_scene": UPSTREAM_SCENE, "fixture": binding.applicability()},
                              asset_root=assets)
    return {"task": TASK, "scene_id": output.name, "scene_directory": str(output),
            "scene_sha256": manifest["content_sha256"], "binding": binding.to_dict()}


def _source_identity(source):
    adapter_files = ("robot-profile.json", "robot_adapters.py", "robot.urdf", "project_runtime_hooks.py",
                     "hinge_fixture.py")
    return content_id({name: digest(source / name) for name in adapter_files})


def _binding(metadata):
    fixture = metadata[PROFILE_FIELD]
    binding = HingeBinding(**fixture["binding"])
    if content_id(fixture["asset_dependencies"]) != binding.asset_sha256:
        raise HingeCapabilityGap("Fixture binding does not match its pinned asset")
    if metadata["layout"][binding.actor]["usd"] != ACTOR:
        raise HingeCapabilityGap("No annotated microwave grasp for the requested asset")
    return binding


def execute_probe(robot, ops, request):
    """RobotAPI callback for collection_control's isolated certification episode."""
    from client.llm_gen.robot_api import MoveConfig, OpenConfig

    source = Path(os.environ["GENIE_RUN_SOURCE"])
    folder = Path(os.environ["GENIE_EPISODE_EVIDENCE"])
    folder.mkdir(parents=True, exist_ok=True)
    if request["robot_id"] != "g2crsb90d" or robot.scene_info["task_template"] != TASK:
        raise HingeCapabilityGap("Hinge validation is outside the loaded fixture and robot")
    binding = _binding(robot.scene_info)
    launch = read_json(source.parent / "launch.json")
    adapter = load_robot(request["robot_id"])
    scope = RuntimeScope.from_robot_api(adapter, robot, image_id=launch["image_id"],
                                        calibration_sha256=digest(source / "robot-calibration.json"),
                                        adapter_sha256=_source_identity(source))
    gap = discover_skills([HINGE_SKILL], adapter, robot_api=robot, scope=scope,
                          registry=SkillRegistry(ROOT / "skill-registry"), bindings={HINGE_SKILL: binding})
    ops.note(robot, "hinge_capability_discovery", **gap)
    if gap["status"] not in ("ready", "validating_skill"):
        return False

    original_playback = robot.client.playback

    def stop_before_rollback(stored_state, *args, **kwargs):
        ops.note(robot, "native_rollback_request_blocked", stored_state=stored_state)
        raise HingeCapabilityGap("Native action failed before rollback")

    # The intentional grasp contact touches these two parts. Remove their static
    # planner proxies during the contact operation, as in the upstream example.
    # The microwave base and table remain planning obstacles. PhysX is unchanged.
    root = robot.get_object_prim_path(binding.actor)
    joint = robot.get_articulation_joint_info(root, joint_prim_path=f"{root}/{binding.joint_path}")
    paths = sorted({joint["body1_path"], f"{root}/{binding.handle_part}"})
    if any(not path.startswith(root + "/") for path in paths) or joint["body0_path"] in paths:
        raise HingeCapabilityGap("Planner contact scope must not exclude the cabinet body")
    robot.client.playback = stop_before_rollback
    try:
        robot.remove_objs_from_obstacle(paths)
        ops.note(robot, "hinge_planner_contact_scope", paths=paths)
        grasp = grasp_hinge_handle(
            robot, MoveConfig, binding,
            geometry=robot.scene_info[PROFILE_FIELD]["grasp_geometry"],
            observe_preflight=lambda witness: ops.note(robot, "hinge_grasp_arc_preflight", **witness),
            observe_attempt=lambda witness: ops.note(robot, "hinge_grasp_attempt", **witness),
        )
        ops.note(robot, "hinge_grasp_probe", **grasp)
        if not grasp["success"]:
            return False
        evidence = execute_hinge(
            robot, OpenConfig, binding, scope=scope,
            observe_segment=lambda witness: ops.note(robot, "hinge_segment_result", **witness),
            verified_grasp=grasp["before_confirmed_pull"],
        )
    except HingeCapabilityGap as exc:
        ops.note(robot, "hinge_probe_failure", reason=exc.reason, **exc.details)
        return False
    finally:
        try:
            robot.add_objs_to_obstacle(paths)
        finally:
            robot.client.playback = original_playback

    (folder / "hinge-observations.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    verification = verify_hinge(evidence, binding, physics_rows=_jsonl_path(folder / "object-physics-120hz.jsonl"),
                                command_rows=_jsonl_path(folder / "native-control-commands.jsonl"))
    ops.note(robot, "hinge_physical_probe", **verification)
    return verification["passed"]


def certify_run(run, registry_directory):
    run = Path(run).resolve()
    launch = read_json(run / "launch.json")
    evidence_path = run / "evidence/hinge-observations.json"
    if (launch["robot_id"] != "g2crsb90d" or launch["launch_returncode"] != 0
            or launch["task"] != "g2crsb90d_finemanip_" + TASK):
        raise ValueError("Only a launched, isolated hinge probe may publish a certificate")
    output = read_json(run / "collection_result.json")
    if output["success"] is not True or len(output["episodes"]) != 1:
        return {"status": "skill_validation_failed", "skill": HINGE_SKILL, "reason": "collection_failed"}
    scene = run / "tasks" / launch["task"] / launch["scene"]
    manifest = verify_manifest(scene, ROOT.resolve() / "GenieSimAssets")
    if manifest["content_sha256"] != launch["scene_sha256"]:
        raise ValueError("Validation scene differs from the frozen launch")
    metadata = read_json(scene / "scene_info.json")
    if any(manifest["asset_dependencies"].get(name) != sha
           for name, sha in metadata[PROFILE_FIELD]["asset_dependencies"].items()):
        raise ValueError("Fixture source paths changed after simulation")
    source = run / "source"
    for name in ("skill_capabilities.py", "hinge_fixture.py", "robot_adapters.py", "robot-profile.json",
                 "robot.urdf", "robot-calibration.json", "project_runtime_hooks.py"):
        if digest(source / name) != launch["source_sha256"].get(name):
            raise ValueError(f"Unverified validation source {name}")
    if digest(source / "skill_capabilities.py") != digest(Path(__file__).with_name("skill_capabilities.py")):
        raise ValueError("The host verifier differs from the approved candidate")
    scope = RuntimeScope(**read_json(evidence_path)["scope"])
    if (scope.image_id != launch["image_id"] or scope.robot_id != launch["robot_id"]
            or scope.calibration_sha256 != digest(source / "robot-calibration.json")
            or scope.adapter_sha256 != _source_identity(source)):
        raise ValueError("The measured API capabilities do not match the frozen collection runtime")
    return SkillRegistry(registry_directory).certify_hinge(scope, _binding(metadata), evidence_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare", help="Generate scene files under the project jobs directory")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--template", type=Path)
    prepare.add_argument("--position-x", type=float, default=.95)
    prepare.add_argument("--position-y", type=float, default=.12)
    certify = actions.add_parser("certify", help="Verify finished probe and publish its scoped skill")
    certify.add_argument("--run", type=Path, required=True)
    certify.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare_fixture(args.output, template=args.template,
                                 position_x=args.position_x, position_y=args.position_y)
    else:
        result = certify_run(args.run, args.registry)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") != "skill_validation_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
