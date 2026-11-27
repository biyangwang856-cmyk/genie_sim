"""Prepare and independently certify a native rotation in a calibrated workcell."""

from pathlib import Path

from robot_adapters import ROOT
from rotation_probe import ACTOR, PROFILE_FIELD, SKILL, SOURCE_IDENTITY_FILES, TASK, RotationBinding
from scene_contract import content_id, digest, read_json, verify_manifest, write_manifest
from skill_capabilities import RuntimeScope, SkillRegistry, _jsonl_path


INSTRUCTION = "Lift the phone and rotate it 15 degrees about the fixed tool position around world +Z"
PACKING_GOAL = {"effects": [{"relation": "inside", "object": "phone", "reference": "receiver"}],
                "preserve": ["insert", "lid"]}


def prepare_fixture(output, adapter, *, completion=None):
    from goal_composition import goal_text
    from scene_composer import compile_scene, save_json

    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / "jobs").resolve()):
        raise ValueError("Create rotation validation scenes inside the project jobs directory")
    assets = (ROOT / "GenieSimAssets").resolve()
    if completion not in (None, "pack_phone"):
        raise ValueError("Unknown validated continuation for the rotation skill")
    goal = PACKING_GOAL if completion == "pack_phone" else None
    output.mkdir(parents=True)
    _, manifest = compile_scene("phone_packaging", "0", adapter, output,
                                mode="existing", seed=20261126, goal=goal)
    metadata = read_json(output / "scene_info.json")
    if metadata["layout"][ACTOR]["usd"] != ACTOR:
        raise ValueError("The calibrated rotation fixture does not contain its annotated phone")
    dependencies = {name: sha for name, sha in manifest["asset_dependencies"].items()
                    if ACTOR in Path(name).parts}
    if not dependencies:
        raise ValueError("Phone geometry dependencies are absent from the frozen scene")
    binding = RotationBinding(actor=ACTOR, arm="right", asset_sha256=content_id(dependencies), angle_deg=15.)
    metadata["scene_id"] = output.name
    metadata[PROFILE_FIELD] = {"binding": binding.to_dict(), "asset_dependencies": dependencies}
    if completion:
        metadata[PROFILE_FIELD]["completion"] = completion
    instruction = INSTRUCTION if goal is None else INSTRUCTION + ". Then " + goal_text(goal)
    metadata["official_instruction"]["instruction"] = instruction
    save_json(output / "scene_info.json", metadata)
    info = read_json(output / "task_info.json")
    info["task_name"] = adapter.config["scene_task_prefix"] + TASK
    info["task_description"]["task_name"] = instruction
    info["task_description"]["english_task_name"] = instruction
    save_json(output / "task_info.json", info)
    manifest = write_manifest(output, (assets / name for name in manifest["asset_dependencies"]),
                              mode="existing", source={**manifest["source"], "candidate_skill": SKILL},
                              asset_root=assets)
    return {"task": TASK, "scene_id": output.name, "scene_directory": str(output),
            "scene_sha256": manifest["content_sha256"], "binding": binding.to_dict()}


def certify_run(run, registry_directory):
    run = Path(run).resolve()
    launch = read_json(run / "launch.json")
    if (launch["robot_id"] != "g2crsb90d" or launch["launch_returncode"] != 0
            or launch["task"] != "g2crsb90d_finemanip_" + TASK):
        raise ValueError("Certification requires the isolated native rotation probe")
    result = read_json(run / "collection_result.json")
    if result["success"] is not True or len(result["episodes"]) != 1:
        return {"status": "skill_validation_failed", "skill": SKILL, "reason": "collection_failed"}
    scene = run / "tasks" / launch["task"] / launch["scene"]
    manifest = verify_manifest(scene, ROOT / "GenieSimAssets")
    metadata = read_json(scene / "scene_info.json")
    profile = metadata[PROFILE_FIELD]
    binding = RotationBinding(**profile["binding"])
    dependencies = profile["asset_dependencies"]
    if (manifest["content_sha256"] != launch["scene_sha256"]
            or binding.asset_sha256 != content_id(dependencies) or metadata["layout"][binding.actor]["usd"] != ACTOR
            or any(manifest["asset_dependencies"].get(name) != sha for name, sha in dependencies.items())):
        raise ValueError("The live rotation asset or scene differs from the frozen binding")
    source = run / "source"
    for name in (*SOURCE_IDENTITY_FILES, "robot-calibration.json"):
        if digest(source / name) != launch["source_sha256"].get(name):
            raise ValueError(f"Unverified rotation source {name}")
    for name in ("rotation_probe.py", "skill_capabilities.py"):
        if digest(source / name) != digest(Path(__file__).with_name(name)):
            raise ValueError(f"Host rotation verifier differs from the archived source {name}")
    observations = run / "evidence/rotation-observations.json"
    scope = RuntimeScope(**read_json(observations)["scope"])
    source_identity = content_id({name: digest(source / name) for name in SOURCE_IDENTITY_FILES})
    if (scope.image_id != launch["image_id"] or scope.robot_id != launch["robot_id"]
            or scope.calibration_sha256 != digest(source / "robot-calibration.json")
            or scope.adapter_sha256 != source_identity):
        raise ValueError("Rotation measurement scope differs from the frozen runtime")
    registry = SkillRegistry(registry_directory)
    continuation = None
    if profile.get("completion") == "pack_phone":
        certificate = registry.lookup(SKILL, scope, binding)
        if certificate is None or metadata.get("project_packing_goal") != PACKING_GOAL:
            raise ValueError("Continuation must reuse a published skill and the frozen packing goal")
        certificate_sha = certificate["certificate_sha256"]
        stages_path = run / "evidence/stage-evidence.jsonl"
        execution_path = run / "evidence/skill-execution.jsonl"
        stages = list(_jsonl_path(stages_path))
        events = list(_jsonl_path(execution_path))
        expected_phases = ("rotation_reuse_receipt_ready", "rotation_reuse_skill_verdict",
                           "rotation_reuse_continuation_verdict")
        phase_indices = []
        for phase in expected_phases:
            matched = [(index, row) for index, row in enumerate(stages) if row.get("phase") == phase]
            if (len(matched) != 1 or matched[0][1].get("certificate_sha256") != certificate_sha
                    or (phase != expected_phases[0] and matched[0][1].get("passed") is not True)):
                raise ValueError(f"Rotation continuation has no verified witness for {phase}")
            phase_indices.append(matched[0][0])
        reused = [index for index, row in enumerate(events) if row.get("event") == "certified_skill_reused"
                  and row.get("certificate_sha256") == certificate_sha]
        verified = [index for index, row in enumerate(events) if row.get("event") == "independent_task_verdict"
                    and row.get("passed") is True]
        if (phase_indices != sorted(phase_indices) or len(reused) != 1 or len(verified) != 1 or reused[0] >= verified[0]
                or not any(row.get("event") == "validated_plan_started" for row in events[reused[0] + 1:verified[0]])):
            raise ValueError("The reuse trace has no verified planner continuation after rotation")
        continuation = {"goal": PACKING_GOAL, "reused_certificate_sha256": certificate_sha,
                        "independent_task_verdict": True,
                        "evidence_sha256": {path.name: digest(path) for path in (stages_path, execution_path)}}
    result = registry.certify_probe(SKILL, scope, binding, observations)
    if continuation is not None and result["status"] == "ready":
        result["continuation"] = continuation
    return result
