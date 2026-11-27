"""Bind an explicit replay request to an immutable, audited simulator recipe."""

from pathlib import Path

from robot_adapters import UnsupportedRequest
from scene_contract import digest, read_json, require_id, verify_manifest
from trajectory_admission import verified_reference
from trajectory_diversity import episode_quotas, scene_descriptor, trajectory_request


REPLAY_METHOD = "immutable_recipe_reexecution"
ASSET_RELEASE_FIELDS = ("dataset", "revision", "release_id", "file_count", "status")
SAME_RUNTIME_FIELDS = (
    "task", "scene", "robot_id", "execution_mode", "seed", "physics_backend",
    "planner_backend", "renderer_backend", "image_id", "initial_plan_sha256",
    "scene_sha256", "collection_slot", "slot_endpoint", "task_loader_sha256", "strict_g2_coupled_wrist",
)


def _sealed_source(run):
    launch_file = run / "launch.json"
    seal_file = run / "automatic-admission.json"
    audit_file = run / "acceptance.json"
    launch, seal, audit = (read_json(path) for path in (launch_file, seal_file, audit_file))
    if (seal.get("passed") is not True or audit.get("accepted") is not True
            or launch.get("launch_returncode") != 0 or not launch.get("container_id")
            or digest(launch_file) != seal.get("launch_sha256")
            or digest(audit_file) != seal.get("acceptance_sha256")
            or any(seal.get(key) != launch.get(key) for key in ("scene_sha256", "initial_plan_sha256", "image_id"))
            or not seal.get("motion_sha256")
            or digest(run / "evidence/scene-physics-preflight.json") != seal.get("physics_preflight_sha256")):
        raise ValueError("The replay reference is missing a valid sealed physical admission")
    if (launch.get("execution_mode") != "structured" or not launch.get("isolated_recording")
            or any(launch.get(field) is None for field in SAME_RUNTIME_FIELDS)):
        raise ValueError("Only sealed, isolated prepared-scene recipes have complete replay provenance")
    if type(launch["seed"]) is not int or not 0 <= launch["seed"] < 2**31:
        raise ValueError("The replay reference has no valid immutable seed")
    source = (run / "source").resolve()
    if source.parent != run:
        raise ValueError("The replay reference source directory escaped its run")
    hashes = launch.get("source_sha256", {})
    if not {"request.json", "prepared-plan.json", "robot-profile.json", "execution-profile.json"} <= hashes.keys():
        raise ValueError("The replay reference has no complete versioned program")
    for name, sha in hashes.items():
        file = (source / name).resolve()
        if file.parent != source or digest(file) != sha:
            raise ValueError(f"Frozen replay program evidence differs for {name}")
    return launch, digest(launch_file), digest(seal_file)


def _replay_source(run, asset_root):
    launch, launch_sha, admission_sha = _sealed_source(run)
    asset_receipt_file = Path(asset_root) / ".geniesim-assets.json"
    asset_receipt = read_json(asset_receipt_file)
    release = {key: asset_receipt.get(key) for key in ASSET_RELEASE_FIELDS}
    release["receipt_sha256"] = digest(asset_receipt_file)
    if release != launch.get("asset_release"):
        raise ValueError("The available asset release differs from the original launch")
    task, scene = require_id(launch["task"]), require_id(launch["scene"])
    folder = run / "tasks" / task / scene
    if folder.resolve() != folder or not folder.is_dir():
        raise ValueError("The replay scene is missing or not owned by its run")
    if digest(folder.parent / "data_gen.py") != launch["task_loader_sha256"]:
        raise ValueError("The archived task loader differs from the replay launch")
    metadata = read_json(folder / "scene_info.json")
    manifest = verify_manifest(folder, asset_root)
    if manifest["content_sha256"] != launch["scene_sha256"] or metadata.get("scene_id") != scene:
        raise ValueError("The replay scene differs from its immutable launch")
    plan_path = run / "source/prepared-plan.json"
    prepared = read_json(plan_path)
    if (prepared.get("validated") is not True or prepared["plan_sha256"] != launch["initial_plan_sha256"]
            or prepared["plan"]["scene_id"] != scene):
        raise ValueError("The replay plan differs from its immutable launch")
    source_request = read_json(run / "source/request.json")
    if (source_request["seed"] != launch["seed"] or source_request["robot_id"] != launch["robot_id"]
            or source_request["scene_id"] != scene or source_request["task_id"] != metadata["task_template"]):
        raise ValueError("The replay request differs from its immutable launch")
    aligned_archive = run / "evidence/aligned-archive.json"
    recording = (
        Path(read_json(aligned_archive)["aligned_recording"]).resolve() if aligned_archive.is_file() else run / "raw"
    )
    if (recording.parent != run or recording.name not in {"raw", "raw-aligned"}
            or not (recording / "state.json").is_file()
            or not (run / "evidence/native-control-commands.jsonl").is_file()):
        raise ValueError("The replay reference no longer has its native motion and control evidence")
    recipe = {
        "method": REPLAY_METHOD, "source_run": str(run), "source_launch_sha256": launch_sha,
        "source_admission_sha256": admission_sha, "plan_path": str(plan_path), "slot": launch["collection_slot"],
        "seed": launch["seed"], "instruction": source_request["instruction"], "physics_backend": launch["physics_backend"],
    }
    return recipe, launch, metadata, manifest, folder


def _resolve_reference(name, storage, asset_root):
    root = (Path(storage) / "collection-runs").resolve()
    source_run = (root / require_id(name)).resolve()
    if source_run.parent != root or source_run.name != name:
        raise ValueError("The replay reference must identify one collection run")
    try:
        return _replay_source(source_run, asset_root)
    except (KeyError, ValueError, OSError) as exc:
        unsupported = UnsupportedRequest(str(exc), missing_capabilities=["compatible_sealed_replay_recipe"])
        unsupported.details.update(code="REPLAY_REFERENCE_UNAVAILABLE", reference_run=name)
        raise unsupported from exc


def reference_task(reference_run, *, storage, asset_root, robot_id):
    """Get the original task from its sealed run when the request only names a replay."""
    _, launch, metadata, _, _ = _resolve_reference(reference_run, storage, asset_root)
    if robot_id != launch["robot_id"]:
        raise ValueError("Replay cannot change the robot of its reference")
    return metadata["task_template"]


def populate_replay(job, *, storage, asset_root, task_name, scene_options=None, per_scene=None, goal=None):
    """Freeze a single reference recipe before any new plan generation."""
    request = trajectory_request(job["trajectory_request"])
    if request["duplicate_policy"] != "replay":
        raise ValueError("A frozen reference can only be used for explicit replay")
    name = request["reference_run"]
    recipe, launch, metadata, manifest, folder = _resolve_reference(name, storage, asset_root)
    if job["robot_id"] != launch["robot_id"] or task_name != metadata["task_template"]:
        raise ValueError("Replay cannot change the robot or task of its reference")
    if goal is not None and goal != metadata.get("project_packing_goal"):
        raise ValueError("Replay cannot change the physical goal of its reference")
    if scene_options:
        provided = scene_options.get("scene_ids")
        compatible = [launch["scene"]]
        if manifest["mode"] == "existing":
            compatible.append(manifest["source"]["source_scene_id"])
        if (scene_options.get("scene_count", 1) != 1 or scene_options.get("mode", "existing") != "existing"
                or (provided is not None and (len(provided) != 1 or provided[0] not in compatible))
                or {"background_query", "background_id", "support_id", "support_query", "variation"}
                & scene_options.keys() or scene_options.get("seed", launch["seed"]) != launch["seed"]):
            raise ValueError("Replay requires the original fixed scene and seed")
    task, scene = launch["task"], launch["scene"]
    quotas = episode_quotas([scene], total_episodes=request.get("total_episodes"), per_scene=per_scene)
    descriptor = scene_descriptor(metadata, asset_digests=manifest["asset_dependencies"],
                                  lighting=manifest["source"].get("lighting_intensity"))
    scene_source = {"scene_id": scene, "directory": str(folder), "sha256": launch["scene_sha256"],
                    "source_scene_id": manifest["source"]["source_scene_id"], "seed": launch["seed"],
                    "diversity_scene": descriptor}
    job.update(
        automatic_admission=True, require_distinct_motion=False, status="skill_plans_prepared",
        scene_request={"mode": "existing", "scene_count": 1, "scene_ids": [scene]},
        scene_preflight={task_name: {"requested": 1, "compiled": 1, "reserve_candidates": [],
                                     "rejected": [], "reference_run": name}},
        replay_recipe=recipe,
        plan={"robot": job["robot_id"], "tasks": [{"task_id": task, "scene_quotas": quotas}]},
        success_counts={task: {scene: 0}},
        task_bindings={task: {"template": task_name, "scenes": [scene], "reserves": [],
                             "scene_sources": {scene: scene_source}}},
        prepared_skill_plans={task: {scene: {"path": recipe["plan_path"],
                                           "plan_sha256": launch["initial_plan_sha256"]}}},
    )
    return job


def recheck_replay_recipe(job, *, storage, asset_root):
    recipe = job["replay_recipe"]
    expected = (Path(storage) / "collection-runs").resolve() / require_id(job["trajectory_request"]["reference_run"])
    source = Path(recipe["source_run"])
    if source != expected or source.resolve() != expected:
        raise ValueError("The replay source no longer matches the frozen request")
    rebuilt, launch, _, _, _ = _replay_source(source, asset_root)
    if rebuilt != recipe:
        raise ValueError("The replay recipe was changed after preparation")
    return launch


def verify_replay_execution(run, recipe):
    """Verify a new launch used the same program, scene, seed and native runtime."""
    source = Path(recipe["source_run"])
    original = read_json(source / "launch.json")
    current = read_json(run / "launch.json")
    if digest(source / "launch.json") != recipe["source_launch_sha256"]:
        raise ValueError("The replay reference changed during collection")
    unequal = [field for field in SAME_RUNTIME_FIELDS if original[field] != current.get(field)]
    if unequal:
        raise ValueError("Recipe re-execution used different settings: " + ", ".join(unequal))
    if (run.resolve() == source or not current.get("container_id")
            or current["container_id"] == original["container_id"] or current.get("launch_returncode") != 0
            or current.get("source_recipe_run") != str(source) or not current.get("isolated_recording")):
        raise ValueError("Replay receipt does not represent a new isolated simulator execution")
    if (current["source_sha256"] != original["source_sha256"] or
            current.get("asset_release") != original.get("asset_release")):
        raise ValueError("Recipe re-execution used different programs or asset release")
    return verified_reference(source)
