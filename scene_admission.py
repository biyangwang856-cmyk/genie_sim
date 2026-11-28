"""Run prepared, immutable scene instances with automatic physical admission."""
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

from collection_control import ROOT, SLOT_BUSY_EXIT, SOCKET, STORAGE
from scene_contract import choose_replacement, content_id, digest, read_json, verify_manifest

SCENE_SENSITIVE_FAILURES = {
    "grasp_unreachable_or_failed", "grasp_candidate_changed_world",
    "probed_grasp_lift_exhausted",
    "no_paired_presentation_candidate", "no_paired_presentation_path",
    "motion_or_tracking_failed", "local_waypoint_plan_or_tracking_failed",
    "native_action_failed_before_rollback", "cap_staging_motion_failed",
    "pen_grasp_lost_before_alignment", "cap_grasp_lost_before_alignment",
    "pen_presentation_recovery_unreachable", "pen_not_stable_before_alignment",
    "cap_preinsert_recovery_exhausted", "shape_transport_grasp_slipped",
    "shape_grasp_not_stable_over_aperture", "axial_insertion_stalled",
    "payload_not_cleared_during_release",
    "failed_passage_evidence_after_containment",
}


def pinned_code():
    from collection_control import SOURCE_FILES
    files = [*SOURCE_FILES, "collection_control.py", "scene_contract.py", "scene_admission.py",
             "batch_collect.py", "audit_episode.py", "audit_task_geometry.py",
             "audit_plan_execution.py", "finalize_job.py", "export_lerobot.py", "task_plugins.py",
             "archive_episode.py", "bundle_raw_trajectories.py", "audit_capture_clock.py",
             "validate_render_alignment.py", "validate_dynamic_render.py", "validate_snapshot_fk.py",
             "reextract_recording.py", "verify_lerobot_reader.py", "metrics_report.py",
             "lerobot_layout.py", "lerobot_v3.py", "verify_download.py",
             "collection_scope.py", "natural_backend.py", "prepare_skill_plans.py",
             "scene_composer.py", "scene_catalog.py", "asset_retrieval.py", "genie_extensions.py",
             "trajectory_diversity.py", "trajectory_admission.py", "trajectory_replay.py",
             "container_goal.py", "container_scene.py",
             "output/insertion-certificate.json", "output/insertion_target_v2.json"]
    for directory in ("pipeline/task_contracts", "pipeline/robots", "configs"):
        files += [str(path.relative_to(ROOT)) for path in (ROOT/directory).glob("*")
                  if path.suffix in {".json",".py"} and path.is_file()]
    return content_id({name: digest(ROOT / name) for name in sorted(set(files))})


def docker_state(label, job_id, env):
    result = subprocess.run(["docker", "inspect", "--format",
                             '{{.State.Running}} {{index .Config.Labels "geniesim.owner-job"}}',
                             "genie-by-" + label], capture_output=True, text=True, env=env)
    if result.returncode:
        if "no such object" in result.stderr.lower() or "no such container" in result.stderr.lower():
            return None
        raise RuntimeError("Project Docker could not confirm trial state: " + result.stderr.strip()[-1000:])
    running, owner = result.stdout.split()
    if owner != job_id:
        raise ValueError("Existing container belongs to a different job")
    return running == "true"


def archive_trial(run, trial, job, scene):
    from batch_collect import audit_recording, motion_fingerprint, write
    report = run / "collection_result.json"
    if not report.is_file():
        trial["failure"] = "native_collection_report_missing"
        return
    try:
        archived, result = audit_recording(run)
    except (ValueError, KeyError, IndexError, TypeError, RuntimeError, OSError) as exc:
        trial["failure"] = "episode_audit_error"
        trial["detail"] = str(exc)
        return
    trial["raw_archive"] = archived
    trial["automatic_acceptance"] = result.get("accepted") is True
    if not archived or not trial["automatic_acceptance"]:
        trial["failure"] = "independent_episode_verification_failed"
        trial["failed_checks"] = [key for key, value in result.get("checks", {}).items() if not value]
        if "training_arm_command_continuity" in trial["failed_checks"]:
            trial["failure"] = "training_arm_command_discontinuity"
        signature = scene_failure(run)
        if signature:
            trial["scene_failure_signature"] = signature
        return
    try:
        fingerprint = motion_fingerprint(run)
    except (ValueError, RuntimeError, OSError) as exc:
        trial.update(failure="invalid_motion_recording", detail=str(exc))
        return
    launch = read_json(run / "launch.json")
    if not launch.get("container_id") or launch.get("launch_returncode")!=0:
        trial["failure"]="launch_failed"
        return
    receipt = {"passed": True, "method": "independent_continuous_physics_and_recording",
               "scene_sha256": scene["sha256"], "initial_plan_sha256": launch["initial_plan_sha256"],
               "code_sha256": job["execution_code_sha256"], "image_id": launch["image_id"],
               "launch_sha256": digest(run / "launch.json"),
               "acceptance_sha256": digest(run / "acceptance.json"),
               "physics_preflight_sha256": digest(run / "evidence/scene-physics-preflight.json"),
               "motion_sha256": fingerprint}
    if "trajectory_request" in job:
        from trajectory_admission import EVIDENCE, admit_episode
        from trajectory_diversity import motion_descriptor_from_run, scene_descriptor_from_run

        reference = None
        replay = job.get("replay_recipe")
        if replay:
            from trajectory_replay import verify_replay_execution

            try:
                reference = verify_replay_execution(run, replay)
            except (ValueError, KeyError, IndexError, OSError, RuntimeError) as exc:
                trial.update(failure="replay_recipe_nonconforming", detail=str(exc))
                return
        try:
            decision = admit_episode(job, trial, scene=scene_descriptor_from_run(run),
                                     motion=motion_descriptor_from_run(run), fingerprint=fingerprint,
                                     reference=reference, lineage_db=STORAGE / "trajectory-admissions.sqlite3")
        except sqlite3.Error as exc:
            trial.update(failure="trajectory_admission_unavailable", detail=str(exc))
            return
        except (ValueError, KeyError, IndexError, OSError, RuntimeError) as exc:
            trial.update(failure="invalid_trajectory_diversity_evidence", detail=str(exc))
            return
        if decision.get("accepted") is not True:
            trial.update(decision)
            if decision["failure"] in {"repeated_scene_content", "exact_motion_repeat", "near_motion_repeat"}:
                trial["scene_failure_signature"] = ("diversity/repeated_scene_content"
                    if decision["failure"] == "repeated_scene_content" else "diversity/redundant_motion")
            return
        record = decision["record"]
        write(run / EVIDENCE, record)
        receipt["diversity_sha256"] = digest(run / EVIDENCE)
        if replay:
            receipt["replay"] = {"method": replay["method"], "reference_run": job["trajectory_request"]["reference_run"],
                                 "reference_launch_sha256": replay["source_launch_sha256"],
                                 "comparison": record["replay_comparison"]}
        trial["diversity_sha256"] = receipt["diversity_sha256"]
        if record.get("duplicate_group"):
            trial["duplicate_group"] = record["duplicate_group"]
    else:
        # Older prepared jobs retain their exact-fingerprint admission rule.
        duplicate = next((row for row in job["attempts"]
                          if row.get("accepted") and row.get("fingerprint") == fingerprint), None)
        if duplicate:
            trial.update(failure="exact_motion_repeat", duplicate_motion_of=duplicate["run_directory"])
            return
    certificate = run / "automatic-admission.json"
    write(certificate, receipt)
    trial.update(accepted=True, fingerprint=fingerprint, certificate_sha256=digest(certificate))

def scene_failure(run):
    import json

    preflight = run / "evidence/scene-physics-preflight.json"
    if preflight.is_file() and read_json(preflight).get("passed") is False:
        return "initial_support_or_stability"
    path = run / "evidence/skill-execution.jsonl"
    if path.is_file():
        for row in reversed(path.read_text().splitlines()):
            try:
                event = json.loads(row)
            except json.JSONDecodeError:
                return None
            if event.get("event") != "step_failed":
                continue
            reason = event.get("error_type")
            if reason not in SCENE_SENSITIVE_FAILURES:
                return None
            step = event["step"]
            params = step.get("parameters", {})
            details = event.get("details") or {}
            return "/".join([reason, step["skill"], params.get("target", params.get("object","")),
                             details.get("phase","")])
    return None


def wait_for_storage(job, job_path, deadline):
    """Defer a launch without consuming a physical attempt while space is low."""
    from batch_collect import write

    minimum = job["budgets"]["minimum_free_bytes"]
    while (now := time.time()) < deadline:
        free = shutil.disk_usage(STORAGE).free
        if free >= minimum:
            if job.get("budget_reason") == "minimum_free_storage":
                job.pop("budget_reason")
                job.pop("storage_wait", None)
                if job["status"] == "waiting_for_storage":
                    job["status"] = "trial"
                write(job_path, job)
            return True

        delay = min(30., deadline - now)
        job["status"] = "waiting_for_storage"
        job["budget_reason"] = "minimum_free_storage"
        job["worker_heartbeat_unix"] = now
        job["storage_wait"] = {"free_bytes": free, "minimum_free_bytes": minimum,
                               "retry_at_unix": now + delay}
        write(job_path, job)
        time.sleep(delay)
    return False


def run_trial(job, job_path, task, scene, trial, slot, deadline, env):
    from batch_collect import write
    run = Path(trial["run_directory"])
    launch_file = run / "launch.json"
    prepared = job["prepared_skill_plans"][task][scene["scene_id"]]
    state = docker_state(trial["label"], job["job_id"], env)
    if state is None and not run.exists():
        replay = job.get("replay_recipe")
        instruction = job["instruction"]
        scene_flags = ["--scene-directory", scene["directory"], "--scene-sha256", scene["sha256"]]
        if replay:
            from trajectory_replay import recheck_replay_recipe

            try:
                recheck_replay_recipe(job, storage=STORAGE, asset_root=ROOT / "GenieSimAssets")
                if trial["seed"] != replay["seed"] or slot != replay["slot"]:
                    raise ValueError("Reserved replay attempt differs from the frozen seed or slot")
            except (ValueError, KeyError, OSError) as exc:
                trial.update(failure="replay_recipe_nonconforming", detail=str(exc))
                return
            instruction = replay["instruction"]
            scene_flags = ["--source-run", replay["source_run"], "--physics-backend", replay["physics_backend"]]
        command = [sys.executable, str(ROOT / "collection_control.py"), job["task_bindings"][task]["template"],
                   "--scene", scene["scene_id"], "--label", trial["label"], "--robot-id", job["robot_id"],
                   "--instruction", instruction, "--execution-mode", "structured", "--slot", slot, *scene_flags,
                   "--prepared-plan", prepared["path"], "--owner-job", job["job_id"], "--seed", str(trial["seed"])]
        log = job_path.parent / "attempts" / (trial["label"] + ".log")
        log.parent.mkdir(exist_ok=True)
        with log.open("a", encoding="utf-8") as output:
            while True:
                if time.time()>=deadline:
                    trial.update(failure="collection_slot_wait_budget_exhausted",launch_log=str(log))
                    return
                if not wait_for_storage(job, job_path, deadline):
                    trial.update(failure="storage_wait_budget_exhausted", launch_log=str(log))
                    return
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, timeout=300)
                output.write(result.stdout)
                output.flush()
                if result.returncode!=SLOT_BUSY_EXIT:
                    break
                wait=json.loads(result.stdout.splitlines()[-1])
                job["status"]=wait["status"]
                job["collection_wait"]=wait
                job["worker_heartbeat_unix"]=time.time()
                write(job_path,job)
                time.sleep(min(15.,max(0.,deadline-time.time())))
        job.pop("collection_wait",None)
        if result.returncode:
            trial.update(failure="launch_failed", launch_log=str(log))
            return
        state = docker_state(trial["label"], job["job_id"], env)
    if not launch_file.is_file():
        if state is True:
            # A surviving launcher may still be atomically publishing its launch receipt.
            for _ in range(15):
                time.sleep(2)
                if launch_file.is_file():break
        if launch_file.is_file():
            return run_trial(job,job_path,task,scene,trial,slot,deadline,env)
        trial["failure"] = "incomplete_launch_receipt"
        return
    launch = read_json(launch_file)
    if (launch.get("owner_job"),launch.get("scene_sha256"),launch.get("initial_plan_sha256"),
            launch.get("collection_slot")) != (job["job_id"],scene["sha256"],prepared["plan_sha256"],trial["slot"]):
        raise ValueError("Launched scene, plan or slot differs from the frozen trial")
    trial["container"] = launch["container"]
    trial["launch_sha256"] = digest(launch_file)
    job["status"] = "collecting"
    while state is True:
        job["worker_heartbeat_unix"] = time.time()
        write(job_path, job)
        if time.time() >= min(deadline, launch["created_unix"] + 4500):
            # Scoped to the container whose job label was just verified.
            subprocess.run(["docker", "stop", "--time", "15", launch["container"]], check=True,
                           stdout=subprocess.DEVNULL, env=env, timeout=40)
            trial["failure"] = "collection_watchdog"
            return
        time.sleep(10)
        state = docker_state(trial["label"], job["job_id"], env)
        if state is None:
            trial["failure"] = "owned_container_disappeared"
            return
    archive_trial(run, trial, job, scene)


def success_counts(job):
    counts = {task["task_id"]: {row["scene_id"]: 0 for row in task["scene_quotas"]}
              for task in job["plan"]["tasks"]}
    for trial in job["attempts"]:
        if trial.get("accepted") is not True:
            continue
        run = Path(trial["run_directory"])
        receipt = read_json(run / "automatic-admission.json")
        if (digest(run / "automatic-admission.json") != trial["certificate_sha256"] or
                digest(run / "launch.json") != trial["launch_sha256"] or
                digest(run / "acceptance.json") != receipt["acceptance_sha256"]):
            raise ValueError("Admitted episode changed since the independent verification")
        if "diversity_sha256" in receipt:
            from trajectory_admission import read_diversity_record

            if trial.get("diversity_sha256") != receipt["diversity_sha256"]:
                raise ValueError("Trajectory admission and diversity evidence differ")
            read_diversity_record(trial)
        counts[trial["task"]][trial["scene"]] += 1
    return counts


def execute(job, job_path, options):
    from batch_collect import write
    from finalize_job import finalize
    from metrics_report import report

    if options.adopt_run or options.source_run:
        raise ValueError("Prepared scene jobs use their own immutable launch receipts")
    replay = job.get("replay_recipe")
    replay_requested = job.get("trajectory_request", {}).get("duplicate_policy") == "replay"
    if replay_requested and replay is None and not options.audit_only:
        # Jobs prepared before recipe binding cannot silently switch to a new plan.
        job.update(status="unsupported", failure={
            "status": "unsupported", "code": "REPLAY_RUNTIME_UNAVAILABLE",
            "reason": "A verified fixed-trajectory re-execution adapter is required for this collection mode",
            "reference_run": job["trajectory_request"]["reference_run"],
            "missing_robot_capabilities": ["verified_fixed_trajectory_reexecution"],
        })
        job["trajectory_outcome"] = {"duplicate_policy": "replay", "published_episodes": 0,
                                     "failure": job["failure"]}
        write(job_path, job)
        return {"status": job["status"], "failure": job["failure"],
                "trajectory_outcome": job["trajectory_outcome"]}
    if replay and (not replay_requested or (options.trial_seed is not None and options.trial_seed != replay["seed"])):
        raise ValueError("Replay requires explicit permission and its original immutable seed")
    code_id = pinned_code()
    if job.get("execution_code_sha256", code_id) != code_id:
        raise ValueError("Execution code differs from this frozen job")
    job["execution_code_sha256"] = code_id
    job.setdefault("execution_started_unix", time.time())
    attempts_per_scene=options.max_attempts_per_scene or int(os.environ.get("GENIE_ATTEMPTS_PER_SCENE","15"))
    if not 1<=attempts_per_scene<=15:
        raise ValueError("Scene attempt budget must be between 1 and 15")
    job.setdefault("budgets", {"wall_seconds": 18*3600, "attempts_per_scene": attempts_per_scene,
                               "minimum_free_bytes": 25*1024**3})
    limits = job["budgets"]
    deadline = job["execution_started_unix"] + limits["wall_seconds"]
    env = dict(os.environ, DOCKER_HOST=SOCKET)
    counts = success_counts(job)
    if options.audit_only:
        return {"status": job["status"], "success_counts": counts}
    job["success_counts"] = counts
    job["preflight_only"] = False
    write(job_path, job)
    for task_plan in job["plan"]["tasks"]:
        task = task_plan["task_id"]
        binding = job["task_bindings"][task]
        if options.tasks and binding["template"] not in options.tasks:
            continue
        for quota in task_plan["scene_quotas"]:
            scene_id, goal = quota["scene_id"], quota["successes"]
            if options.scenes and scene_id not in options.scenes:
                continue
            scene = binding["scene_sources"][scene_id]
            if verify_manifest(scene["directory"], ROOT / "GenieSimAssets")["content_sha256"] != scene["sha256"]:
                raise ValueError("Prepared scene provenance has changed")
            retired = quota.get("retired_scene_ids", [])
            trials = [row for row in job["attempts"] if row["task"]==task and row["scene"] in [scene_id,*retired]]
            unfinished = next((row for row in trials if not row.get("finished_unix")), None)
            while counts[task][scene_id] < goal:
                if unfinished is None:
                    if len(trials) >= limits["attempts_per_scene"] or time.time() >= deadline:
                        break
                    if not wait_for_storage(job, job_path, deadline):
                        break
                    if pinned_code() != code_id:
                        raise ValueError("Code changed while the batch was running")
                    serial = len(job["attempts"])
                    label = "scene-" + content_id([job["job_id"],task,scene_id,serial])[:24]
                    seed = replay["seed"] if replay else options.trial_seed
                    if seed is None:
                        seed = int(content_id([scene["seed"],job["job_id"],serial])[:8],16) % 2**31
                    unfinished = {"task":task,"scene":scene_id,"label":label,"seed":seed,
                                  "run_directory":str(STORAGE/"collection-runs"/label),
                                  "started_unix":time.time(),"slot":replay["slot"] if replay else options.slot}
                    trials.append(unfinished)
                    job["attempts"].append(unfinished)
                    job["status"] = "trial"
                    write(job_path, job)
                run_trial(job, job_path, task, scene, unfinished, unfinished["slot"], deadline, env)
                unfinished["finished_unix"] = time.time()
                if unfinished.get("accepted"):
                    counts[task][scene_id] += 1
                launch_failed = unfinished.get("failure") in {
                    "launch_failed", "replay_recipe_nonconforming", "trajectory_admission_unavailable"}
                initial_support_invalid = unfinished.get("scene_failure_signature") == "initial_support_or_stability"
                unfinished = None
                replacement=None
                if (not counts[task][scene_id] and not options.scenes and options.trial_seed is None
                        and len(trials)<limits["attempts_per_scene"] and time.time()<deadline):
                    replacement=choose_replacement(binding.get("reserves",[]),trials,scene_id)
                if replacement:
                    next_id,reason=replacement
                    alternate=binding["scene_sources"][next_id]
                    if verify_manifest(alternate["directory"], ROOT/"GenieSimAssets")["content_sha256"]!=alternate["sha256"]:
                        raise ValueError("Alternate scene provenance has changed")
                    binding["reserves"].remove(next_id)
                    binding["scenes"][binding["scenes"].index(scene_id)]=next_id
                    quota["retired_scene_ids"]=[*quota.get("retired_scene_ids",[]),scene_id]
                    quota["scene_id"]=next_id
                    job.setdefault("scene_replacements",[]).append(
                        {"task":task,"retired_scene_id":scene_id,"new_scene_id":next_id,"reason":reason})
                    counts[task][next_id]=counts[task].pop(scene_id)
                    scene_id,scene=next_id,alternate
                write(job_path, job)
                if launch_failed or options.trial_seed is not None or (initial_support_invalid and not replacement):
                    break
    total = sum(sum(row.values()) for row in counts.values())
    expected = sum(item["requested"] for item in job["scene_preflight"].values())
    compiled = sum(item["compiled"] for item in job["scene_preflight"].values())
    full = compiled == expected and all(counts[task["task_id"]][row["scene_id"]] >= row["successes"]
        for task in job["plan"]["tasks"] for row in task["scene_quotas"])
    job["scene_outcome"] = {"requested":expected,"compiled":compiled,"admitted_episodes":total,
                            "replacements":job.get("scene_replacements",[]),
                            "failure_counts":dict(Counter(row.get("failure") for row in job["attempts"] if row.get("failure")))}
    if "trajectory_request" in job:
        plan_total = sum(row["successes"] for task in job["plan"]["tasks"] for row in task["scene_quotas"])
        requested_total = job["trajectory_request"].get("total_episodes", plan_total)
        job["scene_outcome"].update(requested_episodes=requested_total,
                                    missing_episodes=max(requested_total - total, 0))
        job["trajectory_outcome"] = {
            "duplicate_policy": job["trajectory_request"]["duplicate_policy"],
            "requested_episodes": requested_total, "planned_episodes": plan_total,
            "distinct_scenes_with_accepted_episodes": sum(n > 0 for rows in counts.values() for n in rows.values()),
            "admitted_episodes": total, "published_episodes": 0, "missing_episodes": requested_total,
        }
        if replay:
            job["trajectory_outcome"].update(replay_method=replay["method"],
                                             reference_run=job["trajectory_request"]["reference_run"])
    job["status"] = "finalizing_dataset" if full else "partial" if total else (
        "scene_search_exhausted" if not compiled else
        "launch_failed" if job["attempts"] and all(a.get("failure")=="launch_failed" for a in job["attempts"])
        else "attempt_budget_exhausted")
    write(job_path, job)
    published_dataset = job.get("partial_dataset") if job.get("raw_trajectory_path") else None
    if full:
        try:
            job["dataset"] = finalize(job, job_path)
            from bundle_raw_trajectories import bundle
            raw = bundle(job["dataset"]["dataset_path"],link=True)
            job["raw_trajectory_path"] = raw["archive"]
            published_dataset = job["dataset"]
        except Exception as exc:
            job.update(status="dataset_finalization_failed",dataset_failure=str(exc))
        else:
            job["status"] = "complete"
            job.pop("dataset_failure", None)
    else:
        report(job_path, print_report=False)
        if total and not (options.tasks or options.scenes or options.trial_seed is not None):
            try:
                dataset=finalize(job,job_path,allow_partial=True)
                from bundle_raw_trajectories import bundle
                raw=bundle(dataset["dataset_path"],link=True)
                job["partial_dataset"]=dataset
                job["raw_trajectory_path"]=raw["archive"]
                published_dataset = dataset
            except Exception as exc:
                job["dataset_failure"]=str(exc)
    if "trajectory_outcome" in job and published_dataset and published_dataset.get("official_reader_verified") is True:
        published = published_dataset["episodes"]
        job["trajectory_outcome"].update(
            published_episodes=published,
            missing_episodes=max(job["trajectory_outcome"]["requested_episodes"] - published, 0),
            distinct_motion_groups=published_dataset.get("trajectory_diversity", {}).get("distinct_motion_groups"),
        )
    write(job_path, job)
    return {"status":job["status"],"success_counts":counts,"dataset":job.get("dataset"),
            "raw_trajectory_path":job.get("raw_trajectory_path"),"dataset_failure":job.get("dataset_failure"),
            "partial_dataset":job.get("partial_dataset"),"trajectory_outcome":job.get("trajectory_outcome")}
