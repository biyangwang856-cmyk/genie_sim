"""Coordinate a durable, isolated RobotAPI skill validation job."""
from contextlib import closing
from datetime import datetime
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from tempfile import NamedTemporaryFile
import time
import uuid

from robot_adapters import ROOT, load_robot
from scene_contract import digest, read_json, require_id
from skill_capabilities import (HINGE_SKILL, PROBE_CAPABILITIES, ROTATION_SKILL, HingeCapabilityGap,
                                RuntimeScope, SkillRegistry, _MAX_BASE_DRIFT_DEG, _MAX_BASE_DRIFT_M,
                                _distance, _rigid_pose, _rotation_deg, discover_skills, probe_binding_type)


JOB_TYPE = "skill_onboarding"
RUN_BUDGET_S = 90 * 60
LAUNCH_TIMEOUT_S = 240
POLL_S = 10
SKILL_SOURCE_FILES = ("hinge_fixture.py", "skill_capabilities.py")
RUN_EVIDENCE = {
    "observations": "hinge-observations.json",
    "physics": "object-physics-120hz.jsonl",
    "controls": "native-control-commands.jsonl",
}
ROTATION_EVIDENCE = {**RUN_EVIDENCE, "observations": "rotation-observations.json"}


def _job_path(job_id):
    require_id(job_id)
    jobs = (ROOT / "jobs").resolve()
    path = jobs / job_id / "job.json"
    if not job_id.startswith(("hinge-skill-", "rotation-skill-")) or path.resolve().parent.parent != jobs:
        raise ValueError("Invalid skill validation job")
    return path


def _write(path, job, **updates):
    job.update(updates, worker_heartbeat_unix=time.time())
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(job, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _summary(job, path):
    fields = ("job_id", "job_type", "status", "phase", "robot_id", "goal", "candidate_apis", "code",
              "reason", "message", "missing_skills", "missing_assets", "missing_robot_capabilities",
              "missing_verifiers", "skill_details", "run_directory", "validation", "collection_wait",
              "runtime_capabilities", "intent_receipts")
    return {**{name: job[name] for name in fields if name in job}, "job_path": str(path)}


def _pinned_code(skill=HINGE_SKILL, completion=None):
    files = SKILL_SOURCE_FILES if skill == HINGE_SKILL else ("rotation_probe.py", "skill_capabilities.py")
    if completion == "pack_phone":
        files += ("rotation_reuse.py", "skill_executor.py")
    return {name: digest(ROOT / name) for name in files}


def prepare(goal, robot_id="g2crsb90d", *, execute=False, on_prepared=None, slot=None, instruction=None,
            scope=None, intent_receipts=()):
    """Prepare a trusted RobotAPI adaptation and record a fresh physical run when requested."""
    if not isinstance(goal, dict) or not set(goal) <= {"skill", "actor", "target_limit", "angle_deg", "completion"}:
        raise ValueError("Unknown field in standalone skill goal")
    skill = require_id(goal.get("skill"))
    adapter = load_robot(robot_id)
    report = discover_skills([skill], adapter, scope=scope)
    if report["status"] in {"unsupported", "capability_gap"}:
        return report
    if skill not in PROBE_CAPABILITIES:
        return {**report, "status": "capability_gap", "code": "SKILL_CAPABILITY_GAP",
                "message": "This registered skill needs an executable task or goal binding",
                "reason": "registered_skill_requires_a_task_binding"}
    fixture_options = {}
    if skill == HINGE_SKILL:
        from hinge_fixture import ACTOR, INSTRUCTION, TASK, prepare_fixture

        actor = require_id(goal.get("actor", ACTOR))
        limit = goal.get("target_limit", "min")
        if actor != ACTOR or robot_id != "g2crsb90d" or limit != "min" or {"angle_deg", "completion"} & goal.keys():
            return {**report, "status": "capability_gap", "code": "SKILL_CAPABILITY_GAP",
                    "message": "The current hinge adapter requires a bound asset and joint target",
                    "reason": "no_trusted_hinge_binding_for_request", "requested_actor": actor,
                    "requested_target_limit": limit, "supported_binding": {"actor": ACTOR, "target_limit": "min"}}
        scene_id, job_prefix, fixture_args = "hinge-0", "hinge-skill-", ()
        bound_goal = {"skill": skill, "actor": actor, "target_limit": limit}
    else:
        from rotation_fixture import INSTRUCTION, prepare_fixture
        from rotation_probe import ACTOR, TASK

        actor = require_id(goal.get("actor", ACTOR))
        angle = goal.get("angle_deg")
        completion = goal.get("completion")
        if (actor != ACTOR or robot_id != "g2crsb90d" or angle != 15. or "target_limit" in goal
                or completion not in (None, "pack_phone")):
            return {**report, "status": "capability_gap", "code": "SKILL_CAPABILITY_GAP",
                    "message": "The rotation adapter requires the annotated phone and its supported angle",
                    "reason": "no_trusted_rotation_binding_for_request", "requested_actor": actor,
                    "requested_angle_deg": angle, "supported_binding": {"actor": ACTOR, "angle_deg": 15.}}
        scene_id, job_prefix, fixture_args = "rotation-0", "rotation-skill-", (adapter,)
        bound_goal = {"skill": skill, "actor": actor, "angle_deg": 15.}
        if completion:
            bound_goal["completion"] = completion
            fixture_options["completion"] = completion

    job_id = job_prefix + uuid.uuid4().hex
    path = _job_path(job_id)
    try:
        fixture = prepare_fixture(path.parent / scene_id, *fixture_args, **fixture_options)
    except HingeCapabilityGap as exc:
        return {**report, "status": "capability_gap", "code": "SKILL_CAPABILITY_GAP", "reason": str(exc)}
    if fixture["task"] != TASK or fixture["scene_id"] != scene_id:
        raise ValueError("Unexpected trusted validation fixture")
    binding = probe_binding_type(skill)(**fixture["binding"])
    report = discover_skills([skill], adapter, scope=scope, registry=SkillRegistry(ROOT / "skill-registry"),
                             bindings={skill: binding})
    job = {**report, "job_id": job_id, "job_type": JOB_TYPE, "automatic_admission": True,
           "phase": "prepared", "instruction": instruction or INSTRUCTION, "robot_id": robot_id,
           "goal": bound_goal, "intent_receipts": list(intent_receipts),
           "fixture": fixture, "slot": require_id(slot or os.environ.get("GENIE_COLLECTION_SLOT", "primary")),
           "created_unix": time.time(), "attempts": [],
           "skill_source_sha256": _pinned_code(skill, bound_goal.get("completion"))}
    _write(path, job)
    if on_prepared is not None:
        on_prepared(job_id)
    return resume(job_id, execute=True) if execute else _summary(job, path)


def _inspect(job_id, env):
    name = "genie-by-" + job_id
    fields = ('{{.Id}}\t{{.State.Status}}\t{{.State.ExitCode}}\t{{.State.FinishedAt}}'
              '\t{{index .Config.Labels "geniesim.owner-job"}}')
    result = subprocess.run(["docker", "inspect", "--format", fields, name], env=env, capture_output=True,
                            text=True, timeout=15)
    if result.returncode:
        if "no such object" in result.stderr.lower() or "no such container" in result.stderr.lower():
            return None
        raise RuntimeError("Project Docker could not confirm the skill collection state")
    container, status, exit_code, finished_at, owner = result.stdout.rstrip("\r\n").split("\t")
    if owner != job_id:
        raise ValueError("Existing skill container belongs to a different job")
    finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00")).timestamp() if status == "exited" else None
    return container, status, int(exit_code), finished


def _failed(path, job, reason, **details):
    _write(path, job, status="skill_validation_failed", code="SKILL_VALIDATION_FAILED", phase="finished",
           message="The candidate did not pass native skill validation", reason=reason,
           validation=details, finished_unix=time.time(), collection_wait=None)
    return _summary(job, path)


def _evidence_rows(run, name):
    try:
        with (run / "evidence" / name).open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    # The running simulator may still be writing the final line.
                    continue
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def _live_capabilities(events):
    for event in reversed(events):
        if event.get("phase") in ("hinge_capability_discovery", "rotation_capability_discovery"):
            fields = ("status", "code", "message", "missing_skills", "candidate_apis",
                      "missing_robot_capabilities", "missing_verifiers", "skill_details")
            return {key: event[key] for key in fields if key in event}
    return {}


def _measured_segment_failure(run, grasp, failed, follow):
    """Corroborate a failed hinge segment with the native drive and object records."""
    try:
        start = float(follow["sim_time"])
        stop = float(failed["sim_time_s"]) if "sim_time_s" in failed else None
        if stop is not None and (not isfinite(stop) or stop <= start):
            return None
        trace = [row for row in _evidence_rows(run, RUN_EVIDENCE["physics"])
                 if start - .03 <= float(row["sim_time"]) <= (stop + .03 if stop is not None else float("inf"))]
        clocks = [float(row["sim_time"]) for row in trace]
        if (len(clocks) < 12 or abs(clocks[0] - start) > .03
                or (stop is not None and abs(clocks[-1] - stop) > .03)
                or any(not 0 < after - before <= .031 for before, after in zip(clocks, clocks[1:]))):
            return None
        base = _rigid_pose(grasp["after"]["base_pose"])
        bases = [_rigid_pose(row["poses"]["hinge_base"]) for row in trace]
        links = [_rigid_pose(row["poses"]["hinge_link"]) for row in trace]
        drift_m = max(_distance(base, pose) for pose in bases)
        drift_deg = max(_rotation_deg(base, pose) for pose in bases)
        peak_swing_deg = max(_rotation_deg(links[0], pose) for pose in links)
        command_end = min(stop, clocks[-1]) if stop is not None else clocks[-1]
        commands = [row["applied_joint_positions"] for row in _evidence_rows(run, RUN_EVIDENCE["controls"])
                    if start - 1e-5 <= float(row["sim_time"]) <= command_end + 1e-5
                    and isinstance(row.get("applied_joint_positions"), list)
                    and row["applied_joint_positions"]]
        has_motion_commands = (
            len(commands) >= 2 and all(len(row) == len(commands[0]) and all(isfinite(v) for v in row)
                                       for row in commands)
            and any(abs(initial - later) > 1e-5 for row in commands[1:]
                    for initial, later in zip(commands[0], row, strict=True))
        )
        diagnostic = {"native_commands_in_segment": len(commands), "physics_samples_in_segment": len(trace),
                      "base_peak_drift_m": drift_m, "base_peak_drift_deg": drift_deg,
                      "hinge_peak_swing_deg": peak_swing_deg}
        if drift_m > _MAX_BASE_DRIFT_M or drift_deg > _MAX_BASE_DRIFT_DEG:
            return "hinge_base_drift", diagnostic
        before, after, commanded = (float(failed[key]) for key in ("before_deg", "after_deg", "commanded_swing_deg"))
        if (follow.get("passed") is True and failed.get("planner_success") is True and has_motion_commands
                and all(isfinite(v) for v in (before, after, commanded))
                and max(abs(after - before), peak_swing_deg) < min(.8, commanded * .5)):
            return "hinge_contact_progress_unverified", {
                **diagnostic, "before_deg": before, "after_deg": after, "commanded_swing_deg": commanded,
            }
    except (TypeError, ValueError, KeyError):
        # Incomplete logs support a generic failure, with no physical cause assigned.
        return None
    return None


def _diagnose_native_failure(run, events):
    """Keep the first evidenced blocking phase in the public job result."""
    failed = next((event for event in reversed(events)
                   if event.get("phase") in {"hinge_motion_failure", "hinge_execution_failure", "hinge_probe_failure"}), {})
    context = {key: failed[key] for key in ("reason", "phase", "sim_time_s", "measured_angle_deg",
                                            "target_angle_deg", "native_commands", "classification")
               if key in failed}
    grasp = next((event for event in reversed(events) if event.get("phase") == "hinge_grasp_probe"), None)
    if grasp is None:
        geometry_reasons = {
            "no_clear_handle_grasp_geometry": "hinge_handle_grasp_geometry_blocked",
            "handle_does_not_fit_native_open_jaws": "hinge_handle_grasp_geometry_blocked",
            "native_gripper_collision_observation_unavailable": "hinge_gripper_observation_unavailable",
            "native_gripper_pad_geometry_unavailable": "hinge_gripper_observation_unavailable",
        }
        if failed.get("reason") in geometry_reasons:
            return geometry_reasons[failed["reason"]], {
                **context, **{key: failed[key] for key in ("measured_opening_m", "collider_links") if key in failed}
            }
        if failed and failed["phase"] == "hinge_probe_failure":
            return "hinge_handle_approach_failed", context
        return "native_collection_failed", context
    if not grasp.get("success"):
        diagnostic = {
            "candidate_count": grasp.get("candidate_count"), "native_motion_ok": grasp.get("native_motion_ok"),
            "contact_error_m": grasp.get("contact_error_m"),
        }
        confirmation = grasp.get("confirmation") or {}
        evidence = ("attempt", "reason", "position_error_m", "rotation_error_deg", "planner_success",
                    "progress_deg", "before_deg", "after_deg", "handle_to_hand_drift_m")
        diagnostic.update({key: confirmation[key] for key in evidence if key in confirmation})
        if confirmation.get("reason") == "grasp_contact_not_confirmed":
            if confirmation.get("planner_success") is False:
                return "hinge_grasp_probe_path_rejected", diagnostic
            if confirmation.get("planner_success") is True and "progress_deg" in confirmation:
                return "hinge_grasp_contact_unverified", diagnostic
        if confirmation.get("reason") == "fixture_shifted_during_closure":
            return "hinge_fixture_shifted_during_closure", diagnostic
        return "hinge_handle_approach_failed", diagnostic
    continuity = _evidence_rows(run, "local-joint-plan-continuity.jsonl")
    when = grasp.get("after", {}).get("sim_time_s")
    if not isinstance(when, (float, int)) or not isfinite(when):
        return "native_collection_failed", {}
    follows = [row for row in continuity if isinstance(row.get("sim_time"), (float, int))
               and isfinite(row["sim_time"]) and row["sim_time"] >= when - 1e-5]
    if follows and follows[-1].get("passed") is False:
        try:
            with (run / "evidence" / RUN_EVIDENCE["controls"]).open(encoding="utf-8") as stream:
                command_times = [float(json.loads(line)["sim_time"]) for line in stream if line.strip()]
            command_rows = len(command_times)
            after_grasp = sum(stamp > when + 1e-5 for stamp in command_times)
        except (OSError, TypeError, ValueError, KeyError):
            command_rows = after_grasp = 0
        if command_rows and after_grasp == 0:
            return "hinge_path_rejected_before_actuation", {
                "native_commands_after_grasp": 0,
                "continuity_limit_rad": follows[-1].get("limit_rad"),
                "rejected_ik_candidates": follows[-1].get("ik_candidates_rejected_by_continuity"),
                "measured_angle_deg": grasp["after"].get("joint", {}).get("current_angle_deg"),
            }
    if failed.get("reason") in {"hinge_hold_or_base_shift", "hinge_segment_did_not_move_as_planned"}:
        diagnosis = _measured_segment_failure(run, grasp, failed, follows[-1]) if follows else None
        if diagnosis is not None:
            reason, measured = diagnosis
            return reason, {**context, **measured}
        context["physical_diagnosis"] = "insufficient_evidence"
    return ("hinge_motion_blocked" if failed else "native_collection_failed"), context


def _launch(path, job, run, deadline, env, busy_exit):
    fixture = job["fixture"]
    scene_id = require_id(fixture["scene_id"])
    scene = (path.parent / scene_id).resolve()
    if Path(fixture["scene_directory"]).resolve() != scene or scene.parent != path.parent.resolve():
        raise ValueError("Skill scene is outside its prepared job")
    command = [sys.executable, str(ROOT / "collection_control.py"), require_id(fixture["task"]), "--scene", scene_id,
               "--scene-directory", str(scene), "--scene-sha256", fixture["scene_sha256"],
               "--label", job["job_id"], "--slot", job["slot"], "--robot-id", job["robot_id"],
               "--owner-job", job["job_id"], "--execution-mode", "structured", "--instruction", job["instruction"]]
    while (remaining := deadline - time.time()) > 0:
        if shutil.disk_usage(run.parent).free < 25 * 1024**3:
            _write(path, job, phase="waiting_for_storage",
                   collection_wait={"status": "waiting_for_storage", "minimum_free_bytes": 25 * 1024**3})
            time.sleep(min(POLL_S, remaining))
            continue
        # Persist the launch intent before starting the subprocess. A restarted job
        # observes this label or waits for the old launcher instead of starting twice.
        if _pinned_code(job["goal"]["skill"], job["goal"].get("completion")) != job["skill_source_sha256"]:
            raise ValueError("Skill code differs from the prepared job")
        _write(path, job, phase="launching", launch_requested_unix=time.time(), collection_wait=None)
        log = path.with_name("collection-launch.log")
        try:
            result = subprocess.run(command, env=env, capture_output=True, text=True,
                                    timeout=min(LAUNCH_TIMEOUT_S, remaining))
        except subprocess.TimeoutExpired:
            return None
        with log.open("a", encoding="utf-8") as stream:
            stream.write(result.stdout)
            stream.write(result.stderr)
        if result.returncode == busy_exit:
            try:
                wait = json.loads(result.stdout.splitlines()[-1])
            except (ValueError, IndexError):
                wait = {"status": "waiting_for_collection_slot"}
            _write(path, job, phase="waiting_for_collection_slot", launch_requested_unix=None, collection_wait=wait)
            time.sleep(max(0., min(POLL_S, deadline - time.time())))
            continue
        return result.returncode
    return None


def _verify_published(job, validation, registry):
    skill = job["goal"]["skill"]
    certificate = validation["certificate"]
    scope = RuntimeScope(**certificate["key"]["runtime"])
    binding = probe_binding_type(skill)(**job["fixture"]["binding"])
    receipt = registry.lookup(skill, scope, binding)
    if scope.robot_id != job["robot_id"] or receipt != certificate:
        raise ValueError("Skill certificate could not be verified after publication")
    evidence = Path(job["run_directory"]) / "evidence"
    evidence_files = RUN_EVIDENCE if skill == HINGE_SKILL else ROTATION_EVIDENCE
    if (validation.get("run_verification", {}).get("passed") is not True
            or validation.get("run_evidence_sha256") !=
            {kind: digest(evidence / name) for kind, name in evidence_files.items()}):
        raise ValueError("The current collection did not provide intact physical validation evidence")
    if job["goal"].get("completion") == "pack_phone":
        proof = validation.get("continuation", {})
        files = ("stage-evidence.jsonl", "skill-execution.jsonl")
        if (proof.get("independent_task_verdict") is not True
                or proof.get("reused_certificate_sha256") != certificate["certificate_sha256"]
                or proof.get("evidence_sha256") != {name: digest(evidence / name) for name in files}):
            raise ValueError("The certified continuation evidence is missing or changed")


def _execute(path, job):
    if job["status"] == "stopped_by_user":
        return _summary(job, path)

    from collection_control import COLLECTION_SLOTS, SLOT_BUSY_EXIT, SOCKET, STORAGE
    skill = job["goal"]["skill"]
    if skill == HINGE_SKILL:
        from hinge_fixture import TASK, certify_run
    elif skill == ROTATION_SKILL:
        from rotation_probe import TASK
        from rotation_fixture import certify_run
    else:
        raise ValueError("No trusted native verifier for the prepared skill")

    if job["slot"] not in COLLECTION_SLOTS:
        raise ValueError("Unknown collection slot")
    registry = SkillRegistry(ROOT / "skill-registry")
    if job["phase"] == "finished":
        if job["status"] == "ready":
            _verify_published(job, job["validation"], registry)
        return _summary(job, path)

    runs = (STORAGE / "collection-runs").resolve()
    runs.mkdir(exist_ok=True)
    run = runs / job["job_id"]
    if run.resolve().parent != runs:
        raise ValueError("Skill run path escaped the project collection directory")
    env = dict(os.environ, DOCKER_HOST=SOCKET)
    env.pop("DOCKER_CONTEXT", None)
    started = job.get("execution_started_unix", time.time())
    deadline = started + RUN_BUDGET_S
    _write(path, job, status="validating_skill", execution_started_unix=started, run_directory=str(run))
    state = _inspect(job["job_id"], env)
    if state is None and not job.get("launch_requested_unix") and not run.exists():
        try:
            returncode = _launch(path, job, run, deadline, env, SLOT_BUSY_EXIT)
        except (OSError, ValueError) as exc:
            return _failed(path, job, "native_launch_not_possible", error=str(exc))
        if returncode not in (None, 0):
            return _failed(path, job, "native_launch_failed", exit_code=returncode)
        state = _inspect(job["job_id"], env)
    while state is None and time.time() < min(deadline, (job.get("launch_requested_unix") or 0) + LAUNCH_TIMEOUT_S):
        time.sleep(max(0., min(POLL_S, deadline - time.time())))
        state = _inspect(job["job_id"], env)
    if state is None:
        reason = "validation_deadline" if time.time() >= deadline else "owned_container_missing"
        return _failed(path, job, reason)

    container, status, exit_code, finished = state
    _write(path, job, phase="collecting", collection_wait=None, attempts=[{"container_id": container,
                                                                         "run_directory": str(run)}])
    while status in {"running", "restarting", "created"} and time.time() < deadline:
        capability = _live_capabilities(list(_evidence_rows(run, "stage-evidence.jsonl")))
        _write(path, job, runtime_capabilities=capability)
        time.sleep(max(0., min(POLL_S, deadline - time.time())))
        state = _inspect(job["job_id"], env)
        if state is None or state[0] != container:
            return _failed(path, job, "owned_container_disappeared")
        _, status, exit_code, finished = state
    if status in {"running", "restarting", "created"}:
        subprocess.run(["docker", "stop", "--time", "15", container], env=env, check=True, timeout=40,
                       stdout=subprocess.DEVNULL)
        return _failed(path, job, "validation_deadline")
    if status != "exited" or exit_code != 0:
        events = list(_evidence_rows(run, "stage-evidence.jsonl"))
        capability = _live_capabilities(events)
        job["runtime_capabilities"] = capability
        if capability.get("status") in {"unsupported", "capability_gap"}:
            _write(path, job, **capability, phase="finished", reason="native_skill_capability_unavailable",
                   finished_unix=time.time(), collection_wait=None)
            return _summary(job, path)
        if skill == HINGE_SKILL:
            reason, diagnostic = _diagnose_native_failure(run, events)
        else:
            diagnostic = next((row for row in reversed(events) if row.get("phase") in
                               ("rotation_probe_failure", "rotation_physical_probe")), {})
            reason = diagnostic.get("reason", "native_rotation_verification_failed")
        return _failed(path, job, reason, container_status=status, exit_code=exit_code, diagnostic=diagnostic)
    if finished > deadline:
        return _failed(path, job, "validation_deadline", native_finished_unix=finished)

    try:
        launch = read_json(run / "launch.json")
        source = {"owner_job": job["job_id"], "scene_sha256": job["fixture"]["scene_sha256"],
                  "task": load_robot(job["robot_id"]).config["scene_task_prefix"] + TASK,
                  "robot_id": job["robot_id"], "collection_slot": job["slot"], "container_id": container}
        if any(launch.get(field) != value for field, value in source.items()):
            raise ValueError("Native skill record does not match its prepared job")
        if (_pinned_code(skill, job["goal"].get("completion")) != job["skill_source_sha256"]
                or any(launch["source_sha256"].get(name) != sha for name, sha in job["skill_source_sha256"].items())):
            raise ValueError("Native validation code differs from the prepared skill")
        validation = certify_run(run, registry.directory)
        if validation["status"] != "ready":
            return _failed(path, job, "independent_physics_verification_failed", outcome=validation)
        _verify_published(job, validation, registry)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return _failed(path, job, "validation_evidence_unavailable", error=str(exc))
    ready = discover_skills([skill], load_robot(job["robot_id"]),
                            scope=RuntimeScope(**validation["certificate"]["key"]["runtime"]),
                            registry=registry, bindings={skill: probe_binding_type(skill)(**job["fixture"]["binding"])})
    if ready["status"] != "ready":
        return _failed(path, job, "scoped_certificate_not_available", outcome=ready)
    for field in ("code", "message", "reason"):
        job.pop(field, None)
    _write(path, job, **ready, phase="finished", validation=validation, finished_unix=time.time(),
           runtime_capabilities=ready)
    return _summary(job, path)


def resume(job_id, *, execute=True):
    """Continue or read a prepared job. Only one worker can advance a job."""
    path = _job_path(job_id)
    job = read_json(path)
    if job.get("job_id") != job_id or job.get("job_type") != JOB_TYPE:
        raise ValueError("Not a skill validation job")
    if not execute or job["status"] == "stopped_by_user":
        return _summary(job, path)
    with closing(sqlite3.connect(path.with_name("launch-lock.sqlite3"), timeout=0.0)) as lock:
        try:
            lock.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            if exc.sqlite_errorcode == sqlite3.SQLITE_BUSY:
                return _summary(read_json(path), path)
            raise
        return _execute(path, read_json(path))
