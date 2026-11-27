"""Transactional diversity admission after the native recording audit."""

from contextlib import closing, contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3

from trajectory_diversity import (different_scene_dimensions, motion_descriptor_from_run,
                                  motion_distance, scene_conflict, scene_descriptor_from_run,
                                  trajectory_request)


EVIDENCE = Path("evidence/trajectory-diversity.json")


def _read(path):
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def read_diversity_record(trial):
    path = Path(trial["run_directory"]) / EVIDENCE
    record, digest = _read(path)
    if digest != trial["diversity_sha256"] or record["motion_sha256"] != trial["fingerprint"]:
        raise ValueError("Trajectory diversity evidence changed after admission")
    return record


def verified_reference(run):
    """Read a previously audited live run as the comparison target for re-execution."""
    run = Path(run)
    seal, _ = _read(run / "automatic-admission.json")
    audit, audit_sha = _read(run / "acceptance.json")
    launch, launch_sha = _read(run / "launch.json")
    if (seal.get("passed") is not True or audit.get("accepted") is not True or
            audit_sha != seal.get("acceptance_sha256") or launch_sha != seal.get("launch_sha256") or
            not launch.get("container_id") or launch.get("launch_returncode") != 0):
        raise ValueError("Replay reference lacks sealed native physical evidence")
    from batch_collect import motion_fingerprint

    if motion_fingerprint(run) != seal["motion_sha256"]:
        raise ValueError("Replay reference motion changed after its physical admission")
    return {
        "run": str(run), "launch_sha256": launch_sha, "reference_label": run.name,
        "robot_id": launch["robot_id"], "task": launch["task"], "motion_sha256": seal["motion_sha256"],
        "scene": scene_descriptor_from_run(run), "motion": motion_descriptor_from_run(run),
    }


def _near(distance):
    return distance is not None and max(
        distance["measured_joint_ratio"], distance["wrist_pose_ratio"] or 0.) <= 1.


def _decide(record, other_records, reference):
    policy = record["policy"]
    distances = []
    for other in other_records:
        if other["run"] == record["run"] or other["launch_sha256"] == record["launch_sha256"]:
            return {"failure": "execution_already_admitted", "duplicate_motion_of": other["run"]}
        # A job can intentionally collect several trajectories in one exact
        # scene. Its separate scene slots still have to differ in content.
        if (other["job_id"] == record["job_id"] and other["task"] == record["task"]
                and other["scene_id"] != record["scene_id"] and other["scene"]):
            conflict = scene_conflict(record["scene"], [other["scene"]], dimensions=record["distinct_dimensions"])
            if conflict:
                return {"failure": conflict["reason"], "scene_conflict": conflict}
        equal = record["motion_sha256"] == other["motion_sha256"]
        distance = motion_distance(record["motion"], other["motion"]) if other["motion"] and not equal else None
        if distance is not None:
            distances.append({"run": other["run"], **distance})
        if equal or _near(distance):
            if policy == "distinct":
                return {"failure": "exact_motion_repeat" if equal else "near_motion_repeat",
                        "duplicate_motion_of": other["run"], "motion_distance": distance}
            record.setdefault("duplicate_group", other.get("duplicate_group") or other["run"])
    if policy == "replay":
        if not reference:
            return {"failure": "replay_reference_unavailable"}
        if (reference["reference_label"] != record["reference_run"] or reference["run"] == record["run"]
                or reference["launch_sha256"] == record["launch_sha256"]
                or reference["robot_id"] != record["robot_id"] or reference["task"] != record["task"]):
            return {"failure": "invalid_replay_reference_or_execution"}
        if different_scene_dimensions(reference["scene"], record["scene"]):
            return {"failure": "replay_scene_differs_from_reference"}
        distance = motion_distance(reference["motion"], record["motion"])
        exact = record["motion_sha256"] == reference["motion_sha256"]
        if not exact and (not _near(distance) or distance["command_position_ratio"] > 1.):
            return {"failure": "replay_motion_differs_from_reference", "motion_distance": distance}
        record.update(duplicate_group=reference["run"], replay_comparison={"exact": exact, "distance": distance})
    record["comparisons"] = distances
    return {"accepted": True, "record": record}


@contextmanager
def _ledger(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=30)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""CREATE TABLE IF NOT EXISTS trajectory_admissions (
            lineage TEXT NOT NULL, launch_sha256 TEXT NOT NULL, job_id TEXT NOT NULL,
            task TEXT NOT NULL, scene_id TEXT NOT NULL, record TEXT NOT NULL,
            loader_receipt_sha256 TEXT, PRIMARY KEY (lineage, launch_sha256)
        )""")
        yield connection


def _quota(job, task, scene_id):
    for plan in job["plan"]["tasks"]:
        if plan["task_id"] == task:
            for quota in plan["scene_quotas"]:
                if quota["scene_id"] == scene_id:
                    return quota["successes"]
    raise ValueError("Trajectory trial has no requested task/scene quota")


def admit_episode(job, trial, *, scene, motion, fingerprint, reference=None, lineage_db=None):
    """Check again under the native job lock, and reserve across sibling jobs if requested."""
    if trial.get("automatic_acceptance") is not True or trial.get("raw_archive") is not True:
        raise ValueError("Trajectory admission requires an independent native audit and raw archive")
    request = trajectory_request(job["trajectory_request"])
    record = {
        "version": motion["version"], "job_id": job["job_id"], "robot_id": job["robot_id"],
        "task": trial["task"], "scene_id": trial["scene"], "run": trial["run_directory"],
        "launch_sha256": trial["launch_sha256"], "motion_sha256": fingerprint, "scene": scene, "motion": motion,
        "policy": request["duplicate_policy"], "distinct_dimensions": request["distinct_dimensions"],
        "reference_run": request.get("reference_run"),
    }
    quota = _quota(job, record["task"], record["scene_id"])
    prior = []
    for previous in job["attempts"]:
        if previous.get("accepted") is not True:
            continue
        if previous.get("diversity_sha256"):
            prior.append(read_diversity_record(previous))
        else:
            # Already accepted in the old format. Preserve the historical
            # receipt and keep exact motion de-duplication for this row.
            prior.append({"job_id": job["job_id"], "task": previous["task"], "scene_id": previous["scene"],
                          "launch_sha256": previous["launch_sha256"], "motion_sha256": previous["fingerprint"],
                          "run": previous["run_directory"], "scene": None, "motion": None})

    def decide(other_records):
        used = sum(row["job_id"] == record["job_id"] and row["task"] == record["task"]
                   and row["scene_id"] == record["scene_id"] for row in other_records)
        if used >= quota:
            return {"failure": "trajectory_quota_already_filled"}
        return _decide(record, other_records, reference)

    lineage = request.get("lineage_id")
    if not lineage:
        return decide(prior)
    if lineage_db is None:
        raise ValueError("The explicit dataset lineage requires a shared admissions database")
    with _ledger(lineage_db) as db:
        stored = {
            sha: json.loads(value) for sha, value in db.execute(
                "SELECT launch_sha256, record FROM trajectory_admissions WHERE lineage = ? ORDER BY rowid", (lineage,)
            )
        }
        if by_launch := stored.get(record["launch_sha256"]):
            # A process can die after reserving the quota but before writing
            # job.json. Its same signed run may finish that admission once.
            if all(by_launch[key] == value for key, value in record.items()):
                return {"accepted": True, "record": by_launch}
            return {"failure": "execution_already_admitted", "duplicate_motion_of": by_launch["run"]}
        stored.update({row["launch_sha256"]: row for row in prior if row["launch_sha256"] not in stored})
        result = decide(stored.values())
        if result.get("accepted"):
            db.execute("INSERT INTO trajectory_admissions VALUES (?, ?, ?, ?, ?, ?, NULL)",
                       (lineage, record["launch_sha256"], record["job_id"], record["task"], record["scene_id"],
                        json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))))
        return result


def publish_lineage(job, receipt, lineage_db):
    """Associate admitted sources with the independently verified dataset receipt."""
    lineage = job.get("trajectory_request", {}).get("lineage_id")
    if not lineage:
        return
    if receipt.get("official_reader_verified") is not True or not receipt.get("loader_receipt_sha256"):
        raise ValueError("Dataset has no official reader proof")
    with _ledger(lineage_db) as db:
        for source in receipt["sources"]:
            changed = db.execute("""UPDATE trajectory_admissions SET loader_receipt_sha256 = ?
                WHERE lineage = ? AND launch_sha256 = ? AND job_id = ?""",
                (receipt["loader_receipt_sha256"], lineage, source["launch_sha256"], job["job_id"]))
            if changed.rowcount != 1:
                raise ValueError("Dataset source has no matching lineage admission")


def summarize_selection(job, runs):
    """Recheck selected sealed evidence before publishing a training dataset."""
    request = trajectory_request(job["trajectory_request"])
    selected = {Path(row["run_directory"]).resolve(): row for row in job["attempts"] if row.get("accepted") is True}
    admitted, groups = [], set()
    for run in runs:
        trial = selected[run]
        seal, seal_sha = _read(run / "automatic-admission.json")
        if (seal.get("passed") is not True or seal_sha != trial["certificate_sha256"] or
                seal["diversity_sha256"] != trial["diversity_sha256"]):
            raise ValueError("Unsealed trajectory diversity receipt")
        record = read_diversity_record(trial)
        if (record["policy"] != request["duplicate_policy"] or record["job_id"] != job["job_id"]
                or record["launch_sha256"] != seal["launch_sha256"]
                or record["motion_sha256"] != seal["motion_sha256"]):
            raise ValueError("Diversity receipt differs from the frozen job or physical admission")
        # Already sealed replay comparisons are considered here once a
        # verified re-execution backend is connected.
        if record["policy"] != "replay":
            result = _decide(record.copy(), admitted, None)
            if result.get("accepted") is not True:
                raise ValueError("Selected diversity evidence conflicts: " + result["failure"])
        elif not record.get("replay_comparison"):
            raise ValueError("Replay trajectory lacks reference comparison")
        admitted.append(record)
        groups.add(record.get("duplicate_group") or record["run"])
    summary = {"policy": request["duplicate_policy"], "lineage_id": request.get("lineage_id"),
            "profile": admitted[0]["version"], "physical_executions": len(admitted),
            "distinct_motion_groups": len(groups), "distinct_dimensions": request["distinct_dimensions"]}
    if recipe := job.get("replay_recipe"):
        summary.update(replay_method=recipe["method"], reference_run=request["reference_run"])
    return summary
