"""Durable intake for one lab worker. Simulation, budgets and dataset validation stay native."""
from contextlib import closing, contextmanager
import json
import logging
from pathlib import Path
import re
import sqlite3
import time
import uuid

from lerobot_layout import FILES_MANIFEST, LEGACY_FORMAT, SUFFIXES
from robot_adapters import ROOT, UnsupportedRequest


class RequestConflict(ValueError):
    pass


def check_dataset_result(record, expected_format):
    """Check publication receipts before exposing them to a version-bound job."""
    if record is None or expected_format is None:
        return record, False
    for field in ("dataset", "partial_dataset"):
        receipt = record.get(field)
        if receipt is None and (field != "dataset" or record.get("status") != "complete"):
            continue
        actual_format = None
        code = "DATASET_RECEIPT_INVALID"
        reason = f"Expected a verified LeRobot {expected_format} receipt in {field}"
        if isinstance(receipt, dict):
            actual_format = receipt.get("dataset_format")
            if actual_format is None and receipt.get("format") == f"LeRobot {LEGACY_FORMAT}":
                actual_format = LEGACY_FORMAT
            path = receipt.get("dataset_path")
            if actual_format != expected_format:
                code = "DATASET_FORMAT_MISMATCH"
                reason = f"Unexpected LeRobot dataset version in {field}"
            elif (receipt.get("official_reader_verified") is True
                  and receipt.get("files_manifest") == FILES_MANIFEST
                  and isinstance(path, str) and Path(path).name.endswith(SUFFIXES[expected_format])):
                continue
        return {**record, "status": "dataset_finalization_failed", "code": code,
                "dataset_failure": reason, "expected_dataset_format": expected_format,
                "actual_dataset_format": actual_format if isinstance(actual_format, str) else None,
                "dataset": None, "partial_dataset": None, "raw_trajectory_path": None}, True
    return record, False


def read_native_summary(job_id, root=ROOT):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", job_id):
        return None
    path = Path(root) / "jobs" / job_id / "job.json"
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        # A native worker can be in the middle of publishing a job update.
        return None
    if job.get("job_id") != job_id:
        return None
    return {"job_id": job_id, "status": job["status"], "success_counts": job.get("success_counts", {}),
            "attempt_count": len(job.get("attempts", [])), "dataset": job.get("dataset"),
            "worker_heartbeat_unix":job.get("worker_heartbeat_unix"),
            "budget_reason":job.get("budget_reason"), "storage_wait":job.get("storage_wait"),
            "automatic_admission":job.get("automatic_admission",False),
            "scene_outcome":job.get("scene_outcome"),"raw_trajectory_path":job.get("raw_trajectory_path"),
            "partial_dataset":job.get("partial_dataset"),"trajectory_outcome":job.get("trajectory_outcome"),
            "phase":job.get("phase"),"missing_skills":job.get("missing_skills",[]),
            "candidate_apis":job.get("candidate_apis",[]),"skill_details":job.get("skill_details",{}),
            "missing_robot_capabilities":job.get("missing_robot_capabilities",[]),
            "missing_verifiers":job.get("missing_verifiers",[]),"failure":job.get("failure"),
            "code":job.get("code"),"validation":job.get("validation"),
            "collection_wait":job.get("collection_wait"),"reason":job.get("reason"),
            "runtime_capabilities":job.get("runtime_capabilities")}


class Jobs:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, request_key TEXT UNIQUE, operation TEXT NOT NULL, payload TEXT NOT NULL,
                state TEXT NOT NULL, native_id TEXT, result TEXT, created REAL NOT NULL, updated REAL NOT NULL
            )""")

    @contextmanager
    def db(self):
        with closing(sqlite3.connect(self.directory / "jobs.sqlite3", timeout=10)) as db:
            db.row_factory = sqlite3.Row
            with db:
                yield db

    def enqueue(self, operation, payload, key=None):
        if key is not None and not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", key):
            raise ValueError("Invalid Idempotency-Key")
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if key is not None:
                prior = db.execute("SELECT * FROM jobs WHERE request_key = ?", (key,)).fetchone()
                if prior:
                    if (prior["operation"], prior["payload"]) != (operation, body):
                        raise RequestConflict("Idempotency-Key already belongs to a different request")
                    return prior["id"]
            job_id = uuid.uuid4().hex
            stamp = time.time()
            db.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, 'queued', NULL, NULL, ?, ?)",
                       (job_id, key, operation, body, stamp, stamp))
            return job_id

    def get(self, job_id):
        with self.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        result = {"id": job_id, "operation": row["operation"], "state": row["state"],
                  "created_unix": row["created"], "updated_unix": row["updated"],
                  "native_job_id": row["native_id"], "result": json.loads(row["result"]) if row["result"] else None}
        dataset_format = json.loads(row["payload"]).get("dataset_format")
        if dataset_format is not None:
            result["dataset_format"] = dataset_format
        if row["operation"] == "isaac_physx" and row["native_id"]:
            native = read_native_summary(row["native_id"])
            result["native"], _ = check_dataset_result(native, dataset_format)
        return result

    @contextmanager
    def worker(self):
        # A separate SQLite file provides a process-scoped OS lock on Linux and Windows.
        # A lost process releases it without locking out readers or enqueuers of jobs.sqlite3.
        with closing(sqlite3.connect(self.directory / "worker.sqlite3", timeout=0)) as guard:
            try:
                guard.execute("BEGIN EXCLUSIVE")
            except sqlite3.OperationalError as exc:
                if exc.sqlite_errorcode in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                    raise RuntimeError("A worker already holds this queue") from exc
                raise
            with self.db() as db:
                running=db.execute("SELECT id,operation,payload,native_id FROM jobs WHERE state='running'").fetchall()
                db.execute("UPDATE jobs SET state='interrupted', updated=? WHERE state='running'", (time.time(),))
                for row in running:
                    if row["operation"]!="isaac_physx":
                        continue
                    native=read_native_summary(row["native_id"]) if row["native_id"] else None
                    # Native execution only starts after the durable id callback.
                    if not row["native_id"] or (native and native.get("automatic_admission")
                                                and native.get("status")!="stopped_by_user"):
                        db.execute("UPDATE jobs SET state='queued', updated=? WHERE id=?",(time.time(),row["id"]))
            yield

    def run_next(self, registry):
        """Call inside worker(). Orphaned jobs retain their native ID and are never blindly replayed."""
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id,operation,payload,native_id FROM jobs WHERE state='queued' ORDER BY rowid LIMIT 1").fetchone()
            if row is None:
                return False
            job_id = row["id"]
            db.execute("UPDATE jobs SET state='running', updated=? WHERE id=?", (time.time(), job_id))

        def native_job(native_id):
            if not isinstance(native_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", native_id):
                raise ValueError("Invalid native job identifier")
            with self.db() as db:
                db.execute("UPDATE jobs SET native_id=?, updated=? WHERE id=?", (native_id, time.time(), job_id))

        try:
            payload=json.loads(row["payload"])
            if row["operation"]=="isaac_physx" and row["native_id"]:
                from natural_backend import resume
                result=resume(row["native_id"],execute=payload["execute"])
            else:
                result = registry.invoke(row["operation"], payload, native_job)
            result, invalid_dataset = check_dataset_result(result, payload.get("dataset_format"))
            encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
            state = "failed" if invalid_dataset else "finished"
        except UnsupportedRequest as exc:
            state, encoded = "failed", json.dumps(exc.details, ensure_ascii=False)
        except Exception as exc:
            logging.exception("Platform job %s failed in %s", job_id, row["operation"])
            state, encoded = "failed", json.dumps({"status": "failed", "error_type": type(exc).__name__})
        with self.db() as db:
            db.execute("UPDATE jobs SET state=?, result=?, updated=? WHERE id=?", (state, encoded, time.time(), job_id))
        return True
