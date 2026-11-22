"""Durable intake for one lab worker. Simulation, budgets and dataset validation stay native."""
from contextlib import closing, contextmanager
import json
import logging
from pathlib import Path
import re
import sqlite3
import time
import uuid

from robot_adapters import ROOT, UnsupportedRequest


class RequestConflict(ValueError):
    pass


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
            "attempt_count": len(job.get("attempts", [])), "dataset": job.get("dataset")}


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
        if row["operation"] == "isaac_physx" and row["native_id"]:
            result["native"] = read_native_summary(row["native_id"])
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
                db.execute("UPDATE jobs SET state='interrupted', updated=? WHERE state='running'", (time.time(),))
            yield

    def run_next(self, registry):
        """Call inside worker(). Orphaned jobs retain their native ID and are never blindly replayed."""
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id,operation,payload FROM jobs WHERE state='queued' ORDER BY rowid LIMIT 1").fetchone()
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
            result = registry.invoke(row["operation"], json.loads(row["payload"]), native_job)
            encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
            state = "finished"
        except UnsupportedRequest as exc:
            state, encoded = "failed", json.dumps(exc.details, ensure_ascii=False)
        except Exception as exc:
            logging.exception("Platform job %s failed in %s", job_id, row["operation"])
            state, encoded = "failed", json.dumps({"status": "failed", "error_type": type(exc).__name__})
        with self.db() as db:
            db.execute("UPDATE jobs SET state=?, result=?, updated=? WHERE id=?", (state, encoded, time.time(), job_id))
        return True
