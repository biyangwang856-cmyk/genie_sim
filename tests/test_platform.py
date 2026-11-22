"""Offline contract checks at the platform boundary; no Qwen calls or physical runs."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from genie_extensions import Registry
from platform_api import create_app
from platform_jobs import Jobs, read_native_summary


class PlatformTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.registry = Registry()
        self.web = create_app(self.root, self.registry)
        self.jobs = Jobs(self.root)

    def call(self, method, path, payload=None, key=None):
        data = json.dumps(payload).encode()
        env = {"REQUEST_METHOD": method, "PATH_INFO": path, "CONTENT_TYPE": "application/json",
               "CONTENT_LENGTH": str(len(data)), "wsgi.input": BytesIO(data)}
        if key:
            env["HTTP_IDEMPOTENCY_KEY"] = key
        response = []
        body = b"".join(self.web(env, lambda status, headers: response.append((status, headers))))
        return int(response[0][0][:3]), json.loads(body), dict(response[0][1])

    def test_discovery_and_unsupported_do_not_schedule(self):
        schema_code, schema, _ = self.call("GET", "/v1/openapi.json")
        self.assertEqual(schema_code, 200)
        self.assertEqual(set(schema["paths"]), {"/v1/openapi.json", "/v1/capabilities", "/v1/jobs",
                                               "/v1/jobs/{id}", "/v1/extensions/{operation}"})
        code, data, _ = self.call("GET", "/v1/capabilities")
        self.assertEqual(code, 200)
        active = [op for op in data["operations"] if op["status"] == "connected"]
        self.assertEqual([op["id"] for op in active], ["isaac_physx"])
        self.assertEqual(active[0]["scenes"], ["0", "1"])
        for req in ({"simulator": "mujoco"}, {"scenes": ["2"]}, {"task_ids": ["unregistered"]}, {"robot_id": "other"}):
            code, reply, _ = self.call("POST", "/v1/jobs", {"instruction": "phone", **req})
            self.assertEqual((code, reply["status"]), (422, "unsupported"))
        self.assertEqual(self.call("POST", "/v1/extensions/scene_admission", {"scene_ref": "s"})[0], 422)
        self.assertEqual(self.call("POST", "/v1/jobs", {"instruction": "x", "execute": "false"})[0], 400)
        with self.jobs.worker():
            self.assertFalse(self.jobs.run_next(self.registry))

    def test_retry_concurrency_worker_and_native_status(self):
        request = {"instruction": "phone", "task_ids": ["phone_packaging"], "execute": True}
        with ThreadPoolExecutor(max_workers=4) as pool:
            answers = list(pool.map(lambda _: self.call("POST", "/v1/jobs", request, "phone-01"), range(8)))
        ids = {a[1]["id"] for a in answers}
        self.assertEqual(len(ids), 1)
        self.assertTrue(all(a[0] == 202 for a in answers))
        url = answers[0][2]["Location"]
        self.assertEqual(self.call("POST", "/v1/jobs", {"instruction": "other"}, "phone-01")[0], 409)
        self.assertEqual(self.call("GET", url)[1]["state"], "queued")

        def submit(instruction, **kwargs):
            self.assertEqual(instruction, "phone")
            self.assertEqual(kwargs["robot_id"], "g2crsb90d")
            self.assertEqual(kwargs["scenes"], ["0", "1"])
            self.assertTrue(kwargs["execute"])
            kwargs["on_prepared"]("native-001")
            self.assertEqual(self.call("GET", url)[1]["native_job_id"], "native-001")
            return {"status": "needs_hardpoint_check", "job_id": "native-001", "dataset": None}

        with patch("natural_backend.submit", side_effect=submit) as backend, self.jobs.worker():
            self.assertTrue(self.jobs.run_next(self.registry))
            self.assertFalse(self.jobs.run_next(self.registry))
        backend.assert_called_once()
        reply = self.call("GET", url)[1]
        self.assertEqual((reply["state"], reply["result"]["status"]), ("finished", "needs_hardpoint_check"))

    def test_extensions_are_explicit_and_use_the_same_worker(self):
        self.registry.register("scene_admission", lambda body, progress: {"validation_ref": "validated/" + body["scene_ref"]})
        self.registry.register("mujoco", lambda body, progress: {"job_id": "fixture", "status": "prepared"},
                               scope={"robot_ids": ["test-arm"], "task_ids": ["fixture"], "scenes": ["A"]})
        wrong, _, _ = self.call("POST", "/v1/jobs", {"instruction": "x", "simulator": "mujoco", "scenes": ["0"]})
        self.assertEqual(wrong, 422)
        status, _, headers = self.call("POST", "/v1/extensions/scene_admission",
                                       {"scene_ref": "sceneA", "robot_id": "fixture", "task_id": "test"})
        self.assertEqual(status, 202)
        with self.jobs.worker():
            self.jobs.run_next(self.registry)
        self.assertEqual(self.call("GET", headers["Location"])[1]["result"], {"validation_ref": "validated/sceneA"})

    def test_interrupted_job_keeps_reference_and_is_not_replayed(self):
        _, body, headers = self.call("POST", "/v1/jobs", {"instruction": "example"})
        with self.jobs.db() as db:
            db.execute("UPDATE jobs SET state='running',native_id='native-1' WHERE id=?", (body["id"],))
        with self.jobs.worker():
            with self.assertRaises(RuntimeError), Jobs(self.root).worker():
                pass
            self.assertFalse(self.jobs.run_next(self.registry))
        reply = self.call("GET", headers["Location"])[1]
        self.assertEqual((reply["state"], reply["native_job_id"]), ("interrupted", "native-1"))

    def test_live_native_status_includes_only_selected_fields(self):
        directory = self.root / "jobs/native-2"
        directory.mkdir(parents=True)
        (directory / "job.json").write_text(json.dumps({"job_id": "native-2", "status": "collecting", "attempts": [{}],
                                                       "instruction": "internal", "success_counts": {"s": 1}}))
        result = read_native_summary("native-2", self.root)
        self.assertEqual(result["attempt_count"], 1)
        self.assertNotIn("instruction", result)
        self.assertIsNone(read_native_summary("../native-2", self.root))
