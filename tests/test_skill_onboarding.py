"""Durable validation flow with simulated launcher and registry responses."""
from pathlib import Path
import json
import shutil
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import skill_onboarding as onboarding
from hinge_fixture import ACTOR, TASK
from robot_adapters import RobotAdapter
from scene_contract import content_id, digest
from skill_capabilities import HINGE_APIS, HINGE_SKILL, HingeBinding, RuntimeScope, SkillRegistry


class SkillOnboardingTests(unittest.TestCase):
    def setUp(self):
        tests = Path(__file__).resolve().parent
        self.root = tests / (".skill-onboarding-test-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.assertEqual(self.root.resolve().parent, tests)
        self.addCleanup(shutil.rmtree, self.root)
        self.adapter = RobotAdapter("g2crsb90d", ("left", "right"), frozenset({"curobo", "friction_grasp"}),
                                    {"scene_task_prefix": "g2crsb90d_finemanip_"})
        self.fixture = {"task": TASK, "scene_id": "hinge-0", "scene_sha256": "a" * 64,
                        "binding": HingeBinding(ACTOR, "native_hinge", "part_01", "left", "min", "b" * 64).to_dict()}
        self.fake_control = SimpleNamespace(COLLECTION_SLOTS={"primary": object()}, SLOT_BUSY_EXIT=75,
                                           SOCKET="unix:///project.sock", STORAGE=self.root)
        self.pinned = {"hinge_fixture.py": "c" * 64, "skill_capabilities.py": "d" * 64}
        self.enterContext(patch.object(onboarding, "_pinned_code", return_value=self.pinned))
        self.enterContext(patch.object(onboarding, "ROOT", self.root))
        self.enterContext(patch.object(onboarding, "load_robot", return_value=self.adapter))
        self.enterContext(patch.dict(sys.modules, {"collection_control": self.fake_control}))
        self.prepare_fixture = self.enterContext(patch("hinge_fixture.prepare_fixture", side_effect=self.build_fixture))
        self.certify = self.enterContext(patch("hinge_fixture.certify_run"))

    def build_fixture(self, output):
        output.mkdir(parents=True)
        return {**self.fixture, "scene_directory": str(output)}

    def scope(self, methods=HINGE_APIS, image="e"):
        return RuntimeScope(self.adapter.robot_id, "sha256:" + image * 64, "1" * 64, "2" * 64, "3" * 64, methods)

    def finished_container(self, job, timestamp=None):
        run = self.root / "collection-runs" / job["job_id"]
        run.mkdir(parents=True)
        receipt = {"owner_job": job["job_id"], "scene_sha256": self.fixture["scene_sha256"],
                   "task": self.adapter.config["scene_task_prefix"] + TASK, "robot_id": self.adapter.robot_id,
                   "collection_slot": "primary", "container_id": "project-container",
                   "source_sha256": self.pinned}
        (run / "launch.json").write_text(json.dumps(receipt), encoding="utf-8")
        return "project-container", "exited", 0, timestamp if timestamp is not None else time.time()

    def test_unknown_and_runtime_missing_api_never_prepare_a_fixture(self):
        requests = [
            ({"skill": "vacuum_pick"}, None, "vacuum_pick"),
            ({"skill": HINGE_SKILL}, self.scope(HINGE_APIS[1:]), HINGE_SKILL),
        ]
        for goal, scope, missing in requests:
            with self.subTest(goal=goal):
                result = onboarding.prepare(goal, execute=True, scope=scope)
                self.assertEqual(result["message"], "Skill is not registered")
                self.assertEqual(result["missing_skills"], [missing])
                self.assertEqual(result["candidate_apis"], [])
        self.prepare_fixture.assert_not_called()
        self.certify.assert_not_called()

    def test_drawer_is_not_bound_to_hinge_api(self):
        result = onboarding.prepare({"skill": HINGE_SKILL, "actor": "prismatic_drawer"}, execute=True)
        self.assertEqual(result["status"], "capability_gap")
        self.assertEqual(result["reason"], "no_trusted_hinge_binding_for_request")
        self.prepare_fixture.assert_not_called()

    def test_registered_skill_without_task_binding_returns_a_gap(self):
        result = onboarding.prepare({"skill": "grasp"}, execute=True)
        self.assertEqual(result["status"], "capability_gap")
        self.assertEqual(result["reason"], "registered_skill_requires_a_task_binding")
        self.assertNotIn("job_id", result)
        self.prepare_fixture.assert_not_called()

    def test_preflight_publishes_job_before_calling_the_queue(self):
        def callback(identifier):
            recorded = json.loads((self.root / "jobs" / identifier / "job.json").read_text(encoding="utf-8"))
            self.assertEqual(recorded["status"], "validating_skill")
            self.assertTrue(recorded["automatic_admission"])
            self.assertEqual(recorded["goal"]["actor"], ACTOR)

        result = onboarding.prepare({"skill": HINGE_SKILL}, on_prepared=callback)
        self.assertEqual(onboarding.resume(result["job_id"], execute=False), result)
        self.certify.assert_not_called()

    def test_completed_native_episode_can_fail_independent_validation(self):
        job = onboarding.prepare({"skill": HINGE_SKILL})
        state = self.finished_container(job)
        self.certify.return_value = {"status": "skill_validation_failed", "reason": "base_displaced"}
        with patch.object(onboarding, "_inspect", return_value=state):
            result = onboarding.resume(job["job_id"])
        self.assertEqual(result["status"], "skill_validation_failed")
        self.assertEqual(result["code"], "SKILL_VALIDATION_FAILED")
        self.assertEqual(result["validation"]["outcome"]["reason"], "base_displaced")
        self.assertFalse((self.root / "skill-registry").exists())
        with patch.object(onboarding, "_inspect") as inspect:
            self.assertEqual(onboarding.resume(job["job_id"]), result)
        inspect.assert_not_called()
        self.certify.assert_called_once()

    def test_exact_certificate_is_reusable_but_explicit_execution_checks_a_new_run(self):
        registry = SkillRegistry(self.root / "skill-registry")
        registry.directory.mkdir()
        scope = self.scope()
        binding = HingeBinding(**self.fixture["binding"])
        key = registry._scope_record(HINGE_SKILL, scope, binding)
        receipt = {"key": key, "verification": {"passed": True}}
        receipt["certificate_sha256"] = content_id(receipt)
        path = registry.directory / (content_id(key) + ".json")
        path.write_text(json.dumps(receipt), encoding="utf-8")
        first = onboarding.prepare({"skill": HINGE_SKILL}, scope=scope)
        mismatch = onboarding.prepare({"skill": HINGE_SKILL}, scope=self.scope(image="f"))
        self.assertEqual(first["status"], "ready")
        self.assertEqual(mismatch["status"], "validating_skill")

        def certify(run, _directory):
            evidence = run / "evidence"
            evidence.mkdir()
            for file in onboarding.RUN_EVIDENCE.values():
                (evidence / file).write_text("test-only raw evidence", encoding="utf-8")
            return {"status": "ready", "certificate": receipt, "certificate_path": str(path),
                    "run_verification": {"passed": True},
                    "run_evidence_sha256": {kind: digest(evidence / file)
                                             for kind, file in onboarding.RUN_EVIDENCE.items()}}

        self.certify.side_effect = certify

        states = [None]

        def launch(_path, job, *_args):
            states.append(self.finished_container(job))
            return 0

        pending = onboarding.prepare({"skill": HINGE_SKILL}, scope=scope)
        self.assertNotEqual(first["job_id"], pending["job_id"])
        with (
            patch.object(onboarding, "_inspect", side_effect=lambda *_args: states.pop(0)),
            patch.object(onboarding, "_launch", side_effect=launch) as started,
        ):
            result = onboarding.resume(pending["job_id"], execute=True)
        self.assertEqual(result["status"], "ready")
        self.assertNotIn("code", result)
        self.assertEqual(result["missing_skills"], [])
        started.assert_called_once()
        self.certify.assert_called_once()

    def test_exited_after_validation_budget_is_not_certified(self):
        job = onboarding.prepare({"skill": HINGE_SKILL})
        state = self.finished_container(job, timestamp=time.time() + onboarding.RUN_BUDGET_S + 120.)
        with patch.object(onboarding, "_inspect", return_value=state):
            result = onboarding.resume(job["job_id"])
        self.assertEqual(result["reason"], "validation_deadline")
        self.certify.assert_not_called()

    def test_stopped_job_never_attempts_another_launch(self):
        prepared = onboarding.prepare({"skill": HINGE_SKILL})
        path = Path(prepared["job_path"])
        job = json.loads(path.read_text(encoding="utf-8"))
        job["status"] = "stopped_by_user"
        path.write_text(json.dumps(job), encoding="utf-8")
        with patch.object(onboarding, "_inspect") as inspect:
            result = onboarding.resume(job["job_id"], execute=True)
        self.assertEqual(result["status"], "stopped_by_user")
        inspect.assert_not_called()
        self.certify.assert_not_called()

    def test_other_jobs_container_is_not_adopted(self):
        output = SimpleNamespace(returncode=0, stdout="other\trunning\t0\tunused\twrong-owner\n", stderr="")
        with patch.object(onboarding.subprocess, "run", return_value=output):
            with self.assertRaisesRegex(ValueError, "different job"):
                onboarding._inspect("hinge-skill-example", {})

    def test_captured_path_refusal_reports_the_block_without_certifying(self):
        prepared = onboarding.prepare({"skill": HINGE_SKILL})
        container, status, _, end = self.finished_container(prepared)
        evidence = self.root / "collection-runs" / prepared["job_id"] / "evidence"
        evidence.mkdir()
        stamp = 12.183333968743682
        (evidence / "stage-evidence.jsonl").write_text(json.dumps({
            "phase": "hinge_grasp_probe", "success": True,
            "after": {"sim_time_s": stamp, "joint": {"current_angle_deg": 0.0}}}) + "\n", encoding="utf-8")
        (evidence / "local-joint-plan-continuity.jsonl").write_text(json.dumps({
            "passed": False, "limit_rad": .18, "ik_candidates_rejected_by_continuity": 6060,
            "sim_time": stamp}) + "\n", encoding="utf-8")
        (evidence / onboarding.RUN_EVIDENCE["controls"]).write_text(
            json.dumps({"sim_time": stamp-.10}) + "\n", encoding="utf-8")
        with patch.object(onboarding, "_inspect", return_value=(container, status, 1, end)):
            result = onboarding.resume(prepared["job_id"])
        self.assertEqual(result["reason"], "hinge_path_rejected_before_actuation")
        self.assertEqual(result["validation"]["diagnostic"]["native_commands_after_grasp"], 0)
        self.certify.assert_not_called()

    def test_grasp_failures_preserve_geometry_and_short_pull_diagnostics(self):
        phase = {"phase": "hinge_probe_failure", "reason": "no_clear_handle_grasp_geometry",
                 "measured_opening_m": .0896893}
        reason, evidence = onboarding._diagnose_native_failure(self.root, [phase])
        self.assertEqual(reason, "hinge_handle_grasp_geometry_blocked")
        self.assertEqual(evidence["measured_opening_m"], .0896893)
        for native, expected in ((False, "hinge_grasp_probe_path_rejected"),
                                 (True, "hinge_grasp_contact_unverified")):
            with self.subTest(native=native):
                confirmation = {"reason": "grasp_contact_not_confirmed", "planner_success": native,
                                "progress_deg": 0.1, "position_error_m": .003}
                phase = {"phase": "hinge_grasp_probe", "success": False, "confirmation": confirmation}
                reason, evidence = onboarding._diagnose_native_failure(self.root, [phase])
                self.assertEqual(reason, expected)
                self.assertEqual(evidence["progress_deg"], 0.1)

    def test_native_segment_failure_uses_applied_commands_and_physical_witnesses(self):
        def pose(x=0.):
            return [[1., 0., 0., x], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]

        prepared = onboarding.prepare({"skill": HINGE_SKILL})
        container, status, _, end = self.finished_container(prepared)
        evidence = self.root / "collection-runs" / prepared["job_id"] / "evidence"
        evidence.mkdir()
        start = 12.

        def write_rows(name, rows):
            (evidence / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

        def physics(drift):
            write_rows(onboarding.RUN_EVIDENCE["physics"], [
                {"sim_time": start + index / 120.,
                 "poses": {"hinge_base": pose(drift * index / 24.), "hinge_link": pose()}}
                for index in range(25)
            ])

        write_rows("stage-evidence.jsonl", [
            {"phase": "hinge_grasp_probe", "success": True, "after": {"sim_time_s": start, "base_pose": pose()}},
            {"phase": "hinge_probe_failure", "reason": "hinge_hold_or_base_shift",
             "before_deg": 0., "after_deg": -.00077, "commanded_swing_deg": 23.901,
             "planner_success": True, "sim_time_s": start + .2},
        ])
        write_rows("local-joint-plan-continuity.jsonl", [{"sim_time": start, "passed": True}])
        write_rows(onboarding.RUN_EVIDENCE["controls"], [
            {"sim_time": start - .1, "applied_joint_positions": [1., .2]},
            {"sim_time": start + .02, "applied_joint_positions": [1.01, .2]},
            {"sim_time": start + .15, "applied_joint_positions": [1.04, .2]},
            {"sim_time": start + .3, "applied_joint_positions": [1.05, .2]},
        ])
        physics(0.)
        run = evidence.parent
        events = list(onboarding._evidence_rows(run, "stage-evidence.jsonl"))
        reason, diagnostic = onboarding._diagnose_native_failure(run, events)
        self.assertEqual(reason, "hinge_contact_progress_unverified")
        self.assertEqual(diagnostic["native_commands_in_segment"], 2)
        self.assertEqual(diagnostic["base_peak_drift_m"], 0.)
        self.assertEqual(diagnostic["hinge_peak_swing_deg"], 0.)

        physics(.012)
        reason, diagnostic = onboarding._diagnose_native_failure(run, events)
        self.assertEqual(reason, "hinge_base_drift")
        self.assertAlmostEqual(diagnostic["base_peak_drift_m"], .012)

        physics(0.)
        write_rows(onboarding.RUN_EVIDENCE["controls"], [])
        reason, diagnostic = onboarding._diagnose_native_failure(run, events)
        self.assertEqual(reason, "hinge_motion_blocked")
        self.assertEqual(diagnostic["physical_diagnosis"], "insufficient_evidence")

        # Replay the same terminal case through the public job result without certification.
        with patch.object(onboarding, "_inspect", return_value=(container, status, 1, end)):
            result = onboarding.resume(prepared["job_id"])
        self.assertEqual(result["reason"], "hinge_motion_blocked")
        self.assertEqual(result["validation"]["diagnostic"]["physical_diagnosis"], "insufficient_evidence")
        self.certify.assert_not_called()
