import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from trajectory_admission import admit_episode, publish_lineage
from trajectory_diversity import (describe_motion, different_scene_dimensions, episode_quotas, motion_distance,
                                  resolve_trajectory_request, scene_conflict, scene_descriptor, trajectory_request)
from trajectory_replay import (ASSET_RELEASE_FIELDS, REPLAY_METHOD, populate_replay, recheck_replay_recipe,
                               reference_task, verify_replay_execution)
from scene_contract import digest, write_manifest


ROBOT = {"joint_names": ["wrist", "finger"], "position_units": ["rad", "m"]}


def path(count=101, *, pace=1., bend=.45, jitter=0.):
    u = np.linspace(0, 1, count) ** pace
    rows = np.column_stack((u + bend * np.sin(2 * np.pi * u), .012 * (1 - u)))
    noise = jitter * np.column_stack((np.cos(65 * u), .05 * np.sin(27 * u)))
    return {"observation.state": rows + noise, "action": rows}


def layout(*, instance="phone_0", room="common__studio__background", dx=0., seed=4):
    return {"scene_id": str(seed), "scene_transfer": {"seed": seed, "background": room,
            "support": {"asset_id": "table", "bounds_xy_m": [[0, 0], [1, 1]], "height_m": .75}},
            "layout": {instance: {"usd": "phone", "xyz": [.3 + dx, .4, .76], "xyzw": [0, 0, 0, 1]},
                       "box_0": {"usd": "box", "xyz": [.7, .4, .76], "xyzw": [0, 0, 0, 1]}}}


class QuotaTests(unittest.TestCase):
    def test_total_allocated_once(self):
        self.assertEqual(episode_quotas(["0", "1"], total_episodes=5),
                         [{"scene_id": "0", "successes": 3}, {"scene_id": "1", "successes": 2}])
        self.assertEqual(episode_quotas(["0", "1"], total_episodes=1),
                         [{"scene_id": "0", "successes": 1}])
        self.assertEqual(episode_quotas(["0", "1"], total_episodes=3, requested_scene_count=3),
                         [{"scene_id": "0", "successes": 1}, {"scene_id": "1", "successes": 1}])
        self.assertEqual(episode_quotas(["0", "1"], per_scene=2),
                         [{"scene_id": "0", "successes": 2}, {"scene_id": "1", "successes": 2}])
        with self.assertRaisesRegex(ValueError, "conflicts"):
            episode_quotas(["0", "1"], total_episodes=3, per_scene=2)

    def test_only_explicit_batch_repeat_intent_enables_repeat(self):
        self.assertEqual(trajectory_request({})["duplicate_policy"], "distinct")
        options = {"duplicate_policy": "allow", "duplicate_intent": "允许动作重复"}
        self.assertEqual(trajectory_request(options, instruction="采三条，允许动作重复")["duplicate_policy"], "allow")
        for text, evidence in (("不允许动作重复", "允许动作重复"), ("除非我允许动作重复", "允许动作重复"),
                               ("重复抓取动作三次", "重复抓取动作"), ("不要采集重复轨迹", "采集重复轨迹")):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "explicit|disallows"):
                trajectory_request({**options, "duplicate_intent": evidence}, instruction=text)
        with self.assertRaisesRegex(ValueError, "reference_run"):
            trajectory_request({"duplicate_policy": "replay"})
        replay = {"duplicate_policy": "replay", "reference_run": "episode-ab", "duplicate_intent": "重放同一条轨迹"}
        self.assertEqual(trajectory_request(replay, instruction="重放同一条轨迹 episode-ab 两次")["reference_run"],
                         "episode-ab")

    def test_structured_opt_in_cannot_override_text_disallowing_repeats(self):
        allow = {"duplicate_policy": "allow", "total_episodes": 2}
        replay = {"duplicate_policy": "replay", "reference_run": "episode-ab"}
        for instruction in ("生成两条，禁止重复轨迹", "录制两条不同的轨迹", "不要回放参考轨迹",
                            "collect two episodes without duplicates"):
            for options in (allow, replay):
                with self.subTest(instruction=instruction, policy=options["duplicate_policy"]):
                    with self.assertRaisesRegex(ValueError, "disallows repeated trajectories"):
                        resolve_trajectory_request(options, instruction=instruction)
        self.assertEqual(resolve_trajectory_request(allow, instruction="录制两条")["duplicate_policy"], "allow")
        self.assertEqual(resolve_trajectory_request(allow, instruction="每次都不要重复抓取动作")["duplicate_policy"],
                         "allow")

    def test_unspecified_structured_fields_preserve_valid_language_intent(self):
        parsed = {"duplicate_policy": "allow", "duplicate_intent": "允许动作重复",
                  "distinct_dimensions": ["support"]}
        result = resolve_trajectory_request({"total_episodes": 2}, parsed,
                                            instruction="录制两条，使用不同桌子，允许动作重复")
        self.assertEqual((result["duplicate_policy"], result["distinct_dimensions"], result["total_episodes"]),
                         ("allow", ["support"], 2))
        with self.assertRaisesRegex(ValueError, "duplicate_policy conflicts"):
            resolve_trajectory_request({"duplicate_policy": "distinct"}, parsed, instruction="允许动作重复")
        with self.assertRaisesRegex(ValueError, "distinct_dimensions conflicts"):
            resolve_trajectory_request({"distinct_dimensions": ["lighting"]}, parsed, instruction="允许动作重复")
        result = resolve_trajectory_request({"duplicate_policy": "allow"}, {"duplicate_policy": "distinct"},
                                            instruction="录制两条")
        self.assertEqual(result["duplicate_policy"], "allow")

    def test_frozen_replay_disallows_plural_requested_scenes(self):
        request = {"duplicate_policy": "replay", "reference_run": "episode-ab", "total_episodes": 1}
        demo = "请按原场景和原配方重复执行一次这一条已验收轨迹。"
        self.assertEqual(resolve_trajectory_request(request, instruction=demo)["duplicate_policy"], "replay")
        for instruction in ("在两种不同的场景复验参考轨迹", "分别用不同房间复验", "replay in different scenes"):
            with self.subTest(instruction=instruction), self.assertRaisesRegex(ValueError, "original fixed scene"):
                resolve_trajectory_request(request, instruction=instruction)
        with self.assertRaisesRegex(ValueError, "original fixed scene"):
            resolve_trajectory_request(request, instruction=demo, distinct_scene_count=2)

    def test_registry_preserves_omission_and_rejects_conflicts_before_enqueue(self):
        from genie_extensions import Registry

        registry = Registry()
        payload = {"instruction": "采集两条，允许轨迹重复", "trajectory_request": {"total_episodes": 2}}
        checked = registry.validate("isaac_physx", payload)
        self.assertEqual(checked["trajectory_request"], {"total_episodes": 2})
        with self.assertRaisesRegex(ValueError, "disallows repeated trajectories"):
            registry.validate("isaac_physx", {"instruction": "采集两条，禁止轨迹重复",
                                              "trajectory_request": {"duplicate_policy": "allow"}})


class SceneDiversityTests(unittest.TestCase):
    def test_instance_alias_seed_quaternion_sign_and_tiny_drift_do_not_make_a_scene(self):
        original = scene_descriptor(layout())
        copy_scene = layout(instance="renamed", dx=.0007, seed=928)
        copy_scene["layout"]["renamed"]["xyzw"][-1] = -1
        self.assertEqual(scene_conflict(scene_descriptor(copy_scene), [original])["scene_id"], "4")
        self.assertIsNone(scene_conflict(scene_descriptor(layout(dx=.03, seed=33)), [original]))

    def test_same_asset_content_with_a_different_label_gets_same_room_identity(self):
        raw = layout(room="old_room")
        alias = layout(room="new_room")
        base_dependencies = {"old_room": "release-checksum"}
        alias_dependencies = {"new_room": "release-checksum"}
        self.assertEqual(different_scene_dimensions(scene_descriptor(raw, asset_digests=base_dependencies),
                         scene_descriptor(alias, asset_digests=alias_dependencies)), set())

    def test_scene_dimension_matches_requested_room_or_lighting(self):
        original = scene_descriptor(layout(), lighting=700)
        lighting_only = scene_descriptor(layout(seed=60), lighting=920)
        self.assertEqual(scene_conflict(lighting_only, [original])["different_dimensions"], ["lighting"])
        self.assertIsNone(scene_conflict(lighting_only, [original], dimensions=["lighting"]))
        self.assertEqual(scene_conflict(scene_descriptor(layout()), [], dimensions=["lighting"])["reason"],
                         "scene_dimension_not_observed")
        new_room = scene_descriptor(layout(room="new_room", seed=303))
        self.assertIsNone(scene_conflict(new_room, [original], dimensions=["background"]))
        self.assertEqual(scene_conflict(new_room, [original], dimensions=["layout"])["reason"],
                         "repeated_scene_content")

    def test_settled_measurement_can_collapse_an_authored_difference(self):
        original = scene_descriptor(layout())
        moved = layout(dx=.03, seed=33)
        self.assertIsNone(scene_conflict(scene_descriptor(moved), [original]))
        matrix = np.eye(4)
        matrix[:3, 3] = [.3, .4, .76]
        settled = scene_descriptor(moved, initial_poses={"phone_0": matrix})
        self.assertEqual(settled["measured_objects"], 1)
        self.assertEqual(scene_conflict(settled, [original])["reason"], "repeated_scene_content")


class MotionTests(unittest.TestCase):
    def test_speed_pause_and_small_measurement_noise_stay_near(self):
        reference = describe_motion(path(), ROBOT)
        arrays = path(281, pace=1.9, jitter=.00025)
        for key, value in arrays.items():
            arrays[key] = np.concatenate((np.repeat(value[:1], 40, axis=0), value[:100],
                                          np.repeat(value[99:100], 40, axis=0), value[100:],
                                          np.repeat(value[-1:], 70, axis=0)))
        distance = motion_distance(reference, describe_motion(arrays, ROBOT))
        self.assertLess(distance["measured_joint_ratio"], 1.)

    def test_different_path_with_same_initial_and_final_pose_separates(self):
        a, b = path(), path(bend=-.2)
        np.testing.assert_allclose(a["observation.state"][[0, -1]], b["observation.state"][[0, -1]])
        self.assertGreater(motion_distance(describe_motion(a, ROBOT), describe_motion(b, ROBOT))
                           ["measured_joint_ratio"], 1.)

    def test_commands_and_native_joint_name_reordering(self):
        arrays = path()
        first = describe_motion(arrays, ROBOT, phases=[("grasp", 0, 43), ("move_held", 43, 101)])
        reordered = {key: val[:, ::-1] for key, val in arrays.items()}
        swapped = {key: values[::-1] for key, values in ROBOT.items()}
        second = describe_motion(reordered, swapped, phases=[("grasp", 0, 43), ("move_held", 43, 101)])
        result = motion_distance(first, second)
        self.assertEqual(result["measured_joint_ratio"], 0.)
        self.assertEqual(result["command_position_ratio"], 0.)
        self.assertEqual(result["compared_phases"], 2)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.db = Path.cwd() / (".trajectory-admission-test-" + uuid.uuid4().hex + ".sqlite3")
        self.addCleanup(self.db.unlink, missing_ok=True)
        self.request = {"lineage_id": "training-dataset"}
        self.default_job = {"job_id": "job-A", "robot_id": "g2crsb90d", "attempts": [],
                            "trajectory_request": self.request,
                            "plan": {"tasks": [{"task_id": "packing", "scene_quotas": [
                                {"scene_id": "0", "successes": 1}]}]}}
        self.scene = scene_descriptor(layout(seed=0))
        self.motion = describe_motion(path(), ROBOT)

    def receive(self, label, *, job=None, motion=None, scene=None, fingerprint=None, reference=None):
        trial = {"task": "packing", "scene": "0", "run_directory": f"/records/{label}",
                 "launch_sha256": "launch:" + label, "raw_archive": True, "automatic_acceptance": True}
        return admit_episode(job or self.default_job, trial, scene=scene or self.scene,
                             motion=motion or self.motion, fingerprint=fingerprint or "motion:" + label,
                             reference=reference, lineage_db=self.db)

    def test_competing_workers_cannot_double_fill_an_episode_quota(self):
        novel = describe_motion(path(bend=-.2), ROBOT)
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(self.receive, "run1")
            second = pool.submit(self.receive, "run2", motion=novel)
        decisions = [first.result(), second.result()]
        self.assertEqual(sum(row.get("accepted", False) for row in decisions), 1)
        self.assertIn("trajectory_quota_already_filled", [row.get("failure") for row in decisions])

    def test_shared_lineage_rejects_near_motion_but_explicit_allow_accepts_new_execution(self):
        a = self.receive("run1")
        self.assertTrue(a["accepted"])
        job = copy.deepcopy(self.default_job)
        job["job_id"] = "job-B"
        other = describe_motion(path(201, pace=1.8), ROBOT)
        failure = self.receive("run2", job=job, motion=other)
        self.assertEqual(failure["failure"], "near_motion_repeat")
        self.assertEqual(failure["duplicate_motion_of"], "/records/run1")
        job["trajectory_request"]["duplicate_policy"] = "allow"
        repeated = self.receive("run2", job=job, motion=other, fingerprint="motion:run1")
        self.assertTrue(repeated["accepted"])
        self.assertEqual(repeated["record"]["duplicate_group"], "/records/run1")
        self.assertTrue(self.receive("run2", job=job, motion=other, fingerprint="motion:run1")["accepted"])
        self.assertEqual(self.receive("run1", job=job)["failure"], "execution_already_admitted")

    def test_shared_lineage_receipt_only_follows_official_readback(self):
        self.assertTrue(self.receive("run1")["accepted"])
        receipt = {"official_reader_verified": True, "loader_receipt_sha256": "loader-checksum",
                   "sources": [{"launch_sha256": "launch:run1"}]}
        publish_lineage(self.default_job, receipt, self.db)
        publish_lineage(self.default_job, receipt, self.db)
        with closing(sqlite3.connect(self.db)) as db:
            self.assertEqual(db.execute("SELECT loader_receipt_sha256 FROM trajectory_admissions").fetchone(),
                             ("loader-checksum",))
        with self.assertRaisesRegex(ValueError, "official reader"):
            publish_lineage(self.default_job, {**receipt, "official_reader_verified": False}, self.db)

    def test_historical_lineage_fingerprint_cannot_fill_another_jobs_quota(self):
        # Ledger signatures do not require the original recording path to
        # remain present after the raw package has moved or been compacted.
        self.assertTrue(self.receive("pruned", fingerprint="historical")["accepted"])
        job = copy.deepcopy(self.default_job)
        job["job_id"] = "job-new"
        repeated = self.receive("redundant", job=job, fingerprint="historical")
        self.assertEqual(repeated["failure"], "exact_motion_repeat")
        self.assertEqual(repeated["duplicate_motion_of"], "/records/pruned")
        self.assertTrue(self.receive("novel", job=job, motion=describe_motion(path(bend=-.2), ROBOT))["accepted"])
        job["job_id"] = "job-independent"
        job["trajectory_request"]["lineage_id"] = "other-dataset"
        self.assertTrue(self.receive("different-lineage", job=job, fingerprint="historical")["accepted"])

    def test_replay_needs_matching_measurements_and_a_new_native_execution(self):
        job = copy.deepcopy(self.default_job)
        job["trajectory_request"].update(duplicate_policy="replay", reference_run="source")
        reference = {"reference_label": "source", "robot_id": "g2crsb90d", "task": "packing",
                     "scene": self.scene, "motion": self.motion, "motion_sha256": "motion:source",
                     "launch_sha256": "launch:source", "run": "/records/source"}
        self.assertEqual(self.receive("missing", job=job)["failure"], "replay_reference_unavailable")
        self.assertEqual(self.receive("source", job=job, reference=reference)["failure"],
                         "invalid_replay_reference_or_execution")
        changed = describe_motion(path(bend=-.2), ROBOT)
        self.assertEqual(self.receive("different", job=job, reference=reference, motion=changed)["failure"],
                         "replay_motion_differs_from_reference")
        commands = path()
        commands["action"] = commands["action"].copy()
        commands["action"][:, 0] += .1
        changed_commands = describe_motion(commands, ROBOT)
        self.assertEqual(self.receive("changed-control", job=job, reference=reference,
                                      motion=changed_commands)["failure"],
                         "replay_motion_differs_from_reference")
        self.assertTrue(self.receive("repeat", job=job, reference=reference)["accepted"])


class ReplayRecipeTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path.cwd().resolve()
        self.storage = self.workspace / (".replay-recipe-test-" + uuid.uuid4().hex)
        self.storage.mkdir()
        self.addCleanup(self.remove_fixture)
        self.assets = self.storage / "assets"
        self.assets.mkdir()
        assets_release = {"release_id": "sealed-fixture-release"}
        self.write(self.assets / ".geniesim-assets.json", assets_release)
        self.original = self.storage / "collection-runs/source"
        self.task = "g2crsb90d_finemanip_phone_packaging"
        self.scene_id = "scene-composed-fixture"
        scene = self.original / "tasks" / self.task / self.scene_id
        scene.mkdir(parents=True)
        metadata = layout()
        metadata.update(scene_id=self.scene_id, task_template="phone_packaging")
        self.write(scene / "scene_info.json", metadata)
        self.write(scene / "task_info.json", {"backend": "cpu"})
        (scene / "scene.usda").write_text("#usda 1.0", encoding="utf-8")
        loader = scene.parent / "data_gen.py"
        loader.write_text("pass", encoding="utf-8")
        manifest = write_manifest(scene, [], mode="compose_from_library", source={"source_scene_id": "0"},
                                  asset_root=self.assets)
        inputs = {
            "request.json": {"seed": 53, "scene_id": self.scene_id, "robot_id": "g2crsb90d",
                             "task_id": "phone_packaging", "instruction": "original instruction"},
            "prepared-plan.json": {"validated": True, "plan_sha256": "known-plan",
                                   "plan": {"scene_id": self.scene_id}},
            "robot-profile.json": {"robot_id": "g2crsb90d"},
            "execution-profile.json": {"collection_slot": "gpu1-extra"},
        }
        source = self.original / "source"
        source.mkdir()
        for name, data in inputs.items():
            self.write(source / name, data)
        (self.original / "raw").mkdir()
        self.write(self.original / "raw/state.json", {"frames": []})
        (self.original / "evidence").mkdir()
        (self.original / "evidence/native-control-commands.jsonl").write_text("", encoding="utf-8")
        preflight = self.original / "evidence/scene-physics-preflight.json"
        self.write(preflight, {"passed": True})
        asset_stamp = {key: assets_release.get(key) for key in ASSET_RELEASE_FIELDS}
        asset_stamp["receipt_sha256"] = digest(self.assets / ".geniesim-assets.json")
        self.launch = {"task": self.task, "scene": self.scene_id, "robot_id": "g2crsb90d",
                       "execution_mode": "structured", "seed": 53, "physics_backend": "cpu",
                       "planner_backend": "curobo_cuda", "renderer_backend": "cuda", "image_id": "fixed-image-id",
                       "initial_plan_sha256": "known-plan", "scene_sha256": manifest["content_sha256"],
                       "collection_slot": "gpu1-extra", "slot_endpoint": {"gpu_uuid": "fixed-gpu"},
                       "task_loader_sha256": digest(loader), "strict_g2_coupled_wrist": False,
                       "launch_returncode": 0, "container_id": "reference", "isolated_recording": True,
                       "asset_release": asset_stamp,
                       "source_sha256": {name: digest(source / name) for name in inputs}}
        self.write(self.original / "launch.json", self.launch)
        self.write(self.original / "acceptance.json", {"accepted": True})
        self.write(self.original / "automatic-admission.json", {
            "passed": True, "launch_sha256": digest(self.original / "launch.json"),
            "acceptance_sha256": digest(self.original / "acceptance.json"), "motion_sha256": "actual-fingerprint",
            "physics_preflight_sha256": digest(preflight), "scene_sha256": self.launch["scene_sha256"],
            "initial_plan_sha256": self.launch["initial_plan_sha256"], "image_id": self.launch["image_id"]})
        self.request = {"robot_id": "g2crsb90d", "trajectory_request": {
            "total_episodes": 2, "duplicate_policy": "replay", "reference_run": "source"}}

    @staticmethod
    def write(path, content):
        path.write_text(json.dumps(content), encoding="utf-8")

    def remove_fixture(self):
        if self.storage.resolve().parent != self.workspace or not self.storage.name.startswith(".replay-recipe-test-"):
            raise ValueError("Unexpected replay test fixture")
        shutil.rmtree(self.storage)

    def prepare(self, **options):
        return populate_replay(copy.deepcopy(self.request), storage=self.storage, asset_root=self.assets,
                               task_name="phone_packaging", **options)

    def test_dynamic_reference_freezes_actual_scene_plan_seed_and_slot(self):
        job = self.prepare()
        recipe = job["replay_recipe"]
        self.assertEqual((recipe["method"], recipe["seed"], recipe["slot"]), (REPLAY_METHOD, 53, "gpu1-extra"))
        self.assertEqual(recipe["instruction"], "original instruction")
        self.assertEqual(recipe["physics_backend"], "cpu")
        self.assertEqual(job["plan"]["tasks"][0]["scene_quotas"], [{"scene_id": self.scene_id, "successes": 2}])
        self.assertIs(recheck_replay_recipe(job, storage=self.storage, asset_root=self.assets)["isolated_recording"],
                      True)
        self.write(self.original / "source/prepared-plan.json", {"plan": {}})
        with self.assertRaisesRegex(ValueError, "Frozen replay program"):
            recheck_replay_recipe(job, storage=self.storage, asset_root=self.assets)

    def test_scene_overrides_do_not_switch_a_composed_reference_to_its_base_template(self):
        with self.assertRaisesRegex(ValueError, "original fixed scene"):
            self.prepare(scene_options={"mode": "existing", "scene_ids": ["0"]})
        with self.assertRaisesRegex(ValueError, "original fixed scene"):
            self.prepare(scene_options={"mode": "compose_from_library", "scene_count": 1})
        self.assertIn("replay_recipe", self.prepare(scene_options={"mode": "existing", "scene_ids": [self.scene_id]}))

    def test_fresh_launch_requires_same_provenance_and_new_container(self):
        recipe = self.prepare()["replay_recipe"]
        new = self.storage / "collection-runs/new"
        new.mkdir()
        result = {**self.launch, "container_id": "new-execution", "source_recipe_run": str(self.original)}
        self.write(new / "launch.json", result)
        with patch("trajectory_replay.verified_reference", return_value={"physical_reference": True}):
            self.assertEqual(verify_replay_execution(new, recipe), {"physical_reference": True})
        self.write(new / "launch.json", {**result, "seed": 99})
        with self.assertRaisesRegex(ValueError, "different settings: seed"):
            verify_replay_execution(new, recipe)
        self.write(new / "launch.json", {**result, "container_id": "reference"})
        with self.assertRaisesRegex(ValueError, "new isolated simulator execution"):
            verify_replay_execution(new, recipe)

    def test_reference_task_routes_without_llm_and_requires_same_asset_release(self):
        options = {"storage": self.storage, "asset_root": self.assets, "robot_id": "g2crsb90d"}
        self.assertEqual(reference_task("source", **options), "phone_packaging")
        with self.assertRaisesRegex(ValueError, "cannot change the robot"):
            reference_task("source", **{**options, "robot_id": "different-robot"})
        self.write(self.assets / ".geniesim-assets.json", {"release_id": "another-release"})
        with self.assertRaisesRegex(ValueError, "asset release differs"):
            reference_task("source", **options)


if __name__ == "__main__":
    unittest.main()
