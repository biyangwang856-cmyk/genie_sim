import json
from pathlib import Path
import runpy
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from container_goal import catalog_identity, matches_phone_packaging, search_assets, validate_goal
from container_scene import cavity
from natural_backend import submit
from robot_adapters import RobotAdapter, UnsupportedRequest
from task_geometry import bbox_corners
from task_plugins import load, source_file


class ContainerGoalTest(unittest.TestCase):
    def test_explicit_geometric_grasp_reaches_robotapi_with_default_annotation(self):
        # The runtime may supply a default annotation even for an unannotated asset.
        native = SimpleNamespace(GraspConfig=SimpleNamespace, MoveConfig=SimpleNamespace,
                                 PlaceConfig=SimpleNamespace, FollowTrajectoryConfig=SimpleNamespace)
        source = Path(__file__).resolve().parents[1] / "collection_programs.py"
        with (patch.dict(sys.modules, {"client.llm_gen.robot_api": native}), patch("signal.signal"),
              patch("signal.SIGUSR1", 10, create=True)):
            pick = runpy.run_path(str(source))["pick"]
        robot = Mock()
        robot.scene_info = {"task_template": "place_in_container", "layout": {
            "item": {"usd": "item", "entity_local_bbox": {"min": [-.035, -.015, 0],
                                                           "max": [.035, .015, .016]}},
        }}
        robot.interactions = {"item": {"passive": {"grasp": {"default": [{}]}}}}
        robot.get_object_pose.return_value = ([.9, -.1, .75], [1, 0, 0, 0])
        robot.get_ee_pose.return_value = np.eye(4)
        robot.client.command_any.return_value = {}
        robot.pick_actor.return_value = False
        options = {"vertical_offsets_m": (-.002,), "yaw_offsets_rad": (0., np.pi)}
        with TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"GENIE_EPISODE_EVIDENCE": directory}), patch("builtins.print"):
                pick(robot, "item", "right", custom_pose_options=options, lift_probe_distance_m=.012)
        supplied = robot.pick_actor.call_args.kwargs["grasp_config"]
        self.assertEqual(supplied.grasp_pose_frame, "world")
        np.testing.assert_allclose(supplied.grasp_pose[:, 2, 2], [-1., -1.])
        np.testing.assert_allclose(supplied.grasp_pose[:, 2, 3], [.756, .756])

    def test_unregistered_goal_enters_the_normal_prepare_and_execute_pipeline(self):
        item, container = "优盘", "包装盒"
        user = f"在桌面上把{item}放进{container}，采一条轨迹"
        goal = validate_goal({"object": {"source": item, "names": ["USB flash drive"]},
                             "container": {"source": container, "names": ["packaging box"]}}, user)
        adapter = RobotAdapter("g2crsb90d", ("right",), frozenset({
            "friction_grasp", "curobo", "held_object_motion", "gripper_opening_control",
        }), {"scene_task_prefix": "g2_"})
        with TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "job.json"
            invocations = []

            def run(command, **kwargs):
                invocations.append(command)
                if Path(command[1]).name == "prepare_skill_plans.py":
                    request = json.loads(Path(command[3]).read_bytes())
                    self.assertEqual(request["goals"]["place_in_container"], goal)
                    self.assertEqual(request["scene_request"]["scene_ids"], ["0"])
                    self.assertEqual(request["trajectory_request"]["total_episodes"], 1)
                    job.write_text(json.dumps({"job_id": "verify", "status": "skill_plans_prepared"}))
                    return SimpleNamespace(returncode=0, stdout=json.dumps({"path": str(job)}))
                self.assertEqual(Path(command[1]).name, "batch_collect.py")
                return SimpleNamespace(returncode=0)

            with (patch("natural_backend.ROOT", root), patch("natural_backend.TASKS", []),
                  patch("natural_backend.load_robot", return_value=adapter),
                  patch("natural_backend.QwenAPI"),
                  patch("natural_backend.resolve_intent", return_value=({"status": "unsupported"}, None, [])),
                  patch("container_goal.resolve_goal", return_value=({"status": "container_goal", "goal": goal}, {})),
                  patch("natural_backend.subprocess.run", side_effect=run)):
                callback = Mock()
                self.assertEqual(submit(user, execute=True, on_prepared=callback)["job_id"], "verify")
            callback.assert_called_once_with("verify")
            self.assertEqual(len(invocations), 2)

    def test_box_color_is_required_and_opening_is_delegated_to_geometry(self):
        row = {"names": ["white hollow packaging box"], "attributes": ["white"]}
        self.assertTrue(catalog_identity(row, "container", "packaging_box_001",
                                         {"description": "A white box with a hollow interior"}))
        self.assertFalse(catalog_identity(row, "container", "packaging_box_001",
                                          {"description": "A red box with a hollow interior"}))

    def test_asset_qualifiers_survive_wording_differences_in_metadata(self):
        row = {"names": ["eraser"], "attributes": ["4B", "white", "with blue-and-white paper sleeve"]}
        description = {"full_description": ["A white rectangular eraser with a blue and white paper wrapper labeled 4B."]}
        self.assertTrue(catalog_identity(row, "object", "eraser_001", description))
        row["attributes"][-1] = "with green-and-white paper sleeve"
        self.assertFalse(catalog_identity(row, "object", "eraser_001", description))

    def test_phone_recipe_cannot_substitute_for_a_usb_drive(self):
        plan = {"tasks": [{"task_id": "phone_packaging", "goal": {"effects": [{}]}}]}
        receiver = {"names": ["packaging box"], "attributes": []}
        self.assertTrue(matches_phone_packaging(plan, {"container": receiver,
                        "object": {"names": ["smartphone"], "attributes": []}}))
        self.assertFalse(matches_phone_packaging(plan, {"container": receiver,
                         "object": {"names": ["USB flash drive"], "attributes": []}}))

    def test_unrequested_asset_id_cannot_bind_to_a_quoted_noun(self):
        goal = {"object": {"source": "口红", "names": ["lipstick"], "asset_id": "some_pen"},
                "container": {"source": "化妆盒", "names": ["makeup organizer"]}}
        with self.assertRaisesRegex(ValueError, "explicitly selected asset ID"):
            validate_goal(goal, "把口红放到化妆盒里")

    def test_embedding_candidates_do_not_replace_missing_assets(self):
        with TemporaryDirectory() as directory:
            asset = Path(directory) / "jar.usda"
            asset.touch()
            rows = [{"asset_id": "cosmetic_jar", "usd": str(asset),
                     "text": json.dumps({"description": {"semantic_name": "cosmetic cream jar"}})}]
            vector = np.ones((1, 1024))
            api = type("Embedding", (), {"embed": lambda self, queries: ([vector[0], vector[0]], {})})()
            goal = validate_goal({"object": {"source": "口红", "names": ["lipstick"]},
                                  "container": {"source": "化妆盒", "names": ["cosmetic organizer"]}},
                                 "把口红放到化妆盒里")
            with patch("asset_retrieval.load_index", return_value=(rows, vector)):
                with self.assertRaises(UnsupportedRequest) as caught:
                    search_assets(goal, api, directory)
        self.assertEqual(caught.exception.details["code"], "ASSET_NOT_FOUND")
        self.assertEqual(caught.exception.details["missing_assets"], ["口红", "化妆盒"])
        self.assertEqual(caught.exception.details["missing_skills"], [])


class CavityTest(unittest.TestCase):
    def setUp(self):
        self.box = np.array([
            [[-.06, -.07, -.01], [.06, .07, 0]],
            [[-.06, -.07, 0], [-.055, .07, .035]],
            [[.055, -.07, 0], [.06, .07, .035]],
            [[-.055, -.07, 0], [.055, -.065, .035]],
            [[-.055, .065, 0], [.055, .07, .035]],
        ])

    def test_compute_cavity_and_reject_closed_or_narrow_receiver(self):
        fit = cavity(self.box, [.030, .050, .015])
        self.assertAlmostEqual(fit["floor_z_m"], 0.)
        self.assertAlmostEqual(fit["rim_z_m"], .035)
        closed = np.concatenate([self.box, [[[-.06, -.07, .035], [.06, .07, .04]]]])
        with self.assertRaisesRegex(ValueError, "No open collision cavity"):
            cavity(closed, [.030, .050, .015])
        with self.assertRaisesRegex(ValueError, "No open collision cavity"):
            cavity(self.box, [.150, .160, .015])

    def test_verifier_rejects_airborne_item_and_item_touching_outside(self):
        fit = {**cavity(self.box, [.030, .050, .015]), "receiver_basis": np.eye(3)}
        corners = bbox_corners([-.015, -.025, 0.], [.015, .025, .015])
        plugin = load("place_in_container")
        item, box = np.eye(4), np.eye(4)
        checks, _ = plugin.measure([(item, box)] * 4, corners, fit)
        self.assertTrue(all(checks.values()))
        item[2, 3] = .009
        checks, _ = plugin.measure([(item, box)] * 4, corners, fit)
        self.assertFalse(checks["supported_by_interior"])
        item[2, 3] = 0
        item[0, 3] = .050
        checks, _ = plugin.measure([(item, box)] * 4, corners, fit)
        self.assertFalse(checks["entire_object_inside"])

    def test_verifier_matches_support_surface_at_measured_item_footprint(self):
        floor_with_raised_edge = np.concatenate([self.box, [[[.038, -.040, 0.], [.055, .040, .012]]]])
        fit = {**cavity(floor_with_raised_edge, [.030, .050, .015]), "receiver_basis": np.eye(3)}
        fit["floor_z_m"] = .012  # Conservative bound for planning above a nearby protrusion.
        plugin = load("place_in_container")
        corners = bbox_corners([-.015, -.025, 0.], [.015, .025, .015])
        item, box = np.eye(4), np.eye(4)
        checks, _ = plugin.measure([(item, box)] * 4, corners, fit)
        self.assertTrue(checks["supported_by_interior"])
        item[2, 3] = .012  # It is this high but does not overlap the raised edge.
        checks, _ = plugin.measure([(item, box)] * 4, corners, fit)
        self.assertFalse(checks["supported_by_interior"])

    def test_general_episode_audit_reads_recorded_scene_and_finger_positions(self):
        from audit_task_geometry import audit
        from scene_contract import digest

        fit = {**cavity(self.box, [.030, .050, .015]), "receiver_basis": np.eye(3).tolist(),
               "object_basis": np.eye(3).tolist(), "operator": "right"}
        scene = {"task_template": "place_in_container",
                 "official_instruction": {"target": {"id1": "item", "id2": "receiver"}},
                 "layout": {"item": {"entity_local_bbox": {"min": [-.015, -.025, 0.],
                                                           "max": [.015, .025, .015]}}},
                 "project_container_placement": fit}
        frames=[]
        for i in range(21):
            frames.append({
                "time_stamp": 0 if i == 0 else 2 + i / 30,
                "objects": {f"World_Objects_{name}_entity": {"pose": np.eye(4).tolist()}
                            for name in ("item", "receiver")},
                "robot": {"joints": {"joint_name": ["right_finger"], "joint_position": [0.]}},
            })
        with TemporaryDirectory() as directory:
            run = Path(directory)
            scene_dir = run / "tasks/robot_place_in_container/scene-demo"
            scene_dir.mkdir(parents=True)
            (scene_dir / "scene_info.json").write_text(json.dumps(scene))
            (run / "source").mkdir()
            snapshot = run / "source/task-contract-place_in_container.py"
            shutil.copyfile(source_file("place_in_container"), snapshot)
            (run / "launch.json").write_text(json.dumps({
                "task": "robot_place_in_container", "scene": "scene-demo",
                "source_sha256": {snapshot.name: digest(snapshot)},
            }))
            evidence = run / "evidence"
            evidence.mkdir()
            (evidence / "skill-execution.jsonl").write_text(json.dumps({
                "event": "container_release_gate", "passed": True, "sim_time_s": 1.0,
            }) + "\n")
            (evidence / "gripper-contract.json").write_text(json.dumps({
                "grippers": {"right": {"joint_names": ["right_finger"], "opened_positions": [0.]}},
            }))
            self.assertTrue(audit(run, run / "raw", frames)["passed"])
            frames[-1]["robot"]["joints"]["joint_position"] = [.5]
            self.assertFalse(audit(run, run / "raw", frames)["passed"])
