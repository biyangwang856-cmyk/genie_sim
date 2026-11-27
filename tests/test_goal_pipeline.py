"""Regression checks for the public skill-gap and declared-goal boundaries."""
import unittest
from unittest.mock import Mock, patch

from genie_extensions import Registry
from goal_composition import PACKING_STAGES
from natural_backend import requested_distinct_scene_count, resolve_intent
from robot_adapters import RobotAdapter, UnsupportedRequest
from skill_registry import validate_plan
from task_geometry import task_spec


PACKING_OBJECTS = (
    "benchmark_phone_000", "benchmark_packaging_box_000", "benchmark_packaging_box_002",
    "benchmark_packaging_box_001",
)


class GoalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.adapter = RobotAdapter(
            "g2crsb90d", ("right",), frozenset({
                "friction_grasp", "curobo", "held_object_motion", "gripper_opening_control",
            }), {},
        )
        self.scene = {
            "task_template": "phone_packaging", "scene_id": "test-scene",
            "official_instruction": {"instruction": "Pack the phone, then its insert and lid"},
            "layout": {name: {"usd": name, "registered": True,
                              "entity_local_bbox": {"min": [0.] * 3, "max": [.04] * 3}}
                       for name in PACKING_OBJECTS},
        }

    def plan(self, spec, indices):
        steps = []

        def add(skill, **params):
            steps.append({"id": str(len(steps)), "skill": skill, "parameters": params})

        for i in indices:
            obj = PACKING_OBJECTS[i]
            add("grasp", object=obj, role="operator", grasp_profile="topdown_box")
            add("move_held", object=obj, role="operator", target=f"packing_{i}_approach")
            add("align_precision", object=obj, role="operator", target=f"packing_{i}_contact")
            add("release", object=obj, role="operator", mode=spec["release_modes"][obj])
            add("retreat", role="operator")
            add("wait_stable", object=obj)
            add("verify_relation", object=obj, relation=f"packing_stage_{i}")
        add("verify_task")
        return {"version": 1, "robot_id": self.adapter.robot_id, "task_id": spec["task_id"],
                "scene_id": spec["scene_id"], "role_bindings": spec["role_bindings"], "steps": steps}

    def test_declared_order_is_enforced_and_normal_packing_uses_all_stages(self):
        normal = task_spec(self.scene, self.adapter.robot_id)
        self.assertEqual(normal["manipulated_objects"], list(PACKING_OBJECTS[:3]))
        validate_plan(self.plan(normal, [0, 1, 2]), normal, self.adapter)

        goal = {"effects": [dict(zip(("relation", "object", "reference"), stage)) for stage in PACKING_STAGES[:2]],
                "preserve": ["lid"]}
        declared = task_spec({**self.scene, "project_packing_goal": goal}, self.adapter.robot_id)
        self.assertEqual(declared["manipulated_objects"], list(PACKING_OBJECTS[:2]))
        self.assertEqual(declared["preserve_objects"], [PACKING_OBJECTS[2]])
        validate_plan(self.plan(declared, [0, 1]), declared, self.adapter)
        with self.assertRaisesRegex(ValueError, "completed in order"):
            validate_plan(self.plan(declared, [1, 0]), declared, self.adapter)

    def test_plan_collects_unknown_and_candidate_skill_together(self):
        spec = task_spec(self.scene, self.adapter.robot_id)
        plan = self.plan(spec, [])
        plan["steps"] = [{"id": "a", "skill": "vacuum_pick", "parameters": {}},
                         {"id": "b", "skill": "open_hinged_lid", "parameters": {}}]
        with self.assertRaises(UnsupportedRequest) as refused:
            validate_plan(plan, spec, self.adapter)
        result = refused.exception.details
        self.assertEqual(result["code"], "SKILL_NOT_REGISTERED")
        self.assertEqual(result["missing_skills"], ["vacuum_pick", "open_hinged_lid"])
        self.assertEqual(result["skill_details"]["vacuum_pick"]["candidate_apis"], [])
        self.assertIn("open", result["skill_details"]["open_hinged_lid"]["candidate_apis"])

    def test_public_registry_preserves_skill_gap_without_allocating_a_job(self):
        def allocated(_job_id):
            self.fail("An unsupported skill must not allocate a native job")

        with (
            patch("natural_backend.load_robot", return_value=self.adapter),
            patch("skill_onboarding.load_robot", return_value=self.adapter),
            self.assertRaises(UnsupportedRequest) as refused,
        ):
            Registry().invoke("isaac_physx", {
                "instruction": "Use a vacuum cup to pick up the item",
                "skill_goal": {"skill": "vacuum_pick"},
                "execute": True,
            }, allocated)
        result = refused.exception.details
        self.assertEqual(result["message"], "Skill is not registered")
        self.assertEqual(result["missing_skills"], ["vacuum_pick"])
        self.assertEqual(result["missing_robot_capabilities"], ["vacuum_actuation"])
        self.assertEqual(result["candidate_apis"], [])

    def test_natural_opening_resolves_asset_binding_after_model_supplies_joint_limit(self):
        instruction = "抓住微波炉门把手，把门打开到开启限位"
        goal = {"skill": "open_hinged_lid", "actor": "benchmark_microwave_oven_003"}
        reply = {"status": "skill_request", "robot": self.adapter.robot_id}
        api = Mock()
        api.plan.side_effect = [
            ({**reply, "skill_goal": {**goal, "target_limit": "max"}}, {}),
            ({**reply, "skill_goal": {**goal, "target_position": "open"}}, {}),
        ]

        plan, scenes, receipts = resolve_intent(api, instruction, [], self.adapter.describe(), None, None)

        self.assertEqual(plan["skill_goal"], goal)
        self.assertIsNone(scenes)
        self.assertEqual(len(receipts), 2)
        self.assertIn("target_position", api.plan.call_args.kwargs["repair"]["error"])

    def test_explicit_scene_count_survives_model_omission_after_repair(self):
        instruction = "我要五条不同场景的机器人把手机装进盒子里的轨迹"
        reply = {"robot": self.adapter.robot_id, "tasks": [{"task_id": "phone_packaging"}],
                 "trajectory_request": {"total_episodes": 5}}
        api = Mock()
        api.plan.side_effect = [(reply, {}), (reply, {})]
        with self.assertRaisesRegex(UnsupportedRequest, "scene_count=5"):
            resolve_intent(api, instruction, [], self.adapter.describe(), None, None)
        self.assertEqual(api.plan.call_count, 2)
        self.assertIn("scene_count=5", api.plan.call_args.kwargs["repair"]["error"])

    def test_repaired_plan_keeps_distinct_scene_count_separate_from_episodes(self):
        instruction = "在两个不同场景中采集五条手机装盒轨迹"
        reply = {"robot": self.adapter.robot_id, "tasks": [{"task_id": "phone_packaging"}],
                 "trajectory_request": {"total_episodes": 5}}
        fixed = {**reply, "scene_request": {"mode": "compose_from_library", "scene_count": 2}}
        api = Mock()
        api.plan.side_effect = [(reply, {}), (fixed, {})]
        plan, scenes, _ = resolve_intent(api, instruction, [], self.adapter.describe(), None, None)
        self.assertEqual(scenes["scene_count"], 2)
        self.assertEqual(plan["trajectory_request"]["total_episodes"], 5)
        self.assertIsNone(requested_distinct_scene_count("在同一场景采五条手机装盒轨迹"))
