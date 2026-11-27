"""Regressions in the public request and immutable scene identity."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from genie_extensions import Registry
from natural_backend import resolve_intent
from robot_adapters import UnsupportedRequest
from scene_contract import SCENE_FILES, choose_replacement, scene_request, verify_manifest, write_manifest


@unittest.skipIf(os.name == 'nt', 'Recording admission uses a Linux file lock')
class RecordingAdmissionReceiptTests(unittest.TestCase):
    def test_rechecking_a_sealed_recording_preserves_its_original_receipt(self):
        from batch_collect import audit_recording

        with TemporaryDirectory() as root:
            run = Path(root)
            original = b'{"accepted": true, "audit_revision": "previous"}'
            launch = b'{"robot_id": "g2crsb90d"}'
            (run / 'acceptance.json').write_bytes(original)
            (run / 'launch.json').write_bytes(launch)
            (run / 'collection_result.json').write_text('{"success": true}')
            (run / 'automatic-admission.json').write_text(json.dumps({
                'passed': True,
                'acceptance_sha256': hashlib.sha256(original).hexdigest(),
                'launch_sha256': hashlib.sha256(launch).hexdigest(),
            }))
            with patch('batch_collect.AUDIT_REVISION', 'current'), \
                    patch('batch_collect.archive', return_value={'archived': True}), \
                    patch('batch_collect.ensure_aligned', return_value=None), \
                    patch('batch_collect.audit', return_value={'accepted': True}) as audit:
                _, result = audit_recording(run)
                self.assertEqual(result['audit_revision'], 'current')
                self.assertEqual((run / 'acceptance.json').read_bytes(), original)
                self.assertEqual(json.loads(
                    (run / 'evidence/acceptance-recheck-current.json').read_text()), result)
                self.assertEqual(audit_recording(run)[1], result)
                audit.assert_called_once()


class CollectionSlotTests(unittest.TestCase):
    def test_slots_on_the_same_gpu_use_different_endpoints_and_ros_domains(self):
        from collection_control import COLLECTION_SLOTS, execution_profile

        self.assertEqual(len(COLLECTION_SLOTS), 4)
        self.assertEqual(len({slot.grpc_host for slot in COLLECTION_SLOTS.values()}), 4)
        self.assertEqual(len({slot.ros_domain for slot in COLLECTION_SLOTS.values()}), 4)
        self.assertEqual(COLLECTION_SLOTS['gpu0-extra'].gpu_uuid, COLLECTION_SLOTS['secondary'].gpu_uuid)
        self.assertEqual(COLLECTION_SLOTS['gpu1-extra'].gpu_uuid, COLLECTION_SLOTS['primary'].gpu_uuid)
        self.assertNotIn('collection_slot', execution_profile())

    def test_extra_workers_reserve_memory_during_the_other_slot_startup(self):
        from collection_control import (
            COLLECTION_SLOTS, EXTRA_MEMORY_BYTES, GIB, HOST_RESERVE_BYTES, CollectionDeferred,
            check_extra_capacity,
        )

        with patch('collection_control.read_host_available', return_value=EXTRA_MEMORY_BYTES + HOST_RESERVE_BYTES), \
                patch('collection_control.running_collections', side_effect=[['genie-by-new'], []]), \
                patch('collection_control.container_working_set', return_value={'genie-by-new': GIB}), \
                patch('collection_control.read_gpu_available', return_value=40 * GIB):
            with self.assertRaises(CollectionDeferred) as defer:
                check_extra_capacity({}, COLLECTION_SLOTS['gpu1-extra'])
            self.assertEqual(defer.exception.status, 'waiting_for_host_memory')
            self.assertEqual(defer.exception.details['starting_worker_reserve_bytes'], EXTRA_MEMORY_BYTES - GIB)

        with patch('collection_control.read_host_available', return_value=2 * EXTRA_MEMORY_BYTES + HOST_RESERVE_BYTES), \
                patch('collection_control.running_collections', return_value=[]), \
                patch('collection_control.read_full_memory_stall', return_value=0.1), \
                patch('collection_control.read_gpu_available', return_value=GIB):
            with self.assertRaises(CollectionDeferred) as defer:
                check_extra_capacity({}, COLLECTION_SLOTS['gpu0-extra'])
            self.assertEqual(defer.exception.status, 'waiting_for_gpu_memory')

    def test_extra_slot_waits_through_sustained_host_memory_stalls(self):
        from collection_control import COLLECTION_SLOTS, GIB, CollectionDeferred, check_extra_capacity

        with patch('collection_control.read_host_available', return_value=30 * GIB), \
                patch('collection_control.running_collections', return_value=[]), \
                patch('collection_control.read_full_memory_stall', return_value=16.66), \
                patch('collection_control.read_gpu_available') as read_gpu:
            with self.assertRaises(CollectionDeferred) as defer:
                check_extra_capacity({}, COLLECTION_SLOTS['gpu1-extra'])
        self.assertEqual(defer.exception.status, 'waiting_for_host_memory_pressure')
        self.assertEqual(defer.exception.details['full_stall_avg10_pct'], 16.66)
        read_gpu.assert_not_called()

    def test_extra_slot_has_only_episode_local_writable_host_mounts(self):
        from collection_control import STORAGE, runtime_mounts

        run = STORAGE / 'collection-runs' / 'test-extra'
        mounts = runtime_mounts(run, private_recording=True, private_storage=True)
        self.assertEqual(
            {a for a, _, mode in mounts if mode == 'rw'},
            {run, run / 'isolated-runtime' / 'recording_data',
             run / 'isolated-runtime' / 'config' / 'robot_cfg'},
        )
        self.assertIn((run / 'isolated-runtime' / 'config' / 'robot_cfg',
                       '/workspace/config/robot_cfg', 'rw'), mounts)
        self.assertIn((STORAGE / 'runtime-collection', '/workspace', 'ro'), mounts)
        self.assertIn((run / 'source', str(run / 'source'), 'ro'), mounts)

    def test_extra_slot_stages_clean_lula_config_for_each_run(self):
        from collection_control import stage_runtime_robot_config

        with TemporaryDirectory() as tmp, patch('collection_control.STORAGE', Path(tmp)):
            storage = Path(tmp)
            original = storage / 'runtime-collection' / 'config' / 'robot_cfg'
            (original / 'G2').mkdir(parents=True)
            name = 'G2_crsB_90D_left'
            config_file = f'G2/{name}.yaml'
            template = b'cspace_to_urdf_rules: []\n'
            (original / config_file).write_bytes(template)
            temp_files = [f'G2/{name}_tmp.yaml', f'G2/{name}_tmp_rotated.yaml']
            for path in temp_files:
                (original / path).write_text('previous run')

            receipt = stage_runtime_robot_config(storage / 'collection-runs' / 'first')
            second = stage_runtime_robot_config(storage / 'collection-runs' / 'second')
            working = Path(receipt['working_directory'])
            self.assertEqual(receipt['source_directory'], str(original))
            self.assertEqual(receipt['container_directory'], '/workspace/config/robot_cfg')
            self.assertEqual(receipt['template_file_sha256'], {config_file: hashlib.sha256(template).hexdigest()})
            self.assertEqual((working / config_file).read_bytes(), template)
            for path in temp_files:
                self.assertFalse((working / path).exists())
                (working / path).write_text('current run')
                self.assertEqual((original / path).read_text(), 'previous run')
                self.assertFalse((Path(second['working_directory']) / path).exists())


class GraspCandidatePriorityTests(unittest.TestCase):
    def test_measured_pen_grasp_priority_matches_exact_scene_and_arm(self):
        from skill_executor import SkillExecutor

        robot_profile = Path(__file__).resolve().parents[1] / 'pipeline/robots/g2crsb90d.json'
        config = json.loads(robot_profile.read_text())
        table = 'scene-98387ba3d89d9f3d'
        clutter = 'scene-f10de168a07f6e10'
        preferred = {'vertical_offsets_m': [-.002], 'yaw_offsets_rad': [.12]}
        pen_backup = config['grasp_candidate_recovery']['pen_barrel'][0]
        cap_backup = config['grasp_candidate_recovery']['cap_body'][0]
        legacy_cap = config['custom_grasp_candidates']['cap_body']['pose_options']
        cases = [
            (table, 'right', 'pen_barrel', [preferred, None, pen_backup]),
            (table, 'left', 'pen_barrel', [None, pen_backup]),
            (clutter, 'right', 'pen_barrel', [None, pen_backup]),
            (clutter, 'left', 'cap_body', [legacy_cap, cap_backup]),
        ]
        for scene_id, arm, profile, candidates in cases:
            with self.subTest(scene=scene_id, arm=arm, profile=profile):
                executor = SkillExecutor.__new__(SkillExecutor)
                executor.spec = {'scene_id': scene_id, 'template': 'put_the_cap_on_the_pen_new',
                                 'support': {'height_m': .75}, 'target_objects': {}}
                executor.profile_scene_id = '0'
                executor.adapter = SimpleNamespace(config=config)
                executor.robot = Mock()
                executor.robot.get_ee_pose.return_value = np.eye(4)
                executor.ops = Mock()
                pose = np.eye(4)
                command_offset = 0
                results = iter([False] * (len(candidates) - 1) + [True])
                executor.ops.local_box.return_value = (np.zeros(3), np.zeros(3))
                executor.ops.pose.side_effect = lambda *args: pose.copy()

                def pick(*args, **kwargs):
                    nonlocal command_offset
                    result = next(results)
                    if result:
                        command_offset += 100
                        pose[2, 3] += .10
                    return result

                executor.ops.pick.side_effect = pick
                executor.holding, executor.held_transforms, executor.facts = {}, {}, set()
                executor.arm = lambda role: arm
                executor.event = Mock()
                executor.folder, executor.started, executor.max_wall_seconds = Path.cwd(), time.monotonic(), 20

                with patch('skill_recovery.native_command_offset', side_effect=lambda _: command_offset):
                    executor.grasp({'object': 'payload', 'role': 'operator', 'grasp_profile': profile})

                calls = executor.ops.pick.call_args_list
                self.assertEqual([row.kwargs.get('custom_pose_options') for row in calls], candidates)
                if profile == 'pen_barrel':
                    self.assertTrue(all(row.kwargs['offset'] == [.005, 0, 0] for row in calls))
                self.assertEqual(executor.holding, {'operator': 'payload'})
                executor.robot.client.playback.assert_not_called()


class CapPreinsertRecoveryTests(unittest.TestCase):
    def test_axis_only_retry_can_use_the_measured_clutter_precontact_alignment(self):
        from skill_executor import SkillExecutor, SkillFailure

        executor = SkillExecutor.__new__(SkillExecutor)
        executor.spec = {'support': {'height_m': .74},
                         'assembly': {'bore_half_width_m': .00919, 'tip_half_width_m': .00451,
                                      'transition_start_depth_m': .02336}}
        executor.robot = Mock()
        executor.ops = Mock()
        executor.event = Mock()
        executor.held_transforms = {'assistant': np.eye(4)}
        executor.arm = lambda role: 'left'
        cap = np.eye(4)
        cap[:3, :3] = Rotation.from_rotvec([.012294761336152243, 0, 0]).as_matrix()
        executor.ops.pose.side_effect = lambda robot, obj: cap.copy() if obj == 'cap' else np.eye(4)
        executor.robot.get_ee_pose.return_value = cap.copy()
        # The exact s06 state reproduced by two clutter attempts on the same scene.
        mouth = {'depth_m': -.013800709714340886, 'radial_m': .002746290753946546,
                 'axis_sine': .011022328276939409, 'rotation_rad': .012294761336152243}
        executor.cap_mouth_measurement = Mock(return_value=mouth.copy())
        failed = SkillFailure('native_action_failed_before_rollback', {'stored_state': 'stage_11'})
        executor.follow_local_pose = Mock(side_effect=[failed, None])
        params = {'role': 'assistant', 'object': 'cap', 'reference': 'pen'}
        depth = -.009493333333333333
        executor.follow_cap_axial_step(params, np.eye(4), depth)

        retries = executor.follow_local_pose.call_args_list
        self.assertEqual([call.args[2] for call in retries],
                         ['cap_axial_waypoint_motion', 'cap_axis_only_approach_waypoints'])
        retry_goal = retries[1].args[1]
        np.testing.assert_allclose(retry_goal[:3, :3], cap[:3, :3])
        np.testing.assert_allclose(retry_goal[:3, 3] - cap[:3, 3], [depth - mouth['depth_m'], 0, 0])

        executor.cap_mouth_measurement.return_value = {**mouth, 'depth_m': -.004}
        self.assertIsNone(executor.cap_axis_only_candidate(params, -.002))
        self.assertIsNone(executor.cap_axis_only_candidate(params, .001, contact_retry=True))

    def test_axis_retry_remains_available_just_before_entry(self):
        from skill_executor import SkillExecutor, SkillFailure

        executor = SkillExecutor.__new__(SkillExecutor)
        executor.spec = {'support': {'height_m': .74}}
        executor.event = Mock()
        executor.ops = Mock()
        executor.robot = Mock()
        executor.arm = lambda role: 'left'
        executor.held_transforms = {'assistant': np.eye(4)}
        executor.ops.pose.side_effect = lambda robot, obj: np.eye(4)
        executor.robot.get_ee_pose.return_value = np.eye(4)
        executor.cap_mouth_measurement = Mock(return_value={
            'depth_m': -.00546, 'axis_sine': .005, 'rotation_rad': .010})
        executor.cap_radial_clearance = Mock(return_value=.00395)
        failure = SkillFailure('native_action_failed_before_rollback', {})
        executor.follow_local_pose = Mock(side_effect=[failure, None])
        params = {'object': 'cap', 'reference': 'pen', 'role': 'assistant'}

        executor.follow_cap_axial_step(params, np.eye(4), -.00129)
        candidate = executor.follow_local_pose.call_args.args[1]
        self.assertAlmostEqual(candidate[0, 3], .00417)
        self.assertEqual(executor.follow_local_pose.call_args.args[2], 'cap_axis_only_approach_waypoints')

    def test_contact_retry_can_use_the_pen_arm_with_the_original_axis_and_clearance(self):
        from skill_executor import SkillExecutor, SkillFailure

        executor = SkillExecutor.__new__(SkillExecutor)
        pen, cap = np.eye(4), np.eye(4)
        cap[0, 3] = -.00105
        executor.spec = {'support': {'height_m': .74}}
        executor.adapter = Mock(config={'held_transport_fixture_margin_m': .0005})
        executor.robot = Mock()
        executor.ops = Mock()
        executor.event = Mock()
        executor.holding = {'operator': 'pen', 'assistant': 'cap'}
        executor.held_transforms = {'operator': np.eye(4), 'assistant': np.eye(4)}
        executor.arm = lambda role: 'right' if role == 'operator' else 'left'
        executor.stable_pen_for_alignment = lambda: pen.copy()
        executor.ops.pose.side_effect = lambda robot, obj: (cap if obj == 'cap' else pen).copy()
        executor.robot.get_ee_pose.side_effect = lambda arm: (cap if arm == 'left' else pen).copy()
        executor.cap_mouth_measurement = lambda pose: {'depth_m': float(pose[0, 3] - pen[0, 3]),
                                                      'axis_sine': .004, 'rotation_rad': .005}
        executor.cap_radial_clearance = Mock(return_value=.003)

        def follow(params, goal, phase):
            if params['role'] == 'assistant':
                raise SkillFailure('native_action_failed_before_rollback', {'stored_state': 'stage_31'})
            self.assertEqual(phase, 'pen_reciprocal_contact_waypoints')
            self.assertAlmostEqual(goal[0, 3] - pen[0, 3], -.0012)
            np.testing.assert_allclose(goal[:3, :3], np.eye(3))
            pen[:] = goal

        executor.follow_local_pose = Mock(side_effect=follow)
        params = {'role': 'assistant', 'object': 'cap', 'reference': 'pen'}
        executor.follow_cap_axial_step(params, cap.copy(), .00281)
        self.assertEqual(executor.follow_local_pose.call_count, 3)
        self.assertGreater(executor.cap_mouth_measurement(cap)['depth_m'], 0.)
        executor.robot.client.playback.assert_not_called()

    def test_local_ik_retry_anchors_optimizer_and_restores_native_seed_settings(self):
        from project_runtime_hooks import solve_from_local_joint_seed

        config = SimpleNamespace(num_seeds=32, use_lm_seed=True, exit_early=True)
        solver = SimpleNamespace(config=config)
        current = Mock()
        anchored = current.position.unsqueeze.return_value.expand.return_value.contiguous.return_value
        options = {'current_state': current, 'return_seeds': 10}

        def check(goal, **kwargs):
            self.assertEqual(goal, 'actual waypoint')
            self.assertIs(kwargs['seed_config'], anchored)
            self.assertFalse(config.use_lm_seed)
            self.assertFalse(config.exit_early)
            raise RuntimeError('native planner did not finish')

        with self.assertRaisesRegex(RuntimeError, 'did not finish'):
            solve_from_local_joint_seed(solver, check, ('actual waypoint',), options)
        current.position.unsqueeze.assert_called_once_with(1)
        current.position.unsqueeze.return_value.expand.assert_called_once_with(-1, 32, -1)
        self.assertEqual(options.keys(), {'current_state', 'return_seeds'})
        self.assertTrue(config.use_lm_seed)
        self.assertTrue(config.exit_early)

    def test_presentation_can_change_wrist_roll_without_changing_assembly_axis(self):
        from skill_executor import SkillExecutor
        from task_geometry import Geometry

        executor = SkillExecutor.__new__(SkillExecutor)
        executor.spec = {'support': {'height_m': .74}, 'bindings': {'cap': 'cap'},
                         'assembly': {'pen_tip_x_m': -.07802, 'cap_mouth_x_m': .02127,
                                      'preinsert_gap_m': .03}}
        geometry = Geometry.__new__(Geometry)
        geometry.spec = executor.spec
        executor.geometry = geometry
        executor.holding = {'operator': 'pen', 'assistant': 'cap'}
        executor.stable_pen_for_alignment = Mock(return_value=np.eye(4))
        executor.pen_presentation_candidates = Mock(return_value=np.eye(4)[None])
        executor.robot = Mock()
        executor.arm = lambda role: 'right' if role == 'operator' else 'left'
        executor.ops = Mock()
        executor.event = Mock()
        goal = np.eye(4)
        poses = []

        def ik(role, obj, candidates):
            poses.append(candidates.copy())
            return np.arange(len(candidates)) == 1

        def move(robot, name, arm, targets, phase):
            if phase == 'plan_pen_present':
                return False
            executor.stable_pen_for_alignment.return_value = targets[0].copy()
            return True

        executor.held_goal_mask = Mock(side_effect=ik)
        executor.ops.move.side_effect = move
        executor.move_pen_presentation({'role': 'operator', 'object': 'pen'}, goal)
        self.assertEqual(len(poses), 2)
        for pen, cap in zip(*poses, strict=True):
            np.testing.assert_allclose(pen[:3, 0], goal[:3, 0])
            np.testing.assert_allclose(cap, geometry.cap_preinsert_pose(pen))
        self.assertFalse(np.allclose(poses[0][1, :3, :3], np.eye(3)))
        executor.robot.client.playback.assert_not_called()

    def test_axial_staging_reuses_collision_checked_move_before_contact(self):
        from skill_executor import SkillExecutor, SkillFailure

        executor = SkillExecutor.__new__(SkillExecutor)
        cap, pen = np.eye(4), np.eye(4)
        cap[0, 3] = -.03115
        executor.spec = {'support': {'height_m': .74}}
        executor.adapter = Mock(config={'held_transport_fixture_margin_m': .0005})
        executor.holding = {'assistant': 'cap', 'operator': 'pen'}
        executor.held_transforms = {'assistant': np.eye(4)}
        executor.arm = lambda role: 'left'
        executor.robot = Mock()
        executor.robot.get_ee_pose.side_effect = lambda arm: cap.copy()
        executor.ops = Mock()
        executor.ops.pose.side_effect = lambda robot, obj: (cap if obj == 'cap' else pen).copy()
        executor.cap_mouth_measurement = lambda pose: {'depth_m': float(pose[0, 3]), 'axis_sine': .004,
                                                      'rotation_rad': .005}
        executor.cap_radial_clearance = lambda mouth, depth: .003
        executor.stable_pen_for_alignment = Mock(return_value=pen.copy())
        executor.event = Mock()
        executor.follow_local_pose = Mock(side_effect=SkillFailure(
            'native_action_failed_before_rollback', {'stored_state': 'stage_10'}))

        def guarded_move(params, target, phase, fixture, margin, slowdown):
            self.assertEqual(fixture, 'pen')
            self.assertEqual(margin, .0005)
            np.testing.assert_allclose(target[:3, :3], np.eye(3))
            cap[:] = target
            return True

        executor.move_held_clear_of_fixture = Mock(side_effect=guarded_move)
        params = {'role': 'assistant', 'object': 'cap', 'reference': 'pen'}
        target = cap.copy()
        target[0, 3] = -.0259
        executor.follow_cap_axial_step(params, target, -.0259)
        self.assertEqual(executor.follow_local_pose.call_count, 2)
        executor.move_held_clear_of_fixture.assert_called_once()
        self.assertAlmostEqual(cap[0, 3], -.0259)
        self.assertFalse(executor.advance_cap_before_contact(params, .004))

        cap[0, 3] = -.03115
        executor.stable_pen_for_alignment.side_effect = lambda: pen.copy()
        executor.cap_mouth_measurement = lambda pose: {'depth_m': float(pose[0, 3] - pen[0, 3]),
                                                      'axis_sine': .004, 'rotation_rad': .005}

        def reciprocal_move(params, target, phase, fixture, margin, slowdown):
            if params['object'] == 'cap':
                raise SkillFailure('native_action_failed_before_rollback', {'stored_state': 'stage_11'})
            self.assertEqual(fixture, 'cap')
            self.assertEqual(params['role'], 'operator')
            pen[:] = target
            return True

        executor.move_held_clear_of_fixture.reset_mock(side_effect=True)
        executor.move_held_clear_of_fixture.side_effect = reciprocal_move
        executor.follow_cap_axial_step(params, target, -.0259)
        self.assertEqual(executor.move_held_clear_of_fixture.call_count, 2)
        self.assertAlmostEqual(cap[0, 3] - pen[0, 3], -.0259)

    def test_native_solver_keeps_searching_after_wrist_invalid_first_path(self):
        from project_runtime_hooks import G2_COUPLED_WRIST_PAIRS, g2_trajectory_candidate_mask

        joints = [joint for pair in G2_COUPLED_WRIST_PAIRS for joint in pair]
        first = np.array([[[[.339, -1.18, .339, 1.181], [-.619, 1.371, .339, 1.181]]]])
        second = first.copy()
        second[0, 0, 1, 1] = 1.24
        retained = None
        for candidate in (first, second):
            if g2_trajectory_candidate_mask(joints, candidate, np.ones((1, 1), dtype=bool)).any():
                retained = candidate
                break
        self.assertIs(retained, second)
        self.assertFalse(g2_trajectory_candidate_mask(joints, second, np.zeros((1, 1), dtype=bool)).any())

    def test_rejected_outboard_translation_uses_a_new_staging_pose(self):
        from skill_executor import SkillExecutor, SkillFailure
        from task_geometry import Geometry

        pen = np.array([[-.234, 0., -.972, -.081], [-.972, 0., .234, -.034],
                        [0., 1., 0., .988], [0., 0., 0., 1.]])
        cap = np.eye(4)
        cap[:3, 3] = [-.204, .130, .849]
        objects = {'pen': pen, 'cap': cap}
        geometry = Geometry.__new__(Geometry)
        geometry.spec = {'assembly': {'pen_tip_x_m': -.07802, 'cap_mouth_x_m': .02127,
                                      'preinsert_gap_m': .03}}
        goal = geometry.cap_preinsert_pose(pen)
        executor = SkillExecutor.__new__(SkillExecutor)
        executor.spec = {'support': {'height_m': .74}, 'bindings': {'pen': 'pen', 'cap': 'cap'},
                         'manipulated_objects': ['pen', 'cap']}
        executor.pen_presentation_pose = pen.copy()
        executor.holding = {'operator': 'pen', 'assistant': 'cap'}
        executor.held_transforms = {'operator': np.eye(4), 'assistant': np.eye(4)}
        executor.facts = set()
        executor.robot = Mock()
        executor.robot.get_ee_pose.side_effect = lambda arm: objects['cap' if arm == 'left' else 'pen'].copy()
        executor.robot.robot.solve_ik.side_effect = lambda poses, **kw: (np.ones(len(poses), dtype=bool), None)
        executor.arm = lambda role: {'operator': 'right', 'assistant': 'left'}[role]
        executor.geometry = Mock()
        executor.geometry.target.return_value = goal, []
        executor.ops = Mock()
        executor.ops.pose.side_effect = lambda robot, name: objects[name].copy()
        executor.event = Mock()
        executor.recover_cap_preinsert = Mock()
        executor.stable_pen_for_alignment = Mock(return_value=pen.copy())
        attempted = []

        def move(robot, obj, arm, target, description, **kwargs):
            attempted.append(target.copy())
            if len(attempted) == 1:
                raise SkillFailure('native_action_failed_before_rollback', {'stored_state': 'stage_8'})
            objects['cap'] = target.copy()
            return True

        executor.ops.move.side_effect = move
        params = {'role': 'assistant', 'object': 'cap', 'target': 'cap_preinsert'}
        try:
            executor.move_cap_preinsert(params, goal)
        except SkillFailure as exc:
            self.fail(f'Staging stopped at the rejected first target: {exc.kind}')
        self.assertEqual(len(attempted), 2)
        self.assertGreater(np.linalg.norm(attempted[0][:3, 3] - attempted[1][:3, 3]), .01)
        np.testing.assert_allclose(attempted[1][:3, :3], np.eye(3))
        self.assertLess((attempted[1][:3, 3] - goal[:3, 3]) @ goal[:3, 0], -.02)
        executor.recover_cap_preinsert.assert_called_once_with(params)
        executor.robot.client.playback.assert_not_called()

    def test_candidate_filter_rejects_either_wrist_and_nonfinite_values(self):
        from project_runtime_hooks import G2_COUPLED_WRIST_PAIRS, g2_coupled_wrist_mask

        names = [joint for pair in G2_COUPLED_WRIST_PAIRS for joint in pair]
        # Measured left and right wrist positions at the V20 failed staging pose.
        positions = np.array([[[.3386, -1.1805, .3393, 1.1809],
                               [.789, -1.2523, 1.0095, 1.3391]],
                              [[.34, -1.10, .9, 1.2],
                               [.34, -1.10, .9, .8]],
                              [[np.nan, .1, .3, .4],
                               [.35, -1.1, .35, 1.1]]])
        np.testing.assert_array_equal(g2_coupled_wrist_mask(names, positions).all(axis=-1),
                                      [False, False, False])
        np.testing.assert_array_equal(g2_coupled_wrist_mask(names, positions[:, 1]),
                                      [False, True, True])

    def test_grasp_candidate_retry_requires_unchanged_live_pose(self):
        from skill_executor import SkillExecutor, SkillFailure

        for first_failure in ('refused', 'moved_without_command', 'false_after_command'):
            with self.subTest(first_failure=first_failure):
                executor = SkillExecutor.__new__(SkillExecutor)
                executor.arm = lambda role: 'right'
                executor.robot = Mock()
                executor.robot.get_ee_pose.return_value = np.eye(4)
                executor.event = Mock()
                executor.ops = Mock()
                executor.ops.local_box.return_value = (np.zeros(3), np.zeros(3))
                executor.holding = {}
                executor.folder, executor.started, executor.max_wall_seconds = Path.cwd(), time.monotonic(), 20
                object_pose = np.eye(4)
                command_offset = 0
                executor.ops.pose.side_effect = lambda robot, obj: object_pose.copy()
                tried = []

                def pick(robot, obj, arm, **options):
                    nonlocal command_offset
                    tried.append(options)
                    if len(tried) == 1:
                        if first_failure == 'moved_without_command':
                            object_pose[2, 3] += .004
                        if first_failure == 'false_after_command':
                            command_offset += 100
                            return False
                        raise SkillFailure('native_action_failed_before_rollback', {})
                    command_offset += 100
                    object_pose[2, 3] += .10
                    return True

                executor.ops.pick.side_effect = pick
                params = {'object': 'cap', 'role': 'assistant', 'grasp_profile': 'cap_body'}
                alternate = {'vertical_offsets_m': [0.], 'yaw_offsets_rad': [.24]}
                with patch('skill_recovery.native_command_offset', side_effect=lambda _: command_offset):
                    if first_failure == 'refused':
                        executor.try_grasp_candidates(params, {}, [alternate])
                        self.assertEqual(tried[-1], {'custom_pose_options': alternate})
                    else:
                        with self.assertRaisesRegex(SkillFailure, 'grasp_candidate_changed_world'):
                            executor.try_grasp_candidates(params, {}, [alternate])
                        self.assertEqual(len(tried), 1)
                executor.robot.client.playback.assert_not_called()

    def test_replans_against_settled_tilted_pen_with_no_reset(self):
        from skill_executor import SkillExecutor, SkillFailure
        from task_geometry import Geometry

        # Reproduce a stable 18.6-degree pen tilt at the docking bottleneck.
        anchor = np.eye(4)
        anchor[:3, :3] = [[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]]
        anchor[:3, 3] = [-.082, -.020, 1.016]
        pen = anchor.copy()
        pen[:3, :3] = Rotation.from_rotvec([.325, 0., 0.]).as_matrix() @ anchor[:3, :3]
        cap = anchor.copy()
        cap[:3, 3] = [-.083, .151, 1.052]
        geometry = Geometry.__new__(Geometry)
        geometry.spec = {'assembly': {'pen_tip_x_m': -.1, 'cap_mouth_x_m': .04,
                                      'preinsert_gap_m': .008}}
        live_target = geometry.cap_preinsert_pose(pen)
        objects = {'pen': pen, 'cap': cap}
        executor = SkillExecutor.__new__(SkillExecutor)
        executor.spec = {'support': {}, 'bindings': {'pen': 'pen', 'cap': 'cap'},
                         'manipulated_objects': ['pen', 'cap']}
        executor.pen_presentation_pose = anchor
        executor.holding = {'operator': 'pen', 'assistant': 'cap'}
        executor.held_transforms = {'operator': pen.copy(), 'assistant': cap.copy()}
        executor.facts = set()
        executor.robot = Mock()
        executor.robot.get_ee_pose.return_value = np.eye(4)
        executor.robot.robot.solve_ik.return_value = ([False, False, True], None)
        executor.arm = lambda role: {'operator': 'right', 'assistant': 'left'}[role]
        executor.adapter = Mock(config={})
        executor.geometry = geometry
        executor.ops = Mock()
        executor.ops.pose.side_effect = lambda robot, name: objects[name].copy()
        executor.ops.move.side_effect = AssertionError('the old pen pose is unreachable')
        executor.event = Mock()
        commands = []

        def motion(params, target, phase, **options):
            commands.append((target, phase))
            if len(commands) == 1:
                raise SkillFailure('local_waypoint_plan_or_tracking_failed', {'phase': phase})
            objects['cap'] = target.copy()

        executor.follow_local_pose = motion
        executor.move_cap_preinsert({'role': 'assistant', 'object': 'cap', 'target': 'cap_preinsert'},
                                    geometry.cap_preinsert_pose(anchor))
        self.assertEqual(len(commands), 2)
        self.assertNotEqual(commands[0][1], commands[1][1])
        np.testing.assert_allclose(commands[1][0], live_target)
        executor.ops.move.assert_not_called()
        executor.robot.client.playback.assert_not_called()

    def test_preinsert_reuses_held_transport_clearance_when_local_branch_is_blocked(self):
        from skill_executor import SkillExecutor, SkillFailure

        executor = SkillExecutor.__new__(SkillExecutor)
        executor.spec = {'bindings': {'pen': 'pen', 'cap': 'cap'}}
        executor.holding = {'assistant': 'cap'}
        executor.held_transforms = {'assistant': np.eye(4)}
        executor.adapter = Mock(config={'held_transport_fixture_margin_m': .0005})
        executor.robot = Mock()
        executor.ops = Mock()
        executor.geometry = Mock()
        executor.event = Mock()
        executor.arm = lambda role: 'left'
        executor.stable_pen_for_alignment = Mock(return_value=np.eye(4))
        executor.connect_cap_outboard = Mock()
        executor.reposition_pen_for_alignment = Mock()
        executor.follow_local_pose = Mock(side_effect=SkillFailure(
            'local_waypoint_plan_or_tracking_failed', {'planner_accepted': False}))
        tool_pose = np.eye(4)
        goal = np.eye(4)
        goal[0, 3] = -.13
        executor.geometry.cap_preinsert_pose.return_value = goal
        executor.ops.pose.side_effect = lambda robot, obj: tool_pose.copy()
        executor.robot.get_ee_pose.side_effect = lambda arm: tool_pose.copy()

        def move(params, target, phase, fixture, margin, slowdown):
            self.assertEqual(fixture, 'pen')
            self.assertEqual(margin, .0005)
            np.testing.assert_allclose(target, goal)
            tool_pose[:] = target
            return True

        executor.move_held_clear_of_fixture = Mock(side_effect=move)
        executor.recover_cap_preinsert({'role': 'assistant', 'object': 'cap'})
        executor.move_held_clear_of_fixture.assert_called_once()
        executor.reposition_pen_for_alignment.assert_not_called()
        executor.robot.client.playback.assert_not_called()


class CaptureReextractionTests(unittest.TestCase):
    def test_render_boundary_audit_requires_every_camera_at_every_retained_frame(self):
        from audit_episode import camera_render_clock_passed

        stamps = [2.1833, 2.2167]
        topics = tuple(f'/camera{index}_compressed' for index in range(3))
        rows = [{'topic': topic, 'header_sim_time_s': t, 'world_sim_time_after_render_s': t + 1/120,
                 'physics_dt_s': 1/120} for t in stamps for topic in topics]
        self.assertTrue(camera_render_clock_passed(rows, stamps, topics))
        self.assertFalse(camera_render_clock_passed(rows[:-1], stamps, topics))
        rows[-1]['world_sim_time_after_render_s'] += 4/30
        self.assertFalse(camera_render_clock_passed(rows, stamps, topics))

    def test_camera_defers_capture_clock_until_matching_native_render(self):
        from project_runtime_hooks import camera_stamp_for_render

        requested = 85.041666667
        self.assertEqual(camera_stamp_for_render(requested, requested + 1/120, 1/120), requested)
        self.assertEqual(camera_stamp_for_render(requested, requested, 1/120), requested)
        with self.assertRaisesRegex(ValueError, 'matching physical capture'):
            camera_stamp_for_render(requested, requested + 4/30, 1/120)

    def test_native_reader_includes_exact_original_messages_at_both_window_ends(self):
        import ast
        from reextract_recording import use_rounded_header_bounds

        native_window = 'def read(current_time, t_start, t_end):\n' \
                        '    if current_time < t_start or current_time > t_end:\n' \
                        '        return False\n' \
                        '    return True\n'
        source = ast.parse(native_window)
        self.assertEqual(use_rounded_header_bounds(source), 1)
        namespace = {}
        exec(compile(ast.fix_missing_locations(source), '<capture extractor>', 'exec'), namespace)
        read = namespace['read']
        self.assertTrue(read(2.416666667, 2.4167, 2.5833))
        self.assertTrue(read(2.583333333, 2.4167, 2.5833))
        self.assertFalse(read(2.383333333, 2.4167, 2.5833))
        self.assertFalse(read(2.616666667, 2.4167, 2.5833))

    def test_idle_cut_uses_the_same_exact_header_normalization_as_audit(self):
        from reextract_recording import retained_capture_times

        previous = 2.383333333
        first_complete = 2.416666667
        following = 2.45
        retained = retained_capture_times([previous, first_complete, following], 2.4167)
        self.assertEqual(retained, [first_complete, following])


class ShapeGraspRecoveryTests(unittest.TestCase):
    def test_scene_seed_changes_right_block_route_leading_lift(self):
        from task_geometry import Geometry

        spec = {'template': 'insert_building_block', 'scene_id': 'composed',
                'profile_scene_id': '1', 'support': {'height_m': .74}, 'manipulated_objects': []}
        def start_height(seed):
            with patch.dict(os.environ, {'GENIE_EPISODE_SEED': str(seed)}):
                geometry = Geometry(Mock(), spec, Mock(), Mock())
            self.assertEqual(geometry.shape_approach_extra_m, 0.)
            return geometry.shape_transport_lift_m

        first = start_height(391017799)
        second = start_height(1445587017)
        self.assertNotEqual(first, second)
        self.assertEqual(first, start_height(391017799))
        self.assertTrue(.008 <= first <= .016 and .008 <= second <= .016)

    def test_shape_entry_recovers_inside_existing_drop_limit(self):
        from skill_executor import SkillExecutor, SkillFailure

        executor = SkillExecutor.__new__(SkillExecutor)
        executor.geometry = SimpleNamespace(cache={})
        executor.arm = lambda role: 'left'
        executor.shape_grasp_profile = Mock(return_value={'maximum_release_gap_m': .035})
        executor.ops = Mock()
        executor.robot = Mock()
        executor.event = Mock()
        actual = np.eye(4)
        executor.ops.pose.side_effect = lambda robot, name: actual.copy()
        executor.held_goal_mask = Mock(return_value=np.array([True, True]))

        def move(robot, name, arm, target, phase):
            if phase == 'plan_shape_entry':
                raise SkillFailure('native_action_failed_before_rollback', {'stored_state': 'stage_3'})
            np.testing.assert_allclose(target[:, 2, 3], [.006, .012])
            actual[:] = target[1]
            return True

        executor.ops.move.side_effect = move
        metrics = {'target_workpiece_clearance_m': .018, 'target_minimum_clearance_m': .008}
        executor.move_shape_entry({'object': 'star', 'role': 'operator'}, np.eye(4), metrics)
        self.assertAlmostEqual(executor.geometry.cache['shape_release_clearance_m'], .02)
        executor.robot.client.playback.assert_not_called()

    def test_verified_grasp_can_finish_lift_in_short_segments(self):
        from skill_executor import SkillExecutor, SkillFailure

        executor = SkillExecutor.__new__(SkillExecutor)
        executor.arm = lambda role: 'left'
        executor.ops = Mock()
        executor.robot = Mock()
        executor.event = Mock()
        hand = np.eye(4)
        hand[2, 3] = .79
        object_in_hand = np.eye(4)
        object_in_hand[2, 3] = -.01
        executor.robot.get_ee_pose.side_effect = lambda arm: hand.copy()
        executor.ops.pose.side_effect = lambda robot, obj: hand @ object_in_hand
        target = hand @ object_in_hand
        target[2, 3] = .87
        params = {'object': 'star', 'role': 'operator'}
        attempted = []

        def follow(params, goal, phase):
            delta = goal[2, 3] - hand[2, 3] + .01
            self.assertGreater(delta, 0)
            self.assertLessEqual(delta, .032)
            attempted.append(goal[2, 3])
            hand[:] = goal @ np.linalg.inv(object_in_hand)

        executor.follow_local_pose = Mock(side_effect=follow)
        executor.recover_probed_lift(params, target, object_in_hand)
        self.assertEqual(len(attempted), 3)
        self.assertAlmostEqual(attempted[-1], .87)

        measured_slip = object_in_hand.copy()
        measured_slip[2, 3] -= .035
        executor.ops.pose.side_effect = lambda robot, obj: hand @ measured_slip
        executor.follow_local_pose.side_effect = lambda *args: None
        with self.assertRaisesRegex(SkillFailure, 'probed_grasp_lift_exhausted'):
            executor.recover_probed_lift(params, target, object_in_hand)
        executor.robot.client.playback.assert_not_called()

    def test_blocked_lift_column_can_continue_inboard_with_a_verified_grasp(self):
        from skill_executor import SkillExecutor, SkillFailure

        executor = SkillExecutor.__new__(SkillExecutor)
        executor.spec = {'support': {'height_m': .75}, 'bindings': {'receiver': 'receiver'}}
        executor.arm = lambda role: 'left'
        executor.ops = Mock()
        executor.robot = Mock()
        executor.event = Mock()
        executor.ops.local_box.return_value = ([-.02] * 3, [.02] * 3)
        executor.robot.client.command_any.return_value = {'robot_base_position': [-.7, .20, 0.]}
        grasp = np.eye(4)
        grasp[2, 3] = -.04
        initial = np.eye(4)
        initial[:3, 3] = [.07, .20, .78]
        hand = initial @ np.linalg.inv(grasp)
        fixture = np.eye(4)
        fixture[:3, 3] = [.01, .045, .822]
        executor.robot.get_ee_pose.side_effect = lambda arm: hand.copy()
        executor.ops.pose.side_effect = lambda robot, obj: (hand @ grasp if obj == 'star' else fixture)
        target = initial.copy()
        target[2, 3] = .868
        commanded = []

        def follow(params, goal, phase, *, via=()):
            actual = hand @ grasp
            if goal[2, 3] > .82 and goal[0, 3] > .06:
                raise SkillFailure('native_action_failed_before_rollback', {})
            if via:
                self.assertEqual(len(via), 1)
                self.assertAlmostEqual(via[0][2, 3], actual[2, 3])
                self.assertLess(via[0][0, 3], actual[0, 3])
                np.testing.assert_allclose(goal[:2, 3], via[0][:2, 3])
                np.testing.assert_allclose(goal[:3, :3], actual[:3, :3])
            commanded.append((phase, goal.copy(), bool(via)))
            hand[:] = goal @ np.linalg.inv(grasp)

        executor.follow_local_pose = Mock(side_effect=follow)
        executor.recover_probed_lift({'object': 'star', 'role': 'operator'}, target, grasp)
        self.assertEqual([via for _, _, via in commanded], [False, True, False])
        self.assertAlmostEqual(commanded[-1][1][0, 3], commanded[-2][1][0, 3])
        self.assertAlmostEqual((hand @ grasp)[2, 3], target[2, 3])
        executor.robot.client.playback.assert_not_called()

    def test_composed_left_grasp_requests_a_real_staged_lift(self):
        from robot_adapters import load_robot
        from skill_executor import SkillExecutor

        executor = SkillExecutor.__new__(SkillExecutor)
        executor.geometry = SimpleNamespace(cache={})
        executor.adapter = load_robot(root=Path(__file__).resolve().parents[1])
        executor.spec = {'template': 'insert_building_block',
                         'support': {'height_m': .74}, 'target_objects': {}}
        executor.profile_scene_id = '0'
        executor.holding, executor.held_transforms, executor.facts = {}, {}, set()
        executor.arm = lambda role: 'left'
        executor.robot = Mock()
        executor.robot.get_ee_pose.return_value = np.eye(4)
        executor.ops = Mock()
        executor.ops.pose.return_value = np.eye(4)
        executor.ops.pick.return_value = True
        executor.grasp({'object': 'star', 'role': 'operator', 'grasp_profile': 'release_compatible'})
        self.assertEqual(executor.ops.pick.call_args.kwargs['lift_probe_distance_m'], .012)
        slip = {'position_error_m': .00114, 'rotation_error_rad': .0851}
        executor.recover_shape_grasp = Mock()
        executor.ops.pick.call_args.kwargs['lift_tracking_recovery'](np.eye(4), slip)
        self.assertEqual(executor.recover_shape_grasp.call_args.args[2], slip)
        self.assertEqual(executor.holding, {'operator': 'star'})

    def test_endpoint_ik_keeps_native_success_and_both_wrist_limits(self):
        from project_runtime_hooks import G2_COUPLED_WRIST_PAIRS, g2_endpoint_ik_seed

        names = [joint for pair in G2_COUPLED_WRIST_PAIRS for joint in pair]
        positions = np.array([
            [[1.18, -.80, 0., 0.], [.71, .42, 0., 0.], [.84, -.62, .75, .50]],
            [[0., 0., 1.04, -.89], [.01, .02, 1., .80], [.01, .02, .92, -1.]],
            [[.40, .60, 0., 0.], [1.35, -.58, .01, .02], [.71, .42, 0., 0.]],
        ])
        native = np.array([[True, False, True], [True, True, True], [False, True, False]])
        indices, admitted = g2_endpoint_ik_seed(names, positions, native)
        np.testing.assert_array_equal(indices, [2, 1, 0])
        np.testing.assert_array_equal(admitted, [True, True, False])

    def test_curobo_fk_accepts_noncontiguous_quaternion_views(self):
        from project_runtime_hooks import tool_pose_matrices

        backing = np.zeros((81, 8))
        backing[:, 0] = 1.
        backing[40, :4] = [0., 0., 0., 1.]
        rotations = backing[:, :4]
        self.assertFalse(rotations.flags.c_contiguous)
        positions = np.repeat([[.03, -.19, .86]], len(rotations), axis=0)
        path = tool_pose_matrices(positions, rotations)
        self.assertEqual(path.shape, (81, 4, 4))
        np.testing.assert_allclose(path[39, :3, :3], np.eye(3))
        np.testing.assert_allclose(path[40, :3, :3], np.diag([-1., -1., 1.]))
        np.testing.assert_allclose(path[-1, :3, 3], [.03, -.19, .86])

    def test_swept_payload_clearance_checks_intermediate_poses_beside_receiver(self):
        from robot_adapters import load_robot
        from project_runtime_hooks import swept_payload_clearance

        receiver = [[-.07234, -.07225, -.07268], [.07292, .07287, .07261]]
        payload = [[-.027, -.027, -.028], [.027, .027, .028]]
        valid_path = np.repeat(np.eye(4)[None], 3, axis=0)
        valid_path[:, :3, 3] = [[.163, .045, -.075], [.013, .107, -.051], [.164, .045, -.075]]
        risky_path = valid_path.copy()
        risky_path[1, 1, 3] = .098
        self.assertGreater(swept_payload_clearance(valid_path, np.eye(4), payload, receiver), .006)
        self.assertLess(swept_payload_clearance(risky_path, np.eye(4), payload, receiver), 0.)
        near_path = valid_path.copy()
        near_path[1, 1, 3] = .100617  # 0.747 mm, measured in the previously admitted route
        adapter = load_robot(root=Path(__file__).resolve().parents[1])
        self.assertGreater(swept_payload_clearance(near_path, np.eye(4), payload, receiver),
                           adapter.config['held_transport_fixture_margin_m'])

    def test_empty_hand_retreat_uses_short_local_motion_after_global_plan_fails(self):
        from skill_executor import SkillExecutor

        executor = SkillExecutor.__new__(SkillExecutor)
        executor.spec = {'template': 'insert_building_block', 'support': {'height_m': .75}}
        executor.facts = {'released:star'}
        executor.holding = {}
        executor.arm = lambda role: 'left'
        executor.event = Mock()
        executor.ops = Mock()
        executor.ops.retreat.return_value = False
        executor.robot = Mock()
        pose = np.eye(4)
        pose[:3, 3] = [-.001, .080, 1.008]
        executor.robot.get_ee_pose.side_effect = lambda arm: pose.copy()
        executor.robot.client.command_any.return_value = {'robot_base_position': [-.68, 0., 0.]}
        executor.restore_contact_planning = Mock()

        def follow(params, goal, phase):
            self.assertEqual(params['object'], 'gripper')
            pose[:] = goal

        executor.follow_local_pose = follow
        try:
            executor.retreat({'role': 'operator'})
        except Exception as exc:
            self.fail(f'Retreat stopped at the global path failure: {exc}')
        self.assertGreaterEqual(pose[2, 3] - 1.008, .04)
        self.assertLessEqual(pose[2, 3] - 1.008, .09)
        self.assertNotIn('operator', executor.holding)

    def test_transfer_selects_left_upward_wrist_while_baseline_keeps_its_grasp(self):
        from robot_adapters import load_robot
        from skill_executor import SkillExecutor

        adapter = load_robot(root=Path(__file__).resolve().parents[1])
        for transferred in (False, True):
            with self.subTest(transferred=transferred):
                executor = SkillExecutor.__new__(SkillExecutor)
                executor.geometry = SimpleNamespace(cache={})
                executor.spec = {'template': 'insert_building_block', 'target_objects': {}}
                if transferred:
                    executor.spec['support'] = {'height_m': .74}
                executor.adapter = adapter
                executor.profile_scene_id = '0'
                executor.arm = lambda role: 'left'
                executor.holding = {}
                executor.held_transforms = {}
                executor.facts = set()
                executor.robot = Mock()
                executor.robot.get_ee_pose.return_value = np.eye(4)
                executor.event = Mock()
                executor.ops = Mock()
                executor.ops.pick.return_value = True
                executor.ops.pose.return_value = np.eye(4)
                executor.grasp({'role': 'operator', 'object': 'star', 'grasp_profile': 'release_compatible'})
                args = executor.ops.pick.call_args.kwargs
                self.assertEqual(args.get('release_tcp_x_points_up', False), transferred)
                self.assertEqual(executor.holding, {'operator': 'star'})

    def test_nearby_detour_keeps_original_target_after_first_route_is_rejected(self):
        from skill_executor import SkillExecutor, SkillFailure

        executor = SkillExecutor.__new__(SkillExecutor)
        executor.event = Mock()
        held = np.eye(4)
        target = held.copy()
        target[0, 3] = -.01666
        paths = []

        def move(params, endpoint, phase, *, via, free_space_rotation):
            np.testing.assert_allclose(endpoint, target)
            paths.append(via)
            if len(paths) == 1:
                raise SkillFailure('native_action_failed_before_rollback', {})

        executor.follow_local_pose = move
        self.assertTrue(executor.try_free_space_waypoint_detours(
            {'role': 'operator', 'object': 'star'}, target, lambda phase: held.copy()))
        self.assertEqual(len(paths), 2)
        self.assertGreater(paths[0][0][2, 3], 0.)
        self.assertNotEqual(paths[0][0][1, 3], paths[1][0][1, 3])

    def test_reconnect_preserves_slowdown_and_rechecks_friction_grasp(self):
        from hashlib import sha256
        from skill_executor import SkillExecutor, SkillFailure

        executor = SkillExecutor.__new__(SkillExecutor)
        executor.spec = {'bindings': {'receiver': 'box'}, 'support': {'height_m': .74}}
        executor.adapter = Mock(config={'held_transport_reconnect_slowdown_factor': 2.})
        executor.arm = lambda role: 'right'
        executor.held_transforms = {'operator': np.eye(4)}
        executor.shape_transport_waypoint_index = 0
        executor.geometry = Mock(cache={})
        executor.event = Mock()
        executor.robot = Mock()
        executor.ops = Mock()
        live = np.eye(4)
        executor.ops.pose.side_effect = lambda robot, obj: live.copy() if obj == 'star' else np.eye(4)
        executor.robot.get_ee_pose.side_effect = lambda arm: live.copy()
        executor.follow_local_pose = Mock(side_effect=SkillFailure('native_action_failed_before_rollback', {}))
        executor.try_free_space_waypoint_detours = Mock(return_value=False)
        goal = np.eye(4)
        goal[0, 3] = .017

        def reconnect(robot, obj, arm, target, action, *, slowdown_factor):
            live[:] = target
            self.assertEqual(slowdown_factor, 2.)
            self.assertEqual(action, 'shape_transport_joint_space_reconnect')
            return True

        executor.ops.move.side_effect = reconnect
        certificate = b'geometry-certificate'
        route = {'receiver': 'box', 'workpiece': 'star', 'source_run': 'source',
                 'insertion_certificate_sha256': sha256(certificate).hexdigest(),
                 'waypoints_box_xyz_wxyz': [[.017, 0., 0., 1., 0., 0., 0.]]}
        with TemporaryDirectory() as tmp, patch.dict('os.environ', {'GENIE_RUN_SOURCE': tmp}):
            (Path(tmp) / 'insertion-certificate.json').write_bytes(certificate)
            executor.move_shape_approach({'role': 'operator', 'object': 'star'}, goal[None], route)

        executor.ops.move.assert_called_once()
        executor.event.assert_any_call('shape_transport_grasp_tracking', phase='waypoint_1',
                                       position_error_m=0., rotation_error_rad=0.)
        self.assertEqual(executor.shape_transport_waypoint_index, 1)

    def test_guarded_approach_replans_archived_waypoint_after_direct_candidates_reject(self):
        from contextlib import nullcontext
        from hashlib import sha256
        from skill_executor import SkillExecutor, SkillFailure

        executor = SkillExecutor.__new__(SkillExecutor)
        executor.spec = {'bindings': {'receiver': 'box'}, 'support': {'height_m': .74},
                         'scene_id': 'scene-room'}
        executor.arm = lambda role: 'left'
        executor.holding = {'operator': 'star'}
        executor.held_transforms = {'operator': np.eye(4)}
        executor.shape_transport_waypoint_index = 0
        executor.geometry = Mock(cache={}, shape_transport_lift_m=0.)
        executor.event = Mock()
        executor.robot = Mock()
        executor.ops = Mock()
        executor.contact_planning = lambda paths: nullcontext()
        live = np.eye(4)
        receiver = np.eye(4)
        receiver[:3, 3] = [.05, .02, .84]
        executor.ops.pose.side_effect = lambda robot, obj: live.copy() if obj == 'star' else receiver
        executor.robot.get_ee_pose.side_effect = lambda arm: live.copy()
        goal = receiver.copy()
        goal[2, 3] += .18
        candidates = np.stack([goal, goal.copy()])
        candidates[1, 0, 3] += .01
        executor.geometry.target.return_value = (candidates, [])
        certificate = b'unchanged aperture'
        route = {'receiver': 'box', 'workpiece': 'star', 'source_run': 'source',
                 'check_payload_clearance': True,
                 'insertion_certificate_sha256': sha256(certificate).hexdigest(),
                 'waypoints_box_xyz_wxyz': [[0., .08, .14, 1., 0., 0., 0.],
                                            [0., 0., .17, 1., 0., 0., 0.]]}
        executor.adapter = Mock(config={'held_transport_fixture_margin_m': .0005})
        executor.shape_grasp_profile = Mock(return_value={'transport_approach_routes_by_scene': {
            'scene-room': route}})
        commanded = []

        def guarded_move(params, target, phase, fixture, margin, slowdown):
            self.assertEqual(fixture, 'box')
            self.assertEqual(margin, .0005)
            commanded.append(np.asarray(target).copy())
            if len(commanded) == 1:
                raise SkillFailure('native_action_failed_before_rollback', {})
            live[:] = target
            return True

        executor.move_held_clear_of_fixture = Mock(side_effect=guarded_move)
        executor.facts = set()
        with TemporaryDirectory() as tmp, patch.dict('os.environ', {'GENIE_RUN_SOURCE': tmp}):
            (Path(tmp) / 'insertion-certificate.json').write_bytes(certificate)
            executor.move_held({'role': 'operator', 'object': 'star', 'target': 'shape_approach'})
        self.assertEqual(len(commanded), 3)
        np.testing.assert_allclose(commanded[0], candidates)
        np.testing.assert_allclose(commanded[1][:3, 3], receiver[:3, 3] + [0., .08, .14])
        np.testing.assert_allclose(commanded[2], goal)
        self.assertIn('target:shape_approach', executor.facts)
        executor.ops.move.assert_not_called()
        executor.robot.client.playback.assert_not_called()

    def test_training_arm_audit_rejects_applied_jump_and_ignores_inactive_targets(self):
        from audit_episode import arm_command_rate_check

        names = ['idx65_arm_r_joint5', 'idx22_arm_l_joint2']
        clock = [0., 1. / 30., 2. / 30.]
        limits = [10., 10.]
        commands = [[-.1, 0.], [-.09, 3.], [-.08, 0.]]
        active = [[True, False]] * len(clock)
        self.assertTrue(arm_command_rate_check(clock, commands, active, names, limits, 0.)['passed'])
        commands[-1][0] = 3.44
        verdict = arm_command_rate_check(clock, commands, active, names, limits, 0.)
        self.assertFalse(verdict['passed'])
        self.assertEqual(verdict['worst_joint'], names[0])
        self.assertEqual(verdict['violations'], 1)

    def test_ik_seed_filter_preserves_a_continuous_solution_when_another_seed_jumps(self):
        from project_runtime_hooks import continuous_ik_candidates

        names = ['idx65_arm_r_joint5', 'idx22_arm_l_joint2']
        current = [[-1.70, .25], [-1.60, .26]]
        candidates = [
            [[2.04441, .25], [-1.68, .25]],
            [[-1.64, .27], [-1.63, -.9]],
        ]
        np.testing.assert_array_equal(continuous_ik_candidates(names, candidates, current),
                                      [[False, True], [True, False]])

    def test_local_joint_plan_detects_recorded_wrist_branch_jump(self):
        from project_runtime_hooks import MAX_LOCAL_ARM_JOINT_STEP_RAD, largest_arm_joint_step

        wrist = 'idx65_arm_r_joint5'
        shoulder = 'idx61_arm_r_joint1'
        names = [wrist, shoulder]
        plan = [[-1.70, .25], [2.04441, .26]]
        result = largest_arm_joint_step(names, plan, [shoulder, wrist], [.23, -1.71])
        self.assertEqual(result['joint'], wrist)
        self.assertEqual(result['waypoint'], 1)
        self.assertAlmostEqual(result['step_rad'], 3.74441)
        self.assertGreater(result['step_rad'], MAX_LOCAL_ARM_JOINT_STEP_RAD)
        safe = largest_arm_joint_step(names, [[-1.70, .25], [-1.69, .26]], names, [-1.71, .23])
        self.assertLess(safe['step_rad'], MAX_LOCAL_ARM_JOINT_STEP_RAD)

    def test_remeasured_grasp_requires_real_lift_and_following(self):
        from skill_executor import SkillExecutor, SkillFailure

        for object_follows in (True, False):
            with self.subTest(object_follows=object_follows):
                executor = SkillExecutor.__new__(SkillExecutor)
                executor.spec = {'support': {'height_m': .74}}
                executor.shape_grasp_probe_used = False
                executor.held_transforms = {'operator': np.eye(4)}
                executor.arm = lambda role: 'right'
                executor.robot = Mock()
                hand = np.eye(4)
                actual = np.eye(4)
                actual[:3, 3] = [.0415, -.1955, .874]
                actual[:3, :3] = Rotation.from_rotvec([.095, 0., 0.]).as_matrix()
                executor.robot.get_ee_pose.side_effect = lambda arm: hand.copy()
                executor.ops = Mock()
                executor.ops.pose.side_effect = lambda robot, obj: actual.copy()
                executor.ops.local_box.return_value = ([-.02, -.02, -.02], [.02, .02, .02])
                executor.sync_planner_payload = Mock()
                executor.event = Mock()

                def move(params, target, phase):
                    hand[2, 3] += .006
                    if object_follows:
                        actual[:] = target

                executor.follow_local_pose = move
                params = {'role': 'operator', 'object': 'star'}
                slip = {'position_error_m': .005, 'rotation_error_rad': .095}
                if object_follows:
                    result = executor.recover_shape_grasp(params, actual.copy(), slip)
                    np.testing.assert_allclose(hand @ executor.held_transforms['operator'], result)
                else:
                    with self.assertRaisesRegex(SkillFailure, 'shape_transport_grasp_slipped'):
                        executor.recover_shape_grasp(params, actual.copy(), slip)
                    np.testing.assert_allclose(executor.held_transforms['operator'], np.eye(4))
                executor.sync_planner_payload.assert_called_once_with('right', ['star'])
                self.assertTrue(executor.shape_grasp_probe_used)


@unittest.skipIf(os.name == 'nt', 'The native collection worker uses Linux process locks')
class CollectionStorageRecoveryTests(unittest.TestCase):
    def test_low_storage_waits_without_using_an_episode_then_continues(self):
        from scene_admission import execute

        job = {'job_id': 'job', 'execution_code_sha256': 'frozen', 'status': 'prepared',
               'plan': {'tasks': [{'task_id': 'task', 'scene_quotas': [{'scene_id': 'scene', 'successes': 1}]}]},
               'task_bindings': {'task': {'template': 'task', 'scene_sources': {'scene': {
                   'scene_id': 'scene', 'directory': '/example', 'sha256': 'scene-digest', 'seed': 42}}}},
               'scene_preflight': {'task': {'requested': 1, 'compiled': 1}},
               'attempts': [], 'budgets': {'wall_seconds': 120, 'attempts_per_scene': 1, 'minimum_free_bytes': 100}}
        options = SimpleNamespace(adopt_run=None, source_run=None, max_attempts_per_scene=1, audit_only=False,
                                  tasks=None, scenes=None, trial_seed=None, slot='secondary')
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'job.json'
            waited = []
            def observe_wait(seconds):
                waited.append(json.loads(path.read_text(encoding='utf-8')))

            def completed_attempt(job, path, task, scene, attempt, slot, deadline, env):
                attempt['failure'] = 'physical_verification_failed'

            with patch.dict(os.environ, {'GENIE_PROJECT_ROOT': os.getcwd()}), \
                    patch('scene_admission.pinned_code', return_value='frozen'), \
                    patch('scene_admission.verify_manifest', return_value={'content_sha256': 'scene-digest'}), \
                    patch('scene_admission.shutil.disk_usage', side_effect=[
                        SimpleNamespace(free=50), SimpleNamespace(free=150)]), \
                    patch('scene_admission.time.sleep', side_effect=observe_wait), \
                    patch('scene_admission.run_trial', side_effect=completed_attempt), \
                    patch('metrics_report.report'):
                execute(job, path, options)
            self.assertEqual(len(waited), 1, 'Low disk space must defer the job and poll again')
            self.assertEqual(waited[0]['status'], 'waiting_for_storage')
            self.assertEqual(waited[0]['attempts'], [])
            persisted = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(len(persisted['attempts']), 1)
            self.assertEqual(persisted['attempts'][0]['scene'], 'scene')
            self.assertNotIn('budget_reason', persisted)


class DynamicSceneTests(unittest.TestCase):
    def test_table_probe_queries_through_task_object_and_keeps_its_evidence(self):
        from project_runtime_hooks import probe_support_ray

        support = '/World/Objects/table'
        table_hit = SimpleNamespace(collision=support + '/entity/collider', rigid_body='',
                                    distance=.20, position=(0., 0., .74), normal=(0., 0., 1.))
        box_hit = SimpleNamespace(collision='/World/Objects/box/entity/body', rigid_body='/World/Objects/box',
                                  distance=.1847, position=(0., 0., .7553), normal=(0., 0., 1.))
        query = Mock()
        query.raycast_all.side_effect = lambda origin, direction, distance, report: [
            self.assertTrue(report(hit)) for hit in (table_hit, box_hit)]
        filtered = probe_support_ray(query, (0., 0., .94), support)
        self.assertTrue(filtered['requested_support'])
        self.assertEqual(filtered['position'][2], .74)
        self.assertEqual(filtered['nearest_occluder']['collision'], box_hit.collision)
        self.assertFalse(probe_support_ray(query, (0., 0., .94), '/World/Objects/missing')['requested_support'])

    def test_new_scene_request_reaches_existing_adapter_contract(self):
        request = Registry().validate("isaac_physx", {"instruction":"Assemble the phone box",
            "task_id":"phone_packaging","scene_request":{"mode":"existing","scene_ids":["2","5"]},
            "successes_per_scene":3})
        self.assertEqual(request["task_ids"],["phone_packaging"])
        self.assertEqual(request["scene_request"]["scene_count"],2)
        self.assertEqual(request["successes_per_scene"],3)

    def test_invalid_counts_ids_and_unvalidated_assembly_swap(self):
        for options in ({"scene_count":True},{"mode":"existing","scene_ids":["../2"]},
                        {"mode":"existing","scene_ids":["2"],"scene_count":3},
                        {"mode":"existing","support_id":"benchmark_table_000"}):
            with self.assertRaises(ValueError):
                scene_request(options)
        with self.assertRaises(UnsupportedRequest):
            scene_request({"object_policy":"any_semantic_match"})

    def test_layout_and_colliders_change_scene_identity(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            library=root/"assets"
            layout=root/"layout"
            library.mkdir()
            layout.mkdir()
            (library/".geniesim-assets.json").write_text('{"release":"test"}')
            mesh=library/"support.usd"
            mesh.write_bytes(b"geometry")
            for filename in SCENE_FILES:
                (layout/filename).write_bytes(b"scene")
            fingerprint=write_manifest(layout,[mesh],mode="existing",source={"seed":1},asset_root=library)
            self.assertEqual(verify_manifest(layout,library),fingerprint)
            (layout/"scene.usda").write_bytes(b"moved")
            with self.assertRaisesRegex(ValueError,"Scene changed"):
                verify_manifest(layout,library)
            changed=write_manifest(layout,[mesh],mode="existing",source={"seed":1},asset_root=library)
            self.assertNotEqual(fingerprint["content_sha256"],changed["content_sha256"])
            mesh.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError,"Asset dependency changed"):
                verify_manifest(layout,library)

    def test_scene_replacement_needs_geometry_failure_evidence(self):
        first={"scene":"original","scene_failure_signature":"grasp_unreachable_or_failed/grasp/cap/"}
        second={"scene":"original"}
        self.assertIsNone(choose_replacement(["reserve"],[first,second],"original"))
        self.assertIsNone(choose_replacement(["reserve"],[first],"original"))
        self.assertEqual(choose_replacement(["reserve"],[first,first.copy()],"original"),
                         ("reserve",first["scene_failure_signature"]))
        preflight={"scene":"original","scene_failure_signature":"initial_support_or_stability"}
        self.assertEqual(choose_replacement(["reserve"],[preflight],"original"),
                         ("reserve","initial_support_or_stability"))
        self.assertIsNone(choose_replacement([],[preflight],"original"))

    def test_llm_scene_schema_invalid_field_gets_one_repair(self):
        correct={"scene_request":{"mode":"compose_from_library","variation":["layout","distractors"]},
                 "successes_per_scene":5}
        invalid={**correct,"scene_request":{**correct["scene_request"],"distractors":["a small decoration"]}}
        api=Mock()
        api.plan.side_effect=[(invalid,{"id":"invalid"}),(correct,{"id":"repaired"})]
        result,scene,receipts=resolve_intent(api,"Compose a new scene, record five successes",[],{},None,None)
        self.assertEqual(result["successes_per_scene"],5)
        self.assertEqual(scene["scene_count"],1)
        self.assertEqual([r["id"] for r in receipts],["invalid","repaired"])
        self.assertIn("distractors",api.plan.call_args.kwargs["repair"]["error"])

    def test_keep_explicit_rearrangement_when_model_only_marks_clutter(self):
        api=Mock()
        api.plan.return_value=({"scene_request":{"mode":"compose_from_library","variation":["distractors"]}}, {})
        _,scene,_=resolve_intent(api,"重新摆放积木和盒子，桌边增加一个小摆件",[],{},None,None)
        self.assertEqual(scene["variation"],["distractors","layout"])
