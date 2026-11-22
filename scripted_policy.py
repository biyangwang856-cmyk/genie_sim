"""Geometric demonstration controller executed by the upstream benchmark.

This controller uses simulator object poses. Its outputs are joint targets sent
through the benchmark's normal physics entry point. Native checkers own success.
"""
from collections import deque
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from geniesim_benchmark.benchmark.policy.base import BasePolicy


def rotation(pose):
    return Rotation.from_quat(np.asarray(pose[3:])[[1, 2, 3, 0]])


class ScriptedPolicy(BasePolicy):
    def __init__(self, benchmark):
        super().__init__(benchmark.args.task_name, benchmark.args.sub_task_name)
        self.benchmark = benchmark
        self.episode_idx = 0
        self.reset()

    def reset(self):
        if getattr(self, 'started', False):
            self._save()
        self.action_buffer = deque()
        self.rows = []
        self.finished = False
        self.started = False
        self.phase = 'initial'
        self.server_episode_done = False
        self.server_episode_termination = None
        self.policy_dispatch_count = 0
        self.chunk_started_this_step = False
        self.chunk_len = 1
        self.program = deque()
        self.held = {}
        self.terminal_state = None

    def set_episode_idx(self, index):
        self.episode_idx = index

    def update_task_status(self, done, progress):
        if done:
            self.finished = True
            (self.folder / 'final_progress.json').write_text(json.dumps(progress, indent=2, default=str))
            self._save()

    def need_infer(self):
        return True

    def inference_due(self):
        return False

    def _read_scene(self):
        api = self.benchmark.api_core

        def capture():
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            root = stage.GetPrimAtPath(api._env_path('/Workspace/Objects'))
            names = [prim.GetName() for prim in root.GetChildren()]
            objects = {}
            for name in names:
                path = '/Workspace/Objects/' + name
                position, quaternion = api.get_obj_world_pose(path)
                objects[name] = {
                    'position': list(map(float, position)),
                    'quaternion_wxyz': list(map(float, quaternion)),
                    'aabb': np.asarray(api.get_obj_aabb(path)).tolist(),
                }
            links = {}
            for name in ('base_link', 'arm_base_link', 'arm_l_end_link', 'arm_r_end_link',
                         'gripper_l_center_link', 'gripper_r_center_link'):
                position, quaternion = api._get_link_world_pose(name)
                links[name] = {'position': list(map(float, position)),
                               'quaternion_wxyz': list(map(float, quaternion))}
            return {'objects': objects, 'links': links}

        return api.run_on_physics_loop(capture)

    def _save(self):
        directory = self.folder
        array = lambda key: np.asarray([row[key] for row in self.rows], dtype=np.float64)
        np.savez_compressed(directory / 'trajectory.npz', state=array('state'), action=array('action'),
                            terminal_state=np.asarray(self.terminal_state if self.terminal_state is not None else []),
                            phase=np.asarray([row['phase'] for row in self.rows]),
                            simulation_time=array('simulation_time'),
                            control_step=np.asarray([row['step'] for row in self.rows]))
        manifest = {'policy': 'geometric_oracle', 'robot': 'g2crsb90d',
                    'frames': len(self.rows), 'task': self.sub_task_name,
                    'official_instance': self.benchmark.env.init_task_config['scene']['scene_instance_id'],
                    'success_owner': 'upstream native checker',
                    'joint_units': 'radians', 'privileged_object_poses': True,
                    'mode': os.environ.get('GENIE_PROJECT_MODE'),
                    'remaining_phases': [phase for phase, _ in self.program],
                    'pending_actions': len(self.action_buffer),
                    'note': 'Acceptance requires native success and complete state/action/video verification.'}
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    def shutdown(self):
        if self.started:
            self._save()

    def _move(self, obs, side, goal, steps=30):
        joints = np.r_[obs['states']['left_arm'], obs['states']['right_arm']]
        initial = obs['eef']
        start = np.asarray(initial[side][:3])
        initial_rotation = rotation(initial[side])
        target_rotation = Rotation.from_matrix(goal[:3, :3])
        interpolate = Slerp([0., 1.], Rotation.concatenate([initial_rotation, target_rotation]))
        effector = self.grippers.copy()
        for index in range(1, steps + 1):
            t = index / steps
            fraction = t * t * (3 - 2 * t)
            position = start + fraction * (goal[:3, 3] - start)
            poses = {}
            for arm in ('left', 'right'):
                poses[arm] = (np.r_[position, interpolate(fraction).as_euler('xyz')]
                              if arm == side else np.r_[initial[arm][:3], rotation(initial[arm]).as_euler('xyz')])
            row = np.r_[poses['left'], poses['right'], effector]
            solved = self.benchmark.env.ikfk_solver.eef_actions_to_joint([row], joints)[0]
            joints = np.asarray(solved[:14])
            self.action_buffer.append({'arm': joints.copy(), 'gripper': effector.copy()})

    def hold_phase(self, steps, obs):
        joints = np.r_[obs['states']['left_arm'], obs['states']['right_arm']]
        for _ in range(steps):
            self.action_buffer.append({'arm': joints.copy(), 'gripper': self.grippers.copy()})

    def grip_phase(self, side, value, obs):
        self.grippers[0 if side == 'left' else 1] = value
        self.hold_phase(35, obs)

    def tip_pose(self, obs, side):
        from task_programs import matrix
        pose = obs['eef'][side]
        wrist = matrix(pose[:3], pose[3:])
        return self.base_link @ obs['arm_base_transform'] @ wrist @ self.wrist_to_tip

    def move_tip(self, obs, side, goal, steps=40):
        arm_base = self.base_link @ obs['arm_base_transform']
        wrist_goal = np.linalg.inv(arm_base) @ goal @ np.linalg.inv(self.wrist_to_tip)
        self._move(obs, side, wrist_goal, steps)

    def approach_phase(self, name, side, long_axis, local_offset, height, obs):
        from task_programs import object_matrix, top_orientation
        record = self._read_scene()['objects'][name]
        transform = object_matrix(record)
        bounds = np.asarray(record['aabb']).reshape(2, 3)
        point = bounds.mean(axis=0)
        if local_offset is not None:
            point = transform[:3, 3] + transform[:3, :3] @ np.asarray(local_offset)
        self.held.setdefault(name, {'start_z': float(bounds[:, 2].mean())})
        goal = np.eye(4)
        goal[:3, :3] = top_orientation(transform[:3, long_axis])
        goal[:3, 3] = point + [0., 0., height + 0.003]
        self.held[name]['last_grasp_goal_world'] = goal.tolist()
        self.move_tip(obs, side, goal, 55 if height else 35)

    def lift_phase(self, name, side, height, obs):
        before = self._read_scene()
        before['measured_grippers'] = np.r_[obs['states']['left_gripper'], obs['states']['right_gripper']].tolist()
        before['commanded_grasp_pose'] = self.held[name]['last_grasp_goal_world']
        (self.folder / f'{name}_prelift.json').write_text(json.dumps(before, indent=2))
        from PIL import Image
        for camera in ('head', f'{side}_hand'):
            Image.fromarray(np.asarray(obs['images'][camera])[:, :, :3].astype('uint8')).save(
                self.folder / f'{name}_prelift_{camera}.png')
        goal = self.tip_pose(obs, side)
        goal[2, 3] += height
        self.move_tip(obs, side, goal, 50)

    def retreat_phase(self, side, height, obs):
        goal = self.tip_pose(obs, side)
        goal[2, 3] += height
        self.move_tip(obs, side, goal, 45)

    def verify_grasp_phase(self, name, side, obs):
        snapshot = self._read_scene()
        (self.folder / f'{name}_lift_check.json').write_text(json.dumps(snapshot, indent=2))
        record = snapshot['objects'][name]
        center = np.asarray(record['aabb']).reshape(2, 3).mean(axis=0)
        lift = center[2] - self.held[name]['start_z']
        print('PROJECT_GRASP_CHECK', name, 'lift_m', float(lift), flush=True)
        if lift < 0.06:
            raise RuntimeError(f'Physical grasp failed for {name}: measured lift {lift:.4f} m')
        self.hold_phase(15, obs)

    def object_pose_phase(self, name, side, target, steps, obs):
        from task_programs import object_matrix
        record = self._read_scene()['objects'][name]
        actual = object_matrix(record)
        goal = target @ np.linalg.inv(actual) @ self.tip_pose(obs, side)
        self.move_tip(obs, side, goal, steps)

    def phone_place_phase(self, name, base, depth, above, obs):
        from task_programs import object_matrix
        objects = self._read_scene()['objects']
        bounds = np.asarray(objects[base]['aabb']).reshape(2, 3)
        target = object_matrix(objects[base])
        target[:2, 3] = bounds[:, :2].mean(axis=0)
        target[2, 3] = bounds[0, 2] + depth + (0.15 if above else 0.)
        if name.endswith('_002'):
            target[:3, :3] = target[:3, :3] @ Rotation.from_euler('z', np.pi).as_matrix()
        self.object_pose_phase(name, 'right', target, 60 if above else 40, obs)

    def block_entry_phase(self, name, side, geometry, height, obs):
        from task_programs import object_matrix
        box = self._read_scene()['objects'][geometry['box']]
        target = object_matrix(box) @ np.asarray(geometry['object_goal_in_box_frame'])
        target[2, 3] += height
        self.object_pose_phase(name, side, target, 65 if height else 50, obs)

    def _start(self, obs):
        output = Path(os.environ['GENIE_PROJECT_OUTPUT'])
        instance = self.benchmark.env.init_task_config['scene']['scene_instance_id']
        self.folder = output / f'instance_{instance}_episode_{self.episode_idx:04d}'
        self.folder.mkdir(parents=True, exist_ok=False)
        scene = self._read_scene()
        scene['observed_states'] = obs['states']
        scene['observed_eef_arm_frame'] = obs['eef']
        (self.folder / 'initial_scene.json').write_text(json.dumps(scene, indent=2))
        from PIL import Image
        for name, pixels in obs.get('images', {}).items():
            if pixels is not None and np.asarray(pixels).size:
                Image.fromarray(np.asarray(pixels)[:, :, :3].astype('uint8')).save(
                    self.folder / f'initial_{name}.png')
        print('PROJECT_INITIAL_SCENE', str(self.folder), json.dumps(scene), flush=True)
        from task_programs import build_program, matrix
        self.grippers = np.r_[obs['states']['left_gripper'], obs['states']['right_gripper']]
        self.arm_base = matrix(scene['links']['arm_base_link']['position'],
                               scene['links']['arm_base_link']['quaternion_wxyz'])
        self.base_link = matrix(scene['links']['base_link']['position'],
                               scene['links']['base_link']['quaternion_wxyz'])
        self.wrist_to_tip = np.eye(4)
        self.wrist_to_tip[2, 3] = 0.185
        path = Path(os.environ['SIM_REPO_ROOT']) / 'source/geniesim_benchmark/src/geniesim_benchmark/benchmark/config/llm_task'
        instructions = json.loads((path / self.sub_task_name / str(instance) / 'instructions.json').read_text())
        self.instruction = instructions['instructions'][0]
        self.started = True
        if os.environ.get('GENIE_PROJECT_MODE') == 'collect':
            self.program = deque(build_program(self, scene))
            return
        goal = np.eye(4)
        goal[:3, :3] = rotation(obs['eef']['right']).as_matrix()
        goal[:3, 3] = obs['eef']['right'][:3]
        goal[2, 3] += 0.03
        self.phase = 'right_wrist_raise_3cm'
        self._move(obs, 'right', goal)
        for _ in range(20):
            self.action_buffer.append(self.action_buffer[-1])

    def act(self, obs, **kwargs):
        if self.finished:
            self.terminal_state = np.r_[obs['states']['left_arm'], obs['states']['right_arm'],
                                         obs['states']['left_gripper'], obs['states']['right_gripper']]
            self._save()
            return None
        if not self.started:
            self._start(obs)
        while not self.action_buffer and self.program:
            self.phase, operation = self.program.popleft()
            print('PROJECT_PHASE', self.phase, flush=True)
            operation(obs)
        if not self.action_buffer:
            self.terminal_state = np.r_[obs['states']['left_arm'], obs['states']['right_arm'],
                                         obs['states']['left_gripper'], obs['states']['right_gripper']]
            self.server_episode_termination = {'reason': 'motion_program_finished'}
            self._save()
            return None
        action = self.action_buffer.popleft()
        state = np.r_[obs['states']['left_arm'], obs['states']['right_arm'],
                      obs['states']['left_gripper'], obs['states']['right_gripper']]
        self.rows.append({'state': state, 'action': np.r_[action['arm'], action['gripper']],
                          'phase': self.phase, 'step': kwargs.get('step_num', len(self.rows)),
                          'simulation_time': self.data_courier._project_latest_simulation_time})
        self.policy_dispatch_count += 1
        if len(self.rows) % 10 == 0:
            self._save()
        return action


def benchmark_factory(config, api_core):
    from geniesim_benchmark.benchmark.task_benchmark import TaskBenchmark

    class ScriptedBenchmark(TaskBenchmark):
        def evaluate_episode(self, robot_cfg, evaluation):
            try:
                return super().evaluate_episode(robot_cfg, evaluation)
            finally:
                if self.policy is not None and getattr(self.policy, 'started', False):
                    self.policy._save()

        def create_env(self, episode_file, instance_id):
            root = Path(os.environ['GENIE_PROJECT_RUNTIME_SCENES']) / self.args.sub_task_name
            self.task_config['scene']['sub_usd_override_root'] = str(root)
            super().create_env(episode_file, instance_id)
            if not hasattr(self.data_courier, '_project_latest_simulation_time'):
                original = self.data_courier.get_obs_bundle

                def observe(*args, **kwargs):
                    bundle = original(*args, **kwargs)
                    self.data_courier._project_latest_simulation_time = float(bundle['simulation_time'])
                    return bundle

                self.data_courier.get_obs_bundle = observe
                self.data_courier._project_latest_simulation_time = 0.

        def create_policy(self):
            if os.environ.get('GENIE_PROJECT_POLICY') != 'geometric':
                return super().create_policy()
            if self.policy is None:
                self.policy = ScriptedPolicy(self)
                self.policy.set_data_courier(self.data_courier)
            else:
                self.policy.reset()

    return ScriptedBenchmark(config, api_core)
