"""Archive measured state and applied targets at native video submission times."""
import json
import os
from pathlib import Path

import numpy as np


def install_frame_journal(benchmark):
    core = benchmark.api_core
    recorder = core.local_recorder
    original = recorder.write_frames
    root = Path(os.environ['GENIE_PROJECT_OUTPUT']).parent / 'frame-journals'

    def write_frames(interface, current_step_index, *args, **kwargs):
        before = recorder._step_counter
        dropped = {name: encoder.frames_dropped for name, encoder in recorder._encoders.items()}
        original(interface, current_step_index, *args, **kwargs)
        if recorder._step_counter == before:
            return
        instance = benchmark.env.init_task_config['scene']['scene_instance_id']
        episode = recorder._current_episode_idx
        folder = root / f'instance_{instance}_episode_{episode:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        articulation = core._get_articulation()
        names = list(articulation.dof_names)
        positions = np.asarray(articulation.get_joint_positions(), dtype=float)
        applied = articulation.get_applied_action()
        targets = np.full(len(names), np.nan)
        if applied is not None and applied.joint_positions is not None:
            values = np.asarray(applied.joint_positions, dtype=float)
            indices = applied.joint_indices
            if indices is None:
                targets[:] = values
            else:
                targets[np.asarray(indices, dtype=int)] = values
        record = {'frame_index': before, 'physics_step': int(current_step_index),
                  'simulation_time': float(core.ui_builder.my_world.current_time),
                  'joint_names': names if before == 0 else None,
                  'state': positions.tolist(),
                  'applied_position_targets': [float(value) if np.isfinite(value) else None for value in targets],
                  'dropped_total': {name: encoder.frames_dropped for name, encoder in recorder._encoders.items()},
                  'dropped_during_submit': {name: recorder._encoders[name].frames_dropped - count
                                             for name, count in dropped.items()}}
        with (folder / 'frames.jsonl').open('a') as output:
            output.write(json.dumps(record) + '\n')

    recorder.write_frames = write_frames
