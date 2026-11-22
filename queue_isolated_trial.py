"""Launch and audit one trial in a selected simulation slot."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from batch_collect import recipe_sources, verified_recipe
from collection_control import ROOT, SOCKET, STORAGE, running_collections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--scene', choices=['0', '1'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--slot', choices=['primary', 'secondary'], default='secondary')
    parser.add_argument('--after', type=Path)
    parser.add_argument('--source-run', type=Path)
    args = parser.parse_args()

    env = dict(os.environ, DOCKER_HOST=SOCKET)
    deadline = time.monotonic() + 3600
    while running_collections(env, args.slot):
        if time.monotonic() >= deadline:
            raise TimeoutError(f'{args.slot} slot did not finish the previous trial')
        time.sleep(10)

    if args.after:
        predecessor = args.after.resolve()
        if predecessor.parent != STORAGE/'collection-runs':
            parser.error('The previous run must belong to this project')
        result_path = predecessor/'collection_result.json'
        if result_path.exists() and json.loads(result_path.read_text()).get('success') is True:
            print('Previous episode has a physical success report. Review it before retrying.', flush=True)
            return

    job = json.loads(args.job.read_text())
    task_id = next(k for k, value in job['task_bindings'].items() if value['template'] == args.task)
    task = next(t for t in job['plan']['tasks'] if t['task_id'] == task_id)
    quota = next(q['successes'] for q in task['scene_quotas'] if str(q['scene_id']) == args.scene)
    if job['success_counts'][task_id][args.scene] >= quota:
        print('The scene quota is already met.', flush=True)
        return

    plan = job['prepared_skill_plans'][task_id][args.scene]
    stamp = time.strftime('%Y%m%d%H%M%S')
    label = f'{args.task.replace("_", "-")}-s{args.scene}-iso-{args.variant}-{stamp}'
    command = [sys.executable, str(ROOT/'collection_control.py'), args.task, '--scene', args.scene,
               '--label', label, '--slot', args.slot, '--robot-id', job['robot_id'],
               '--execution-mode', job['execution_mode'], '--instruction', job['instruction'],
               '--prepared-plan', plan['path'], '--seed', str(args.seed)]
    if args.source_run:
        launch = json.loads((args.source_run/'launch.json').read_text())
        if not verified_recipe(args.job.parent, task_id, args.scene, recipe_sources(launch)):
            raise ValueError('The source for a quota repeat needs a passing hardpoint review')
        command += ['--source-run', str(args.source_run)]
    subprocess.run(command, env=env, check=True)

    run = STORAGE/'collection-runs'/label
    print(f'STARTED {run}', flush=True)
    subprocess.run([sys.executable, str(ROOT/'watch_isolated_run.py'), str(run)], env=env, check=True)


if __name__ == '__main__':
    main()
