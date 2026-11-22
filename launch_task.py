"""Launch a project-scoped native Genie Sim task and retain its run receipt."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil


TASKS = {
    'phone_packaging': 'g2crsb90d_finemanip_phone_packaging',
    'put_the_cap_on_the_pen_new': 'g2crsb90d_finemanip_put_the_cap_on_the_pen_new',
    'insert_building_block': 'g2crsb90d_finemanip_insert_building_block',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=TASKS)
    parser.add_argument('--instances', default='0')
    parser.add_argument('--episodes', default=1, type=int)
    parser.add_argument('--mode', choices=['preview', 'motion_probe', 'collect'], default='preview')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or any(item not in ('0', '1') for item in args.instances.split(',')):
        parser.error('Use instances 0 or 1, and a positive episode count')
    project = Path(__file__).resolve().parent
    repository = project / 'genie_sim-main'
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    ros = project / '.venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy'
    env.update(SIM_REPO_ROOT=str(repository), SIM_ASSETS=str(project / 'assets'),
               GENIESIM_ASSETS_PATH=str(project / 'assets'),
               GENIESIM_SKIP_AUTOBOOT='1', GENIE_PROJECT_GPU='1',
               GENIE_PROJECT_POLICY='' if args.mode == 'preview' else 'geometric',
               GENIE_PROJECT_MODE=args.mode,
               GENIE_PROJECT_OUTPUT=str(args.output / 'native'),
               GENIESIM_BENCHMARK_OUTPUT_ROOT=str(args.output / 'benchmark'),
               GENIE_PROJECT_RUNTIME_SCENES=str(project / 'output/runtime-scenes'),
               GENIESIM_KIT_RUNTIME_DIR=str(project / 'cache/kit-policy' / args.output.name),
               GENIESIM_OMNI_DOCUMENTS_DIR=str(project / 'cache/Documents'),
               GENIESIM_PREVIEW_OUTPUT_DIR=str(args.output / 'previews'),
               XDG_CACHE_HOME=str(project / 'cache'), XDG_DATA_HOME=str(project / 'cache/data'),
               OMNI_KIT_ACCEPT_EULA='YES', ROS_DISTRO='jazzy', RMW_IMPLEMENTATION='rmw_fastrtps_cpp')
    env['PATH'] = str(project / '.venv/bin') + os.pathsep + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = str(ros / 'lib') + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    config = repository / 'source/geniesim_benchmark/src/geniesim_benchmark/config' / (TASKS[args.task] + '.yaml')
    command = [str(project / '.venv/bin/python'), '-u', str(project / 'run_official.py'),
               f'--config={config}', '--app.headless=true',
               f'--benchmark.instance_ids={args.instances}', f'--benchmark.num_episode={args.episodes}',
               '--benchmark.preview=' + str(args.mode == 'preview').lower(),
               '--benchmark.record=' + str(args.mode != 'preview').lower(),
               '--benchmark.pause_physics_during_inference=false']
    sources = ('run_official.py', 'scripted_policy.py', 'task_programs.py', 'launch_task.py')
    snapshot = args.output / 'source'
    snapshot.mkdir()
    for name in sources:
        shutil.copy2(project / name, snapshot / name)
    command[2] = str(snapshot / 'run_official.py')
    receipt = {'task': args.task, 'mode': args.mode, 'instances': args.instances,
               'episodes_requested': args.episodes, 'command': command,
               'started_utc': datetime.now(timezone.utc).isoformat(),
               'source_sha256': {name: hashlib.sha256((project / name).read_bytes()).hexdigest()
                                 for name in sources}}
    path = args.output / 'launch.json'
    path.write_text(json.dumps(receipt, indent=2))
    with (args.output / 'runtime.log').open('w') as log:
        run = subprocess.run(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt.update(exit_code=run.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
    runtime_log = (args.output / 'runtime.log').read_text(errors='replace')
    fatal = [line for line in runtime_log.splitlines() if any(marker in line for marker in (
        'Unhandled exception in episode:', 'Episode loop stopped due to fatal episode error:',
        'Main loop watchdog', 'Episode recording failed:'))]
    receipt['runtime_errors'] = fatal[-8:]
    receipt['execution_ok'] = run.returncode == 0 and not fatal
    results = []
    for result_path in (args.output / 'benchmark').rglob('evaluate_ret_*.json'):
        report = json.loads(result_path.read_text())
        for detail in report.get('details', []):
            results.append({'instruction': detail.get('task_instruction'),
                            'native_e2e': detail.get('result', {}).get('scores', {}).get('E2E', 0)})
    receipt['native_results'] = results
    receipt['native_success_count'] = sum(row['native_e2e'] == 1 for row in results)
    path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'output': str(args.output), 'exit_code': run.returncode}), flush=True)
    raise SystemExit(run.returncode or (0 if receipt['execution_ok'] else 1))


if __name__ == '__main__':
    main()
