"""Resolve language requests against the three official FineManip templates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


TASKS = {
    'phone_packaging': ('手机装盒', '手机放入盒', '手机放进盒', '手机包装', 'phone packaging'),
    'put_the_cap_on_the_pen_new': ('盖笔帽', '套笔帽', '笔帽套', '笔帽盖', 'cap the pen'),
    'insert_building_block': ('积木插入', '插入积木', '把积木插', 'insert building block'),
}
PREFIX = 'g2crsb90d_finemanip_'


def request(prompt):
    normalized = prompt.casefold()
    tasks = [task for task, aliases in TASKS.items()
             if task in normalized or any(alias in normalized for alias in aliases)]
    if not tasks:
        raise ValueError('没有匹配到已登记的任务。支持手机装盒、盖笔帽、积木插入。')
    if re.search(r'不要|不做|排除|不需要|不生成|\b(?:without|except|exclude|not)\b', normalized):
        raise ValueError('请直接列出要采集的任务。')
    counts = {int(n) for n in re.findall(r'(\d+)\s*条', normalized)}
    counts.update(int(n) for n in re.findall(r'(\d+)\s+(?:successful\s+)?(?:episodes|trajectories)', normalized))
    if len(counts) > 1:
        raise ValueError('本次请求请为各任务指定同一个数量。')
    target = counts.pop() if counts else 5
    if not 5 <= target <= 10:
        raise ValueError('首批每个任务的成功目标应为 5–10 条。')
    return tasks, target


def prepare(repo, output, prompt):
    tasks, target = request(prompt)
    base = repo / 'source/geniesim_benchmark/src/geniesim_benchmark'
    result = {'prompt': prompt, 'robot': 'g2crsb90d', 'status': 'prepared',
              'trajectory_success_count': 0, 'tasks': []}
    for task in tasks:
        task_id = PREFIX + task
        config = base / 'config' / f'{task_id}.yaml'
        task_out = output / task_id
        task_out.mkdir()
        shutil.copy2(config, task_out)
        task_record = {'task_id': task_id, 'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                       'target_successes': target, 'scenes': []}
        for scene_id, quota in ((0, (target + 1) // 2), (1, target // 2)):
            source = base / 'benchmark/config/llm_task' / task / str(scene_id)
            destination = task_out / str(scene_id)
            destination.mkdir()
            sources = {}
            for name in ('scene.usda', 'scene_info.json', 'instructions.json', 'problems.json'):
                path = source / name
                shutil.copy2(path, destination)
                sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            layer = (destination / 'scene.usda').read_text()
            if not layer.startswith('#usda'):
                raise ValueError(f'Invalid USD layer for {task_id}/{scene_id}')
            instructions = json.loads((destination / 'instructions.json').read_text())
            task_record['scenes'].append({
                'instance_id': scene_id, 'target_successes': quota,
                'instructions': [item['instruction'] for item in instructions['instructions']],
                'top_layer_references': sorted(set(re.findall(r'@([^@\n]+)@', layer))),
                'sha256': sources,
            })
        result['tasks'].append(task_record)
    (output / 'request.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt')
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', action='store_true', help='Run upstream scene preview after preparation')
    parser.add_argument('--cli', default='geniesim', help='Genie Sim CLI in the configured simulation runtime')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = prepare(args.repo, args.output, args.prompt)
    if args.preview:
        output = args.output.resolve()
        for task in result['tasks']:
            task_id = task['task_id']
            command = [args.cli, 'benchmark', 'run', task_id, '--',
                       '--app.headless=true', '--benchmark.preview=true',
                       '--benchmark.instance_ids=0,1', '--benchmark.record=false',
                       f'--benchmark.output_dir={output / "preview"}']
            with (output / task_id / 'preview.log').open('w', encoding='utf-8') as log:
                try:
                    run = subprocess.run(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT)
                    task['preview_exit_code'] = run.returncode
                except FileNotFoundError as error:
                    task['preview_exit_code'] = 127
                    log.write(str(error))
        result['status'] = ('preview_commands_finished' if all(
            task['preview_exit_code'] == 0 for task in result['tasks']) else 'preview_failed')
        (output / 'request.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'tasks': len(result['tasks']),
                      'scenes': sum(len(task['scenes']) for task in result['tasks']),
                      'output': str(args.output), 'trajectory_success_count': 0}))
    if result['status'] == 'preview_failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
