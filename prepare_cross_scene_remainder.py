"""Certify released actions and compute the unfilled frozen scene quotas."""
import argparse
import json
from pathlib import Path

from audit_episode import g2_arm_command_continuity
from batch_collect import write
from collection_control import STORAGE
from export_lerobot import recording_path
from lerobot_layout import raw_archive_path
from scene_contract import digest, read_json


def formal_jobs(root):
    jobs = {}
    for summary in sorted(root.glob('project-v*/jobs/acceptance-*/validation.json')):
        for case, group in read_json(summary)['groups'].items():
            if not group.get('job_path'):
                continue
            path = Path(group['job_path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f'Job outside the validation experiment: {path}')
            if jobs.setdefault(path, case) != case:
                raise ValueError(f'Conflicting case for job: {path}')
    return jobs


def published_sources(root, jobs, retired=()):
    datasets = []
    for job in jobs.values():
        dataset = job.get('dataset') or job.get('partial_dataset')
        if dataset:
            datasets.append(dataset)
    for path in sorted(root.glob('project-v*/jobs/*/release-preview.json')):
        datasets.append(read_json(path)['dataset'])
    sources = {}
    for dataset in datasets:
        if not dataset.get('official_reader_verified'):
            continue
        path = Path(dataset['dataset_path']).resolve()
        if path.parent != STORAGE/'exports':
            raise ValueError(f'Dataset outside the project exports directory: {path}')
        if path in retired:
            continue
        loader = path/'official_loader_receipt.json'
        archive = raw_archive_path(path)
        manifest = read_json(archive/'manifest.json')
        if (digest(loader) != dataset['loader_receipt_sha256'] or not read_json(loader)['passed']
                or manifest['episodes'] != dataset['episodes'] or manifest['training_dataset'] != str(path)):
            raise ValueError(f'Invalid official reader or original trajectory receipt: {path}')
        for source in dataset['sources']:
            sources[source['training_motion_sha256']] = {
                'dataset': str(path), 'source_run': source['run'], 'raw_archive': str(archive),
            }
    return sources


def certify(root, matrix):
    cases_by_job = formal_jobs(root)
    jobs = {path: read_json(path) for path in cases_by_job}
    record = root/'accepted-native/pruned-artifacts.json'
    retired_runs, retired_datasets = set(), set()
    if record.is_file():
        retention = read_json(record)
        if retention['experiment'] != str(root) or retention['status'] != 'complete':
            raise ValueError('Invalid or unfinished recording retention receipt')
        retired_runs = {Path(name) for name in retention['retired_recording_directories']}
        retired_datasets = {Path(name) for name in retention['retired_dataset_directories']}
        if (any(path.parent != STORAGE/'collection-runs' for path in retired_runs)
                or any(path.parent != STORAGE/'exports' for path in retired_datasets)):
            raise ValueError('Recording retention receipt points outside the project')
    publications = published_sources(root, jobs, retired_datasets)
    quotas = {case['id']: case['request']['successes_per_scene'] for case in matrix['cases']}
    counts = {case: 0 for case in quotas}
    rows = {}
    for path, job in jobs.items():
        case = cases_by_job[path]
        for trial in job['attempts']:
            if trial.get('accepted') is not True:
                continue
            run = Path(trial['run_directory']).resolve()
            motion = trial['fingerprint']
            if run.parent != STORAGE/'collection-runs':
                raise ValueError(f'Episode outside the project recording directory: {run}')
            if run in retired_runs:
                continue
            if motion in rows:
                if rows[motion]['case'] != case:
                    raise ValueError(f'One action recording counted in two frozen cases: {run}')
                continue
            certificate = run/'automatic-admission.json'
            admission = read_json(certificate)
            if (digest(certificate) != trial['certificate_sha256']
                    or digest(run/'launch.json') != admission['launch_sha256']
                    or digest(run/'acceptance.json') != admission['acceptance_sha256']):
                raise ValueError(f'Physical admission evidence changed: {run}')
            with (run/'evidence/native-control-commands.jsonl').open(encoding='utf-8') as stream:
                commands = [json.loads(line) for line in stream]
            timestamps = [frame['time_stamp'] for frame in read_json(recording_path(run)/'state.json')['frames']]
            continuity = g2_arm_command_continuity(run, commands, timestamps, enforce_coupled_wrist=bool(
                read_json(run/'launch.json').get('strict_g2_coupled_wrist')))
            publication = publications.get(motion)
            if publication and Path(publication['source_run']).resolve() != run:
                raise ValueError(f'Published action digest points to a different run: {run}')
            rows[motion] = {'case': case, 'run': str(run), 'command_continuity': continuity,
                            'official_publication': publication}
            if continuity['passed'] and publication:
                counts[case] += 1
    return {'episodes': rows, 'qualified_released': counts,
            'remaining': {case: max(0, quota - counts[case]) for case, quota in quotas.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('matrix', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--remaining-matrix', type=Path)
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    matrix = read_json(args.matrix)
    result = certify(args.experiment.resolve(), matrix)
    write(args.report, result)
    if args.remaining_matrix:
        selected = []
        for case in matrix['cases']:
            request = case['request']
            if args.tasks and request['task_id'] not in args.tasks:
                continue
            remaining = result['remaining'][case['id']]
            if remaining:
                old = request['successes_per_scene']
                request['instruction'] = request['instruction'].replace(f'采集{old}条', f'采集{remaining}条')
                request['successes_per_scene'] = remaining
                selected.append(case)
        write(args.remaining_matrix, {**matrix, 'cases': selected})
    print('Certified training episodes:', sum(result['qualified_released'].values()), flush=True)


if __name__ == '__main__':
    main()
