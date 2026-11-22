"""Turn a quota-complete job into a reader-verified training dataset."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from collection_control import ROOT, STORAGE

READER = STORAGE / 'lerobot-reader-0cf8648/bin/python'
READER_COMMIT = '0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'


def quota_runs(job):
    requested = [(task['task_id'], str(quota['scene_id']), quota['successes'])
                 for task in job['plan']['tasks'] for quota in task['scene_quotas']]
    runs = []
    for task, scene, count in requested:
        matching = [entry for entry in job['attempts'] if entry['task']==task and str(entry['scene'])==scene
                    and entry.get('accepted') is True and entry.get('automatic_acceptance') is True
                    and entry.get('raw_archive') is True]
        if len(matching)<count:
            raise ValueError(f'Incomplete quota for {task}/{scene}: {len(matching)} of {count}')
        for entry in matching[:count]:
            run = Path(entry['run_directory']).resolve()
            if run.parent!=STORAGE/'collection-runs' or run in runs:
                raise ValueError(f'Invalid or duplicate accepted run: {run}')
            launch = json.loads((run/'launch.json').read_text())
            if (launch['task'],str(launch['scene']),launch['robot_id'])!=(task,scene,job['plan']['robot']):
                raise ValueError(f'Accepted run does not match the job: {run}')
            runs.append(run)
    if not runs:raise ValueError('No accepted episodes in the job')
    return runs


def finalize(job, job_path):
    runs = quota_runs(job)
    name = job['job_id']
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',name):raise ValueError('Invalid job ID for dataset directory')
    output = STORAGE/'exports'/(name+'-v2-1')
    sources = [{'run':str(run),'launch_sha256':hashlib.sha256((run/'launch.json').read_bytes()).hexdigest()} for run in runs]
    if job.get('require_distinct_motion'):
        from batch_collect import motion_fingerprint
        fingerprints=[motion_fingerprint(run) for run in runs]
        if len(set(fingerprints))!=len(runs):
            raise ValueError('Dataset selection still contains repeated measured state/action motion')
        for source,fingerprint in zip(sources,fingerprints,strict=True):
            source['training_motion_sha256']=fingerprint
    if output.exists():
        receipt = json.loads((output/'finalization.json').read_text())
        if receipt['sources']!=sources or receipt['job_id']!=name:
            raise FileExistsError(f'Dataset directory contains a different episode selection: {output}')
        loader_file = output/'official_loader_receipt.json'
        loader = json.loads(loader_file.read_text())
        if loader.get('passed') is not True or loader['episodes']!=len(runs):
            raise ValueError(f'Existing dataset lacks a passing loader receipt: {output}')
        if hashlib.sha256(loader_file.read_bytes()).hexdigest()!=receipt['loader_receipt_sha256']:
            raise ValueError(f'Existing loader receipt was changed: {output}')
        return receipt

    if not READER.is_file():raise FileNotFoundError(f'Project LeRobot reader is unavailable: {READER}')
    from export_lerobot import export
    from metrics_report import report
    staging = output.parent/('.'+name+'-staging-'+uuid.uuid4().hex[:10])
    try:
        export(runs, staging)
        with (staging/'official-loader.log').open('w') as log:
            subprocess.run([str(READER),str(ROOT/'verify_lerobot_reader.py'),str(staging)],
                           env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2'),
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        loader = json.loads((staging/'official_loader_receipt.json').read_text())
        if loader.get('passed') is not True or loader['episodes']!=len(runs) or loader.get('lerobot_commit')!=READER_COMMIT:
            raise ValueError('Reader receipt did not verify the complete episode selection')
        report(job_path, print_report=False)
        shutil.copy2(Path(job_path).parent/'metrics.json',staging/'metrics.json')
        receipt = {'job_id':name,'dataset_path':str(output),'format':'LeRobot v2.1','sources':sources,
                   'episodes':len(runs),'frames':loader['frames'],'official_reader_verified':True,
                   'loader_receipt_sha256':hashlib.sha256((staging/'official_loader_receipt.json').read_bytes()).hexdigest()}
        if job.get('require_distinct_motion'):
            receipt['all_training_motions_distinct']=True
        (staging/'finalization.json').write_text(json.dumps(receipt,indent=2))
        if output.exists():raise FileExistsError(output)
        staging.rename(output)
    except Exception as exc:
        raise RuntimeError(f'Dataset finalization failed. Preserved work directory: {staging}. {exc}') from exc
    return receipt
