"""Turn a quota-complete job into a reader-verified training dataset."""
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from collection_control import ROOT, STORAGE
from lerobot_layout import DATASET_FORMAT, FILES_MANIFEST, LEGACY_FORMAT, READER_VERSION, dataset_info, dataset_name
from verify_download import write_dataset_inventory

READER = STORAGE / 'lerobot-reader-v3/bin/python'


def check_loader(dataset, version, episodes):
    info = dataset_info(dataset)
    loader = json.loads((dataset/'official_loader_receipt.json').read_bytes())
    if (info['codebase_version'] != version or loader.get('format') != version
            or loader.get('passed') is not True or loader['episodes'] != episodes
            or info['total_episodes'] != episodes or loader['frames'] != info['total_frames']):
        raise ValueError(f'Reader receipt did not verify the complete episode selection: {dataset}')
    if version == DATASET_FORMAT and (loader.get('reader_distribution') != READER_VERSION
                                      or loader.get('numeric_readback_exact') is not True
                                      or loader.get('video_samples_match_shards') is not True):
        raise ValueError(f'Dataset requires the verified LeRobot {READER_VERSION} reader: {dataset}')
    return loader


def quota_runs(job, *, allow_partial=False):
    requested = [(task['task_id'], str(quota['scene_id']), quota['successes'])
                 for task in job['plan']['tasks'] for quota in task['scene_quotas']]
    runs = []
    for task, scene, count in requested:
        matching = [entry for entry in job['attempts'] if entry['task']==task and str(entry['scene'])==scene
                    and entry.get('accepted') is True and entry.get('automatic_acceptance') is True
                    and entry.get('raw_archive') is True]
        if len(matching)<count and not allow_partial:
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


def finalize(job, job_path, *, allow_partial=False):
    requested_format = job.get('dataset_format')
    if requested_format is not None and requested_format != DATASET_FORMAT:
        raise ValueError(f'Collection requires LeRobot Dataset {DATASET_FORMAT}')
    runs = quota_runs(job, allow_partial=allow_partial)
    name = job['job_id']
    selection = hashlib.sha256(json.dumps([str(run) for run in runs]).encode()).hexdigest()[:14]
    if allow_partial:name += '-partial-'+selection
    output = STORAGE/'exports'/dataset_name(name)
    sources = [{'run':str(run),'launch_sha256':hashlib.sha256((run/'launch.json').read_bytes()).hexdigest()} for run in runs]
    trajectory = job.get('trajectory_request')
    diversity_summary = None
    if trajectory is not None:
        from trajectory_admission import publish_lineage, summarize_selection

        diversity_summary = summarize_selection(job, runs)
    if trajectory is not None or job.get('require_distinct_motion'):
        from batch_collect import motion_fingerprint
        fingerprints=[motion_fingerprint(run) for run in runs]
        if job.get('require_distinct_motion') and len(set(fingerprints))!=len(runs):
            raise ValueError('Dataset selection still contains repeated measured state/action motion')
        for source,fingerprint in zip(sources,fingerprints,strict=True):
            if trajectory is not None:
                seal = json.loads((Path(source['run'])/'automatic-admission.json').read_bytes())
                if fingerprint != seal['motion_sha256']:
                    raise ValueError('Training motion changed after diversity admission')
            source['training_motion_sha256']=fingerprint
    reusable_formats = (DATASET_FORMAT, LEGACY_FORMAT) if requested_format is None else (DATASET_FORMAT,)
    for version in reusable_formats:
        published = output.with_name(dataset_name(name, version))
        if not published.exists():
            continue
        receipt = json.loads((published/'finalization.json').read_bytes())
        if receipt['sources']!=sources or receipt['job_id']!=name:
            raise FileExistsError(f'Dataset directory contains a different episode selection: {published}')
        loader_file = published/'official_loader_receipt.json'
        loader = check_loader(published, version, len(runs))
        if (receipt.get('official_reader_verified') is not True or receipt['frames'] != loader['frames']
                or Path(receipt['dataset_path']).resolve() != published.resolve()):
            raise ValueError(f'Existing publication receipt does not match the dataset: {published}')
        if version == DATASET_FORMAT and not (published/FILES_MANIFEST).is_file():
            raise ValueError(f'Existing dataset has no integrity inventory: {published}')
        if hashlib.sha256(loader_file.read_bytes()).hexdigest()!=receipt['loader_receipt_sha256']:
            raise ValueError(f'Existing loader receipt was changed: {published}')
        if diversity_summary is not None:
            if receipt.get('trajectory_diversity') != diversity_summary:
                raise ValueError('Existing dataset has different trajectory diversity evidence')
            publish_lineage(job, receipt, STORAGE/'trajectory-admissions.sqlite3')
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
        loader = check_loader(staging, DATASET_FORMAT, len(runs))
        report(job_path, print_report=False)
        shutil.copy2(Path(job_path).parent/'metrics.json',staging/'metrics.json')
        receipt = {'job_id':name,'dataset_path':str(output),'format':f'LeRobot {DATASET_FORMAT}',
                   'dataset_format':DATASET_FORMAT,'files_manifest':FILES_MANIFEST,'sources':sources,
                   'episodes':len(runs),'frames':loader['frames'],'official_reader_verified':True,
                   'loader_receipt_sha256':hashlib.sha256((staging/'official_loader_receipt.json').read_bytes()).hexdigest()}
        if job.get('require_distinct_motion'):
            receipt['all_training_motions_distinct']=True
        if diversity_summary is not None:
            receipt['trajectory_diversity'] = diversity_summary
        if allow_partial:
            receipt['source_job_id']=job['job_id']
            receipt['requested_quotas_complete']=False
        (staging/'finalization.json').write_text(json.dumps(receipt,indent=2))
        write_dataset_inventory(staging)
        if output.exists():raise FileExistsError(output)
        staging.rename(output)
    except Exception as exc:
        raise RuntimeError(f'Dataset finalization failed. Preserved work directory: {staging}. {exc}') from exc
    if diversity_summary is not None:
        publish_lineage(job, receipt, STORAGE/'trajectory-admissions.sqlite3')
    return receipt
