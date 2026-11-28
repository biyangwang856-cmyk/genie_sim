"""Package the accepted native recordings in the order of a verified LeRobot export."""

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

from collection_control import STORAGE
from export_lerobot import digest, recording_path
from lerobot_layout import DATASET_FORMAT, FILES_MANIFEST, dataset_identity, dataset_info, raw_archive_path


RUNS = STORAGE / 'collection-runs'
EXPORTS = STORAGE / 'exports'
DIRECTORIES = ('raw', 'raw-aligned', 'source', 'tasks', 'evidence', 'review', 'postprocess-source')
DATASET_RECEIPTS = ('finalization.json', 'export_receipt.json', 'official_loader_receipt.json', 'metrics.json')

README = """# Genie Sim 原始轨迹

回合目录名 episode_XXXXXX 和正式 LeRobot 数据集的 episode_index 一致。
根目录 manifest.json 可查每个回合的任务、场景、同步录像位置和文件校验值。

- raw/ 是首次原始归档，含 MCAP、状态、视频和记录元数据。
- 有 raw-aligned/ 时，正式训练使用从同一 MCAP 提取的同步版本。它在包内复用同一份 MCAP 文件。
- evidence/native-control-commands.jsonl 记录实际下发的原生关节命令。
- source/、tasks/、evidence/ 和 review/ 保存计划、场景、控制、物理验收和复核证据。
- dataset-receipts/ 保存正式训练集的验收及调用统计。

原始记录的元数据保留当时的路径。读取这份归档时，请用 manifest.json 中相对本目录的数据路径。
"""


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def episode_files(run):
    files = {path.relative_to(run): path for path in run.iterdir()
             if path.is_file() and path.suffix in {'.json', '.log'}}
    for name in DIRECTORIES:
        directory = run / name
        if directory.is_symlink():
            raise ValueError(f'Unexpected trajectory directory link: {directory}')
        if not directory.is_dir():
            continue
        for path in directory.rglob('*'):
            relative = path.relative_to(run)
            if '__pycache__' in relative.parts or path.name == 'audit.lock':
                continue
            if path.is_symlink() and not path.is_file():
                raise ValueError(f'Invalid link in trajectory: {path}')
            if path.is_file():
                files[relative] = path
    return files


def episode_plan(dataset_source, final_source):
    run = Path(final_source['run']).resolve()
    if run.parent != RUNS or str(run) != dataset_source['run']:
        raise ValueError(f'Invalid raw trajectory source: {run}')
    if digest(run / 'launch.json') != final_source['launch_sha256']:
        raise ValueError(f'Launch receipt changed for {run}')
    acceptance = read_json(run / 'acceptance.json')
    raw_archive = read_json(run / 'evidence/raw-archive.json')
    if not acceptance['accepted'] or not raw_archive['archived'] or acceptance['frames'] != dataset_source['frames']:
        raise ValueError(f'Raw trajectory does not match the accepted sample: {run}')

    active = recording_path(run).resolve()
    if active.parent != run or active.name not in {'raw', 'raw-aligned'} or str(active) != acceptance['raw_directory']:
        raise ValueError(f'Invalid training recording directory for {run}')
    files = episode_files(run)
    originals = {Path('raw') / row['path']: row for row in raw_archive['files']}
    if not originals.keys() <= files.keys():
        raise FileNotFoundError(f'An original recording file is missing from {run}')

    record = active.relative_to(run)
    cameras = read_json(active / 'recording_info.json')['camera_info']
    videos = {name: record / 'observations/videos' / camera['output']['video'] for name, camera in cameras.items()}
    state = record / 'state.json'
    commands = Path('evidence/native-control-commands.jsonl')
    bags = sorted(path for path in originals if path.suffix == '.mcap')
    if not bags or len(videos) != 3 or not {state, commands, *videos.values()} <= files.keys():
        raise ValueError(f'MCAP, original commands, state, or camera data is incomplete: {run}')

    root = Path(f"episode_{dataset_source['episode_index']:06d}")
    info = {
        'episode_index': dataset_source['episode_index'],
        'task': dataset_source['task_id'],
        'scene': str(dataset_source['scene_id']),
        'frames': dataset_source['frames'],
        'source_run': run.name,
        'training_motion_sha256': final_source.get('training_motion_sha256'),
        'directory': root.as_posix(),
        'first_recording': (root / 'raw').as_posix(),
        'synchronized_recording': (root / record).as_posix(),
        'mcap': [(root / path).as_posix() for path in bags],
        'joint_and_scene_observations': (root / state).as_posix(),
        'native_commands': (root / commands).as_posix(),
        'camera_videos': {name: (root / path).as_posix() for name, path in videos.items()},
    }
    return run, files, originals, info


def copy_verified(source, dest, expected=None, *, link=False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    checksum = digest(source)
    size = source.stat().st_size
    if expected and (checksum, size) != (expected['sha256'], expected['bytes']):
        raise ValueError(f'Raw archive checksum changed: {source}')
    if link:
        os.link(source,dest)
    else:
        shutil.copy2(source,dest)
    if dest.stat().st_size != size or digest(dest) != checksum:
        raise OSError(f'Copy verification failed: {dest}')
    return {'sha256': checksum, 'bytes': size}


def copy_episode(staging, run, files, originals, episode, *, link=False):
    root = staging / episode['directory']
    contents = {}
    for relative, source in sorted(files.items()):
        if not source.is_symlink():
            contents[relative] = copy_verified(source, root / relative, originals.get(relative),link=link and relative.parts[0] in ('raw','raw-aligned'))
    for relative, source in sorted(files.items()):
        if source.is_symlink():
            target = source.resolve().relative_to(run)
            if target not in contents or relative in originals:
                raise ValueError(f'Unexpected source link: {source}')
            # Recreate the aligned MCAP reference inside this independent copy.
            link = root / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            os.link(root / target, link)
            contents[relative] = {**contents[target], 'same_file_as': target.as_posix()}

    episode['files'] = {path.as_posix(): record for path, record in sorted(contents.items())}
    print(json.dumps({k: episode[k] for k in ('episode_index', 'source_run', 'frames')}), flush=True)
    return sum(row['bytes'] for row in contents.values() if 'same_file_as' not in row)


def bundle(dataset, *, link=False):
    dataset = Path(dataset).resolve()
    if dataset.parent != EXPORTS:
        raise ValueError('Select a published LeRobot dataset in the project exports directory')
    _, version = dataset_identity(dataset)
    info = dataset_info(dataset)
    final = read_json(dataset / 'finalization.json')
    export = read_json(dataset / 'export_receipt.json')
    loader = read_json(dataset / 'official_loader_receipt.json')
    if not final['official_reader_verified'] or not loader['passed'] or not export['official_reader_verified']:
        raise ValueError('Only a reader-verified dataset can be bundled')
    if (info['codebase_version'] != version or loader['format'] != version
            or final.get('dataset_format', version) != version
            or info['total_frames'] != loader['frames'] or info['total_episodes'] != loader['episodes']):
        raise ValueError('Dataset metadata and reader receipt disagree')
    if digest(dataset / 'official_loader_receipt.json') != final['loader_receipt_sha256']:
        raise ValueError('Reader receipt has changed')
    if len(final['sources']) != final['episodes'] or final['episodes'] != loader['episodes']:
        raise ValueError('Dataset episode selection does not match the reader receipt')
    plans = [episode_plan(source, verified)
             for source, verified in zip(export['sources'], final['sources'], strict=True)]
    indexes = [p[3]['episode_index'] for p in plans]
    if indexes != list(range(final['episodes'])) or sum(p[3]['frames'] for p in plans) != loader['frames']:
        raise ValueError('Source episode indices or frame counts do not match the dataset')

    output = raw_archive_path(dataset)
    if output.exists():
        manifest=read_json(output/'manifest.json')
        if (manifest['training_dataset']==str(dataset) and manifest['episodes']==len(plans)
            and manifest['dataset_receipts']['finalization.json']['sha256']==digest(dataset/'finalization.json')):
            return {'archive':str(output),'episodes':manifest['episodes'],'frames':manifest['frames'],
                    'bytes':manifest['copied_episode_file_bytes']}
        raise FileExistsError(output)
    size = sum(p.stat().st_size for _, files, _, _ in plans for rel,p in files.items()
               if not p.is_symlink() and (not link or rel.parts[0] not in ('raw','raw-aligned')))
    if shutil.disk_usage(output.parent).free < size:
        raise OSError(f'Insufficient storage for {size} bytes of raw trajectories')
    staging = output.parent / f'.{output.name}-staging-{uuid.uuid4().hex[:10]}'
    staging.mkdir()
    try:
        receipt_names = DATASET_RECEIPTS + ((FILES_MANIFEST,) if version == DATASET_FORMAT else ())
        receipts = {name: copy_verified(dataset / name, staging / 'dataset-receipts' / name)
                    for name in receipt_names}
        copied = sum(copy_episode(staging, *plan,link=link) for plan in plans)
        manifest = {
            'training_dataset': str(dataset),
            'dataset_format': version,
            'episodes': final['episodes'],
            'frames': loader['frames'],
            'copied_episode_file_bytes': copied,
            'all_files_checksum_verified': True,
            'storage_method':'local_hardlinks_for_recordings' if link else 'independent_copy',
            'dataset_receipts': receipts,
            'trajectory_index': [p[3] for p in plans],
        }
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        note='\n服务器归档中的录制文件使用同盘硬链接。复制此目录后可独立读取。\n' if link else ''
        (staging / 'README.md').write_text(README+note, encoding='utf-8')
        staging.rename(output)
    except Exception as exc:
        raise RuntimeError(f'Raw bundle interrupted. Partial data is at {staging}') from exc
    receipt={'archive': str(output), 'episodes': len(plans), 'frames': loader['frames'], 'bytes': copied}
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    bundle(parser.parse_args().dataset)
