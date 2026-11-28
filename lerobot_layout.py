"""Dataset identities shared by publication, readers, and raw trajectory archives."""

import json
import re
from pathlib import Path


LEGACY_FORMAT = "v2.1"
DATASET_FORMAT = "v3.0"
READER_VERSION = "0.7.0"
SUFFIXES = {LEGACY_FORMAT: "-v2-1", DATASET_FORMAT: "-v3-0"}
FILES_MANIFEST = "dataset-files.json"
CHUNK_SIZE = 1000
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODE_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def dataset_name(job, version=DATASET_FORMAT):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job):
        raise ValueError("Invalid job ID for dataset directory")
    return job + SUFFIXES[version]


def dataset_identity(path):
    name = Path(path).name
    for version, suffix in SUFFIXES.items():
        if name.endswith(suffix):
            job = name.removesuffix(suffix)
            if dataset_name(job, version) == name:
                return job, version
    raise ValueError(f"Unknown LeRobot dataset directory: {name}")


def raw_archive_path(dataset):
    path = Path(dataset)
    job, _ = dataset_identity(path)
    return path.with_name(job + "-raw-trajectories")


def dataset_info(root):
    info = json.loads((Path(root) / "meta/info.json").read_bytes())
    if info.get("codebase_version") not in SUFFIXES:
        raise ValueError("Unsupported LeRobot dataset format in meta/info.json")
    return info
