"""Independently verify a native export with its matching official CPU reader."""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from lerobot_layout import DATASET_FORMAT, FILES_MANIFEST, LEGACY_FORMAT, dataset_info


TIME_TOLERANCE_S = 0.001


def episode_rows(root, version):
    if version == LEGACY_FORMAT:
        with (root / "meta/episodes.jsonl").open(encoding="utf-8") as source:
            return [json.loads(line) for line in source]
    paths = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    return [row for path in paths for row in pq.read_table(path).to_pylist()]


def compare_numeric(dataset, root, info, episodes):
    """Read each Parquet shard and verify all official reader values in order."""
    v3 = info["codebase_version"] == DATASET_FORMAT
    observed = dataset.hf_dataset.with_format("numpy")
    numeric = [key for key, spec in info["features"].items() if spec["dtype"] != "video"]
    offset = ep_index = 0
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(path)
        end = offset + len(table)
        actual = observed[offset:end]
        for key in numeric:
            np.testing.assert_array_equal(actual[key], np.asarray(table[key].to_pylist(),
                                          dtype=info["features"][key]["dtype"]), err_msg=key)
        local = 0
        while local < len(table):
            episode = episodes[ep_index]
            length = episode["length"]
            if length < 1 or local + length > len(table) or episode["episode_index"] != ep_index:
                raise ValueError(f"Invalid episode range in {path}")
            if v3:
                expected = root / info["data_path"].format(chunk_index=episode["data/chunk_index"],
                                                          file_index=episode["data/file_index"])
                if (path, offset + local, offset + local + length) != (
                        expected, episode["dataset_from_index"], episode["dataset_to_index"]):
                    raise ValueError(f"Episode {ep_index} points to incorrect row range")
            rows = table.slice(local, length)
            np.testing.assert_array_equal(rows["episode_index"].to_numpy(), np.full(length, ep_index))
            np.testing.assert_array_equal(rows["index"].to_numpy(), np.arange(offset + local, offset + local + length))
            np.testing.assert_array_equal(rows["frame_index"].to_numpy(), np.arange(length))
            np.testing.assert_allclose(rows["timestamp"].to_numpy(), np.arange(length) / info["fps"],
                                       atol=TIME_TOLERANCE_S, rtol=0)
            local += length
            ep_index += 1
        offset = end
    if offset != info["total_frames"] or offset != len(dataset) or ep_index != len(episodes):
        raise ValueError("Reader, Parquet shards and episode metadata disagree on totals")


def reference_frame(path, timestamp):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, math.floor((timestamp + TIME_TOLERANCE_S) / stream.time_base)),
                       stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            if abs(frame.time - timestamp) <= TIME_TOLERANCE_S:
                return frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
            if frame.time > timestamp:
                break
    raise ValueError(f"No camera frame at {timestamp}s in {path}")


def video_references(root, info, episodes):
    offset = 0
    previous = {}
    camera_keys = [key for key, spec in info["features"].items() if spec["dtype"] == "video"]
    for episode in episodes:
        length = episode["length"]
        cameras = {}
        if info["codebase_version"] == DATASET_FORMAT:
            for key in camera_keys:
                prefix = f"videos/{key}/"
                chunk, file = (episode[prefix + name] for name in ("chunk_index", "file_index"))
                start, end = (episode[prefix + name] for name in ("from_timestamp", "to_timestamp"))
                last_order, last_end = previous.get(key, ((-1, -1), 0))
                order = (chunk, file)
                expected_start = last_end if order == last_order else 0
                if (order < last_order or not math.isclose(start, expected_start, abs_tol=1e-8)
                        or not math.isclose(end - start, length / info["fps"], abs_tol=1e-8)):
                    raise ValueError(f"Invalid {key} offsets for episode {episode['episode_index']}")
                previous[key] = (order, end)
                path = root / info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file)
                cameras[key] = (path, start)
        for index in sorted({0, length // 2, length - 1}):
            yield offset + index, index / info["fps"], cameras, episode["tasks"]
        offset += length


def verify(root, receipt_path=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch

    root = Path(root).resolve()
    if receipt_path is None and any((root / name).exists() for name in ("finalization.json", FILES_MANIFEST)):
        raise ValueError("Use --receipt-path outside the dataset to recheck a published dataset")
    if receipt_path is not None and Path(receipt_path).resolve().is_relative_to(root):
        raise ValueError("Recheck receipts must be written outside the verified dataset")
    info = dataset_info(root)
    version = info["codebase_version"]
    options = {}
    if version == DATASET_FORMAT:
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

        options = {"return_uint8": True, "tolerance_s": TIME_TOLERANCE_S}
    else:
        from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != version:
        raise ValueError(f"Installed reader {CODEBASE_VERSION} cannot verify dataset {version}")
    torch.set_num_threads(2)
    dataset = LeRobotDataset(repo_id="local/geniesim_native", root=root, video_backend="pyav", **options)
    episodes = episode_rows(root, version)
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Episode metadata count differs from meta/info.json")
    compare_numeric(dataset, root, info, episodes)
    samples = []
    camera_keys = [key for key, spec in info["features"].items() if spec["dtype"] == "video"]
    for index, timestamp, cameras, tasks in video_references(root, info, episodes):
        sample = dataset[index]
        shapes = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite loaded sample: {key}")
                shapes[key] = list(value.shape)
        for key in camera_keys:
            height, width, channels = info["features"][key]["shape"]
            if shapes.get(key) != [channels, height, width]:
                raise ValueError(f"Unexpected image shape for {key} at {index}")
            if key in cameras:
                path, start = cameras[key]
                np.testing.assert_array_equal(sample[key].numpy(), reference_frame(path, start + timestamp),
                                              err_msg=f"{key} at dataset row {index}")
        if sample.get("task") not in tasks:
            raise ValueError(f"Row {index} resolved to an incorrect task label")
        samples.append({"index": index, "shapes": shapes, "task": sample["task"]})
    distribution = importlib.metadata.distribution("lerobot")
    source = json.loads(distribution.read_text("direct_url.json") or "{}")
    receipt = {"passed": True, "format": version, "reader_distribution": distribution.version,
               "lerobot_commit": source.get("vcs_info", {}).get("commit_id"),
               "torch_version": importlib.metadata.version("torch"),
               "torchvision_version": importlib.metadata.version("torchvision"),
               "datasets_version": importlib.metadata.version("datasets"), "pyav_version": av.__version__,
               "frames": len(dataset), "episodes": len(episodes), "numeric_readback_exact": True,
               "video_samples_match_shards": version == DATASET_FORMAT,
               "samples": samples, "used_gpu": False, "pi_action_adapter_verified": False}
    if receipt_path is not None:
        with Path(receipt_path).open("x", encoding="utf-8") as result:
            json.dump(receipt, result, ensure_ascii=False, indent=2)
    else:
        (root / "official_loader_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        export = root / "export_receipt.json"
        if export.is_file():
            result = json.loads(export.read_bytes())
            result["official_reader_verified"] = True
            export.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "samples"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--receipt-path", type=Path)
    args = parser.parse_args()
    verify(args.root, args.receipt_path)
