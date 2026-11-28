"""Native LeRobot v3.0 shards for accepted Genie Sim recordings."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_layout import CHUNK_SIZE, DATA_PATH, EPISODE_PATH, VIDEO_PATH


DATA_SIZE_MB = 100
VIDEO_SIZE_MB = 200
MIB = 1024 * 1024
TIME_TOLERANCE_S = 0.001


def shard_location(path_pattern, number, **kwargs):
    chunk, file = divmod(number, CHUNK_SIZE)
    return path_pattern.format(chunk_index=chunk, file_index=file, **kwargs), chunk, file


def verify_frame_time(frame, index, fps, path):
    if frame.time is None or abs(frame.time - index / fps) > TIME_TOLERANCE_S:
        raise ValueError(f"Invalid timestamp for frame {index} in {path}: {frame.time}")


def probe_clip(path, length, fps):
    """Check native camera timing and collect per-episode RGB stats and packet evidence."""
    counts = np.zeros(3)
    squares = np.zeros(3)
    lo = np.full(3, 255)
    hi = np.zeros(3)
    encoded = hashlib.sha256()
    frames = packets = 0
    with av.open(str(path)) as container:
        if len(container.streams) != 1 or len(container.streams.video) != 1:
            raise ValueError(f"Expected one video stream in {path}")
        stream = container.streams.video[0]
        context = stream.codec_context
        codec = context.name
        if stream.average_rate != fps or codec not in {"h264", "hevc"}:
            raise ValueError(f"Unexpected native video codec or frame rate in {path}")
        signature = (codec, stream.width, stream.height, stream.pix_fmt, stream.time_base,
                     context.extradata)
        for packet in container.demux(stream):
            if packet.size:
                if packets == 0 and not packet.is_keyframe:
                    raise ValueError(f"Recording starts without a keyframe: {path}")
                data = bytes(packet)
                encoded.update(len(data).to_bytes(8, "little"))
                encoded.update(data)
                packets += 1
            for frame in packet.decode():
                verify_frame_time(frame, frames, fps, path)
                pixels = frame.reformat(width=64, height=64, format="rgb24",
                                        interpolation="POINT").to_ndarray().reshape(-1, 3)
                lo = np.minimum(lo, pixels.min(axis=0))
                hi = np.maximum(hi, pixels.max(axis=0))
                counts += pixels.sum(axis=0)
                squares += np.square(pixels, dtype=np.float64).sum(axis=0)
                frames += 1
        if frames != length:
            raise ValueError(f"Expected {length} frames, decoded {frames} in {path}")
        height, width = stream.height, stream.width
        info = {"video.height": height, "video.width": width, "video.codec": codec,
                "video.pix_fmt": stream.pix_fmt, "video.is_depth_map": False,
                "video.fps": fps, "video.channels": 3, "has_audio": False}
    mean = counts / (length * 64 * 64)
    std = np.sqrt(np.maximum(squares / (length * 64 * 64) - mean**2, 0))
    statistics = {key: (value / 255).reshape(3, 1, 1).tolist()
                  for key, value in (("min", lo), ("max", hi), ("mean", mean), ("std", std))}
    statistics["count"] = [length]
    return {"dtype": "video", "shape": [height, width, 3],
            "names": ["height", "width", "channels"], "info": info}, statistics, (
                path, length, signature, packets, encoded.hexdigest())


def check_video_shard(path, clips, fps):
    frames = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        packets = (packet for packet in container.demux(stream) if packet.size)
        for _, _, _, count, expected_hash in clips:
            encoded = hashlib.sha256()
            for _ in range(count):
                packet = next(packets, None)
                if packet is None:
                    raise ValueError(f"Truncated video shard: {path}")
                data = bytes(packet)
                encoded.update(len(data).to_bytes(8, "little"))
                encoded.update(data)
                for frame in packet.decode():
                    verify_frame_time(frame, frames, fps, path)
                    frames += 1
            if encoded.hexdigest() != expected_hash:
                raise ValueError(f"Encoded camera payload changed in {path}")
        if next(packets, None) is not None:
            raise ValueError(f"Unexpected extra video packets: {path}")
        for frame in stream.codec_context.decode(None):
            verify_frame_time(frame, frames, fps, path)
            frames += 1
    if frames != sum(clip[1] for clip in clips):
        raise ValueError(f"Decoded shard frame count differs from episode lengths: {path}")


def write_video_shard(path, clips, fps, ffmpeg):
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(clips) == 1:
        shutil.copy2(clips[0][0], path)
    else:
        playlist = path.with_suffix(".ffconcat")
        entries = ["ffconcat version 1.0"]
        for source, length, *_ in clips:
            filename = source.as_posix()
            if "\n" in filename or "\r" in filename:
                raise ValueError("Invalid camera filename for FFmpeg")
            quoted = filename.replace("'", "'\\''")
            entries.extend((f"file '{quoted}'", f"duration {length / fps:.9f}"))
        playlist.write_text("\n".join(entries) + "\n", encoding="utf-8")
        try:
            subprocess.run([str(ffmpeg), "-nostdin", "-v", "error", "-f", "concat", "-safe", "0",
                            "-auto_convert", "0", "-i", str(playlist), "-map", "0:v:0",
                            "-c:v", "copy", "-movflags", "+faststart", str(path)],
                           check=True, stdin=subprocess.DEVNULL)
        finally:
            playlist.unlink()
    check_video_shard(path, clips, fps)


def aggregate_stats(episode_stats):
    result = {}
    for key in episode_stats[0]:
        rows = [row[key] for row in episode_stats]
        counts = np.asarray([row["count"][0] for row in rows], dtype=np.float64)
        means = np.asarray([row["mean"] for row in rows])
        mean = np.average(means, axis=0, weights=counts)
        variance = np.average(np.asarray([row["std"] for row in rows])**2 + (means - mean)**2,
                              axis=0, weights=counts)
        result[key] = {"min": np.min([row["min"] for row in rows], axis=0).tolist(),
                       "max": np.max([row["max"] for row in rows], axis=0).tolist(),
                       "mean": mean.tolist(), "std": np.sqrt(variance).tolist(),
                       "count": [int(counts.sum())]}
    return result


def write_tasks(path, tasks):
    # The official reader uses the Pandas index for task names, even when Arrow writes the file.
    table = pa.table({"task_index": pa.array(range(len(tasks)), type=pa.int64()), "task": tasks})
    metadata = {"index_columns": ["task"], "column_indexes": [],
                "columns": [{"name": "task_index", "field_name": "task_index",
                             "pandas_type": "int64", "numpy_type": "int64", "metadata": None},
                            {"name": "task", "field_name": "task", "pandas_type": "unicode",
                             "numpy_type": "object", "metadata": None}],
                "creator": {"library": "pyarrow", "version": pa.__version__}}
    pq.write_table(table.replace_schema_metadata({b"pandas": json.dumps(metadata).encode()}), path)


class V3Shards:
    """Bounded, episode-aligned buffers for numeric and camera shards."""

    def __init__(self, root, fps, ffmpeg):
        self.root = Path(root)
        self.fps = fps
        self.ffmpeg = ffmpeg
        self.data_tables = []
        self.data_bytes = 0
        self.data_number = 0
        self.videos = {}

    def add_data(self, table):
        if self.data_tables and self.data_bytes + table.nbytes > DATA_SIZE_MB * MIB:
            self.flush_data()
        self.data_tables.append(table)
        self.data_bytes += table.nbytes
        _, chunk, file = shard_location(DATA_PATH, self.data_number)
        return {"data/chunk_index": chunk, "data/file_index": file}

    def flush_data(self):
        if not self.data_tables:
            return
        relative, _, _ = shard_location(DATA_PATH, self.data_number)
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.concat_tables(self.data_tables)
        pq.write_table(table, path, compression="zstd")
        if not pq.read_table(path).equals(table):
            raise ValueError(f"Numeric shard changed on Parquet readback: {path}")
        self.data_tables = []
        self.data_bytes = 0
        self.data_number += 1

    def add_video(self, key, clip):
        buffer = self.videos.setdefault(key, {"number": 0, "clips": [], "frames": 0, "bytes": 0})
        path, length, signature, *_ = clip
        size = path.stat().st_size
        if buffer["clips"] and (buffer["clips"][0][2] != signature
                                or buffer["bytes"] + size > VIDEO_SIZE_MB * MIB
                                or len(buffer["clips"]) == CHUNK_SIZE):
            self.flush_video(key)
        start = buffer["frames"]
        buffer["clips"].append(clip)
        buffer["frames"] += length
        buffer["bytes"] += size
        _, chunk, file = shard_location(VIDEO_PATH, buffer["number"], video_key=key)
        return {f"videos/{key}/chunk_index": chunk, f"videos/{key}/file_index": file,
                f"videos/{key}/from_timestamp": start / self.fps,
                f"videos/{key}/to_timestamp": (start + length) / self.fps}

    def flush_video(self, key):
        buffer = self.videos[key]
        if buffer["clips"]:
            relative, _, _ = shard_location(VIDEO_PATH, buffer["number"], video_key=key)
            write_video_shard(self.root / relative, buffer["clips"], self.fps, self.ffmpeg)
            buffer.update(number=buffer["number"] + 1, clips=[], frames=0, bytes=0)

    def finish(self, tasks, episodes, episode_stats):
        self.flush_data()
        for key in self.videos:
            self.flush_video(key)
        write_tasks(self.root / "meta/tasks.parquet", tasks)
        for start in range(0, len(episodes), CHUNK_SIZE):
            relative, chunk, file = shard_location(EPISODE_PATH, start // CHUNK_SIZE)
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for episode, stats in zip(episodes[start:start + CHUNK_SIZE],
                                      episode_stats[start:start + CHUNK_SIZE], strict=True):
                flat = {f"stats/{key}/{stat}": value for key, group in stats.items()
                        for stat, value in group.items()}
                rows.append({**episode, "meta/episodes/chunk_index": chunk,
                             "meta/episodes/file_index": file, **flat})
            pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        (self.root / "meta/stats.json").write_text(
            json.dumps(aggregate_stats(episode_stats), indent=2), encoding="utf-8")
