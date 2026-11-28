"""Regression coverage for shared episode shards and download integrity."""

import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest

from lerobot_layout import DATASET_FORMAT, FILES_MANIFEST, LEGACY_FORMAT, dataset_identity, dataset_name, raw_archive_path
from verify_download import records, verify_files, write_dataset_inventory


class DatasetLayoutTests(unittest.TestCase):
    def test_published_versions_share_the_job_raw_archive_name(self):
        newest = Path("exports") / dataset_name("test-job")
        legacy = Path("exports") / dataset_name("test-job", LEGACY_FORMAT)
        self.assertEqual(dataset_identity(newest), ("test-job", DATASET_FORMAT))
        self.assertEqual(dataset_identity(legacy), ("test-job", LEGACY_FORMAT))
        self.assertEqual(raw_archive_path(newest), Path("exports/test-job-raw-trajectories"))
        self.assertEqual(raw_archive_path(newest), raw_archive_path(legacy))

    def test_v3_download_rejects_changed_metadata_or_extra_shards(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = ["meta/tasks.parquet", "meta/episodes/chunk-000/file-000.parquet", "meta/stats.json",
                     "official_loader_receipt.json", "export_receipt.json", "finalization.json",
                     "data/chunk-000/file-000.parquet", "videos/observation.images.cam_high/chunk-000/file-000.mp4"]
            for relative in files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"sample")
            (root / "meta/info.json").write_text(json.dumps({"codebase_version": DATASET_FORMAT}), encoding="utf-8")
            write_dataset_inventory(root)
            self.assertTrue((root / FILES_MANIFEST).is_file())
            self.assertEqual(verify_files(root, records("lerobot", root))["verified_files"], len(files) + 1)
            path = root / files[1]
            path.write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "Missing, incomplete or incorrect"):
                verify_files(root, records("lerobot", root))
            path.write_bytes(b"sample")
            path.with_name("file-001.parquet").write_bytes(b"sample")
            with self.assertRaisesRegex(ValueError, "layout differs"):
                verify_files(root, records("lerobot", root))


class SharedShardTests(unittest.TestCase):
    def test_shared_video_keeps_episode_frame_order_and_offsets(self):
        try:
            import av
            import numpy as np
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            self.skipTest(str(exc))
        env_ffmpeg = Path(sys.executable).parent / "ffmpeg"
        ffmpeg = env_ffmpeg if env_ffmpeg.is_file() else shutil.which("ffmpeg")
        if ffmpeg is None:
            self.skipTest("FFmpeg is required for the shared-video regression")

        from export_lerobot import stats
        from lerobot_v3 import V3Shards, probe_clip

        # The lab FFmpeg launcher runs in a container with the project disk mounted.
        with TemporaryDirectory(prefix=".test-v3-", dir=Path.cwd().resolve()) as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            writer = V3Shards(root, 30, ffmpeg)
            key = "observation.images.cam_high"
            episodes, statistics = [], []
            offset = 0
            for index, (length, intensity) in enumerate(((3, 20), (5, 220))):
                source = root / f"input-{index}.mp4"
                with av.open(str(source), "w") as container:
                    stream = container.add_stream("libx264", rate=30)
                    stream.width = stream.height = 32
                    stream.pix_fmt = "yuv420p"
                    stream.codec_context.max_b_frames = 0
                    for frame in range(length):
                        pixels = np.full((32, 32, 3), intensity + frame, dtype=np.uint8)
                        for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
                values = np.arange(offset, offset + length)
                _, video_stats, clip = probe_clip(source, length, 30)
                episodes.append({"episode_index": index, "tasks": [f"Task {index}"], "length": length,
                                 "dataset_from_index": offset, "dataset_to_index": offset + length,
                                 **writer.add_data(pa.table({"index": values})), **writer.add_video(key, clip)})
                statistics.append({"index": stats(values), key: video_stats})
                offset += length

            writer.finish(["Task 0", "Task 1"], episodes, statistics)
            files = list((root / "videos").rglob("*.mp4"))
            self.assertEqual(len(files), 1)
            with av.open(str(files[0])) as container:
                observed = [frame.to_ndarray(format="rgb24").mean() for frame in container.decode(video=0)]
            self.assertEqual(len(observed), 8)
            self.assertLess(observed[2], 40)
            self.assertGreater(observed[3], 200)
            rows = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
            self.assertEqual((rows[1]["dataset_from_index"], rows[1]["dataset_to_index"]), (3, 8))
            self.assertEqual((rows[0]["data/file_index"], rows[1]["data/file_index"]), (0, 0))
            self.assertAlmostEqual(rows[1][f"videos/{key}/from_timestamp"], 3 / 30)
            self.assertEqual(rows[1][f"videos/{key}/file_index"], 0)
            global_stats = json.loads((root / "meta/stats.json").read_bytes())
            self.assertEqual(global_stats["index"]["mean"], [3.5])
            self.assertEqual(global_stats[key]["count"], [8])
