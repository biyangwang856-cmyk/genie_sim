"""Catch modified download bytes and drift in the committed scene/calibration fixtures."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from collection_scope import TASKS, SCENES
from manage_runtime_inputs import prepare, ROOT
from verify_download import records, verify_files


class ReleaseTests(unittest.TestCase):
    def test_scene_and_input_fingerprints(self):
        self.assertEqual(prepare(), {})
        for task in TASKS:
            for scene in SCENES:
                path = ROOT / "tasks" / ("g2crsb90d_finemanip_" + task) / scene
                metadata = json.loads((path / "scene_info.json").read_bytes())
                self.assertEqual((metadata["task_template"], metadata["scene_id"]), (task, scene))
                self.assertTrue(metadata["layout"])
                self.assertTrue((path / "scene.usda").is_file())

    def test_public_inventories_match_release(self):
        assets = list(records("assets", ROOT))
        training = list(records("lerobot", ROOT))
        self.assertEqual((len(assets), sum(x["size"] for x in assets)), (20435, 43177589868))
        self.assertEqual((len(training), sum(x["size"] for x in training)), (442, 763358526))

    def test_same_size_corruption_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = {"path": "item", "size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
            (root / "item").write_bytes(b"abc")
            self.assertEqual(verify_files(root, [entry])["verified_files"], 1)
            (root / "item").write_bytes(b"bad")
            with self.assertRaises(ValueError):
                verify_files(root, [entry])
