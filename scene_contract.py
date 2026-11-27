"""Validated requests and immutable inputs for a specific task/scene instance."""
import hashlib
import json
from pathlib import Path
import re

from robot_adapters import UnsupportedRequest


NEW_SCENE_MODES = ("existing", "compose_from_library")
VARIATIONS = ("background", "support", "layout", "distractors", "lighting")
SCENE_FILES = ("scene.usda", "scene_info.json", "task_info.json")
SCENE_REQUEST_FIELDS = {"mode", "scene_count", "scene_ids", "background_query", "background_id",
                        "support_query", "support_id", "object_policy", "variation", "seed"}


def is_id(value):
    return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value) is not None


def require_id(value):
    if not is_id(value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def count(value, name, upper):
    if type(value) is not int or not 1 <= value <= upper:
        raise ValueError(f"{name} must be an integer from 1 to {upper}")
    return value


def scene_request(raw):
    if not isinstance(raw, dict):
        raise ValueError("scene_request must be an object")
    if unknown := raw.keys() - SCENE_REQUEST_FIELDS:
        raise ValueError(f"Unknown scene_request fields: {sorted(unknown)}")
    mode = raw.get("mode", "compose_from_library")
    if mode not in NEW_SCENE_MODES:
        raise ValueError("Unsupported scene_request mode")
    result = dict(raw, mode=mode)
    for key in ("background_query", "support_query"):
        if key in raw and (not isinstance(raw[key], str) or not raw[key].strip() or len(raw[key]) > 240):
            raise ValueError(f"{key} must be a nonempty description of at most 240 characters")
    for key in ("background_id", "support_id"):
        if key in raw:
            require_id(raw[key])
    if "scene_ids" in raw:
        ids = raw["scene_ids"]
        if mode != "existing" or not isinstance(ids, list) or not ids or any(not is_id(v) for v in ids) or len(set(ids)) != len(ids):
            raise ValueError("scene_ids must be distinct IDs for existing mode")
        if raw.get("scene_count", len(ids)) != len(ids):
            raise ValueError("scene_count conflicts with scene_ids")
    if mode == "existing" and {"background_id", "support_id", "variation", "support_query"} & raw.keys():
        raise ValueError("Existing scenes are immutable. Use compose_from_library for layout or asset changes.")
    result["scene_count"] = count(raw.get("scene_count", len(raw.get("scene_ids", [])) or 1), "scene_count", 12)
    if "seed" in raw and (type(raw["seed"]) is not int or not 0 <= raw["seed"] < 2**31):
        raise ValueError("seed must be a nonnegative 31-bit integer")
    if raw.get("object_policy", "validated_assembly_pairs") != "validated_assembly_pairs":
        raise UnsupportedRequest("No object replacement policy beyond verified assembly pairs",
                                 missing_assets=["verified_assembly_pair"])
    result["object_policy"] = "validated_assembly_pairs"
    if mode == "compose_from_library":
        variation = raw.get("variation", ["background", "layout"])
        if not isinstance(variation, list) or not variation or any(v not in VARIATIONS for v in variation):
            raise ValueError(f"variation must select from {VARIATIONS}")
        result["variation"] = list(dict.fromkeys(variation))
    return result


def read_json(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def choose_replacement(reserves, trials, scene_id):
    history = [row.get("scene_failure_signature") for row in trials if row["scene"] == scene_id]
    if not history or not reserves:
        return None
    latest = history[-1]
    if latest and (latest == "initial_support_or_stability" or history[-2:] == [latest, latest]):
        return reserves[0], latest
    return None


def write_manifest(directory, dependency_paths, *, mode, source, asset_root):
    directory, asset_root = Path(directory), Path(asset_root).resolve()
    dependencies = {}
    for dependency in sorted({Path(path).resolve() for path in dependency_paths}):
        relative = dependency.relative_to(asset_root).as_posix()
        dependencies[relative] = digest(dependency)
    manifest = {"schema": 1, "mode": mode, "source": source,
                "files": {name: digest(directory / name) for name in SCENE_FILES},
                "asset_release_sha256": digest(asset_root / ".geniesim-assets.json"),
                "asset_dependencies": dependencies}
    manifest["content_sha256"] = content_id(manifest)
    (directory / "scene-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_manifest(directory, asset_root):
    directory, asset_root = Path(directory), Path(asset_root).resolve()
    manifest = read_json(directory / "scene-manifest.json")
    expected = manifest["content_sha256"]
    actual = content_id({key: value for key, value in manifest.items() if key != "content_sha256"})
    if manifest["schema"] != 1 or expected != actual or set(manifest["files"]) != set(SCENE_FILES):
        raise ValueError("Invalid scene manifest")
    for filename, sha in manifest["files"].items():
        if digest(directory / filename) != sha:
            raise ValueError(f"Scene changed after preparation: {filename}")
    if digest(asset_root / ".geniesim-assets.json") != manifest["asset_release_sha256"]:
        raise ValueError("Asset release changed after preparation")
    for relative, sha in manifest["asset_dependencies"].items():
        path = (asset_root / relative).resolve()
        if not path.is_relative_to(asset_root) or digest(path) != sha:
            raise ValueError(f"Asset dependency changed: {relative}")
    return manifest
