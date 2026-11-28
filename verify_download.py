"""Verify a release or job download without a simulator or network."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile

from lerobot_layout import DATASET_FORMAT, FILES_MANIFEST, dataset_info


RELEASE = Path(__file__).resolve().parent / "release"
IMAGE_ID = "21b42560fa351db97478fa77851199fcfabb7af01ad3d35af1a8a80e05ac6732"
IMAGE_TAG = "geniesim-by-collection:a100-20261119"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_dataset_inventory(root):
    files = [{"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": digest(path)}
             for path in sorted(root.rglob("*")) if path.is_file() and path != root / FILES_MANIFEST]
    inventory = {"dataset_format": DATASET_FORMAT, "files": files}
    with (root / FILES_MANIFEST).open("x", encoding="utf-8") as stream:
        json.dump(inventory, stream, indent=2)


def records(kind, root, *, manifest_sha256=None):
    release = json.loads((RELEASE / "reference.json").read_bytes())
    if kind == "lerobot" and (root / "meta/info.json").is_file():
        if dataset_info(root)["codebase_version"] == DATASET_FORMAT:
            manifest = root / FILES_MANIFEST
            if manifest_sha256 is not None and digest(manifest) != manifest_sha256:
                raise ValueError("Dataset manifest does not match the supplied checksum")
            inventory = json.loads(manifest.read_bytes())
            if inventory["dataset_format"] != DATASET_FORMAT:
                raise ValueError("Dataset manifest version differs from meta/info.json")
            names = {entry["path"] for entry in inventory["files"]}
            required = {"meta/info.json", "meta/tasks.parquet", "meta/stats.json", "official_loader_receipt.json",
                        "export_receipt.json", "finalization.json"}
            if not required <= names or len(names) != len(inventory["files"]):
                raise ValueError("Dataset manifest is incomplete or contains duplicate paths")
            layout_files = {path.relative_to(root).as_posix() for directory in ("meta", "data", "videos")
                            for path in (root / directory).rglob("*") if path.is_file()}
            if layout_files != {name for name in names if name.split("/", 1)[0] in {"meta", "data", "videos"}}:
                raise ValueError("Dataset layout differs from the published inventory")
            yield from inventory["files"]
            return
    if kind != "raw":
        name = "asset-files.json.gz" if kind == "assets" else "lerobot-files.json.gz"
        if digest(RELEASE / name) != release[kind]["inventory_sha256"]:
            raise ValueError("Release file inventory has changed")
        with gzip.open(RELEASE / name, "rt", encoding="utf-8") as src:
            yield from json.load(src)["files"]
        return
    manifest = root / "manifest.json"
    checksum = digest(manifest)
    data = json.loads(manifest.read_bytes())
    if Path(data["training_dataset"]).name == Path(release["lerobot"]["lab_root"]).name:
        if checksum != release["raw"]["manifest_sha256"]:
            raise ValueError("Raw manifest differs from the published release")
    if manifest_sha256 is not None and checksum != manifest_sha256:
        raise ValueError("Raw manifest does not match the supplied checksum")
    if (data.get("all_files_checksum_verified") is not True
            or data["episodes"] < 1 or data["episodes"] != len(data["trajectory_index"])):
        raise ValueError("Raw archive manifest does not describe a verified episode selection")
    for episode in data["trajectory_index"]:
        for item, row in episode["files"].items():
            yield {"path": episode["directory"] + "/" + item, "size": row["bytes"], "sha256": row["sha256"]}
    for item, row in data["dataset_receipts"].items():
        yield {"path": "dataset-receipts/" + item, "size": row["bytes"], "sha256": row["sha256"]}


def verify_files(root, entries):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Download directory not found: {root}")
    count = 0
    size = 0
    for entry in entries:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"File outside the download directory: {entry['path']}")
        if not path.is_file() or path.stat().st_size != entry["size"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Missing, incomplete or incorrect file: {entry['path']}")
        count += 1
        size += entry["size"]
    if count == 0:
        raise ValueError("Empty integrity inventory")
    return {"verified_files": count, "logical_bytes": size}


def verify_image(path):
    # docker image save writes a Docker-compatible manifest, including on OCI-capable engines.
    with tarfile.open(path, "r:") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        images = [i for i in manifest if IMAGE_TAG in (i.get("RepoTags") or [])]
        if len(images) != 1:
            raise ValueError(f"Expected exactly one entry tagged {IMAGE_TAG}")
        item = images[0]
        if hashlib.sha256(archive.extractfile(item["Config"]).read()).hexdigest() != IMAGE_ID:
            raise ValueError("Image configuration differs from the accepted lab image")
        for layer in item["Layers"]:
            if not archive.getmember(layer).isfile():
                raise ValueError(f"Missing image layer: {layer}")
    return {"image_config_id": "sha256:" + IMAGE_ID, "layer_entries": len(item["Layers"]),
            "next_check": "docker image load -i <tar> and inspect the image ID to validate layer contents"}


def restore_asset_receipt(directory):
    reference = json.loads((RELEASE / "reference.json").read_bytes())["assets"]
    data = gzip.decompress((RELEASE / "asset-files.json.gz").read_bytes())
    if hashlib.sha256(data).hexdigest() != reference["lab_receipt_sha256"]:
        raise ValueError("Pinned asset receipt hash mismatch")
    target = directory / ".geniesim-assets.json"
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError("Existing asset receipt belongs to a different acquisition")
        return
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        stream.write(data)
        temp = Path(stream.name)
    try:
        os.chmod(temp, 0o644)
        if os.name == "nt":
            os.rename(temp, target)
        else:
            os.link(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("assets", "lerobot", "raw", "image"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--restore-asset-receipt", action="store_true",
                        help="After full asset verification, install the exact lab release receipt if absent")
    parser.add_argument("--manifest-sha256", help="Expected SHA-256 of a published v3.0 or raw job inventory")
    args = parser.parse_args()
    if args.restore_asset_receipt and args.kind != "assets":
        parser.error("--restore-asset-receipt applies only to assets")
    if args.manifest_sha256 is not None and args.kind not in {"lerobot", "raw"}:
        parser.error("--manifest-sha256 applies only to LeRobot and raw job downloads")
    try:
        result = (verify_image(args.path) if args.kind == "image" else
                  verify_files(args.path, records(args.kind, args.path, manifest_sha256=args.manifest_sha256)))
        if args.restore_asset_receipt:
            restore_asset_receipt(args.path)
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        parser.exit(1, f"Verification failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
