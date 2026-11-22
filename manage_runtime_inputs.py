"""Verify the checked-in FineManip inputs. Optionally stage them into a separate lab checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verified(path, expected):
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError(f"Unexpected file digest: {path}")
    return data


def prepare(project=None, sdk=None):
    lock = json.loads((ROOT / "runtime_inputs/inputs.lock.json").read_bytes())
    pending = {}
    for rel, expected in lock["scene_and_calibration_sha256"].items():
        data = verified(ROOT / rel, expected)
        if project:
            pending[project / rel] = (data, None)
    for rel, row in lock["inputs"].items():
        data = verified(ROOT / row["file"], row["sha256"])
        if project:
            pending[project / rel] = (data, None)
    for rel, row in lock["sdk_patches"].items():
        patch = ROOT / row["patch"]
        verified(patch, row["patch_sha256"])
        if sdk:
            live = (sdk / rel).read_bytes()
            if sha(live) == row["after_sha256"]:
                continue
            if sha(live) != row["before_sha256"]:
                raise ValueError(f"SDK differs from both pinned versions: {rel}")
            with tempfile.TemporaryDirectory() as temp:
                candidate = Path(temp) / rel
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.write_bytes(live)
                subprocess.run(["git", "-c", "core.autocrlf=false", "-C", temp, "apply", str(patch)], check=True)
                pending[sdk / rel] = (verified(candidate, row["after_sha256"]), row["before_sha256"])
    changes = {}
    for path, (data, before) in pending.items():
        if path.exists():
            current = path.read_bytes()
            if current == data:
                continue
            if before is None or sha(current) != before:
                raise ValueError(f"Existing project file differs from release: {path}")
        changes[path] = (data, before)
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--sdk-root", type=Path, help="External data_collection runtime directory")
    parser.add_argument("--install", action="store_true", help="Stage missing release inputs and apply pinned SDK patches")
    args = parser.parse_args()
    if bool(args.project_root) != bool(args.sdk_root) or (args.install and not args.project_root):
        parser.error("Provide --project-root and --sdk-root together for deployment checks or installation")
    changes = prepare(args.project_root, args.sdk_root)
    if args.install:
        for path, (data, before) in changes.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            previous_mode = path.stat().st_mode if path.exists() else 0o644
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                handle.write(data)
                staged = Path(handle.name)
            try:
                os.chmod(staged, previous_mode)
                if before is None:
                    if os.name == "nt":
                        os.rename(staged, path)  # Windows rejects an existing destination.
                    else:
                        os.link(staged, path)
                else:
                    verified(path, before)
                    os.replace(staged, path)
            finally:
                staged.unlink(missing_ok=True)
        if prepare(args.project_root, args.sdk_root):
            raise RuntimeError("Installed files failed verification")
    elif changes:
        for path in changes:
            print(path)
        parser.exit(1, "Release inputs require staging; use --install on a dedicated runtime.\n")
    print("Runtime input digests verified.")


if __name__ == "__main__":
    main()
