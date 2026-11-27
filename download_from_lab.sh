#!/usr/bin/env bash
# Run on the receiving Linux/WSL machine. SSH uses the member's existing lab account.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'Usage: bash download_from_lab.sh {assets|image|raw|lerobot} SSH_ALIAS DEST_DIRECTORY' >&2
  exit 2
fi
kind=$1
host=$2
dest=$3
if [[ ! $host =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.@-]*$ ]]; then
  echo 'Use an SSH host or configured Host alias; configure the port in ~/.ssh/config.' >&2
  exit 2
fi
root=$(cd "$(dirname "$0")" && pwd)
dataset=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["accepted_job"])' "$root/release/reference.json")
case "$kind" in
  assets) remote_dir=/home/vipuser/by/genie_sim/GenieSimAssets; folder=GenieSimAssets ;;
  raw) folder="$dataset-raw-trajectories"; remote_dir="/data/by/genie_sim/exports/$folder" ;;
  lerobot) folder="$dataset-v2-1"; remote_dir="/data/by/genie_sim/exports/$folder" ;;
  image)
    mkdir -p -- "$dest"
    target="$dest/geniesim-by-collection-a100-20261119.tar"
    if [[ -e $target ]]; then
      python3 "$root/verify_download.py" image "$target"
      exit
    fi
    partial=$(mktemp "$dest/.geniesim-image.XXXXXXXX")
    trap 'rm -f -- "$partial"' EXIT
    ssh -T "$host" 'DOCKER_HOST=unix:///data/by/genie_sim/dockerhub-sidebuild-20261119/docker.sock /data/by/genie_sim/recovery-tools/docker/docker image save geniesim-by-collection:a100-20261119' > "$partial"
    python3 "$root/verify_download.py" image "$partial"
    mv -n -- "$partial" "$target"
    echo "Image archive available at $target"
    exit ;;
  *) echo 'Unknown resource' >&2; exit 2 ;;
esac
mkdir -p -- "$dest/$folder"
# Same-size damaged local files can be repaired by rsync --checksum after a failed verification.
filters=()
if [[ $kind == assets ]]; then
  filters+=(--exclude=/.geniesim-download/ --exclude=/.geniesim-download.lock --exclude=/__pycache__/)
fi
rsync -aH --partial --info=progress2 "${filters[@]}" "$host:$remote_dir/" "$dest/$folder/"
python3 "$root/verify_download.py" "$kind" "$dest/$folder"
