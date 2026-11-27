"""Loopback trajectory workbench for the existing Genie Sim platform API."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
from pathlib import Path
import re
from socketserver import ThreadingMixIn
import subprocess
import sys
import threading
import time
from wsgiref.simple_server import WSGIServer, make_server


WEB = Path(__file__).resolve().parent
CAMERAS = ("head", "hand_left", "hand_right")
TASK_LABELS = {
    "phone_packaging": "手机装盒",
    "insert_building_block": "积木插入",
    "put_the_cap_on_the_pen_new": "盖笔帽",
    "place_in_container": "桌面抓放入盒",
}
SAMPLE_EXPORTS = (
    ("20261126-152211-skills-f805ead3f9e74334", [0, 1]),
    ("initial-scenes-20261126-one-per-task-scene", [0, 1, 4, 5]),
)
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/studio.css": ("studio.css", "text/css; charset=utf-8"),
    "/studio.js": ("studio.js", "text/javascript; charset=utf-8"),
}
JOB_ID = re.compile(r"[a-f0-9]{32}")

class VideoCache:
    def __init__(self, directory, python):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.python = str(python)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = {}
        self.lock = threading.Lock()

    def file_for(self, source):
        stat = source.stat()
        key = hashlib.sha256(f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
        return self.directory / (key + ".mp4")

    def prepare(self, source):
        target = self.file_for(source)
        if target.is_file():
            return "ready", target
        with self.lock:
            future = self.pending.get(target)
            if future is None:
                future = self.executor.submit(self._encode, source, target)
                self.pending[target] = future
            if future.done():
                try:
                    future.result()
                except Exception:
                    logging.exception("Video conversion failed")
                    return "failed", target
                return "ready", target
        return "preparing", target

    def _encode(self, source, target):
        subprocess.run([self.python, str(WEB / "transcode.py"), str(source), str(target)],
                       check=True, timeout=3600)


def response(start, status, body, content_type, *, head=False, headers=()):
    start(status, [("Content-Type", content_type), ("Content-Length", str(len(body))),
                   ("X-Content-Type-Options", "nosniff"), *headers])
    return [] if head else [body]


def json_response(start, data, status="200 OK"):
    return response(start, status, json.dumps(data, ensure_ascii=False).encode(),
                    "application/json; charset=utf-8", headers=[("Cache-Control", "no-store")])


def export_directory(value, exports):
    path = Path(value).resolve()
    if path.parent != exports or not path.is_dir():
        raise ValueError("Export is outside the studio library")
    return path


def verified_episodes(archive, dataset, exports, expected_hash=None):
    raw, training = (export_directory(path, exports) for path in (archive, dataset))
    receipt_bytes = (training / "official_loader_receipt.json").read_bytes()
    if expected_hash and hashlib.sha256(receipt_bytes).hexdigest() != expected_hash:
        raise ValueError("Loader receipt verification failed")
    receipt = json.loads(receipt_bytes)
    manifest = json.loads((raw / "manifest.json").read_bytes())
    items = manifest["trajectory_index"]
    if (receipt.get("passed") is not True or manifest.get("all_files_checksum_verified") is not True
            or Path(manifest["training_dataset"]).resolve() != training
            or receipt["episodes"] != len(items) or manifest["episodes"] != len(items)):
        raise ValueError("Archive and reader receipt disagree")
    episodes = []
    for item in items:
        cameras = {}
        for name in CAMERAS:
            video = (raw / item["camera_videos"][name]).resolve()
            if not video.is_relative_to(raw) or video.suffix.lower() != ".mp4" or not video.is_file():
                raise ValueError("Verified video is missing from the archive")
            cameras[name] = video
        episodes.append({**{k: item[k] for k in ("task", "scene", "frames")}, "cameras": cameras})
    return episodes


def sample_library(exports):
    library = {}
    for name, indices in SAMPLE_EXPORTS:
        try:
            episodes = verified_episodes(exports / (name + "-raw-trajectories"),
                                         exports / (name + "-v2-1"), exports)
            for index in indices:
                episode = episodes[index]
                episode["scene_label"] = ("场景 A" if index == 0 else "场景 B") if name.startswith("2026") else f"场景 {episode['scene']}"
                library[str(len(library))] = episode
        except (OSError, IndexError, KeyError, TypeError, ValueError) as exc:
            logging.warning("Sample export %s unavailable: %s", name, exc)
    return library


def episode_info(key, ep, *, sample=False):
    base = "/studio/media/" + ("samples/" if sample else "jobs/") + key
    return {
        "id": key, "task": ep["task"], "title": TASK_LABELS.get(ep["task"], ep["task"]),
        "scene": ep["scene"], "scene_label": ep.get("scene_label", f"场景 {ep['scene']}"),
        "frames": ep["frames"], "videos": {name: f"{base}/{name}.mp4" for name in CAMERAS},
        "poster": f"/studio/posters/{key}.jpg" if sample else None,
    }


def completed_episodes(job, exports):
    for record in (job.get("result"), job.get("native")):
        if not isinstance(record, dict) or record.get("status") not in ("complete", "partial"):
            continue
        dataset = record.get("dataset") or record.get("partial_dataset")
        if not isinstance(dataset, dict) or dataset.get("official_reader_verified") is not True:
            continue
        try:
            return verified_episodes(record["raw_trajectory_path"], dataset["dataset_path"],
                                     exports, dataset["loader_receipt_sha256"])
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return []


def stream_video(env, start, path):
    size = path.stat().st_size
    media_range = env.get("HTTP_RANGE")
    first, last = 0, size - 1
    headers = [("Accept-Ranges", "bytes"), ("Cache-Control", "private, max-age=3600")]
    status = "200 OK"
    if media_range:
        match = re.fullmatch(r"bytes=(\d{0,18})-(\d{0,18})", media_range)
        if match and (match[1] or match[2]):
            if match[1]:
                first = int(match[1])
                last = min(last, int(match[2])) if match[2] else last
            else:
                first = max(0, size - int(match[2]))
        if not match or not (match[1] or match[2]) or first > last:
            return response(start, "416 Range Not Satisfiable", b"", "video/mp4",
                            headers=[*headers, ("Content-Range", f"bytes */{size}")])
        status = "206 Partial Content"
        headers.append(("Content-Range", f"bytes {first}-{last}/{size}"))
    length = last - first + 1
    start(status, [("Content-Type", "video/mp4"), ("Content-Length", str(length)), *headers])
    if env["REQUEST_METHOD"] == "HEAD":
        return []

    def chunks():
        remaining = length
        with path.open("rb") as stream:
            stream.seek(first)
            while remaining:
                chunk = stream.read(min(remaining, 256 * 1024))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return chunks()


def create_studio(api, jobs, ready, exports, posters, library, cache):
    def video_source(kind, key, index, camera):
        if kind == "samples" and key in library:
            return library[key]["cameras"].get(camera)
        if kind == "jobs" and JOB_ID.fullmatch(key) and (job := jobs.get(key)):
            episodes = completed_episodes(job, exports)
            if index is not None and index < len(episodes):
                return episodes[index]["cameras"].get(camera)
        return None

    def app(env, start):
        path, method = env["PATH_INFO"], env["REQUEST_METHOD"]
        host = env.get("HTTP_HOST", "")
        if not re.fullmatch(r"(?:localhost|127\.0\.0\.1)(?::\d+)?", host):
            return json_response(start, {"error": "本工作台仅允许从本机访问"}, "403 Forbidden")
        if path == "/v1/jobs" and method == "POST":
            if env.get("HTTP_ORIGIN", f"http://{host}") != f"http://{host}":
                return json_response(start, {"error": "提交来源不匹配"}, "403 Forbidden")
            if not ready():
                return json_response(start, {"error": "采集服务暂不可用"}, "503 Service Unavailable")
            return api(env, start)
        if path == "/v1/capabilities" and method == "GET":
            return api(env, start)
        if method in ("GET", "HEAD") and path in STATIC:
            file, mime = STATIC[path]
            return response(start, "200 OK", (WEB / file).read_bytes(), mime, head=method == "HEAD",
                            headers=[("Cache-Control", "no-cache")])
        if method == "GET" and path == "/studio/health":
            return json_response(start, {"service": "nexus-trajectory-studio", "worker_ready": ready()})
        if method == "GET" and path == "/studio/samples":
            return json_response(start, [episode_info(key, ep, sample=True) for key, ep in library.items()])
        job_key = path.removeprefix("/studio/jobs/")
        if method == "GET" and path.startswith("/studio/jobs/") and JOB_ID.fullmatch(job_key):
            job = jobs.get(job_key)
            if job is None:
                return json_response(start, {"error": "找不到此作业"}, "404 Not Found")
            episodes = completed_episodes(job, exports)
            return json_response(start, {**job, "episodes": [
                episode_info(f"{job_key}/{i}", ep) for i, ep in enumerate(episodes)]})
        if method in ("GET", "HEAD"):
            match = re.fullmatch(r"/studio/posters/([0-9]+)\.jpg", path)
            if match and match[1] in library:
                poster = posters / f"{match[1]}.jpg"
                if poster.is_file():
                    return response(start, "200 OK", poster.read_bytes(), "image/jpeg",
                                    head=method == "HEAD", headers=[("Cache-Control", "private, max-age=3600")])
            match = re.fullmatch(r"/studio/(prepare|media)/(samples)/([0-9]+)/(head|hand_left|hand_right)\.mp4", path)
            if match:
                operation, kind, key, camera = match.groups()
                source = video_source(kind, key, None, camera)
            else:
                match = re.fullmatch(r"/studio/(prepare|media)/(jobs)/([a-f0-9]{32})/([0-9]{1,6})/(head|hand_left|hand_right)\.mp4", path)
                if match:
                    operation, kind, key, index, camera = match.groups()
                    source = video_source(kind, key, int(index), camera)
            if match and source is not None:
                state, target = cache.prepare(source)
                if operation == "prepare":
                    return json_response(start, {"state": state})
                if state == "ready":
                    return stream_video(env, start, target)
                return json_response(start, {"state": state}, "503 Service Unavailable")
        return json_response(start, {"error": "资源不存在"}, "404 Not Found")

    return app


class StudioServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


def make_posters(library, posters):
    import av

    posters.mkdir(parents=True, exist_ok=True)
    for key, ep in library.items():
        destination = posters / f"{key}.jpg"
        if destination.exists():
            continue
        temporary = posters / f"{key}.staging.jpg"
        with av.open(str(ep["cameras"]["head"])) as container:
            container.seek(int(ep["frames"] / 100 * av.time_base))
            frame = next(container.decode(video=0))
            image = frame.to_image()
            image.thumbnail((640, 640))
            image.save(temporary, format="JPEG", quality=84)
        temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--media-python", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18764)
    parser.add_argument("--posters-only", action="store_true")
    args = parser.parse_args()
    exports = args.exports.resolve()
    posters = args.state / "posters"
    library = sample_library(exports)
    if args.posters_only:
        make_posters(library, posters)
        print(f"Prepared posters for {len(library)} verified episodes")
        return

    sys.path.insert(0, str(args.runtime.resolve()))
    from genie_extensions import Registry
    from platform_api import create_app
    from platform_jobs import Jobs

    ready = threading.Event()
    state = args.state / "queue"
    jobs = Jobs(state)

    def work():
        try:
            registry = Registry()
            with jobs.worker():
                ready.set()
                while True:
                    if not jobs.run_next(registry):
                        time.sleep(1)
        finally:
            ready.clear()

    cache = VideoCache(args.state / "media-cache", args.media_python)
    app = create_studio(create_app(state), jobs, ready.is_set, exports, posters, library, cache)
    with make_server("127.0.0.1", args.port, app, server_class=StudioServer) as server:
        threading.Thread(target=work, name="studio-worker", daemon=True).start()
        if "0" in library:
            cache.prepare(library["0"]["cameras"]["head"])
        print(f"Trajectory studio at http://127.0.0.1:{args.port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
