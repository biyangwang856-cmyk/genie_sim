"""Genie Sim v1 platform interface. Web and worker run as separate processes."""
import argparse
from http import HTTPStatus
import json
import os
from pathlib import Path
import re
import time
from wsgiref.simple_server import make_server

from genie_extensions import Registry
from lerobot_layout import DATASET_FORMAT
from platform_jobs import Jobs, RequestConflict
from robot_adapters import UnsupportedRequest


class HttpError(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def read_body(env):
    if env.get("CONTENT_TYPE", "").partition(";")[0].strip().lower() != "application/json":
        raise HttpError(415, "Expected application/json")
    length = env.get("CONTENT_LENGTH") or ""
    if not re.fullmatch(r"[0-9]{1,9}", length):
        raise HttpError(411, "Content-Length required")
    size = int(length)
    if size > 65536:
        raise HttpError(413, "Request exceeds 64 KiB")
    data = env["wsgi.input"].read(size)
    if len(data) != size:
        raise ValueError("Incomplete JSON request")
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def create_app(directory, registry=None):
    registry = registry if registry is not None else Registry()
    jobs = Jobs(directory)

    def route(env):
        path, method = env["PATH_INFO"], env["REQUEST_METHOD"]
        if path == "/v1/openapi.json":
            if method != "GET":
                raise HttpError(405, "Use GET")
            schema = Path(__file__).resolve().parent / "docs/openapi.json"
            return 200, json.loads(schema.read_bytes()), []
        if path == "/v1/capabilities":
            if method != "GET":
                raise HttpError(405, "Use GET")
            return 200, {"version": "v1", "dataset_format": DATASET_FORMAT, "operations": registry.describe()}, []
        match = re.fullmatch(r"/v1/jobs/([a-f0-9]{32})", path)
        if match:
            if method != "GET":
                raise HttpError(405, "Use GET")
            job = jobs.get(match.group(1))
            if job is None:
                raise HttpError(404, "Unknown job")
            return 200, job, []
        extension = re.fullmatch(r"/v1/extensions/([a-z0-9_]+)", path)
        if path != "/v1/jobs" and not extension:
            raise HttpError(404, "Unknown route")
        if method != "POST":
            raise HttpError(405, "Use POST")
        payload = read_body(env)
        name = extension.group(1) if extension else payload.pop("simulator", "isaac_physx")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]{1,64}", name):
            raise ValueError("Invalid operation identifier")
        spec = registry.specs.get(name)
        if spec and bool(extension) == spec.collection:
            raise ValueError("Operation belongs to a different endpoint")
        request = registry.validate(name, payload)
        job_id = jobs.enqueue(name, request, env.get("HTTP_IDEMPOTENCY_KEY"))
        url = "/v1/jobs/" + job_id
        receipt = {"id": job_id, "status_url": url}
        if not extension:
            receipt["dataset_format"] = request["dataset_format"]
        return 202, receipt, [("Location", url)]

    def app(env, start_response):
        try:
            code, body, headers = route(env)
        except UnsupportedRequest as exc:
            code, body, headers = 422, exc.details, []
        except RequestConflict as exc:
            code, body, headers = 409, {"error": str(exc)}, []
        except HttpError as exc:
            code, body, headers = exc.code, {"error": str(exc)}, []
            if code == 405:
                headers.append(("Allow", str(exc).split()[-1]))
        except (ValueError, RecursionError) as exc:
            code, body, headers = 400, {"error": str(exc)}, []
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        start_response(f"{code} {HTTPStatus(code).phrase}", [
            ("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(data))), *headers])
        return [data]
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path(os.environ.get("GENIE_PLATFORM_STATE", ".platform-state")))
    commands = parser.add_subparsers(dest="command", required=True)
    web = commands.add_parser("serve")
    web.add_argument("--port", type=int, default=8786)
    worker = commands.add_parser("work")
    worker.add_argument("--once", action="store_true", help="Process at most one request and exit")
    args = parser.parse_args()
    if args.command == "serve":
        with make_server("127.0.0.1", args.port, create_app(args.state_dir)) as server:
            print(f"Genie Sim v1 listening on http://127.0.0.1:{server.server_port}", flush=True)
            server.serve_forever()
    else:
        registry = Registry()
        jobs = Jobs(args.state_dir)
        with jobs.worker():
            while True:
                processed = jobs.run_next(registry)
                if args.once:
                    break
                if not processed:
                    time.sleep(1)


if __name__ == "__main__":
    main()
