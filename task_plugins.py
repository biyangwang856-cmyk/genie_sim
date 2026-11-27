"""Load trusted task contracts. Plans can reference registered IDs and data only."""
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import re

from robot_adapters import ROOT, UnsupportedRequest


DIRECTORY = Path(__file__).resolve().parent / "pipeline/task_contracts"
FUNCTIONS = ("bind", "resolve_target", "validate_steps", "verify_relation", "verify_task", "audit")


def registered():
    tasks = {}
    for path in sorted(DIRECTORY.glob("*.json")):
        task_id = path.stem
        code = path.with_suffix(".py")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,61}", task_id) or not code.is_file():
            raise ValueError(f"Task descriptor has no matching trusted implementation: {path.name}")
        descriptor = json.loads(path.read_bytes())
        if descriptor.get("task_id") != task_id or not isinstance(descriptor.get("description"), str):
            raise ValueError(f"Invalid task descriptor: {path.name}")
        tasks[task_id] = descriptor
    return tasks


def source_file(task_id, snapshot=None):
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,61}", task_id):
        raise ValueError("Invalid task contract ID")
    archive = Path(snapshot or os.environ.get("GENIE_RUN_SOURCE", "/nonexistent"))
    file = archive / ("task-contract-" + task_id + ".py")
    if file.is_file():
        return file
    if snapshot is None and task_id in registered():
        return DIRECTORY / (task_id + ".py")
    return None


def load(task_id, snapshot=None):
    file = source_file(task_id, snapshot)
    if file is None:
        raise UnsupportedRequest("Task contract is not registered", missing_skills=[task_id])
    spec = spec_from_file_location("project_task_" + task_id, file)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [key for key in FUNCTIONS if not callable(getattr(module, key, None))]
    if missing:
        raise ValueError(f"Task contract {task_id} is missing {missing}")
    return module
