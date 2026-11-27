"""Validate collection counts and compare scenes and measured trajectories."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
from scipy.spatial.transform import Rotation


SCENE_DIMENSIONS = ("background", "support", "layout", "distractors", "lighting")
DEFAULT_SCENE_DIMENSIONS = SCENE_DIMENSIONS[:-1]
REQUEST_FIELDS = {
    "total_episodes", "duplicate_policy", "duplicate_intent", "reference_run",
    "distinct_dimensions", "lineage_id",
}
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
ALLOW_INTENT = re.compile(
    r"(?:允许|可以|可接受)\s*(?:采集|生成|录制|出现|有)?\s*重复(?:的)?(?:轨迹|路径|回合)"
    r"|(?:轨迹|路径|回合|动作)\s*(?:允许|可以|可接受)\s*重复"
    r"|(?:允许|可以|可接受)\s*(?:轨迹|路径|回合|动作)\s*重复"
    r"|\b(?:allow|accept) (?:exact )?(?:duplicates|repeats|(?:duplicate|repeated) (?:trajectories|episodes))\b"
    r"|\b(?:trajectories|episodes) (?:may|can) (?:be )?(?:repeat|repeated|duplicates?)\b",
    re.IGNORECASE,
)
REPLAY_INTENT = re.compile(
    r"(?:复验|重放|回放|复现|重复\s*(?:执行|采集|录制|运行))"
    r"|(?:同一|相同|指定)\S{0,12}(?:重复|复验|重放)"
    r"|(?:按|沿)\S{0,18}(?:轨迹|路径)\S{0,6}执行"
    r"|\b(?:replay|rerun|reexecute|re-execute|repeat)\b",
    re.IGNORECASE,
)
NEGATED_INTENT = re.compile(
    r"禁止|不(?:要|允许|可以|得)|不能|不可|避免|杜绝|无需|不用|除非|如果|假如"
    r"|\b(?:no|never|cannot|don't|do not|not allowed|unless|if)\b",
    re.IGNORECASE,
)
REFERENCE_INTENT = re.compile(r"同一|相同|指定|参考|这条|原轨迹|\b(?:same|given|reference|recorded)\b",
                              re.IGNORECASE)
DISTINCT_MOTION_INTENT = re.compile(
    r"(?:禁止|拒绝|不(?:要|得|准|能|允许|可以|接受)|杜绝|避免|拒收)\s*"
    r"(?:采集|生成|录制|出现|接受|有)?\s*(?:任何)?\s*(?:重复|相同|同样)的?\s*(?:轨迹|路径|回合|动作)"
    r"|(?:轨迹|路径|回合|动作)\s*(?:互不重复|各不相同|不(?:能|得|准|许|允许|可以)?\s*重复)"
    r"|(?:禁止|拒绝|不(?:要|得|准|能|允许|可以|接受)|杜绝|避免|拒收)\s*(?:轨迹|路径|回合|动作)\s*重复"
    r"|(?:禁止|拒绝|不(?:要|得|准|能|允许|可以|接受)|杜绝|避免)\s*"
    r"(?:复验|重放|回放|复现|重执行|重复\s*(?:执行|运行|采集|录制))[^。；，,.!?;\n]{0,18}(?:轨迹|路径|回合)"
    r"|(?:条|次|种)\s*(?:不重复|不同|独立不同|互异|各不相同)的?\s*(?:轨迹|路径|回合)"
    r"|\b(?:no (?:exact )?duplicates|no repeat(?:ed)? (?:trajectories|episodes|paths))\b"
    r"|\b(?:do not|don't|never|must not|cannot|can't) (?:allow |accept |generate |collect )?"
    r"(?:duplicate(?:s|\b)|repeat(?:ed)? (?:trajectories|episodes|paths)|replay (?:it|the same|the reference))\b"
    r"|\b(?:different|distinct|unique|non[- ]duplicate) (?:trajectories|episodes|paths)\b"
    r"|\bwithout (?:duplicates|repeated (?:trajectories|episodes|paths))\b",
    re.IGNORECASE,
)
FIXED_REFERENCE_CONFLICT = re.compile(
    r"(?:不同|互异|各不相同)的?\s*(?:(?:运行|作业|初始)\s*)?(?:场景|布局|房间|桌子)"
    r"|\b(?:different|distinct|unique) (?:initial |existing )?(?:scenes|layouts|rooms|tables)\b",
    re.IGNORECASE,
)

PROFILE = {
    "version": "genie-trajectory-diversity-v1",
    "scene_position_m": 0.010,
    "scene_rotation_rad": 0.060,
    "scene_lighting_fraction": 0.10,
    "joint_angle_rad": 0.024,
    "joint_linear_m": 0.004,
    "end_effector_position_m": 0.008,
    "end_effector_rotation_rad": 0.060,
}


def _count(value, field, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{field} must be an integer from 1 to {maximum}")
    return value


def trajectory_request(raw, *, instruction=None):
    """Check an explicit request, or cite user wording for a model-inferred opt-in."""
    if not isinstance(raw, dict) or raw.keys() - REQUEST_FIELDS:
        raise ValueError("Invalid trajectory_request fields")
    policy = raw.get("duplicate_policy", "distinct")
    if policy not in ("distinct", "allow", "replay"):
        raise ValueError("duplicate_policy must be distinct, allow, or replay")
    dimensions = raw.get("distinct_dimensions", [])
    if not isinstance(dimensions, list) or any(value not in SCENE_DIMENSIONS for value in dimensions):
        raise ValueError("distinct_dimensions must select supported scene variations")
    result = {**raw, "duplicate_policy": policy, "distinct_dimensions": list(dict.fromkeys(dimensions))}
    for label in ("reference_run", "lineage_id"):
        if label in raw and (not isinstance(raw[label], str) or not ID_PATTERN.fullmatch(raw[label])):
            raise ValueError(f"Invalid {label}")
    if "total_episodes" in raw:
        _count(raw["total_episodes"], "total_episodes", 144)
    if (policy == "replay") != ("reference_run" in raw):
        raise ValueError("A reference_run is required and only supported for replay")
    if policy == "replay" and dimensions:
        raise ValueError("Replay requires a fixed scene")
    evidence = raw.get("duplicate_intent")
    if evidence is not None and (
        not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 512
    ):
        raise ValueError("duplicate_intent must be an exact user clause")
    if instruction is not None and policy != "distinct":
        pattern = ALLOW_INTENT if policy == "allow" else REPLAY_INTENT
        clauses = re.split(r"[。；，,.!?;\n]+", instruction)
        if DISTINCT_MOTION_INTENT.search(instruction):
            raise ValueError("The instruction disallows repeated trajectories")
        # Evaluate the full clause too, so extracting a positive substring
        # from “do not allow duplicate trajectories” cannot grant permission.
        has_intent = evidence and any(
            evidence in clause and pattern.search(evidence) and not NEGATED_INTENT.search(clause)
            and (policy == "allow" or REFERENCE_INTENT.search(clause) or raw["reference_run"] in clause)
            for clause in clauses)
        if not has_intent:
            raise ValueError("Repeated trajectories require explicit user intent")
    return result


def resolve_trajectory_request(structured=None, inferred=None, *, instruction, distinct_scene_count=None):
    """Merge explicit fields with validated language intent without changing their meaning."""
    structured = {} if structured is None else structured
    inferred = {} if inferred is None else inferred
    supplied = trajectory_request(structured)
    suggested = trajectory_request(inferred, instruction=instruction)
    for key in structured.keys() & inferred.keys() & {"total_episodes", "reference_run", "distinct_dimensions"}:
        if key == "distinct_dimensions":
            # A list of explicitly supplied dimensions can further narrow a
            # request. An omitted or empty model list imposes no restriction.
            if not set(suggested[key]) <= set(supplied[key]):
                raise ValueError("trajectory_request.distinct_dimensions conflicts with instruction")
        elif supplied[key] != suggested[key]:
            raise ValueError(f"trajectory_request.{key} conflicts with instruction")
    policy = supplied["duplicate_policy"] if "duplicate_policy" in structured else suggested["duplicate_policy"]
    if ("duplicate_policy" in structured and suggested.get("duplicate_intent")
            and suggested["duplicate_policy"] != "distinct" and policy != suggested["duplicate_policy"]):
        raise ValueError("trajectory_request.duplicate_policy conflicts with instruction")
    resolved = trajectory_request({**inferred, **structured, "duplicate_policy": policy})
    if policy != "distinct" and DISTINCT_MOTION_INTENT.search(instruction):
        raise ValueError("The instruction disallows repeated trajectories")
    if policy == "replay" and ((distinct_scene_count is not None and distinct_scene_count > 1)
                               or FIXED_REFERENCE_CONFLICT.search(instruction)):
        raise ValueError("Replay requires the original fixed scene")
    return resolved


def episode_quotas(scene_ids, *, total_episodes=None, per_scene=None, requested_scene_count=None,
                   legacy_defaults=None):
    """Assign a total to stable candidate positions, preserving any missing slots."""
    ids = list(scene_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated scene IDs")
    if total_episodes is not None:
        _count(total_episodes, "total_episodes", 144)
    if per_scene is not None:
        _count(per_scene, "successes_per_scene", 12)
    if requested_scene_count is None and total_episodes is not None and per_scene is None:
        ids = ids[:total_episodes]
    count = len(ids) if requested_scene_count is None else _count(requested_scene_count, "scene_count", 12)
    if len(ids) > count:
        raise ValueError("More scenes than requested slots")
    if total_episodes is not None and per_scene is not None and total_episodes != count * per_scene:
        raise ValueError("total_episodes conflicts with scene_count and successes_per_scene")
    if not count:
        return []
    if total_episodes is not None:
        share, remainder = divmod(total_episodes, count)
        if share < 1 or share + bool(remainder) > 12:
            raise ValueError("The requested total cannot provide 1–12 trajectories per requested scene")
        counts = [share + (index < remainder) for index in range(count)]
    else:
        counts = [per_scene or (legacy_defaults or {}).get(scene_id, 1) for scene_id in ids]
    return [{"scene_id": scene_id, "successes": successes} for scene_id, successes in zip(ids, counts)]


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _asset_identity(identifier, dependencies, *, background=False):
    if identifier is None:
        return None
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Scene asset with invalid identity")
    if identifier in dependencies:
        return {"id": identifier, "sha256": dependencies[identifier]}
    parts = tuple(identifier.split("__"))
    paths = {}
    for file, checksum in dependencies.items():
        path = PurePosixPath(file.replace("\\", "/"))
        if background:
            root = ("background", *parts[:-1])
            if path.parts[:len(root)] == root:
                paths["/".join(path.parts[len(root):])] = checksum
        elif identifier in path.parts[:-1]:
            root_index = path.parts.index(identifier)
            paths["/".join(path.parts[root_index + 1:])] = checksum
    return {"id": identifier, "sha256": _sha(paths) if paths else None}


def _same_asset(left, right):
    if left is None or right is None:
        return left is right
    if left["sha256"] and right["sha256"]:
        return left["sha256"] == right["sha256"]
    return left["id"] == right["id"]


def _vector(values, length):
    vector = np.asarray(values, dtype=float)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError("Invalid position, rotation or support vector")
    return vector


def _quat(values):
    value = _vector(values, 4)
    length = float(np.linalg.norm(value))
    if length < 1e-8:
        raise ValueError("Invalid zero quaternion")
    value = value / length
    return -value if value[np.argmax(np.abs(value))] < 0 else value


def _rotation_distance(left, right):
    q1, q2 = np.asarray(left), np.asarray(right)
    return 2.0 * np.arccos(np.minimum(1.0, np.abs(np.sum(q1 * q2, axis=-1))))


def _pose(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Invalid measured object pose")
    return matrix[:3, 3], _quat(Rotation.from_matrix(matrix[:3, :3]).as_quat())


def scene_descriptor(scene_info, *, initial_poses=None, asset_digests=None, lighting=None):
    """Describe effective content. Authored and settled poses use the same geometry."""
    transfer = scene_info.get("scene_transfer", {})
    support = scene_info.get("support") or transfer.get("support") or {}
    dependencies = asset_digests or {}
    layout = scene_info["layout"]
    support_name = support.get("instance")
    if not support and not support_name:
        tables = [name for name, row in layout.items() if re.search(r"table|desk|workbench", row["usd"])]
        support_name = tables[0] if len(tables) == 1 else None
    support_row = layout.get(support_name, {})
    background = transfer.get("background") or scene_info.get("background_id")
    if not background:
        backgrounds = [PurePosixPath(name) for name in dependencies
                       if name.startswith("background/") and PurePosixPath(name).name in
                       ("background.usda", "scene_full.usda")]
        if len(backgrounds) == 1:
            background = "__".join((*backgrounds[0].parts[1:-1], backgrounds[0].stem))
    surface = None
    if "bounds_xy_m" in support:
        bounds = np.concatenate([_vector(row, 2) for row in support["bounds_xy_m"]])
        surface = [*bounds.tolist(), float(support["height_m"])]
        if len(surface) != 5 or not np.isfinite(surface).all():
            raise ValueError("Invalid scene support geometry")
    origin = np.asarray([(surface[0] + surface[2]) / 2, (surface[1] + surface[3]) / 2, surface[4]]) \
        if surface else np.zeros(3)
    result = {
        "schema": PROFILE["version"], "scene_id": str(scene_info.get("scene_id", "")),
        "background": _asset_identity(background, dependencies, background=True),
        "support": {"asset": _asset_identity(support.get("asset_id", support_row.get("usd")), dependencies),
                    "surface": surface},
        "layout": [], "distractors": [], "lighting": lighting,
        "measured_objects": 0,
    }
    if lighting is not None and (
        not isinstance(lighting, (int, float)) or not np.isfinite(lighting) or lighting <= 0
    ):
        raise ValueError("Invalid realized scene light intensity")
    for name, row in layout.items():
        if name == support_name:
            continue
        if name in (initial_poses or {}):
            position, quaternion = _pose(initial_poses[name])
            result["measured_objects"] += 1
        else:
            position, quaternion = _vector(row["xyz"], 3), _quat(row["xyzw"])
        category = "distractors" if row.get("distractor") or name.startswith("collection_distractor_") else "layout"
        result[category].append({
            "asset": _asset_identity(row["usd"], dependencies),
            "relative": np.round(position - origin, 7).tolist(),
            "world": np.round(position, 7).tolist(),
            "quaternion_xyzw": quaternion.tolist(),
        })
    for dimension in ("layout", "distractors"):
        result[dimension].sort(key=lambda row: (row["asset"]["id"], row["relative"], row["quaternion_xyzw"]))
    return result


def _poses_match(left, right, profile):
    if len(left) != len(right):
        return False
    possible = []
    for first in left:
        matches = []
        for index, second in enumerate(right):
            if not _same_asset(first["asset"], second["asset"]):
                continue
            if all(np.linalg.norm(np.subtract(first[key], second[key])) <= profile["scene_position_m"]
                   for key in ("relative", "world")) and _rotation_distance(
                       first["quaternion_xyzw"], second["quaternion_xyzw"]) <= profile["scene_rotation_rad"]:
                matches.append(index)
        possible.append(matches)

    assigned = {}

    def assign(left_index, visited):
        for right_index in possible[left_index]:
            if right_index in visited:
                continue
            visited.add(right_index)
            if right_index not in assigned or assign(assigned[right_index], visited):
                assigned[right_index] = left_index
                return True
        return False

    return all(assign(index, set()) for index in range(len(left)))


def different_scene_dimensions(left, right, *, profile=None):
    profile = {**PROFILE, **(profile or {})}
    changed = set()
    if left["background"] and right["background"] and not _same_asset(left["background"], right["background"]):
        changed.add("background")
    first, second = left["support"], right["support"]
    if (first["asset"] and second["asset"] and not _same_asset(first["asset"], second["asset"])) or (
        first["surface"] is not None and second["surface"] is not None and
        np.max(np.abs(np.subtract(first["surface"], second["surface"]))) > profile["scene_position_m"]
    ):
        changed.add("support")
    for dimension in ("layout", "distractors"):
        if not _poses_match(left[dimension], right[dimension], profile):
            changed.add(dimension)
    a, b = left["lighting"], right["lighting"]
    if a is not None and b is not None and abs(a - b) > profile["scene_lighting_fraction"] * max(a, b):
        changed.add("lighting")
    return changed


def scene_conflict(candidate, others, *, dimensions=(), profile=None):
    """An explicit field must vary on every pair; otherwise any physical/layout field may vary."""
    if set(dimensions) - set(SCENE_DIMENSIONS):
        raise ValueError("Unknown scene diversity dimension")
    if ("background" in dimensions and candidate["background"] is None
            or "support" in dimensions and candidate["support"]["asset"] is None
            or "lighting" in dimensions and candidate["lighting"] is None):
        return {"reason": "scene_dimension_not_observed", "required_dimensions": list(dimensions)}
    for prior in others:
        different = different_scene_dimensions(candidate, prior, profile=profile)
        matches = set(dimensions) <= different if dimensions else bool(set(DEFAULT_SCENE_DIMENSIONS) & different)
        if not matches:
            return {"reason": "repeated_scene_content", "scene_id": prior["scene_id"],
                    "different_dimensions": sorted(different), "required_dimensions": list(dimensions)}
    return None


def _load_json(path):
    return json.loads(path.read_bytes())


def _load_json_lines(path):
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def _run_scene(run):
    launch = _load_json(run / "launch.json")
    folder = run / "tasks" / launch["task"] / str(launch["scene"])
    return folder, _load_json(folder / "scene_info.json")


def scene_descriptor_from_run(run):
    """Recheck the compiled scene against the first measured pre-motion object poses."""
    run = Path(run)
    folder, metadata = _run_scene(run)
    event = next((row for row in _load_json_lines(run / "evidence/skill-execution.jsonl")
                  if row.get("event") == "step_started"), {})
    measured = event.get("state", {}).get("object_poses", {})
    proof = run / "evidence/scene-physics-preflight.json"
    if proof.is_file():
        measured.update(_load_json(proof).get("initial_poses", {}))
    manifest = folder / "scene-manifest.json"
    dependencies = _load_json(manifest).get("asset_dependencies", {}) if manifest.is_file() else {}
    return scene_descriptor(metadata, initial_poses=measured, asset_digests=dependencies,
                            lighting=metadata.get("scene_transfer", {}).get("lighting_intensity"))


def _resample(joints, scales, arrays, samples):
    # Ignore standstill and position noise below one fifth of the comparison
    # tolerance. Parameterize by distance travelled, not timestamp.
    normalized = joints / scales
    selected = [0]
    for index, row in enumerate(normalized[1:], start=1):
        if np.max(np.abs(row - normalized[selected[-1]])) > 0.20:
            selected.append(index)
    if selected[-1] != len(joints) - 1:
        selected.append(len(joints) - 1)
    segments = np.linalg.norm(np.diff(normalized[selected], axis=0), axis=1)
    distance = np.r_[0., segments.cumsum()]
    keep = np.r_[True, np.diff(distance) > 1e-10]
    positions = distance[keep]
    select = np.asarray(selected)[keep]
    samples_at = np.linspace(0., positions[-1], samples)
    return {
        key: np.round(np.column_stack([np.interp(samples_at, positions, column[select])
                                       for column in matrix.T]), 6).tolist()
        for key, matrix in arrays.items()
    }


def describe_motion(arrays, spec, *, phases=(), observed_paths=None, profile=None):
    """Describe consistently ordered measured joints, command positions and tracked poses."""
    settings = {**PROFILE, **(profile or {})}
    name_order = np.argsort(spec["joint_names"])
    names = [spec["joint_names"][index] for index in name_order]
    units = [spec["position_units"][index] for index in name_order]
    if len(set(names)) != len(names) or not names or set(units) - {"m", "rad"}:
        raise ValueError("Missing or incompatible measured joint units")
    scales = np.array([settings["joint_linear_m"] if unit == "m" else settings["joint_angle_rad"] for unit in units])
    matrix = {}
    for target, source in (("measured_joints", "observation.state"), ("position_commands", "action")):
        data = np.asarray(arrays[source], dtype=float)
        if data.ndim != 2 or data.shape[1] != len(names) or len(data) < 2 or not np.isfinite(data).all():
            raise ValueError("Invalid measured or commanded joint series")
        matrix[target] = data[:, name_order]
    joints = matrix["measured_joints"]
    if len(matrix["position_commands"]) != len(joints):
        raise ValueError("Measured joints and commands have different frame counts")
    for key, values in (observed_paths or {}).items():
        values = np.asarray(values, dtype=float)
        if values.shape != (len(joints), 7) or not np.isfinite(values).all():
            raise ValueError("Tracked pose must align to the joint frame clock")
        # Quaternions and their negations denote the same rotation. Maintain
        # a continuous representation during interpolation of pose channels.
        values = values.copy()
        values[0, 3:] = _quat(values[0, 3:])
        for idx in range(1, len(values)):
            q = _quat(values[idx, 3:])
            values[idx, 3:] = -q if np.dot(q, values[idx - 1, 3:]) < 0 else q
        matrix[key] = values
    begin = np.flatnonzero(np.max(np.abs((joints - joints[0]) / scales), axis=1) > .20)
    end = np.flatnonzero(np.max(np.abs((joints - joints[-1]) / scales), axis=1) > .20)
    if not len(begin) or not len(end):
        raise ValueError("No substantive measured motion")
    # Include the endpoints on both sides of a meaningful movement. Subtle
    # initial/final movement may overlap, for example a monotone 0.3-scale path.
    span = slice(min(int(begin[0]) - 1, int(end[-1])), max(int(begin[0]) + 1, int(end[-1]) + 2))
    primary = _resample(joints[span], scales, {key: data[span] for key, data in matrix.items()}, 64)
    phase_paths = []
    for label, begin, end in phases:
        if 0 <= begin < end - 1 < len(joints):
            part = slice(begin, end)
            phase_paths.append({"skill": label, "paths": _resample(joints[part], scales,
                                {key: data[part] for key, data in matrix.items()}, 16)})
    return {"version": settings["version"], "thresholds": settings,
            "joint_names": names, "position_units": units, "paths": primary, "phases": phase_paths}


def _phase_ranges(times, events):
    active, phases = None, []
    for row in events:
        stamp = row.get("sim_time", row.get("sim_time_s"))
        if not isinstance(stamp, (float, int)) or not np.isfinite(stamp):
            continue
        if row.get("event") == "step_started":
            active = (row["step"]["skill"], row["step"]["id"], stamp)
        elif row.get("event") in ("step_succeeded", "step_failed") and active:
            skill, token, start = active
            active = None
            if row.get("step_id", row.get("step", {}).get("id")) != token:
                continue
            # Only compare phases measured on the same clock as observations.
            if not times[0] - .10 <= start <= stamp <= times[-1] + .10:
                continue
            begin = int(np.searchsorted(times, start, side="left"))
            end = int(np.searchsorted(times, stamp + .000051, side="right"))
            if end - begin >= 3:
                phases.append((skill, begin, end))
    return phases


def _measured_paths(run, frames, times):
    paths = {}
    base = None
    for event in _load_json_lines(run / "evidence/stage-evidence.jsonl"):
        snapshot = event.get("snapshot") or {}
        if not event.get("phase", "").startswith("model_snapshot") or "robot_base_position" not in snapshot:
            continue
        pose = np.eye(4)
        pose[:3, 3] = _vector(snapshot["robot_base_position"], 3)
        q = _quat(np.asarray(snapshot["robot_base_quaternion_wxyz"])[[1, 2, 3, 0]])
        pose[:3, :3] = Rotation.from_quat(q).as_matrix()
        base = np.linalg.inv(pose)
        break

    def channel(matrices):
        return [np.concatenate(_pose(matrix)) for matrix in matrices]

    def object_transforms(name):
        key = f"World_Objects_{name}_entity"
        measured = [frame.get("objects", {}).get(key, {}).get("pose") for frame in frames]
        return np.asarray(measured) if all(value is not None for value in measured) else None

    spec = run / "evidence/task-spec.json"
    if spec.is_file():
        spec = _load_json(spec)
        bindings = spec.get("bindings", {})
        role_names = {name: role for role, name in sorted(bindings.items())}
        reference_name = bindings.get("receiver") or bindings.get("pen")
        reference = object_transforms(reference_name) if reference_name else None
        reference_inverse = np.linalg.inv(reference) if reference is not None else None
        tracked = spec.get("manipulated_objects", ())
        for index, object_name in enumerate(tracked):
            raw = object_transforms(object_name)
            if raw is None:
                continue
            label = role_names.get(object_name, str(index))
            paths["object:workcell:" + label] = channel(base @ raw if base is not None else raw)
            if reference_inverse is not None and object_name != reference_name:
                paths["object:reference:" + label] = channel(reference_inverse @ raw)
    samples = list(_load_json_lines(run / "evidence/render-motion-samples.jsonl"))
    if len(samples) < 2:
        return paths
    stamps = np.array([row["sim_time"] for row in samples], dtype=float)
    if not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0) or (
        stamps[0] > times[0] + .16 or stamps[-1] < times[-1] - .16
    ):
        return paths
    for side in ("l", "r"):
        measured = np.asarray([row["links"][side]["physics"] for row in samples])
        tracked = channel(base @ measured if base is not None else measured)
        for prev, sample in zip(tracked, tracked[1:]):
            if np.dot(prev[3:], sample[3:]) < 0:
                sample[3:] = -sample[3:]
        frame = "base" if base is not None else "world"
        paths[f"wrist:{frame}:{side}"] = np.column_stack([
            np.interp(times, stamps, col) for col in np.asarray(tracked).T])
    return paths


def motion_descriptor_from_run(run, *, profile=None):
    from export_lerobot import native_arrays, recording_path

    run = Path(run)
    arrays, spec = native_arrays(run)
    frames = _load_json(recording_path(run) / "state.json")["frames"]
    times = np.array([frame["time_stamp"] for frame in frames], dtype=float)
    phases = _phase_ranges(times, _load_json_lines(run / "evidence/skill-execution.jsonl"))
    return describe_motion(arrays, spec, phases=phases, observed_paths=_measured_paths(run, frames, times), profile=profile)


def _ratio(left, right, scale):
    return float(np.quantile(np.max(np.abs(np.subtract(left, right)) / scale, axis=1), .95))


def _pose_ratio(left, right, profile):
    a, b = np.asarray(left), np.asarray(right)
    translation = np.linalg.norm(a[:, :3] - b[:, :3], axis=1) / profile["end_effector_position_m"]
    qa = a[:, 3:] / np.linalg.norm(a[:, 3:], axis=1, keepdims=True)
    qb = b[:, 3:] / np.linalg.norm(b[:, 3:], axis=1, keepdims=True)
    orientation = _rotation_distance(qa, qb) / profile["end_effector_rotation_rad"]
    return float(np.quantile(np.maximum(translation, orientation), .95))


def motion_distance(left, right):
    """Measure pace-invariant departure from a recorded path in tolerance units."""
    if (left["joint_names"], left["position_units"]) != (right["joint_names"], right["position_units"]):
        return None
    profile = left["thresholds"]
    if left["version"] != right["version"] or profile != right["thresholds"]:
        raise ValueError("Cannot compare differently calibrated motion descriptors")
    scales = np.array([profile["joint_linear_m"] if unit == "m" else profile["joint_angle_rad"]
                       for unit in left["position_units"]])
    spanning = [(left["paths"], right["paths"])]
    if left["phases"] and [p["skill"] for p in left["phases"]] == [p["skill"] for p in right["phases"]]:
        spanning.extend((a["paths"], b["paths"]) for a, b in zip(left["phases"], right["phases"]))
    measured, commanded, wrist, objects = [], [], [], []
    for a, b in spanning:
        measured.append(_ratio(a["measured_joints"], b["measured_joints"], scales))
        commanded.append(_ratio(a["position_commands"], b["position_commands"], scales))
        for key in a.keys() & b.keys():
            if key.startswith("wrist:"):
                wrist.append(_pose_ratio(a[key], b[key], profile))
            elif key.startswith("object:"):
                objects.append(_pose_ratio(a[key], b[key], profile))
    return {"measured_joint_ratio": max(measured), "command_position_ratio": max(commanded),
            "wrist_pose_ratio": max(wrist, default=None), "object_pose_ratio": max(objects, default=None),
            "compared_phases": len(spanning) - 1}
