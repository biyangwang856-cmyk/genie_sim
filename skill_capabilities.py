"""Trusted RobotAPI capability discovery and physics-scoped skill certificates."""

from dataclasses import asdict, dataclass
import inspect
import json
from math import isfinite
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.spatial.transform import Rotation

from scene_contract import content_id, digest, read_json, require_id


HINGE_SKILL = "open_hinged_lid"
ROTATION_SKILL = "rotate_held_about_tool"
BUILTIN_APIS = {
    "grasp": ("pick_actor", "get_object_pose"),
    "move_held": ("move_actor_to_target_pose", "get_object_pose"),
    "align_precision": ("follow_trajectory", "get_object_pose"),
    "insert_axial": ("follow_trajectory", "get_object_pose"),
    "release": ("open_gripper", "get_object_pose"),
    "retreat": ("move_actor_to_target_pose", "get_ee_pose"),
    "wait_stable": ("get_object_pose",),
    "verify_relation": ("get_object_pose",),
    "sync_assembly_proxy": ("attach_obj",),
    "verify_task": ("get_object_pose",),
}
HINGE_APIS = (
    "open", "get_articulation_joint_info", "get_object_prim_path", "get_object_pose", "get_ee_pose",
)
HINGE_REQUIRES = ("curobo", "friction_grasp")
ROTATION_APIS = ("rotate_actor", "get_object_prim_path", "get_object_pose", "get_ee_pose")
PROBE_CAPABILITIES = {
    HINGE_SKILL: (HINGE_APIS, HINGE_REQUIRES, "hinge_asset_binding_required"),
    ROTATION_SKILL: (ROTATION_APIS, ("curobo", "friction_grasp", "held_object_motion"),
                     "rotation_asset_binding_required"),
}
KNOWN_API_NAMES = tuple(sorted({
    *(api for apis, _, _ in PROBE_CAPABILITIES.values() for api in apis),
    "move_actor_to_target_pose", *(api for apis in BUILTIN_APIS.values() for api in apis),
}))
_PROBE_CALLS = {
    "open": ((), {"actor": "", "joint_name": "", "open_config": None, "arm": ""}),
    "get_articulation_joint_info": (("",), {"joint_prim_path": ""}),
    "get_object_prim_path": (("",), {}),
    "get_object_pose": (("",), {"articulation_part": ""}),
    "get_ee_pose": ((), {"arm": ""}),
    "move_actor_to_target_pose": (("", None), {"arm": "", "move_config": None}),
    "rotate_actor": (("", None), {"arm": "", "rotate_config": None}),
}
_VERSION = 1
_TRACE_KEYS = ("hinge_base", "hinge_link", "hinge_handle")
_MAX_BASE_DRIFT_M = .006
_MAX_BASE_DRIFT_DEG = 2.
_MAX_SETTLE_DRIFT_DEG = 2.
_MIN_SWING_DEG = 5.
_MAX_LIMIT_ERROR_DEG = 5.
_MAX_HANDLE_CONTACT_ERROR_M = .045
_MAX_HANDLE_SLIP_M = .025
_MAX_HANDLE_SLIP_DEG = 15.


def _sha256(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Expected an immutable SHA-256 digest")
    return value


def runtime_api_methods(robot_api):
    """Validate published call signatures without invoking an actuator."""
    names = []
    for name in KNOWN_API_NAMES:
        method = getattr(robot_api, name, None)
        if not callable(method):
            continue
        if name in _PROBE_CALLS:
            args, kwargs = _PROBE_CALLS[name]
            try:
                inspect.signature(method).bind(*args, **kwargs)
            except (TypeError, ValueError):
                continue
        names.append(name)
    return tuple(names)


def _api_fingerprint(robot_api, methods):
    source = inspect.getsourcefile(type(robot_api))
    if source is None:
        raise ValueError("Cannot identify the loaded RobotAPI source")
    signatures = {}
    for name in methods:
        if name in _PROBE_CALLS:
            parameters = inspect.signature(getattr(robot_api, name)).parameters.values()
            # Avoid the memory addresses of upstream config instances used as defaults.
            signatures[name] = [(parameter.name, parameter.kind.name, parameter.default is not parameter.empty)
                                for parameter in parameters]
    return content_id({"source_sha256": digest(source), "signatures": signatures, "methods": list(methods)})


@dataclass(frozen=True)
class RuntimeScope:
    robot_id: str
    image_id: str
    calibration_sha256: str
    adapter_sha256: str
    api_sha256: str
    api_methods: tuple[str, ...]

    def __post_init__(self):
        require_id(self.robot_id)
        image_digest = self.image_id.removeprefix("sha256:") if isinstance(self.image_id, str) else self.image_id
        object.__setattr__(self, "image_id", "sha256:" + _sha256(image_digest))
        for value in (self.calibration_sha256, self.adapter_sha256, self.api_sha256):
            _sha256(value)
        if (not isinstance(self.api_methods, (list, tuple))
                or any(not isinstance(name, str) or name not in KNOWN_API_NAMES for name in self.api_methods)
                or len(set(self.api_methods)) != len(self.api_methods)):
            raise ValueError("Invalid RobotAPI capability list")
        object.__setattr__(self, "api_methods", tuple(sorted(self.api_methods)))

    def to_dict(self):
        return dict(asdict(self), api_methods=list(self.api_methods))

    @classmethod
    def from_robot_api(cls, adapter, robot_api, *, image_id, calibration_sha256, adapter_sha256):
        methods = runtime_api_methods(robot_api)
        return cls(adapter.robot_id, image_id, calibration_sha256, adapter_sha256,
                   _api_fingerprint(robot_api, methods), methods)

    def matches(self, robot_api):
        methods = runtime_api_methods(robot_api)
        return self.api_methods == methods and self.api_sha256 == _api_fingerprint(robot_api, methods)


def _part_path(path):
    if not isinstance(path, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*", path):
        raise ValueError("Invalid articulation prim path")
    return path


@dataclass(frozen=True)
class HingeBinding:
    actor: str
    joint_path: str | None
    handle_part: str
    arm: str
    target_limit: str
    asset_sha256: str
    handle_contact_local_m: tuple[float, float, float] = (0., 0., 0.)
    handle_fixed_joint_path: str | None = None

    def __post_init__(self):
        require_id(self.actor)
        _part_path(self.handle_part)
        _sha256(self.asset_sha256)
        if self.arm not in ("left", "right") or self.target_limit not in ("min", "max"):
            raise ValueError("Expected an arm and a native min/max joint limit")
        root = f"/World/Objects/{self.actor}/entity/"
        for field in ("joint_path", "handle_fixed_joint_path"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _part_path(value.removeprefix(root) if isinstance(value, str) else value))
        point = _array(self.handle_contact_local_m, (3,))
        object.__setattr__(self, "handle_contact_local_m", tuple(point.tolist()))

    def to_dict(self):
        return dict(asdict(self), handle_contact_local_m=list(self.handle_contact_local_m))

    def applicability(self):
        return {key: value for key, value in self.to_dict().items() if key != "actor"}


def probe_binding_type(skill):
    if skill == HINGE_SKILL:
        return HingeBinding
    if skill == ROTATION_SKILL:
        from rotation_probe import RotationBinding
        return RotationBinding
    raise ValueError(f"Unknown native skill probe {skill!r}")


def describe_capabilities(adapter, *, robot_api=None, scope=None):
    """Return the supported mappings, including mappings awaiting a live probe."""
    from skill_registry import SKILLS

    methods = _scope_methods(adapter, robot_api, scope)
    rows = {name: {
        "effect": SKILLS[name]["success"], "requires": list(SKILLS[name]["requires"]),
        "parameters": list(SKILLS[name]["parameters"]), "candidate_apis": list(apis),
        "status": "registered",
    } for name, apis in BUILTIN_APIS.items() if name in SKILLS}
    rows[HINGE_SKILL] = {
        "effect": "Move a grasped revolute part to its selected joint limit, with a stationary base",
        "requires": list(HINGE_REQUIRES), "parameters": list(HingeBinding.__dataclass_fields__),
        "candidate_apis": list(HINGE_APIS), "status": "candidate",
        "units": {"joint_angle": "deg", "position": "m", "orientation": "wxyz"},
        "verifier": "hinge_joint_limit_and_native_physics_trace",
    }
    rows[ROTATION_SKILL] = {
        "effect": "Rotate a physically held rigid object about the tool at a fixed TCP position",
        "requires": list(PROBE_CAPABILITIES[ROTATION_SKILL][1]),
        "parameters": ["actor", "arm", "asset_sha256", "angle_deg"],
        "candidate_apis": list(ROTATION_APIS), "status": "candidate",
        "units": {"rotation_angle": "deg", "position": "m"},
        "verifier": "signed_tool_rotation_and_native_physics_trace",
    }
    for row in rows.values():
        row["api_availability"] = {name: None if methods is None else name in methods
                                   for name in row["candidate_apis"]}
    return rows


def _scope_methods(adapter, robot_api, scope):
    if scope is not None:
        if scope.robot_id != adapter.robot_id or robot_api is not None and not scope.matches(robot_api):
            raise ValueError("RobotAPI runtime scope does not match the active adapter")
        return scope.api_methods
    return runtime_api_methods(robot_api) if robot_api is not None else None


def discover_skills(skills, adapter, *, robot_api=None, scope=None, registry=None, bindings=None):
    """Resolve all requested skills before formal plan validation."""
    from skill_registry import SKILLS

    if isinstance(skills, str):
        raise ValueError("Pass a sequence of skill identifiers")
    names = list(dict.fromkeys(require_id(name) for name in skills))
    methods = _scope_methods(adapter, robot_api, scope)
    report = {"status": "ready", "missing_skills": [], "candidate_apis": [],
              "missing_robot_capabilities": [], "missing_verifiers": [], "skill_details": {}}
    blocked_statuses = set()
    for name in names:
        if name in PROBE_CAPABILITIES:
            row = _discover_probe(name, adapter, methods, scope, registry, (bindings or {}).get(name))
        elif name in SKILLS:
            missing = sorted(set(SKILLS[name]["requires"]) - adapter.capabilities)
            row = {"status": "capability_gap" if missing else "ready",
                   "candidate_apis": list(BUILTIN_APIS.get(name, ())), "missing_robot_capabilities": missing}
        else:
            row = {"status": "unsupported", "reason": "no_compatible_robot_api",
                   "candidate_apis": [],
                   "missing_robot_capabilities": ["vacuum_actuation"] if name == "vacuum_pick" else []}
        report["skill_details"][name] = row
        if row["status"] != "ready":
            blocked_statuses.add(row["status"])
            report["missing_skills"].append(name)
            for field in ("candidate_apis", "missing_robot_capabilities", "missing_verifiers"):
                report[field].extend(row.get(field, ()))
    for field in ("candidate_apis", "missing_robot_capabilities", "missing_verifiers"):
        report[field] = sorted(set(report[field]))
    if "unsupported" in blocked_statuses:
        report.update(status="unsupported", code="SKILL_NOT_REGISTERED", message="Skill is not registered")
    elif "capability_gap" in blocked_statuses:
        report.update(status="capability_gap", code="SKILL_CAPABILITY_GAP",
                      message="Skill needs compatible physical capabilities or observations")
    elif blocked_statuses:
        report.update(status="validating_skill", code="SKILL_VALIDATION_REQUIRED",
                      message="A candidate skill requires independent physical validation")
    return report


def _discover_probe(skill, adapter, methods, scope, registry, binding):
    apis, requires, binding_gap = PROBE_CAPABILITIES[skill]
    row = {"candidate_apis": [api for api in apis if methods is None or api in methods],
           "api_probed": methods is not None}
    if methods is not None and apis[0] not in methods:
        return dict(row, status="unsupported", reason="no_compatible_robot_api", candidate_apis=[])
    if isinstance(binding, dict):
        binding = probe_binding_type(skill)(**binding)
    if binding is not None and not isinstance(binding, probe_binding_type(skill)):
        raise ValueError("Skill binding type does not match the requested effect")
    missing = sorted(set(requires) - adapter.capabilities)
    if binding is not None and binding.arm not in adapter.arms:
        missing.append("arm:" + binding.arm)
    measurements = [] if methods is None else [name for name in apis[1:] if name not in methods]
    if missing or measurements:
        return dict(row, status="capability_gap", reason="missing_robot_capability_or_observation",
                    missing_robot_capabilities=missing, missing_verifiers=measurements)
    if binding is None:
        return dict(row, status="validating_skill", reason=binding_gap)
    certificate = registry.lookup(skill, scope, binding) if registry is not None and scope is not None else None
    if certificate is not None:
        return dict(row, status="ready", certificate_sha256=certificate["certificate_sha256"])
    return dict(row, status="validating_skill", reason="physical_certificate_required")


class HingeCapabilityGap(ValueError):
    """The live scene or runtime cannot satisfy a hinge skill precondition."""

    def __init__(self, reason, **details):
        super().__init__(reason)
        self.reason = reason
        self.details = details


def _array(value, shape):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise HingeCapabilityGap(f"Expected finite physical coordinates with shape {shape}")
    return array


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise HingeCapabilityGap("Expected a finite numeric measurement")
    return float(value)


def _rigid_pose(value):
    matrix = _array(value, (4, 4))
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
        raise HingeCapabilityGap("Invalid rigid physical pose")
    return matrix


def _object_pose(robot_api, actor, part):
    position, quat_wxyz = robot_api.get_object_pose(actor, articulation_part=part)
    pose = np.eye(4)
    pose[:3, 3] = _array(position, (3,))
    pose[:3, :3] = Rotation.from_quat(_array(quat_wxyz, (4,))[[1, 2, 3, 0]]).as_matrix()
    return pose


def _part_under_actor(absolute, root):
    if not isinstance(absolute, str) or not absolute.startswith(root + "/"):
        raise HingeCapabilityGap("Hinge metadata references a prim outside the bound asset")
    return _part_path(absolute[len(root) + 1:])


def _hinge_info(robot_api, binding):
    missing = set(HINGE_APIS) - set(runtime_api_methods(robot_api))
    if missing:
        raise HingeCapabilityGap(f"RobotAPI lacks expected signatures for {sorted(missing)}")
    root = robot_api.get_object_prim_path(binding.actor)
    if root != f"/World/Objects/{binding.actor}/entity":
        raise HingeCapabilityGap("Actor binding does not resolve to its scene prim")
    requested = f"{root}/{binding.joint_path}" if binding.joint_path is not None else None
    info = robot_api.get_articulation_joint_info(root, joint_prim_path=requested)
    if not isinstance(info, dict) or not {"joint_path", "body0_path", "body1_path", "axis",
                                          "lower_limit_deg", "upper_limit_deg", "current_angle_deg"} <= info.keys():
        raise HingeCapabilityGap("No verified revolute joint observation for the requested asset")
    _part_under_actor(info["joint_path"], root)
    base_part = _part_under_actor(info["body0_path"], root)
    link_part = _part_under_actor(info["body1_path"], root)
    if requested is not None and info["joint_path"] != requested:
        raise HingeCapabilityGap("The live joint differs from the requested revolute joint")
    handle_on_link = binding.handle_part == link_part or binding.handle_part.startswith(link_part + "/")
    if (info.get("joint_type", "revolute") not in ("revolute", "PhysicsRevoluteJoint")
            or info["axis"] not in ("X", "Y", "Z") or base_part == link_part
            or not (handle_on_link or binding.handle_fixed_joint_path is not None)):
        raise HingeCapabilityGap("The handle must lie on the moving side of the revolute joint")
    lower = _finite(info["lower_limit_deg"])
    upper = _finite(info["upper_limit_deg"])
    angle = _finite(info["current_angle_deg"])
    # The native viewport angle wraps at 180 degrees. This adapter is scoped to door hinges.
    if (not -180 < lower < upper < 180 or not _MIN_SWING_DEG <= upper - lower <= 180
            or not lower - _MAX_LIMIT_ERROR_DEG <= angle <= upper + _MAX_LIMIT_ERROR_DEG):
        raise HingeCapabilityGap("Revolute joint angle or limits are outside the supported door range")
    return info, base_part, link_part


def _snapshot(robot_api, binding):
    info, base_part, link_part = _hinge_info(robot_api, binding)
    joint_paths = [info["joint_path"]]
    if binding.handle_fixed_joint_path is not None:
        root = robot_api.get_object_prim_path(binding.actor)
        joint_paths.append(f"{root}/{binding.handle_fixed_joint_path}")
    snapshot = robot_api.client.command_any({"command": "ProjectSnapshot", "joint_paths": joint_paths})
    if (not isinstance(snapshot, dict)
            or any(snapshot.get("joint_enabled", {}).get(path) is not True for path in joint_paths)):
        raise HingeCapabilityGap("Native hinge or handle joint is disabled or unobservable")
    return {"joint": info, "joint_enabled": True,
            "enabled_joint_paths": joint_paths,
            "base_pose": _object_pose(robot_api, binding.actor, base_part).tolist(),
            "link_pose": _object_pose(robot_api, binding.actor, link_part).tolist(),
            "handle_pose": _object_pose(robot_api, binding.actor, binding.handle_part).tolist(),
            "ee_pose": _rigid_pose(robot_api.get_ee_pose(arm=binding.arm)).tolist(),
            "sim_time_s": _finite(snapshot["sim_time"])}


def _contact_error(snapshot, binding):
    handle = _rigid_pose(snapshot["handle_pose"])
    contact = (handle @ np.r_[binding.handle_contact_local_m, 1.])[:3]
    return float(np.linalg.norm(_rigid_pose(snapshot["ee_pose"])[:3, 3] - contact))


def _rank_handle_grasps(robot_api, binding, goals, *, observe):
    snapshot = _snapshot(robot_api, binding)
    positions, orientations = _native_hinge_arc(robot_api, binding, snapshot)
    sampled = np.unique(np.rint(np.linspace(0, len(positions) - 1, 7)).astype(int))
    transforms = np.repeat(np.eye(4)[None], len(sampled), axis=0)
    transforms[:, :3, 3] = positions[sampled]
    transforms[:, :3, :3] = Rotation.from_quat(orientations[sampled][:, [1, 2, 3, 0]]).as_matrix()
    # A world-frame hinge delta moves any TCP rigidly held at the start pose.
    relative = transforms @ np.linalg.inv(transforms[0])
    candidates = np.asarray(goals)
    targets = relative[None, ...] @ candidates[:, None, ...]
    flags, _ = robot_api.solve_ik(targets.reshape(-1, 4, 4), arm=binding.arm, ee_type="gripper", type="AvoidObs")
    flags = np.asarray(flags, dtype=bool)
    if flags.shape != (len(candidates) * len(sampled),):
        raise HingeCapabilityGap("invalid_native_hinge_grasp_preflight")
    reachable = flags.reshape(len(candidates), len(sampled))
    complete = reachable.all(axis=1)
    prefix = np.cumprod(reachable, axis=1).sum(axis=1)
    observe({"arc_waypoints": len(positions), "sampled_arc_indices": sampled.tolist(),
             "grasps_tested": len(candidates), "complete_coarse_grasps": int(complete.sum()),
             "best_contiguous_feasible_samples": int(prefix.max()),
             "reachable_by_waypoint": reachable.sum(axis=0).tolist()})
    # Coarse IK ranks the alternatives. Native continuity planning still guards every motion.
    return np.argsort(-prefix, kind="stable")


def _handle_grasp_targets(robot_api, binding, geometry, ee_pose):
    """Project the native open jaws onto the annotated handle section."""
    low, high = _array(geometry["contact_box_local_m"], (2, 3))
    door_front = _finite(geometry["door_front_y_m"])
    if np.any(high - low < .004) or door_front >= low[1]:
        raise HingeCapabilityGap("invalid_handle_contact_geometry")
    measured = robot_api.client.command_any({"command": "ProjectGripperCollisionBounds", "arm": binding.arm})
    if not isinstance(measured, dict) or measured.get("success") is not True:
        raise HingeCapabilityGap("native_gripper_collision_observation_unavailable")
    to_tool = np.linalg.inv(_rigid_pose(ee_pose))
    tool_shapes = {}
    for shape in measured["shapes"]:
        path = shape["path"]
        corners = _array(shape["corners_world_m"], (8, 3))
        link = path.split("/")[2]
        tool_shapes.setdefault(link, []).append(corners @ to_tool[:3, :3].T + to_tool[:3, 3])
    pads = [np.concatenate(shapes) for link, shapes in tool_shapes.items() if "support" in link]
    if len(pads) != 2:
        raise HingeCapabilityGap("native_gripper_pad_geometry_unavailable", collider_links=sorted(tool_shapes))
    near, far = sorted(pads, key=lambda group: float(group[:, 1].mean()))
    gap_start, gap_end = float(near[:, 1].max()), float(far[:, 1].min())
    overlap_low = np.maximum(near.min(axis=0), far.min(axis=0))
    overlap_high = np.minimum(near.max(axis=0), far.max(axis=0))
    opening = gap_end - gap_start
    if opening < high[2] - low[2] + .004 or np.any(overlap_high[[0, 2]] <= overlap_low[[0, 2]]):
        raise HingeCapabilityGap("handle_does_not_fit_native_open_jaws", measured_opening_m=opening)
    contact_in_tool = (overlap_low + overlap_high) / 2
    contact_in_tool[1] = (gap_start + gap_end) / 2
    all_corners = np.concatenate([points for shapes in tool_shapes.values() for points in shapes])
    # In the handle frame, +X runs along the bar and +Y is outward. Close
    # along the short +Z axis and approach the handle from the clear side.
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    center = (low + high) / 2
    goals = []
    shift = min(.006, (high[0] - low[0]) / 4)
    for offset in (0., -shift, shift):
        contact = center + [offset, 0., 0.]
        for axes in (rotation, rotation @ np.diag([-1., -1., 1.])):
            target = np.eye(4)
            target[:3, :3] = axes
            target[:3, 3] = contact - axes @ contact_in_tool
            corners_in_handle = all_corners @ axes.T + target[:3, 3]
            if corners_in_handle[:, 1].min() > door_front + .002:
                goals.append(target)
    if not goals:
        raise HingeCapabilityGap("no_clear_handle_grasp_geometry", measured_opening_m=opening)
    return np.asarray(goals), opening


def _grasp_recovery_state_unchanged(first, current):
    base1, base2 = (_rigid_pose(row["base_pose"]) for row in (first, current))
    angle1, angle2 = (row["joint"]["current_angle_deg"] for row in (first, current))
    return (_distance(base1, base2) <= .002 and _rotation_deg(base1, base2) <= .5
            and abs(angle2 - angle1) <= .5)


def _probe_hinge_grasp(robot_api, binding, before):
    """Test a short opening stroke before allowing a full-arc motion."""
    _track_hinge(robot_api, binding, before)
    positions, orientations = _native_hinge_arc(robot_api, binding, before)
    count = min(4, len(positions))
    planner_ok = _follow_hinge_prefix(robot_api, binding, positions[:count], orientations[:count])
    robot_api.client.run_step(24)
    after = _snapshot(robot_api, binding)
    base1, base2 = (_rigid_pose(row["base_pose"]) for row in (before, after))
    contacts = [np.linalg.inv(_rigid_pose(row["handle_pose"])) @ _rigid_pose(row["ee_pose"])
                for row in (before, after)]
    direction = -1 if binding.target_limit == "min" else 1
    progress = direction * (after["joint"]["current_angle_deg"] - before["joint"]["current_angle_deg"])
    slip = _distance(*contacts)
    slip_angle = _rotation_deg(*contacts)
    passed = (planner_ok and .7 <= progress <= 4. and slip <= .006 and slip_angle <= 5.
              and _distance(base1, base2) <= .002 and _rotation_deg(base1, base2) <= .5)
    return after, {"passed": passed, "planner_success": planner_ok, "progress_deg": progress,
                   "handle_to_hand_drift_m": slip, "handle_to_hand_drift_deg": slip_angle,
                   "before_deg": before["joint"]["current_angle_deg"],
                   "after_deg": after["joint"]["current_angle_deg"],
                   "start_sim_time_s": before["sim_time_s"], "end_sim_time_s": after["sim_time_s"]}


def grasp_hinge_handle(robot_api, move_config_type, binding, *, geometry, observe_preflight=None,
                       observe_attempt=None):
    """Verify that the independently observed door follows a real native grasp."""
    initial = _snapshot(robot_api, binding)
    if not all(callable(getattr(robot_api, method, None)) for method in (
            "move_actor_to_target_pose", "open_gripper", "close_gripper")):
        raise HingeCapabilityGap("native_handle_grasp_control_unavailable")
    robot_api.open_gripper(arm=binding.arm)
    robot_api.client.run_step(30)
    current = _snapshot(robot_api, binding)
    goals, opening = _handle_grasp_targets(robot_api, binding, geometry, current["ee_pose"])
    order = range(len(goals))
    if observe_preflight is not None and callable(getattr(robot_api, "solve_ik", None)):
        world_goals = _rigid_pose(current["handle_pose"]) @ goals
        order = _rank_handle_grasps(robot_api, binding, world_goals, observe=observe_preflight)
    config = move_config_type(
        gripper_action=None, extra_params={"slowdown_factor": 2.},
        action_description={"zh": "对准铰链把手", "en": "approach the hinge handle"},
    )
    result = {}
    for attempt, goal_index in enumerate(list(order)[:3], start=1):
        if not _grasp_recovery_state_unchanged(initial, current):
            break
        goal = _rigid_pose(current["handle_pose"]) @ goals[goal_index]
        try:
            moved = robot_api.move_actor_to_target_pose("gripper", goal[None], arm=binding.arm, move_config=config)
        except HingeCapabilityGap as error:
            if error.reason != "Native action failed before rollback":
                raise
            moved = False
        robot_api.client.run_step(24)
        current = _snapshot(robot_api, binding)
        live_ee = _rigid_pose(current["ee_pose"])
        error_m, error_deg = _distance(goal, live_ee), _rotation_deg(goal, live_ee)
        witness = {"passed": False, "reason": "grasp_pose_unreached", "attempt": attempt,
                   "native_motion_ok": bool(moved), "target_pose": goal.tolist(),
                   "position_error_m": error_m, "rotation_error_deg": error_deg,
                   "measured_opening_m": opening}
        if (moved and error_m <= .008 and error_deg <= 6.
                and _grasp_recovery_state_unchanged(initial, current)):
            robot_api.close_gripper(arm=binding.arm)
            robot_api.client.run_step(36)
            before = _snapshot(robot_api, binding)
            if _grasp_recovery_state_unchanged(initial, before):
                current, contact = _probe_hinge_grasp(robot_api, binding, before)
                witness.update(contact, reason=None if contact["passed"] else "grasp_contact_not_confirmed")
            else:
                current = before
                witness["reason"] = "fixture_shifted_during_closure"
        if observe_attempt is not None:
            observe_attempt(witness)
        result = {"success": witness["passed"], "native_motion_ok": bool(moved),
                  "contact_error_m": _contact_error(current, binding), "after": current,
                  "candidate_count": len(goals), "attempt_count": attempt, "confirmation": witness}
        if witness["passed"]:
            result["before_confirmed_pull"] = before
            break
        if not _grasp_recovery_state_unchanged(initial, current):
            break
        robot_api.open_gripper(arm=binding.arm)
        robot_api.client.run_step(30)
        current = _snapshot(robot_api, binding)
    if not result:
        raise HingeCapabilityGap("fixture_moved_before_handle_grasp")
    return result


def _native_hinge_arc(robot_api, binding, snapshot):
    joint = snapshot["joint"]
    result = robot_api.client.command_any({
        "command": "GetRevoluteJointTrajectory",
        "articulation_object_prim_path": robot_api.get_object_prim_path(binding.actor),
        "joint_prim_path": joint["joint_path"],
        "object_prim_path": binding.arm + "_gripper",
        "target_limit": binding.target_limit,
        "trajectory_step_deg": 1.,
    })
    if not isinstance(result, dict):
        raise HingeCapabilityGap("invalid_native_hinge_arc")
    positions = result.get("trajectory_positions")
    orientations = result.get("trajectory_orientations")
    if not isinstance(positions, list) or not 2 <= len(positions) <= 182:
        raise HingeCapabilityGap("invalid_native_hinge_arc_count")
    positions = _array(positions, (len(positions), 3))
    orientations = _array(orientations, (len(positions), 4))
    if not np.allclose(np.linalg.norm(orientations, axis=1), 1., atol=.002):
        raise HingeCapabilityGap("invalid_native_hinge_arc_orientation")
    first = np.eye(4)
    first[:3, 3] = positions[0]
    first[:3, :3] = Rotation.from_quat(orientations[0, [1, 2, 3, 0]]).as_matrix()
    live_ee = _rigid_pose(snapshot["ee_pose"])
    position_error = _distance(first, live_ee)
    rotation_error = _rotation_deg(first, live_ee)
    if position_error > .008 or rotation_error > 4.:
        raise HingeCapabilityGap("native_hinge_arc_frame_mismatch",
                                 start_error_m=position_error, start_error_deg=rotation_error)
    return positions, orientations


def _follow_hinge_prefix(robot_api, binding, positions, orientations):
    robot_api.client.set_frame_state(
        "open", 0, "gripper", binding.actor, False, arm=binding.arm,
        action_description={"action_text": "沿铰链轨迹打开门板", "english_action_text": "follow the door hinge"},
    )
    result = robot_api.client.command_any({
        "command": "FollowTrajectory",
        "trajectory_positions": positions.tolist(),
        "trajectory_orientations": orientations.tolist(),
        "solvable_threshold": float(np.nextafter(1., 0.)),
        "max_consecutive_failures": 1,
        "is_right": binding.arm == "right",
    })
    return isinstance(result, dict) and result.get("success") is True


def follow_hinge_segments(robot_api, binding, before, *, observe):
    """Run collision-checked prefixes of the native arc and remeasure after each."""
    direction = -1 if binding.target_limit == "min" else 1
    limit = "lower_limit_deg" if direction < 0 else "upper_limit_deg"
    target = before["joint"][limit]
    reference_base = _rigid_pose(before["base_pose"])
    reference_contact = np.linalg.inv(_rigid_pose(before["handle_pose"])) @ _rigid_pose(before["ee_pose"])
    current = before
    step_deg = 24.
    for index in range(32):
        angle = current["joint"]["current_angle_deg"]
        remaining = direction * (target - angle)
        if remaining <= 2.:
            return current
        positions, orientations = _native_hinge_arc(robot_api, binding, current)
        count = min(len(positions), max(2, 1 + int(step_deg * (len(positions) - 1) / remaining)))
        reached = _follow_hinge_prefix(robot_api, binding, positions[:count], orientations[:count])
        observed = _snapshot(robot_api, binding)
        actual = observed["joint"]["current_angle_deg"]
        progress = direction * (actual - angle)
        commanded = remaining * (count - 1) / (len(positions) - 1)
        witness = {"segment": index + 1, "before_deg": angle, "after_deg": actual,
                   "commanded_swing_deg": commanded, "waypoints": count,
                   "original_arc_waypoints": len(positions), "planner_success": reached,
                   "progress_deg": progress}
        observe(witness)
        base = _rigid_pose(observed["base_pose"])
        contact = np.linalg.inv(_rigid_pose(observed["handle_pose"])) @ _rigid_pose(observed["ee_pose"])
        if (_distance(reference_base, base) > _MAX_BASE_DRIFT_M
                or _rotation_deg(reference_base, base) > _MAX_BASE_DRIFT_DEG
                or _distance(reference_contact, contact) > _MAX_HANDLE_SLIP_M
                or _rotation_deg(reference_contact, contact) > _MAX_HANDLE_SLIP_DEG
                or _contact_error(observed, binding) > _MAX_HANDLE_CONTACT_ERROR_M):
            raise HingeCapabilityGap("hinge_hold_or_base_shift", **witness)
        if not reached:
            if (step_deg <= 4. or abs(progress) > .6
                    or _distance(_rigid_pose(current["ee_pose"]), _rigid_pose(observed["ee_pose"])) > .004):
                raise HingeCapabilityGap("hinge_segment_unreachable", **witness)
            step_deg = max(4., step_deg / 2)
        elif progress < min(.8, commanded * .5):
            raise HingeCapabilityGap("hinge_segment_did_not_move_as_planned", **witness)
        else:
            step_deg = min(24., step_deg * 1.5)
        current = observed
    raise HingeCapabilityGap("hinge_segment_budget_exhausted",
                             after_deg=current["joint"]["current_angle_deg"], target_deg=target)


def _track_hinge(robot_api, binding, snapshot):
    joint = snapshot["joint"]
    tracked = robot_api.client.command_any({
        "command": "ProjectTrackObjects", "tcp_arm": binding.arm,
        "paths": dict(zip(_TRACE_KEYS, [joint["body0_path"], joint["body1_path"],
                                       robot_api.get_object_prim_path(binding.actor) + "/" + binding.handle_part],
                          strict=True)),
    })
    if not isinstance(tracked, dict) or tracked.get("success") is not True:
        raise HingeCapabilityGap("Native hinge trajectory recording is unavailable")


def execute_hinge(robot_api, open_config_type, binding, *, scope=None, observe_segment=None,
                  verified_grasp=None):
    """Run the bounded native action and capture raw telemetry for a separate verifier."""
    if scope is not None and not scope.matches(robot_api):
        raise HingeCapabilityGap("The loaded RobotAPI changed after capability discovery")
    before = _snapshot(robot_api, binding)
    if _contact_error(before, binding) > _MAX_HANDLE_CONTACT_ERROR_M:
        raise HingeCapabilityGap("The selected hand is not at the annotated handle contact")
    joint = before["joint"]
    target_name = "lower_limit_deg" if binding.target_limit == "min" else "upper_limit_deg"
    if abs(joint["current_angle_deg"] - joint[target_name]) < _MIN_SWING_DEG:
        raise HingeCapabilityGap("The hinge is already at the target and cannot certify motion")
    _track_hinge(robot_api, binding, before)
    if observe_segment is None:
        config = open_config_type(target_limit=binding.target_limit, action_description={
            "zh": "转动铰链至开位", "en": "open the articulated part to the requested limit",
        })
        reached = robot_api.open(actor=binding.actor, joint_name=joint["joint_path"],
                                 arm=binding.arm, open_config=config)
        after = _snapshot(robot_api, binding)
    else:
        after = follow_hinge_segments(robot_api, binding, before, observe=observe_segment)
        reached = True
    robot_api.client.run_step(36)
    settled = _snapshot(robot_api, binding)
    return {"schema": _VERSION, "skill": HINGE_SKILL, "binding": binding.to_dict(),
            "scope": scope.to_dict() if scope is not None else None,
            "native_result": bool(reached), "before": verified_grasp or before,
            "after": after, "settled": settled}


def _rotation_deg(first, second):
    relative = first[:3, :3].T @ second[:3, :3]
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def _distance(first, second):
    return float(np.linalg.norm(first[:3, 3] - second[:3, 3]))


def _trace_window(rows, start, stop, *, tolerance=0.03):
    selected = [(stamp, row) for row in rows if start - tolerance <= (stamp := _finite(row["sim_time"])) <= stop + tolerance]
    if len(selected) < 2:
        raise HingeCapabilityGap("Missing native trace of hinge motion")
    return np.asarray([stamp for stamp, _ in selected]), [row for _, row in selected]


def verify_hinge(evidence, binding, *, physics_rows, command_rows):
    """Independently calculate physical acceptance from raw measurements and the drive journal."""
    checks = {}
    metrics = {}
    try:
        if (evidence["schema"] != _VERSION or evidence["skill"] != HINGE_SKILL
                or evidence["binding"] != binding.to_dict()):
            raise HingeCapabilityGap("Evidence is for a different hinge binding")
        before, after, settled = (evidence[name] for name in ("before", "after", "settled"))
        t0, t1, t2 = (_finite(row["sim_time_s"]) for row in (before, after, settled))
        checks["native_returned_success"] = evidence["native_result"] is True
        checks["simulated_and_settled"] = 0 <= t0 < t1 and t2 - t1 >= .15
        joint = before["joint"]
        keys = ("joint_path", "body0_path", "body1_path", "axis", "lower_limit_deg", "upper_limit_deg")
        checks["same_enabled_joint"] = all(row["joint_enabled"] is True and
                                           all(row["joint"][key] == joint[key] for key in keys)
                                           for row in (before, after, settled))
        actor_root = f"/World/Objects/{binding.actor}/entity"
        joint_relative = _part_under_actor(joint["joint_path"], actor_root)
        base_part = _part_under_actor(joint["body0_path"], actor_root)
        link_part = _part_under_actor(joint["body1_path"], actor_root)
        if binding.handle_fixed_joint_path is not None:
            fixed_joint = f"{actor_root}/{binding.handle_fixed_joint_path}"
            checks["native_fixed_handle_enabled"] = all(
                fixed_joint in row.get("enabled_joint_paths", []) for row in (before, after, settled))
        checks["revolute_binding"] = (joint["axis"] in ("X", "Y", "Z") and base_part != link_part
                                      and binding.joint_path in (None, joint_relative)
                                      and (binding.handle_part == link_part
                                           or binding.handle_part.startswith(link_part + "/")
                                           or binding.handle_fixed_joint_path is not None))
        lower, upper = (_finite(joint[key]) for key in ("lower_limit_deg", "upper_limit_deg"))
        target = lower if binding.target_limit == "min" else upper
        angle0, angle1, angle2 = (_finite(row["joint"]["current_angle_deg"]) for row in (before, after, settled))
        direction = -1 if binding.target_limit == "min" else 1
        metrics.update(target_deg=target, before_deg=angle0, after_deg=angle1, settled_deg=angle2)
        checks["actual_hinge_opening"] = (-180 < lower < upper < 180
                                          and _MIN_SWING_DEG <= upper - lower <= 180
                                          and all(lower - _MAX_LIMIT_ERROR_DEG <= angle <= upper + _MAX_LIMIT_ERROR_DEG
                                                  for angle in (angle0, angle1, angle2))
                                          and direction * (angle2 - angle0) >= _MIN_SWING_DEG
                                          and abs(angle0 - target) >= _MIN_SWING_DEG
                                          and max(abs(angle1 - target), abs(angle2 - target)) <= _MAX_LIMIT_ERROR_DEG
                                          and abs(angle2 - angle1) <= _MAX_SETTLE_DRIFT_DEG)
        base0 = _rigid_pose(before["base_pose"])
        base_poses = [_rigid_pose(row["base_pose"]) for row in (after, settled)]
        metrics["base_drift_m"] = max(_distance(base0, pose) for pose in base_poses)
        metrics["base_drift_deg"] = max(_rotation_deg(base0, pose) for pose in base_poses)
        checks["base_stayed_put"] = (metrics["base_drift_m"] <= _MAX_BASE_DRIFT_M
                                     and metrics["base_drift_deg"] <= _MAX_BASE_DRIFT_DEG)
        relations = [np.linalg.inv(_rigid_pose(row["handle_pose"])) @ _rigid_pose(row["ee_pose"])
                     for row in (before, after, settled)]
        metrics["handle_slip_m"] = max(_distance(relations[0], relation) for relation in relations[1:])
        metrics["handle_slip_deg"] = max(_rotation_deg(relations[0], relation) for relation in relations[1:])
        checks["handle_followed_hand"] = (max(_contact_error(row, binding) for row in (before, after, settled))
                                            <= _MAX_HANDLE_CONTACT_ERROR_M
                                         and metrics["handle_slip_m"] <= _MAX_HANDLE_SLIP_M
                                         and metrics["handle_slip_deg"] <= _MAX_HANDLE_SLIP_DEG)
        clocks, trace = _trace_window(physics_rows, t0, t2)
        metrics["physics_samples"] = len(trace)
        deltas = np.diff(clocks)
        raw_poses = [[_rigid_pose(row["poses"][key]) for row in trace] for key in _TRACE_KEYS]
        raw_base, raw_link, raw_handle = raw_poses
        handle_in_link = [np.linalg.inv(link) @ handle for link, handle in zip(raw_link, raw_handle, strict=True)]
        checks["handle_remained_part_of_link"] = all(
            _distance(handle_in_link[0], relation) <= .004
            and _rotation_deg(handle_in_link[0], relation) <= 3. for relation in handle_in_link[1:])
        reference = {key: _rigid_pose(settled[key]) for key in ("base_pose", "link_pose", "handle_pose")}
        checks["raw_coverage_and_continuity"] = (len(trace) >= 12 and abs(clocks[0] - t0) <= .03
                                                  and abs(clocks[-1] - t2) <= .03
                                                  and bool(np.all((deltas > 0) & (deltas <= .031)))
                                                  and all(_rotation_deg(prev, next_) <= 12.
                                                          for prev, next_ in zip(raw_link, raw_link[1:])))
        checks["independent_raw_hinge_motion"] = (
            all(_distance(base0, pose) <= _MAX_BASE_DRIFT_M
                and _rotation_deg(base0, pose) <= _MAX_BASE_DRIFT_DEG for pose in raw_base)
            and abs(_rotation_deg(raw_link[0], raw_link[-1]) - abs(angle2 - angle0)) <= 6.
            and all(_distance(poses[-1], reference[key]) <= .008
                    and _rotation_deg(poses[-1], reference[key]) <= 6.
                    for poses, key in zip(raw_poses, reference, strict=True)))
        command_clock, commands = _trace_window(command_rows, t0, t1)
        targets = [row["applied_joint_positions"] for row in commands]
        if not isinstance(targets[0], list) or not targets[0]:
            raise HingeCapabilityGap("Native joint targets are missing")
        resolved = np.asarray([_array(target, (len(targets[0]),)) for target in targets])
        metrics["drive_commands"] = len(commands)
        checks["live_drive_motion"] = (bool(np.all(np.diff(command_clock) >= 0))
                                        and bool(np.any(np.abs(np.diff(resolved, axis=0)) > 1e-5)))
    except (KeyError, TypeError, ValueError, AttributeError, np.linalg.LinAlgError) as exc:
        return {"passed": False, "checks": checks, "metrics": metrics, "error": str(exc)}
    return {"passed": bool(checks) and all(checks.values()), "checks": checks, "metrics": metrics}


def _jsonl_path(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


class SkillRegistry:
    """Publish immutable, content-checked receipts for a specific runtime and asset scope."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def _scope_record(self, skill, scope, binding):
        verifier_file = __file__
        if skill == ROTATION_SKILL:
            from rotation_probe import verify_rotation
            verifier_file = inspect.getsourcefile(verify_rotation)
        return {"schema": _VERSION, "skill": skill, "runtime": scope.to_dict(),
                "binding": binding.applicability(), "verifier_sha256": digest(verifier_file)}

    def lookup(self, skill, scope, binding):
        if skill not in PROBE_CAPABILITIES or scope is None or not isinstance(binding, probe_binding_type(skill)):
            return None
        key = self._scope_record(skill, scope, binding)
        path = self.directory / (content_id(key) + ".json")
        try:
            receipt = read_json(path)
        except FileNotFoundError:
            return None
        if (receipt["key"] != key or receipt["verification"]["passed"] is not True
                or content_id({name: value for name, value in receipt.items() if name != "certificate_sha256"})
                != receipt["certificate_sha256"]):
            raise ValueError("Skill registry certificate is invalid")
        return receipt

    def certify_hinge(self, scope, binding, evidence_path, *, physics_path=None, controls_path=None):
        return self.certify_probe(HINGE_SKILL, scope, binding, evidence_path, physics_path=physics_path,
                                  controls_path=controls_path)

    def certify_probe(self, skill, scope, binding, evidence_path, *, physics_path=None, controls_path=None):
        if not isinstance(binding, probe_binding_type(skill)):
            raise ValueError("Certification requires the binding type declared for this skill")
        if set(PROBE_CAPABILITIES[skill][0]) - set(scope.api_methods):
            raise ValueError("Certification requires all observed RobotAPI interfaces for the skill")
        verify = verify_hinge
        if skill == ROTATION_SKILL:
            from rotation_probe import verify_rotation
            verify = verify_rotation
        evidence_path = Path(evidence_path)
        physics_path = Path(physics_path) if physics_path is not None else evidence_path.parent / "object-physics-120hz.jsonl"
        controls_path = Path(controls_path) if controls_path is not None else evidence_path.parent / "native-control-commands.jsonl"
        sources = {"observations": evidence_path, "physics": physics_path, "controls": controls_path}
        try:
            hashes = {name: digest(path) for name, path in sources.items()}
            evidence = read_json(evidence_path)
            if evidence.get("scope") != scope.to_dict():
                raise HingeCapabilityGap("Observations were collected with a different RobotAPI runtime")
            verification = verify(evidence, binding, physics_rows=_jsonl_path(physics_path),
                                  command_rows=_jsonl_path(controls_path))
            if any(digest(path) != hashes[name] for name, path in sources.items()):
                raise HingeCapabilityGap("The recording changed during physical certification")
        except (OSError, TypeError, ValueError, AttributeError) as exc:
            return {"status": "skill_validation_failed", "skill": skill, "error": str(exc)}
        if not verification["passed"]:
            return {"status": "skill_validation_failed", "skill": skill, "verification": verification}
        key = self._scope_record(skill, scope, binding)
        receipt = {"key": key, "verification": verification, "evidence_sha256": hashes,
                   "evidence_files": {name: str(path.resolve()) for name, path in sources.items()}}
        receipt["certificate_sha256"] = content_id(receipt)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / (content_id(key) + ".json")
        with NamedTemporaryFile(dir=self.directory, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                temporary.write(json.dumps(receipt, indent=2, allow_nan=False).encode())
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary.close()
                try:
                    if os.name == "nt":
                        temporary_path.rename(path)
                    else:
                        os.link(temporary_path, path)
                except FileExistsError:
                    pass  # Another verifier published the same scoped skill first.
            finally:
                temporary_path.unlink(missing_ok=True)
        return {"status": "ready", "skill": skill, "certificate_path": str(path),
                "certificate": self.lookup(skill, scope, binding),
                "run_verification": verification, "run_evidence_sha256": hashes}
