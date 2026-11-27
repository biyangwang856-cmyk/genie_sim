"""Bounded, observation-driven retries within a validated skill step."""
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
import math
import time
from typing import Any

import numpy as np


class FailureKind(str, Enum):
    PLANNING_REFUSED = "planning_refused"
    TRACKING_ERROR = "tracking_error"
    GRASP_FAILED = "grasp_failed"
    GRASP_LOST = "grasp_lost"
    PHYSICAL_FAILURE = "physical_failure"
    PRECONDITION_FAILED = "precondition_failed"
    OBSERVATION_MISSING = "observation_missing"
    BUDGET_EXHAUSTED = "recovery_budget_exhausted"
    UNKNOWN = "unknown_failure"


class RecoveryFailure(RuntimeError):
    def __init__(self, kind: FailureKind, details: Mapping[str, Any] | None = None):
        self.kind = FailureKind(kind)
        self.details = dict(details or {})
        super().__init__(self.kind.value)


@dataclass(frozen=True)
class PhysicalState:
    """A fresh world observation. Offsets address one append-only native command log."""

    poses: Mapping[str, np.ndarray]
    joints: Mapping[str, float] = field(default_factory=dict)
    command_offset: int | None = None
    holding: Mapping[str, str] = field(default_factory=dict)
    sim_time: float | None = None

    def __post_init__(self):
        poses = {}
        for name, value in self.poses.items():
            pose = np.array(value, dtype=float, copy=True)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"Missing or invalid measured pose for {name}")
            rotation = pose[:3, :3]
            if (not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-5) or
                    not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or
                    np.linalg.det(rotation) < 0.):
                raise ValueError(f"Measured pose for {name} is not a rigid transform")
            pose.setflags(write=False)
            poses[name] = pose
        joints = {name: float(value) for name, value in self.joints.items()}
        if not (poses or joints) or not all(map(math.isfinite, joints.values())):
            raise ValueError("Physical state needs finite measured poses or joint positions")
        if self.command_offset is not None and (type(self.command_offset) is not int or self.command_offset < 0):
            raise ValueError("Invalid native command offset")
        if self.sim_time is not None and not math.isfinite(self.sim_time):
            raise ValueError("Invalid simulation time")
        object.__setattr__(self, "poses", MappingProxyType(poses))
        object.__setattr__(self, "joints", MappingProxyType(joints))
        object.__setattr__(self, "holding", MappingProxyType(dict(self.holding)))

    def evidence(self):
        return {
            "poses": {name: pose.tolist() for name, pose in self.poses.items()},
            "joints": dict(self.joints),
            "holding": dict(self.holding),
            "native_command_offset": self.command_offset,
            "sim_time": self.sim_time,
        }


@dataclass(frozen=True)
class RecoveryCandidate:
    """An approved action variant. Keep its name stable across plans and retries."""

    name: str
    action: Callable[[PhysicalState], Any]
    after: frozenset[FailureKind] = frozenset({FailureKind.PLANNING_REFUSED})
    applicable: Callable[[PhysicalState], bool] | None = None
    require_command: bool = True


def native_command_offset(path: Path):
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None


def physical_change(before: PhysicalState, after: PhysicalState):
    if before.poses.keys() != after.poses.keys() or before.joints.keys() != after.joints.keys():
        raise RecoveryFailure(FailureKind.OBSERVATION_MISSING, {"reason": "observation_channels_changed"})
    if before.sim_time is not None and after.sim_time is not None and after.sim_time < before.sim_time:
        raise RecoveryFailure(FailureKind.OBSERVATION_MISSING, {"reason": "simulation_clock_reversed"})
    commands = None
    if before.command_offset is not None and after.command_offset is not None:
        commands = after.command_offset - before.command_offset
        if commands < 0:
            raise RecoveryFailure(FailureKind.OBSERVATION_MISSING, {"reason": "native_command_log_reversed"})
    translation = 0.
    rotation = 0.
    for name, pose in before.poses.items():
        measured = after.poses[name]
        translation = max(translation, float(np.linalg.norm(measured[:3, 3] - pose[:3, 3])))
        relative_trace = np.sum(pose[:3, :3] * measured[:3, :3])
        rotation = max(rotation, math.acos(float(np.clip((relative_trace - 1.) / 2., -1., 1.))))
    return {
        "translation_m": translation,
        "rotation_rad": rotation,
        "joint_delta": max((abs(after.joints[key] - value) for key, value in before.joints.items()), default=0.),
        "holding_changed": before.holding != after.holding,
        "native_command_bytes": commands,
    }


def materially_changed(change, position_tolerance_m, rotation_tolerance_rad, joint_tolerance):
    return (change["translation_m"] > position_tolerance_m or change["rotation_rad"] > rotation_tolerance_rad
            or change["joint_delta"] > joint_tolerance or change["holding_changed"])


def classify_failure(kind, details, *, world_changed, commands_written):
    """A refusal can enter the no-motion retry path only with measured evidence."""
    no_motion = commands_written == 0 and not world_changed
    if kind == FailureKind.PLANNING_REFUSED or kind in {
        "native_action_failed_before_rollback", "native_action_false", "local_waypoint_segment_out_of_range",
        "no_paired_presentation_candidate", "packing_release_ik_infeasible",
    }:
        return FailureKind.PLANNING_REFUSED if no_motion else FailureKind.PHYSICAL_FAILURE
    if kind == "local_waypoint_plan_or_tracking_failed":
        if details.get("planner_accepted") is True and commands_written:
            return FailureKind.TRACKING_ERROR
        return FailureKind.PLANNING_REFUSED if no_motion and details.get("planner_accepted") is False else FailureKind.PHYSICAL_FAILURE
    if kind == "grasp_unreachable_or_failed":
        return FailureKind.PLANNING_REFUSED if no_motion else FailureKind.GRASP_FAILED
    if kind in {"grasp_candidate_changed_world", "probed_grasp_lift_exhausted"}:
        return FailureKind.GRASP_FAILED
    if kind in {"not_held", "shape_transport_grasp_slipped", "cap_grasp_lost_before_alignment",
                "pen_grasp_lost_before_alignment"}:
        return FailureKind.GRASP_LOST
    if kind in {"physical_task_incomplete", "unstable", "release_precondition_failed",
                "relation_not_satisfied", "axial_insertion_stalled"}:
        return FailureKind.PHYSICAL_FAILURE
    try:
        return FailureKind(kind)
    except ValueError:
        return FailureKind.UNKNOWN


def _read_state(observe):
    try:
        state = observe()
        if not isinstance(state, PhysicalState):
            raise TypeError("Expected PhysicalState from the simulator observer")
        return state
    except Exception as exc:
        raise RecoveryFailure(FailureKind.OBSERVATION_MISSING, {"reason": str(exc)}) from exc


def run_recoverable_skill(
    step_id: str,
    candidates: Sequence[RecoveryCandidate],
    *,
    observe: Callable[[], PhysicalState],
    verify: Callable[[PhysicalState, PhysicalState, Any], bool],
    event: Callable[..., None],
    deadline: float,
    max_attempts: int = 3,
    history: MutableMapping[tuple[str, str], list[tuple[PhysicalState, RecoveryFailure]]] | None = None,
    retry_key: str | None = None,
    position_tolerance_m: float = .001,
    rotation_tolerance_rad: float = .02,
    joint_tolerance: float = .01,
):
    """Execute physical actions in-place and admit only the verified effect.

    'observe' and 'verify' must be read-only. Non-planning remedies must opt in to
    specific failure kinds and check current prerequisites using 'applicable'.
    For cross-plan deduplication, share 'history' and use a semantic 'retry_key'.
    """
    if type(max_attempts) is not int or max_attempts < 1 or not candidates:
        raise ValueError("A positive attempt budget and at least one candidate are required")
    if not math.isfinite(deadline):
        raise ValueError("A finite monotonic deadline is required")
    tolerances = (position_tolerance_m, rotation_tolerance_rad, joint_tolerance)
    if not all(math.isfinite(value) and value > 0. for value in tolerances):
        raise ValueError("Physical change tolerances must be positive and finite")
    history = {} if history is None else history
    attempts = 0
    failure = None
    for candidate in candidates:
        context = {"step_id": step_id, "candidate": candidate.name}
        if time.monotonic() >= deadline or attempts >= max_attempts:
            raise RecoveryFailure(FailureKind.BUDGET_EXHAUSTED,
                                  {**context, "attempts": attempts, "last_failure": failure and failure.kind.value})
        if failure is not None and failure.kind not in candidate.after:
            event("skill_attempt_skipped", **context, reason="failure_not_applicable", failure_kind=failure.kind.value)
            continue
        if (failure is not None and failure.kind != FailureKind.PLANNING_REFUSED
                and candidate.applicable is None):
            event("skill_attempt_skipped", **context, reason="missing_recovery_precondition", failure_kind=failure.kind.value)
            continue
        before = _read_state(observe)
        if candidate.require_command and before.command_offset is None:
            raise RecoveryFailure(FailureKind.OBSERVATION_MISSING, {**context, "reason": "missing_native_command_log"})
        if candidate.applicable is not None and not candidate.applicable(before):
            if failure is None:
                failure = RecoveryFailure(FailureKind.PRECONDITION_FAILED, context)
            event("skill_attempt_skipped", **context, reason="measured_precondition_failed")
            continue
        failed_starts = history.setdefault((retry_key or step_id, candidate.name), [])
        duplicate = next((prior_failure for prior, prior_failure in failed_starts
                          if not materially_changed(physical_change(prior, before), *tolerances)), None)
        if duplicate is not None:
            failure = duplicate
            event("skill_attempt_skipped", **context, reason="same_candidate_without_physical_progress",
                  failure_kind=duplicate.kind.value, observed=before.evidence())
            continue

        if time.monotonic() >= deadline:
            raise RecoveryFailure(FailureKind.BUDGET_EXHAUSTED, {**context, "attempts": attempts})
        attempts += 1
        event("skill_attempt_started", **context, attempt=attempts, observed=before.evidence())
        native_error = None
        result = None
        try:
            result = candidate.action(before)
        except Exception as exc:
            if not isinstance(exc, RecoveryFailure) and not isinstance(getattr(exc, "kind", None), str):
                event("skill_attempt_failed", **context, attempt=attempts,
                      failure_kind=FailureKind.UNKNOWN.value, reason=str(exc))
                raise
            native_error = exc
        try:
            after = _read_state(observe)
            change = physical_change(before, after)
            commands = change["native_command_bytes"]
            if candidate.require_command and commands is None:
                raise RecoveryFailure(FailureKind.OBSERVATION_MISSING, {"reason": "missing_native_command_log"})
        except RecoveryFailure as exc:
            event("skill_attempt_failed", **context, attempt=attempts,
                  failure_kind=exc.kind.value, details=exc.details)
            raise

        native_refused = isinstance(result, (bool, np.bool_)) and not result
        if native_error is not None or native_refused:
            if isinstance(native_error, RecoveryFailure):
                reason = native_error.kind.value
            else:
                reason = native_error.kind if native_error else "native_action_false"
            details = getattr(native_error, "details", {}) or {}
            if not isinstance(details, Mapping):
                details = {"message": str(details)}
            kind = classify_failure(reason, details, world_changed=materially_changed(change, *tolerances),
                                    commands_written=commands)
            outcome = RecoveryFailure(kind, {**details, "source_failure": str(reason), **change})
        elif candidate.require_command and not commands:
            outcome = RecoveryFailure(FailureKind.PHYSICAL_FAILURE, {**change, "reason": "no_new_native_command"})
        else:
            try:
                passed = verify(before, after, result)
            except RecoveryFailure as exc:
                kind = classify_failure(exc.kind, exc.details, world_changed=materially_changed(change, *tolerances),
                                        commands_written=commands)
                outcome = RecoveryFailure(kind, {**exc.details, **change})
            else:
                verified = isinstance(passed, (bool, np.bool_)) and bool(passed)
                outcome = None if verified else RecoveryFailure(
                    FailureKind.PHYSICAL_FAILURE, {**change, "reason": "measured_effect_not_verified"})

        if time.monotonic() >= deadline:
            outcome = RecoveryFailure(FailureKind.BUDGET_EXHAUSTED, {
                **change, "reason": "execution_deadline",
                "last_failure": outcome.kind.value if outcome else None,
            })
        if outcome is None:
            event("skill_attempt_recovered" if failure else "skill_attempt_succeeded", **context,
                  attempt=attempts, prior_failure=failure.kind.value if failure else None,
                  observed=after.evidence(), change=change)
            return result

        failed_starts.append((before, outcome))
        event("skill_attempt_failed", **context, attempt=attempts, failure_kind=outcome.kind.value,
              details=outcome.details, observed=after.evidence())
        if outcome.kind in {FailureKind.OBSERVATION_MISSING, FailureKind.BUDGET_EXHAUSTED, FailureKind.UNKNOWN}:
            raise outcome
        failure = outcome

    raise RecoveryFailure(failure.kind if failure else FailureKind.PRECONDITION_FAILED,
                          {**(failure.details if failure else {}), "step_id": step_id,
                           "attempts": attempts, "recovery": "no_remaining_eligible_candidate"})
