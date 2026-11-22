"""Trusted project adapters for Genie Sim operations. Importing this catalog needs no SDK."""
from dataclasses import dataclass
from typing import Callable

from collection_scope import DEFAULT_ROBOT, SCENES, TASKS
from robot_adapters import UnsupportedRequest


# An adapter may report the ID of its durable native job before a long run.
Progress = Callable[[str], None]
Handler = Callable[[dict, Progress], dict]


@dataclass(frozen=True)
class Operation:
    id: str
    upstream: str
    inputs: tuple
    outputs: tuple
    collection: bool = False


def simulator(name, upstream):
    return Operation(name, upstream, ("instruction",), ("status", "job_id"), collection=True)


ROS = "source/geniesim_ros/src/ros_ws/src/"
OPERATIONS = (
    simulator("isaac_physx", "source/data_collection"),
    simulator("mujoco", "source/geniesim_benchmark"),
    simulator("newton", ROS + "genie_sim_engine_newton"),
    simulator("libuipc", ROS + "genie_sim_engine_libuipc"),
    Operation("scene_authoring", "source/geniesim_scene_agent", ("instruction", "asset_catalog_ref"), ("scene_ref",)),
    Operation("scene_admission", "source/geniesim_benchmark", ("scene_ref", "task_id", "robot_id"), ("validation_ref",)),
    Operation("scene_reconstruction", "source/scene_reconstruction", ("observations_ref",), ("scene_ref",)),
    Operation("world_generation", "source/geniesim_world", ("panorama_ref",), ("world_ref",)),
    Operation("teleoperation", "source/geniesim_teleop", ("scene_ref", "robot_id"), ("recording_ref",)),
    Operation("benchmark", "source/geniesim_benchmark", ("task_ref", "policy_ref"), ("evaluation_ref",)),
    Operation("ros_control", "source/geniesim_ros", ("scene_ref", "robot_id"), ("session_ref",)),
    Operation("moveit_planning", ROS + "genie_sim_moveit", ("state_ref", "goal_ref", "robot_id"), ("plan_ref",)),
    Operation("agent2act", "source/agent2act", ("instruction", "observation_ref"), ("action_plan_ref",)),
    Operation("dream", "source/geniesim_dream", ("episode_ref", "model_ref"), ("prediction_ref",)),
    Operation("rl_training", "source/rlinf_geniesim", ("task_ref", "policy_ref"), ("checkpoint_ref",)),
)


def isaac_collection(payload, on_native_job):
    # Worker-only import: catalog, web and status run without the GPU/Python stack.
    from natural_backend import submit
    return submit(payload["instruction"], robot_id=payload["robot_id"],
                  task_ids=payload.get("task_ids"), scenes=payload["scenes"],
                  execute=payload["execute"], on_prepared=on_native_job)


class Registry:
    def __init__(self):
        self.specs = {spec.id: spec for spec in OPERATIONS}
        self.handlers = {"isaac_physx": isaac_collection}
        self.scopes = {"isaac_physx": {"robot_ids": [DEFAULT_ROBOT], "task_ids": list(TASKS), "scenes": list(SCENES)}}

    def register(self, name, handler: Handler, *, scope=None):
        """Called by trusted deployment code at startup, in both web and worker."""
        if name not in self.specs or name in self.handlers:
            raise ValueError(f"Unknown or already registered operation: {name}")
        if not callable(handler):
            raise TypeError("The operation requires a callable handler")
        if self.specs[name].collection:
            if not isinstance(scope, dict) or set(scope) != {"robot_ids", "task_ids", "scenes"}:
                raise ValueError("Collection adapters must declare robot_ids, task_ids and scenes")
            if not all(isinstance(v, list) and v and all(isinstance(s, str) and s for s in v) for v in scope.values()):
                raise ValueError("Collection scopes require nonempty lists of supported IDs")
            self.scopes[name] = {key: list(value) for key, value in scope.items()}
        self.handlers[name] = handler

    def describe(self):
        return [{"id": spec.id, "type": "collection" if spec.collection else "extension",
                 "status": "connected" if spec.id in self.handlers else "reserved",
                 "upstream": spec.upstream, "required_inputs": list(spec.inputs),
                 "required_outputs": list(spec.outputs), **self.scopes.get(spec.id, {})}
                for spec in self.specs.values()]

    def validate(self, name, payload):
        spec = self.specs.get(name)
        if spec is None or name not in self.handlers:
            reason = "Unregistered operation" if spec is None else "Adapter reserved and not connected"
            error = UnsupportedRequest(f"{reason}: {name}", missing_capabilities=[name])
            error.details["operation"] = name
            raise error
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        required = set(spec.inputs)
        allowed = required | ({"robot_id", "task_ids", "scenes", "execute"} if spec.collection else set())
        if required - payload.keys() or payload.keys() - allowed:
            raise ValueError(f"Missing fields {sorted(required - payload.keys())}; unknown fields {sorted(payload.keys() - allowed)}")
        for field in spec.inputs:
            value = payload[field]
            if not isinstance(value, str) or not value.strip() or len(value) > 8192:
                raise ValueError(f"Expected a nonempty string of at most 8192 characters for {field}")
        if not spec.collection:
            return dict(payload)
        scope = self.scopes[name]
        cfg = {"robot_id": scope["robot_ids"][0], "scenes": list(scope["scenes"]), "execute": False, **payload}
        if type(cfg["execute"]) is not bool or not isinstance(cfg["robot_id"], str):
            raise ValueError("execute must be boolean and robot_id must be a string")
        if cfg["robot_id"] not in scope["robot_ids"]:
            raise UnsupportedRequest("Robot unavailable for this simulator", missing_capabilities=[cfg["robot_id"]])
        for key in ("scenes", "task_ids"):
            if key not in cfg:
                continue
            items = cfg[key]
            if not isinstance(items, list) or not items or not all(isinstance(v, str) and v for v in items):
                raise ValueError(f"{key} must be a nonempty list of string IDs")
            if len(set(items)) != len(items):
                raise ValueError(f"Duplicate IDs in {key}")
            missing = sorted(set(items) - set(scope[key]))
            if missing:
                if key == "scenes":
                    raise UnsupportedRequest("Unregistered scene", missing_assets=missing)
                raise UnsupportedRequest("Unregistered task", missing_skills=missing)
        return cfg

    def invoke(self, name, payload, on_native_job):
        payload = self.validate(name, payload)
        result = self.handlers[name](payload, on_native_job)
        if not isinstance(result, dict) or any(k not in result for k in self.specs[name].outputs):
            raise ValueError(f"Adapter violated its result contract: {name}")
        return result
