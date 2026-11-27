"""Bind declarative goals to existing asset and physical relation contracts."""
from robot_adapters import UnsupportedRequest


PACKING_STAGES = (
    ("inside", "phone", "receiver"),
    ("inside", "insert", "receiver"),
    ("covers", "lid", "receiver"),
)


def packing_goal(raw):
    """Validate a nonempty prefix of physically supported packing relations."""
    if not isinstance(raw, dict) or set(raw) != {"effects", "preserve"}:
        raise ValueError("A goal requires effects and preserve")
    effects, preserve = raw["effects"], raw["preserve"]
    if not isinstance(effects, list) or not 1 <= len(effects) <= len(PACKING_STAGES):
        raise ValueError("Provide one to three ordered packing effects")
    for effect, stage in zip(effects, PACKING_STAGES):
        if (not isinstance(effect, dict) or set(effect) != {"relation", "object", "reference"}
                or tuple(effect[key] for key in ("relation", "object", "reference")) != stage):
            raise UnsupportedRequest("No verified asset relation for this goal",
                                     missing_skills=["verified_goal_relation"])
    unchanged = {role for _, role, _ in PACKING_STAGES[len(effects):]}
    if not isinstance(preserve, list) or any(not isinstance(role, str) or role not in unchanged for role in preserve):
        raise ValueError("Preserved objects must be outside the manipulated stages")
    return {"effects": effects, "preserve": sorted(set(preserve))}


def goal_text(goal):
    descriptions = ["place the phone inside the box base", "fit the insert into the box base",
                    "cover the box with the lid"]
    text = ". Then ".join(descriptions[:len(goal["effects"])])
    if goal["preserve"]:
        text += ". Leave " + " and ".join(goal["preserve"]) + " at their initial poses"
    return text[0].upper() + text[1:] + "."


def bind_packing_goal(spec, raw):
    goal = packing_goal(raw)
    required = len(goal["effects"])
    spec["goal"] = goal
    spec["official_instruction"] = goal_text(goal)
    spec["manipulated_objects"] = spec["manipulated_objects"][:required]
    spec["relations"] = spec["relations"][:required]
    spec["target_objects"] = {key: obj for key, obj in spec["target_objects"].items()
                              if obj in spec["manipulated_objects"]}
    spec["targets"] = list(spec["target_objects"])
    spec["grasp_profiles"] = {obj: spec["grasp_profiles"][obj] for obj in spec["manipulated_objects"]}
    spec["preserve_objects"] = [spec["bindings"][role] for role in goal["preserve"]]
    return spec


def packing_rule_indices(scene):
    goal = scene.get("project_packing_goal")
    return list(range(1, 2 * len(packing_goal(goal)["effects"]), 2)) if goal is not None else [1, 3, 5]
