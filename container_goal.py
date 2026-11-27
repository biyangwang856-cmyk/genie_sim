"""Parse and ground an unregistered tabletop containment request."""

import json
from pathlib import Path
import re

from robot_adapters import UnsupportedRequest


TASK_ID = "place_in_container"
REQUIRED_SKILLS = ("grasp", "move_held", "release", "retreat", "wait_stable", "verify_relation", "verify_task")
ROLE_NAMES = ("object", "container")
IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{0,79}")


def phrases(values, limit=6):
    if not isinstance(values, list) or not 1 <= len(values) <= limit:
        raise ValueError("Expected a short list of identity phrases")
    if any(not isinstance(value, str) or not 1 <= len(value.strip()) <= 120 for value in values):
        raise ValueError("Invalid identity phrase")
    return list(dict.fromkeys(value.strip() for value in values))


def validate_goal(goal, instruction):
    if not isinstance(goal, dict) or set(goal) != set(ROLE_NAMES):
        raise ValueError("A containment goal requires one object and one container")
    normalized = {}
    for role in ROLE_NAMES:
        row = goal[role]
        if not isinstance(row, dict) or not {"source", "names"} <= row.keys() <= {
            "source", "names", "attributes", "asset_id"
        }:
            raise ValueError("Unrecognized object reference in containment goal")
        source = row["source"]
        if not isinstance(source, str) or not 1 <= len(source.strip()) <= 160 or source not in instruction:
            raise ValueError("The object reference must quote the original request")
        asset_id = row.get("asset_id")
        if asset_id is not None and (
            not isinstance(asset_id, str) or not IDENTIFIER.fullmatch(asset_id) or asset_id not in source
        ):
            raise ValueError("An explicitly selected asset ID must be in the original request")
        names = [name for name in phrases(row["names"]) if re.search("[a-zA-Z]", name)]
        if not names:
            if asset_id is None:
                raise ValueError("Expected an English identity term for the asset index")
            names = [asset_id]
        attributes = row.get("attributes", [])
        normalized[role] = {"source": source.strip(), "names": names,
                            "attributes": phrases(attributes) if attributes else [],
                            **({"asset_id": asset_id} if asset_id else {})}
    return normalized


def resolve_goal(api, instruction, adapter, scene_options, quota):
    """Give the model only the task kind that has a geometric compiler."""
    from scene_contract import SCENE_REQUEST_FIELDS

    prompt = (
        "The exact registered task catalog did not handle this request. Determine whether its entire "
        "manipulation is one rigid item picked up and released fully inside an upward-open, stationary "
        "container on a tabletop. Return JSON with status=container_goal and goal={object: {source, "
        "names, attributes?, asset_id?}, container: {source, names, attributes?, asset_id?}} if supported. "
        "For each role copy its exact original noun phrase into source, including any explicit asset ID. "
        "names is an array of 1 to 6 English translations or exact synonyms for the noun identity "
        "without descriptive modifiers. For '白色中空包装盒' use names=['packaging box'] and "
        "attributes=['white']; an open cavity is checked by collision geometry. "
        "Never use only a material, shape, parent category, or a substitute. For lipstick, lipstick, "
        "lip stick or lipstick tube are suitable names; cosmetic bottle, makeup, or pen are not. For makeup "
        "box, use makeup box, cosmetic storage box, makeup organizer or vanity case; do not broaden to box, "
        "jewelry box or a cosmetic jar. attributes is an optional array of required English qualifiers "
        "such as a specified color. Use short noun phrases without words such as 'with' or 'a'. "
        "Add asset_id only if explicitly requested in that role's source phrase. "
        "Opening or closing a lid, several items, tools, suction, deformation, orientation requirements, "
        "insertion into a keyed hole, or an additional operation need other skills. For those return "
        "status=unsupported with reason and missing_skills containing the exact missing canonical skill "
        "names where known (e.g. vacuum_pick, open_container, multi_object_placement). Do not omit a "
        "requested operation. Reject ambiguous source or destination rather than inventing one. "
        "Honor the requested robot and scene details. For the default tabletop omit scene_request. "
        "For explicit layout or room changes use optional scene_request with mode=compose_from_library "
        "and scene_count, support_query, background_query, variation or explicit IDs as applicable. "
        "For named existing scene IDs use mode=existing and scene_ids. Allowed scene_request fields are "
        f"{sorted(SCENE_REQUEST_FIELDS)}. Honor supplied scene constraints. "
        "Map a total number of episodes to trajectory_request.total_episodes; map an explicit number "
        "per scene to successes_per_scene. Keep scene count separate. Default to distinct trajectories. "
        "Repeat modes require the user's exact clause in trajectory_request.duplicate_intent. "
        "Do not provide numeric coordinates, assets absent from the instruction, or motion commands."
    )
    result, receipt = api.request("chat/completions", {
        "model": "qwen-plus",
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps({
            "instruction": instruction, "robot": adapter.robot_id, "scene_constraints": scene_options,
            "episode_quota": quota,
        }, ensure_ascii=False)}],
        "response_format": {"type": "json_object"}, "temperature": 0.0, "enable_thinking": False,
    })
    raw = result["choices"][0]["message"]["content"]
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("status") not in ("container_goal", "unsupported"):
        raise ValueError("Unrecognized response from the containment intent planner")
    if value["status"] == "container_goal":
        allowed = {"status", "goal", "scene_request", "trajectory_request", "successes_per_scene"}
        if value.keys() - allowed:
            raise ValueError("Unexpected containment request fields")
        value["goal"] = validate_goal(value["goal"], instruction)
    directory = api.audit_directory / "container-intents"
    directory.mkdir(exist_ok=True)
    (directory / (receipt["id"] + ".json")).write_text(json.dumps({
        "instruction": instruction, "receipt": receipt, "response": raw, "parsed": value,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return value, receipt


def skill_requirements(adapter):
    from skill_registry import SKILLS

    missing = [skill for skill in REQUIRED_SKILLS if skill not in SKILLS]
    if missing:
        error = UnsupportedRequest("The grasp-and-place sequence needs unregistered skills", missing_skills=missing)
        error.details["code"] = "SKILL_NOT_REGISTERED"
        raise error
    adapter.require({capability for skill in REQUIRED_SKILLS for capability in SKILLS[skill]["requires"]})


def asset_gap(reason, missing, *, code="ASSET_NOT_FOUND", candidates=None):
    error = UnsupportedRequest(reason, missing_assets=missing)
    error.details["code"] = code
    if candidates:
        error.details["asset_candidates"] = candidates
    return error


def normalized(text):
    # Keep exact noun phrases while accepting common inflections in the metadata.
    words = re.findall("[a-z0-9]+", text.casefold())
    return " " + " ".join(word[:-1] if len(word) > 3 and word.endswith("s") else word for word in words) + " "


def normalized_attributes(text):
    words = " ".join(word for word in normalized(text).split()
                     if word not in {"a", "an", "the", "and", "with"})
    # These are equivalent packaging descriptions; "sleeve" on its own can
    # also describe clothing and should not be substituted.
    words = re.sub(r"\bpaper (?:sleeve|wrap)\b", "paper wrapper", words)
    return " " + words + " "


def catalog_identity(row, role, identifier, description):
    metadata = identifier + " " + json.dumps(description, ensure_ascii=False)
    text = normalized(metadata)
    qualifiers = set(normalized(" ".join(row["attributes"])).split())
    if role == "container":
        # The collision compiler, rather than the name, establishes an open cavity.
        qualifiers.update({"hollow", "empty", "open", "opened", "top"})
    identity = [" ".join(word for word in normalized(name).split() if word not in qualifiers)
                for name in row["names"]]
    attributes = normalized_attributes(metadata)
    return (any(term and normalized(term) in text for term in identity)
            and all(normalized_attributes(term) in attributes for term in row["attributes"]))


def matches_phone_packaging(plan, goal):
    """Keep the proven single-component phone recipe only for its original asset roles."""
    tasks = plan.get("tasks", [])
    if len(tasks) != 1 or tasks[0].get("task_id") != "phone_packaging":
        return False
    if len(tasks[0].get("goal", {}).get("effects", [])) != 1:
        return False
    bindings = {
        "object": ("benchmark_phone_000", ("phone", "cell phone", "mobile phone", "smartphone")),
        "container": ("benchmark_packaging_box_001", ("box", "packaging box", "phone box",
                      "phone packaging box", "phone box base", "packaging box base", "phone packaging box base")),
    }
    for role, (asset_id, labels) in bindings.items():
        row = goal[role]
        if row.get("asset_id") not in (None, asset_id) or row["attributes"]:
            return False
        if not set(map(normalized, row["names"])) & set(map(normalized, labels)):
            return False
    return True


def search_assets(goal, api, asset_root, *, limit=8):
    """Rank semantic candidates, then require the requested identity in release metadata."""
    import numpy as np
    from asset_retrieval import load_index

    records, matrix = load_index()
    rows = [goal[role] for role in ROLE_NAMES]
    queries = [" ".join([*row["names"], *row["attributes"]]) for row in rows]
    vectors, receipt = api.embed(queries)
    vectors = np.asarray(vectors, dtype=float)
    if vectors.shape != (len(rows), matrix.shape[1]) or not np.isfinite(vectors).all():
        raise ValueError("Invalid object-retrieval embeddings")
    lengths = np.linalg.norm(vectors, axis=1)
    if (lengths == 0).any():
        raise ValueError("Empty object-retrieval embedding")
    scores = matrix @ (vectors / lengths[:, None]).T
    root = Path(asset_root).resolve()
    matches, audit, missing = {}, {}, []
    for column, role in enumerate(ROLE_NAMES):
        row = goal[role]
        ordered = np.argsort(-scores[:, column], kind="stable")
        hits = []
        for i in ordered:
            record = records[i]
            identifier = record["asset_id"]
            if row.get("asset_id"):
                if identifier != row["asset_id"]:
                    continue
            else:
                metadata = json.loads(record["text"])
                if not catalog_identity(row, role, identifier, metadata["description"]):
                    continue
            path = Path(record["usd"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                continue
            hits.append({"asset_id": identifier, "usd": str(path), "score": float(scores[i, column])})
            if len(hits) == limit:
                break
        matches[role] = hits
        audit[role] = {"query": queries[column], "matches": hits,
                       "semantic_candidates": [records[i]["asset_id"] for i in ordered[:5]]}
        if not hits:
            missing.append(row["source"])
    if missing:
        raise asset_gap("现有资产索引未找到与需求匹配的对象：" + "、".join(missing), missing,
                        candidates={role: row["semantic_candidates"] for role, row in audit.items()})
    return matches, {"model": "text-embedding-v4", "receipt": receipt, "queries": audit}
