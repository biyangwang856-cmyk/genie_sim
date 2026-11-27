"""Discover usable task scenes and rank background/support assets by release metadata."""
from collections import defaultdict
from pathlib import Path
import re

import numpy as np

from asset_retrieval import load_index
from robot_adapters import ROOT, UnsupportedRequest


ASSETS = ROOT / "GenieSimAssets"
BACKGROUND_FILES = {"background.usda", "scene_full.usda"}
DESK_CATEGORIES = {"table", "desk", "workbench"}
SCENE_QUERY_TERMS = {"棚拍": "studio", "居家": "home", "客厅": "living room",
                     "办公室": "office", "教室": "classroom", "工作桌": "workbench",
                     "实验室": "laboratory"}
BACKGROUND_TYPES = (
    frozenset({"studio"}), frozenset({"home","living","apartment"}),
    frozenset({"office","workplace"}), frozenset({"classroom"}),
    frozenset({"laboratory","lab"}), frozenset({"kitchen"}),
)


def records(asset_root=ASSETS):
    root = Path(asset_root).resolve()
    for path in sorted((root / "background").rglob("*.usda")):
        if path.name in BACKGROUND_FILES:
            parts = path.relative_to(root).parts[1:-1]
            name = "__".join((*parts, path.stem))
            yield {"kind": "background", "id": name, "path": path,
                   "description": "environment " + " ".join(parts).replace("_", " ")}
    for path in sorted((root / "objects").rglob("Aligned.usda")):
        name = path.parent.name
        groups = {part.lower() for part in path.relative_to(root).parts[1:-2]}
        if groups & DESK_CATEGORIES or re.fullmatch(r"(?:table|desk)(?:_number)?_\d+", name):
            yield {"kind": "support", "id": name, "path": path,
                   "description": "desk table horizontal support " + name.replace("_", " ")}


def rank(records, kind, api, *, query=None, explicit=None):
    candidates = [row for row in records if row["kind"] == kind]
    if explicit:
        chosen = [row for row in candidates if row["id"] == explicit]
        if len(chosen) != 1:
            raise UnsupportedRequest(f"Missing or ambiguous {kind} asset", missing_assets=[explicit])
        return chosen, None
    by_id = defaultdict(list)
    for row in candidates:
        by_id[row["id"]].append(row)
    candidates = [group[0] for group in by_id.values() if len(group) == 1]
    if not candidates:
        raise UnsupportedRequest(f"No local {kind} assets", missing_assets=[kind])
    if query is None:
        # Stable defaults drawn from the same byte-pinned library.
        return candidates, None
    hints = " ".join(term for word,term in SCENE_QUERY_TERMS.items() if word in query)
    words = set(re.findall(r"[a-z0-9]+", (query + " " + hints).lower()))
    if kind == "background":
        required = set().union(*(group for group in BACKGROUND_TYPES if words & group))
        if required:
            candidates = [row for row in candidates
                          if required & set(re.findall(r"[a-z0-9]+",row["description"].lower()))]
            if not candidates:
                raise UnsupportedRequest("No indexed background has the requested category",
                                         missing_assets=[query])
    def overlap(row):
        return len(words & set(re.findall(r"[a-z0-9]+", row["description"].lower())))
    candidates.sort(key=lambda row: (-overlap(row), row["id"]))
    indexed = {}
    if kind == "support":
        object_rows, vectors = load_index()
        indexed = {row["asset_id"]: vector for row, vector in zip(object_rows, vectors, strict=True)}
    scores, receipts = {}, []
    # Rank a bounded semantic shortlist using the already-built object index when possible.
    shortlist = candidates[:10]
    missing = [row for row in shortlist if row["id"] not in indexed]
    vectors, receipt = api.embed([query, *(row["description"] for row in missing[:9])])
    receipts.append(receipt)
    for row, vector in zip(missing, vectors[1:]):
        indexed[row["id"]] = vector
    if len(missing) == 10:
        extra, receipt = api.embed([missing[-1]["description"]])
        receipts.append(receipt)
        indexed[missing[-1]["id"]] = extra[0]
    query_vector = np.asarray(vectors[0], dtype=float)
    if query_vector.shape != (1024,) or not np.isfinite(query_vector).all() or np.linalg.norm(query_vector) == 0:
        raise ValueError("Embedding query returned an invalid vector")
    query_vector /= np.linalg.norm(query_vector)
    for row in shortlist:
        vector = np.asarray(indexed[row["id"]], dtype=float)
        if vector.shape != query_vector.shape or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("Asset embedding returned an invalid vector")
        scores[row["id"]] = float(query_vector @ vector / np.linalg.norm(vector))
    ranked = sorted(shortlist, key=lambda row: -(scores[row["id"]] + .08*overlap(row)))
    return ranked, {"model": "text-embedding-v4", "query": query, "receipts": receipts,
                    "scores": scores, "lexical_hints":hints, "ordered_ids":[row["id"] for row in ranked]}
