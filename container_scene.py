"""Prepare an open-container scene from pinned assets and authored colliders."""

from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, maximum_filter, minimum_filter, uniform_filter
from scipy.spatial.transform import Rotation

from container_goal import TASK_ID, asset_gap, search_assets
from task_geometry import bbox_corners


PLACEMENT_CLEARANCE_M = .003
MAX_DROP_M = .085
MIN_PAYLOAD_SPAN_M = .012


def shape_data(path, *, receiver):
    """Read dimensions in the asset's Z-up frame without changing its colliders."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from scene_composer import world_bounds

    stage = Usd.Stage.Open(str(path))
    if not stage or not stage.GetDefaultPrim() or UsdGeom.GetStageUpAxis(stage) != "Z" or not np.isclose(
        UsdGeom.GetStageMetersPerUnit(stage), 1.0
    ):
        raise ValueError("Expected a canonical Z-up USD asset in metres")
    root = stage.GetDefaultPrim()
    entity = stage.GetPrimAtPath(root.GetPath().AppendChild("entity"))
    if not entity or not entity.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("No native rigid-body entity")
    transform = UsdGeom.XformCache()
    root_transform = np.asarray(transform.GetLocalToWorldTransform(root)).T
    frame = np.asarray(transform.GetLocalToWorldTransform(entity)).T
    basis = frame[:3, :3]
    if not np.allclose(root_transform, np.eye(4), atol=1e-6) or not np.allclose(
        basis.T @ basis, np.eye(3), atol=1e-6
    ):
        raise ValueError("Asset transform cannot be placed with its original scale")
    prims = list(Usd.PrimRange(entity))
    if (any(p.IsA(UsdPhysics.Joint) or p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims[1:])
            or any(p.GetAttribute("physxDeformable:deformableEnabled").Get() for p in prims)):
        raise ValueError("A single rigid entity is required")
    collisions = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)
                  and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
    if not collisions:
        raise ValueError("Asset has no enabled collision geometry")
    bounds = [world_bounds(stage, p.GetPath()) for p in collisions]
    lower, upper = np.asarray(world_bounds(stage, entity.GetPath()))
    lower = np.minimum(lower, np.min([pair[0] for pair in bounds], axis=0))
    upper = np.maximum(upper, np.max([pair[1] for pair in bounds], axis=0))
    local_points = (bbox_corners(lower, upper) - frame[:3, 3]) @ basis
    local = np.array([local_points.min(axis=0), local_points.max(axis=0)])
    if receiver:
        # Reconstruct only documented box colliders. A convex hull or visual
        # AABB alone cannot establish that the container has empty interior.
        for prim in collisions:
            if not prim.IsA(UsdGeom.Cube) and prim.GetAttribute("physics:approximation").Get() != "boundingCube":
                raise ValueError("Container lacks supported independent wall and floor box colliders")
            transform_basis = np.asarray(transform.GetLocalToWorldTransform(prim)).T[:3, :3]
            if (np.sum(np.abs(transform_basis) > 1e-5, axis=0) != 1).any() or (
                    np.sum(np.abs(transform_basis) > 1e-5, axis=1) != 1).any():
                raise ValueError("Container box collision axes are not aligned in its rest frame")
        if len(bounds) < 4 or len(bounds) > 96:
            raise ValueError("Container wall and floor collider count is unsupported")
    if not receiver and UsdPhysics.RigidBodyAPI(entity).GetKinematicEnabledAttr().Get():
        raise ValueError("The workpiece cannot move under physics")
    return {"basis": basis, "extent": upper - lower, "bottom_z": float(lower[2]),
            "bbox": local, "world_points": bbox_corners(lower, upper),
            "colliders": np.asarray(bounds) - frame[:3, 3]}


def workpiece_orientations(shape, aperture):
    """Try the authored rest pose and side faces. Runtime checks settling before any action."""
    rotations = [(0., 0., 0.)]
    if shape["extent"][2] > 1.2 * min(shape["extent"][:2]):
        rotations.extend([(np.pi / 2, 0., 0.), (-np.pi / 2, 0., 0.),
                          (0., np.pi / 2, 0.), (0., -np.pi / 2, 0.)])
    for angles in rotations:
        rotation = Rotation.from_rotvec(angles)
        corners = rotation.apply(shape["world_points"])
        extent = np.ptp(corners, axis=0)
        if MIN_PAYLOAD_SPAN_M <= min(extent[:2]) <= aperture - .006:
            yield {**shape, "extent": extent, "bottom_z": float(corners[:, 2].min()),
                   "rest_rotation": rotation, "basis": rotation.as_matrix() @ shape["basis"]}


def height_map(colliders):
    """Project active box chunks from above, conservatively occupying intersected cells."""
    lower = colliders[:, 0].min(axis=0)
    upper = colliders[:, 1].max(axis=0)
    step = max(.001, float(max(upper[:2] - lower[:2])) / 250)
    origin = lower[:2] - 4 * step
    shape = np.ceil((upper[:2] - origin) / step).astype(int) + 5
    surface = np.full(shape, -np.inf)
    for low, high in colliders:
        start = np.floor((low[:2] - origin) / step).astype(int)
        stop = np.ceil((high[:2] - origin) / step).astype(int)
        cells = surface[start[0]:stop[0], start[1]:stop[1]]
        np.maximum(cells, high[2], out=cells)
    return origin, step, surface


def expanded_bounds(free, cell, radii):
    """Find a rectangular clearance region around the selected release column."""
    start, stop = cell - radii, cell + radii + 1
    for axis in (0, 1):
        for direction in (-1, 1):
            while True:
                a, b = start.copy(), stop.copy()
                if direction < 0:
                    a[axis] -= 1
                else:
                    b[axis] += 1
                if (a < 0).any() or (b > free.shape).any() or not free[a[0]:b[0], a[1]:b[1]].all():
                    break
                start, stop = a, b
    return start, stop


def cavity(colliders, payload):
    """Find a vertical insertion column with enclosed walls and floor support."""
    payload = np.asarray(payload, dtype=float)
    origin, step, surface = height_map(colliders)
    midpoint = (np.asarray(surface.shape) - 1) / 2
    highest = float(colliders[:, 1, 2].max())
    best = None
    for rim in sorted(set(colliders[:, 1, 2]), reverse=True):
        walls = surface >= rim - .0005
        # Small seams between authored chunks never enlarge the free opening.
        walls = binary_closing(walls, structure=np.ones((3, 3))) | walls
        free = binary_fill_holes(walls) & ~walls
        if not free.any():
            continue
        for yaw, footprint in ((0., payload[:2]), (np.pi / 2, payload[:2][::-1])):
            radii = np.ceil((footprint / 2 + PLACEMENT_CLEARANCE_M) / step).astype(int)
            window = tuple(2 * radii + 1)
            floor = maximum_filter(surface, size=window, mode="constant", cval=-np.inf)
            coverage = uniform_filter(np.isfinite(surface).astype(float), size=window, mode="constant")
            feasible = (minimum_filter(free, size=window, mode="constant")
                        & (coverage >= .55) & (rim - floor >= payload[2] + .001)
                        & (highest - floor <= MAX_DROP_M - .012))
            cells = np.argwhere(feasible)
            if not len(cells):
                continue
            cell = cells[np.linalg.norm(cells - midpoint, axis=1).argmin()]
            start, stop = expanded_bounds(free, cell, radii)
            span = (stop - start) * step
            floor_z = float(floor[tuple(cell)])
            score = float(np.min(span - footprint)) - .08 * (highest - floor_z)
            if best is None or score > best[0]:
                best = score, {
                    "center_xy_m": (origin + (cell + .5) * step).tolist(),
                    "opening_xy_m": [list(origin + start * step), list(origin + stop * step)],
                    "rim_z_m": float(rim), "highest_collision_z_m": highest,
                    "floor_z_m": floor_z, "object_yaw_rad": yaw,
                    "grid_pitch_m": step, "collider_count": len(colliders),
                }
    if best is None:
        raise ValueError("No open collision cavity with sufficient width, depth and support for this object")
    geometry = best[1]
    # Inside the same cavity, ridges and recesses can support different parts of
    # the payload at different heights. Keep the authored surfaces by footprint.
    geometry["support_surfaces_m"] = [
        [lo[:2].tolist(), hi[:2].tolist(), float(hi[2])]
        for lo, hi in colliders if hi[2] < geometry["rim_z_m"] - .001
    ]
    return geometry


def choose_pair(goal, adapter, api, asset_root):
    matches, receipt = search_assets(goal, api, asset_root)
    prepared = {}
    rejected = []
    aperture = adapter.config["gripper"].get("max_grasp_width_m", .085)
    for role, rows in matches.items():
        prepared[role] = []
        for row in rows:
            try:
                shape = shape_data(row["usd"], receiver=role == "container")
                shapes = list(workpiece_orientations(shape, aperture)) if role == "object" else [shape]
                if not shapes:
                    raise ValueError("Workpiece is outside the gripper span used by the grasp skill")
                prepared[role].extend({**row, **variant} for variant in shapes)
            except (OSError, RuntimeError, ValueError) as exc:
                rejected.append({"role": role, "asset_id": row["asset_id"], "reason": str(exc)})
    for workpiece in prepared["object"]:
        for receiver in prepared["container"]:
            if workpiece["asset_id"] == receiver["asset_id"]:
                continue
            try:
                geometry = cavity(receiver["colliders"], workpiece["extent"])
            except ValueError as exc:
                rejected.append({"object": workpiece["asset_id"], "container": receiver["asset_id"],
                                 "reason": str(exc)})
                continue
            receipt.update(selected={role: row["asset_id"] for role, row in
                                     (("object", workpiece), ("container", receiver))}, rejected=rejected)
            return {"object": workpiece, "container": receiver, "geometry": geometry,
                    "instruction": f"Place {goal['object']['names'][0]} inside {goal['container']['names'][0]}."}, receipt
    missing = [goal[role]["source"] for role in matches if not prepared[role]]
    error = asset_gap("找到了相关资产，但缺少可用于抓放的碰撞几何或尺寸匹配",
                      missing or [goal["object"]["source"], goal["container"]["source"]],
                      code="ASSET_GEOMETRY_INCOMPATIBLE", candidates=rejected)
    raise error


def author_objects(stage, metadata, task_info, selection, support, adapter):
    """Reuse the calibrated tabletop, preserving USD assets at their actual scale."""
    from pxr import UsdGeom
    from scene_composer import set_pose, world_bounds

    layout = metadata["layout"]
    # The two calibrated starting regions are reachable from the operator arm.
    object_anchor = layout["benchmark_phone_000"]
    container_anchor = layout["benchmark_packaging_box_001"]
    receiver_rotation = Rotation.from_quat(container_anchor["xyzw"])
    object_rotation = (receiver_rotation * Rotation.from_rotvec([0, 0, selection["geometry"]["object_yaw_rad"]])
                       * selection["object"]["rest_rotation"])
    object_path = stage.GetPrimAtPath("/World/Objects")
    for prim in list(object_path.GetChildren()):
        if not stage.RemovePrim(prim.GetPath()):
            raise ValueError("Cannot replace a source workpiece in the generated scene")
    composed = {}
    footprints = []
    for role, anchor, rotation in (("object", object_anchor, object_rotation),
                                    ("container", container_anchor, receiver_rotation)):
        asset = selection[role]
        name = asset["asset_id"]
        prim = UsdGeom.Xform.Define(stage, f"/World/Objects/{name}").GetPrim()
        prim.GetPayloads().AddPayload(str(Path(asset["usd"])))
        xyz = [*anchor["xyz"][:2], support["height_m"] - asset["bottom_z"] + .0008]
        quaternion = rotation.as_quat().tolist()
        set_pose(prim, xyz, quaternion)
        if not stage.GetPrimAtPath(prim.GetPath().AppendChild("entity")):
            raise ValueError(f"Asset payload failed to compose: {name}")
        lower, upper = world_bounds(stage, prim.GetPath().AppendChild("entity"))
        plane = np.asarray(support["bounds_xy_m"])
        if np.any(lower[:2] < plane[0] + .005) or np.any(upper[:2] > plane[1] - .005):
            raise ValueError(f"{name} does not fit the usable tabletop")
        footprints.append((lower, upper))
        composed[name] = {"usd": name, "xyz": xyz, "xyzw": quaternion, "registered": True,
                          "entity_local_bbox": {k: v.tolist() for k, v in zip(("min", "max"), asset["bbox"])}}
    a, b = footprints
    if not np.any(np.minimum(a[1][:2], b[1][:2]) - np.maximum(a[0][:2], b[0][:2]) < -.012):
        raise ValueError("Initial placement leaves insufficient space between the assets")
    item, box = (selection[role]["asset_id"] for role in ("object", "container"))
    metadata.update(task_template=TASK_ID, layout=composed, official_subtask_steps=[],
                    official_instruction={"instruction": selection["instruction"], "target": {"id1": item, "id2": box}},
                    project_container_placement={**selection["geometry"],
                        "receiver_basis": selection["container"]["basis"].tolist(),
                        "object_basis": selection["object"]["basis"].tolist(),
                        "operator": "right" if "right" in adapter.arms else adapter.arms[0]})
    task_info.update(task_name=adapter.config["scene_task_prefix"] + TASK_ID, task_metric={},
                     task_description={"task_name": selection["instruction"],
                                       "english_task_name": selection["instruction"], "init_scene_text": ""})
