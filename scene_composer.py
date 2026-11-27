"""Build runtime task packages from pinned USD assets and verified task geometry."""
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from robot_adapters import ROOT, UnsupportedRequest
from scene_catalog import ASSETS, rank, records
from scene_contract import SCENE_FILES, content_id, digest, read_json, write_manifest
from task_geometry import task_spec

KNOWN_RENDERER_ASSETS = {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl", "OmniPBR_ClearCoat.mdl", "OmniPBR_Opacity.mdl"}
SOURCE_PHONE_SUPPORT = "/World/background/benchmark_table_019"
SOURCE_PHONE_PLANE = "/World/background/Plane"
MAX_SUPPORT_DELTA_M = .046


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def world_bounds(stage, path, *, local=False):
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError(f"Missing scene prim {path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    bounds = (cache.ComputeUntransformedBound(prim) if local else cache.ComputeWorldBound(prim)).ComputeAlignedRange()
    lower, upper = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
    if bounds.IsEmpty() or not np.isfinite([lower, upper]).all() or (upper < lower).any():
        raise ValueError(f"Empty or invalid bounds on {path}")
    return lower, upper


def set_pose(prim, xyz, xyzw):
    from pxr import Gf, UsdGeom
    xform = UsdGeom.Xformable(prim)
    scale = prim.GetAttribute("xformOp:scale")
    if scale and not np.allclose(scale.Get(),1.):
        raise ValueError(f"Expected unit instance scale: {prim.GetPath()}")
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble, opSuffix="project_layout").Set(Gf.Vec3d(*map(float, xyz)))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble, opSuffix="project_layout").Set(
        Gf.Quatd(float(xyzw[3]), *map(float, xyzw[:3])))


def source_support(stage, metadata):
    # Original phone scene supplies its operative plane in the background.
    from pxr import UsdPhysics
    if metadata["task_template"] == "phone_packaging" and stage.GetPrimAtPath(SOURCE_PHONE_PLANE):
        path = SOURCE_PHONE_SUPPORT
        lower, upper = world_bounds(stage, path)
        height = float(world_bounds(stage, SOURCE_PHONE_PLANE)[1][2])
        return {"prim": path, "instance": None, "asset_id": "benchmark_table_019",
                "bounds_xy_m": [lower[:2].tolist(), upper[:2].tolist()], "height_m": height}
    candidates = [name for name, item in metadata["layout"].items()
                  if "table" in item["usd"].lower() or "desk" in item["usd"].lower()]
    if len(candidates) != 1:
        raise ValueError("Source task requires exactly one recognizable support")
    name = candidates[0]
    path = "/World/Objects/" + name
    lower, upper = world_bounds(stage, path + "/entity")
    if not any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in stage.Traverse()
               if str(prim.GetPath()).startswith(path + "/")):
        raise ValueError("Source support lacks collision geometry")
    return {"prim": path, "instance": name, "asset_id": metadata["layout"][name]["usd"],
            "bounds_xy_m": [lower[:2].tolist(), upper[:2].tolist()], "height_m": float(upper[2])}


def asset_dependencies(stage, metadata):
    from pxr import Sdf, UsdUtils
    filename = stage.GetRootLayer().realPath
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(filename))
    missing = [item for item in unresolved if str(item) not in KNOWN_RENDERER_ASSETS]
    if missing:
        raise ValueError(f"Unresolved USD assets: {missing[:6]}")
    dependencies = set()
    root = ASSETS.resolve()
    for path in [*(layer.realPath for layer in layers), *assets]:
        absolute = Path(path).resolve()
        if absolute == Path(filename).resolve():
            continue
        if not absolute.is_file() or not absolute.is_relative_to(root):
            raise ValueError(f"USD dependency outside pinned asset library: {absolute.name}")
        dependencies.add(absolute)
    for layer in stage.GetUsedLayers():
        if layer.realPath and Path(layer.realPath).resolve().is_relative_to(root):
            dependencies.add(Path(layer.realPath).resolve())
    for row in metadata["layout"].values():
        asset_id = row["usd"]
        if not isinstance(asset_id,str) or "/" in asset_id or "\\" in asset_id:
            raise ValueError("Invalid catalog asset identifier")
        folder=root/"interaction"/asset_id
        if folder.is_dir():
            dependencies.update(p.resolve() for p in folder.rglob("*") if p.is_file())
    return sorted(dependencies)


def rebase(layer, source_layer):
    """Canonicalize library references before moving a scene into a run snapshot."""
    from pxr import Sdf, UsdUtils

    def rewrite(path):
        if "/geniesim_assets/" in path.replace("\\", "/"):
            path = str(ASSETS / path.replace("\\", "/").split("/geniesim_assets/", 1)[1])
        elif path.startswith("."):
            path = Sdf.ComputeAssetPathRelativeToLayer(source_layer, path)
        candidate = Path(path)
        return str(candidate.resolve()) if candidate.exists() and candidate.resolve().is_relative_to(ASSETS.resolve()) else path

    UsdUtils.ModifyAssetPaths(layer, rewrite)


def source_directory(task, scene, adapter):
    source_task = adapter.config["scene_task_prefix"] + task
    local = ROOT / "tasks" / source_task / scene
    if local.is_dir():
        return local
    return ROOT / "official-task-scenes" / task / scene


def existing_ids(task, adapter):
    roots = [source_directory(task, "0", adapter).parent, ROOT / "official-task-scenes" / task]
    identifiers = {p.name for root in roots if root.is_dir() for p in root.iterdir()
                   if p.is_dir() and p.name.isdecimal() and (p / "scene.usda").is_file()}
    return sorted(identifiers, key=int)


def import_source(task, source_id, adapter, destination):
    from pxr import Sdf, Usd, UsdGeom
    directory = source_directory(task, source_id, adapter)
    if not (directory / "scene.usda").is_file():
        raise UnsupportedRequest("No matching complete task scene", missing_assets=[f"{task}/{source_id}"])
    template = ROOT / "tasks" / (adapter.config["scene_task_prefix"] + task) / "0"
    source_scene = directory / "scene_info.json"
    metadata = read_json(source_scene)
    layer = Sdf.Layer.FindOrOpen(str(directory / "scene.usda"))
    if layer is None:
        raise ValueError("Source scene USD could not be opened")
    output = Sdf.Layer.CreateNew(str(destination / "scene.usda"))
    output.TransferContent(layer)
    # Official unadapted instances use an asset-root placeholder.
    rebase(output, layer)
    stage = Usd.Stage.Open(output)
    task_info = read_json(directory / "task_info.json" if (directory / "task_info.json").is_file()
                          else template / "task_info.json")
    template_meta = read_json(template / "scene_info.json")
    if "official_instruction" not in metadata:
        instructions = read_json(directory / "instructions.json")
        metadata.update(official_instruction=instructions["instructions"][0],
                        official_subtask_steps=instructions.get("subtask_steps", []),
                        task_template=task, scene_id=source_id)
        if task == "phone_packaging" and not stage.GetPrimAtPath(SOURCE_PHONE_SUPPORT):
            scene_root = stage.GetDefaultPrim()
            scene_root.GetReferences().AddReference(str((ASSETS / "background/home/phone_packaging/background.usda").resolve()))
            # Preserve the source task's surface collision as stored in the task package.
        objects = stage.GetPrimAtPath("/World/Objects")
        if not objects:
            raise ValueError("No task object group in existing scene")
        # Derive geometry for distinct official instance labels from their payloads.
        layout = {}
        cache = UsdGeom.XformCache()
        available = {p.parent.name for p in ASSETS.rglob("Aligned.usda")}
        for obj in objects.GetChildren():
            name = obj.GetName()
            row = metadata.get("layout", {}).get(name, {})
            entry = row.get("usd")
            if not entry:
                payloads = obj.GetMetadata("payload")
                if payloads and payloads.GetAddedOrExplicitItems():
                    entry = Path(payloads.GetAddedOrExplicitItems()[0].assetPath).parent.name
            transform = cache.GetLocalToWorldTransform(obj)
            rotation = transform.ExtractRotationQuat()
            lower, upper = world_bounds(stage, str(obj.GetPath()) + "/entity", local=True)
            layout[name] = {**row, "usd": entry, "xyz": list(transform.ExtractTranslation()),
                            "xyzw": [*rotation.GetImaginary(), rotation.GetReal()],
                            "registered": entry in available,
                            "entity_local_bbox": {"min": lower.tolist(), "max": upper.tolist()}}
        metadata["layout"] = layout
    if metadata.get("task_template") != task:
        raise ValueError("Existing scene belongs to another task")
    # Asset pair verification is meaningful even when the instance has a different label.
    source = compatible_template(metadata, adapter, template_meta)
    return stage, metadata, task_info, source


def compatible_template(metadata, adapter, fallback):
    task = metadata["task_template"]
    from task_plugins import registered, load
    if task in registered():
        load(task).bind(metadata,adapter.robot_id)
        return metadata["scene_id"]
    reference = ROOT / "tasks" / (adapter.config["scene_task_prefix"] + task)
    if metadata["scene_id"] in ("0", "1"):
        return metadata["scene_id"]
    # Refinement of labels / geometry is safe only for exactly the verified pairs.
    wanted = metadata["official_instruction"].get("target", {})
    entries = metadata["layout"]
    source_ids = ("0", "1")
    if task == "insert_building_block" and metadata["official_instruction"]["gripper"]["1"]["side"] == "right":
        source_ids = ("1", "0")
    for source_id in source_ids:
        template = read_json(reference / source_id / "scene_info.json") if source_id == "1" else fallback
        target = template["official_instruction"].get("target", {})
        if task == "phone_packaging":
            ids = {name: row["usd"] for name, row in template["layout"].items()}
            if all(entries.get(name, {}).get("usd") == asset for name, asset in ids.items()):
                return source_id
        elif all(wanted.get(key) in entries and target.get(key) in template["layout"]
                 and entries[wanted[key]]["usd"] == template["layout"][target[key]]["usd"]
                 for key in (("id1", "id2") if task == "put_the_cap_on_the_pen_new" else ("id1",))):
            if task != "insert_building_block":
                return source_id
            from insertion_physics import certificate
            pair = certificate()
            if (wanted["id1"] == pair["star"] and pair["box"] in entries
                    and entries[pair["box"]]["usd"] == template["layout"][pair["box"]]["usd"]):
                return source_id
    raise UnsupportedRequest("Scene uses an unverified task object or assembly pair",
                             missing_assets=["verified_assembly_pair"])


def introduce_background(stage, new):
    from pxr import Usd, UsdGeom
    if new is None:
        return None
    background = Usd.Stage.Open(str(new["path"]))
    if not background or not background.GetDefaultPrim():
        raise ValueError("Background has no default prim")
    if UsdGeom.GetStageUpAxis(background) != "Z" or abs(UsdGeom.GetStageMetersPerUnit(background) - 1.) > 1e-6:
        raise ValueError("Background units or axes are incompatible")
    # The referenced room must not contribute a second copy of the source task table.
    world = stage.GetDefaultPrim()
    world.GetReferences().ClearReferences()
    world.GetReferences().AddReference(str(new["path"]))
    return new["id"]


def read_support_asset(row):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(row["path"]))
    if not stage or not stage.GetDefaultPrim() or UsdGeom.GetStageUpAxis(stage) != "Z" or abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.) > 1e-6:
        raise ValueError("Support USD must use Z up and metres")
    if not stage.GetPrimAtPath("/World/entity"):
        raise ValueError("Support is missing the runtime entity prim")
    lower, upper = world_bounds(stage, "/World/entity")
    if (upper[:2] - lower[:2] < [.4, .7]).any() or (upper[:2] - lower[:2] > [1.4, 2.1]).any():
        raise ValueError("Support footprint outside tested table range")
    # A horizontal upper collision face is required. Do not trust the visual AABB alone.
    cache = UsdGeom.XformCache()
    planar_area = sum(horizontal_collision_area(prim, cache, upper[2]) for prim in stage.Traverse()
                      if enabled_collision_mesh(prim))
    if planar_area < .82 * np.prod(upper[:2] - lower[:2]):
        raise ValueError("Support needs a near-full horizontal collider at its top height")
    return lower, upper


def enabled_collision_mesh(prim):
    from pxr import UsdGeom, UsdPhysics
    return (prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)
            and prim.GetAttribute("physics:collisionEnabled").Get() is not False)


def horizontal_collision_area(prim, cache, height):
    from pxr import UsdGeom
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        return 0.
    transform = np.asarray(cache.GetLocalToWorldTransform(prim)).T
    points = points @ transform[:3, :3].T + transform[:3, 3]
    face_start = 0
    area = 0.
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    for count in mesh.GetFaceVertexCountsAttr().Get():
        face = points[indices[face_start:face_start + count]]
        face_start += count
        if len(face) >= 3 and np.all(abs(face[:, 2] - height) < .004):
            area += .5 * abs(float(np.sum(face[:, 0] * np.roll(face[:, 1], -1) -
                                         face[:, 1] * np.roll(face[:, 0], -1))))
    return area


def align_room_floor(stage, support):
    """Match the room's load-bearing floor to the unchanged robot/table datum."""
    from pxr import Gf, Usd, UsdGeom
    footprint = np.asarray(support["bounds_xy_m"])
    footprint_area = float(np.prod(footprint[1] - footprint[0]))
    cache = UsdGeom.XformCache()
    floors = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(("/World/background/", "/World/Background/")) or not enabled_collision_mesh(prim):
            continue
        lower, upper = world_bounds(stage, path)
        if (lower[2] > .025 or not .004 < upper[2] < .25 or
                not (lower[:2] <= footprint[0]+.006).all() or not (upper[:2] >= footprint[1]-.006).all()):
            continue
        if horizontal_collision_area(prim, cache, upper[2]) >= .85 * footprint_area:
            floors.append((float(upper[2]), path))
    if not floors:
        return None
    height, floor_path = max(floors)
    root = stage.GetPrimAtPath("/" + "/".join(floor_path.split("/")[1:3]))
    roots = []
    # Some library backgrounds have a Scope above their movable components.
    components = iter(Usd.PrimRange(root))
    for component in components:
        transform = UsdGeom.Xformable(component)
        if not transform:
            continue
        matrix = cache.GetLocalToWorldTransform(component)
        move = matrix.GetInverse().TransformDir(Gf.Vec3d(0., 0., -height))
        transform.AddTranslateOp(opSuffix="projectFloorDatum",
                                 precision=UsdGeom.XformOp.PrecisionDouble).Set(move)
        roots.append(str(component.GetPath()))
        components.PruneChildren()
    _, final_upper = world_bounds(stage, floor_path)
    if abs(final_upper[2]) > .002:
        raise ValueError("Composed room collision floor could not be aligned")
    return {"floor":floor_path, "roots":roots, "previous_height_m":height,
            "final_height_m":float(final_upper[2]), "translation_world_z_m":-height}


def background_obstacles(stage, support):
    paths = []
    work = np.asarray(support["bounds_xy_m"])
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(("/World/background/", "/World/Background/")):
            continue
        if not enabled_collision_mesh(prim):
            continue
        lo, hi = world_bounds(stage,path)
        if (hi[:2] >= work[0]-.6).all() and (lo[:2] <= work[1]+.6).all() and hi[2] >= support["height_m"]-.12:
            paths.append(path)
    return paths


def replace_support(stage, metadata, source, row):
    from pxr import Usd, UsdGeom, UsdPhysics
    lower, upper = read_support_asset(row)
    if source["instance"]:
        stage.RemovePrim(source["prim"])
        metadata["layout"].pop(source["instance"], None)
    elif stage.GetPrimAtPath(SOURCE_PHONE_PLANE):
        # Remove the replaced environment furniture and its original tabletop plane.
        for old in (source["prim"], SOURCE_PHONE_PLANE):
            if stage.GetPrimAtPath(old):
                stage.OverridePrim(old).SetActive(False)
    # Support assets keep their native mesh, material and size.
    target = (lower + upper) / 2
    center = np.asarray(source["bounds_xy_m"]).mean(axis=0)
    top = float(upper[2] - lower[2])
    if abs(top - source["height_m"]) > MAX_SUPPORT_DELTA_M:
        raise ValueError("Support exceeds the initial height transfer range")
    translation = [float(center[0] - target[0]), float(center[1] - target[1]), -float(lower[2])]
    name = "collection_support_000"
    path = "/World/Objects/" + name
    support = UsdGeom.Xform.Define(stage, path).GetPrim()
    support.GetPayloads().AddPayload(str(row["path"]))
    set_pose(support, translation, [0., 0., 0., 1.])
    entity = stage.GetPrimAtPath(path + "/entity")
    # A work table is a fixed scene fixture. PhysX convex decomposition pushed
    # the real support plane 10 mm above the authored mesh in a library scene.
    # Keep the original triangles, transforms and material as static colliders.
    exact_colliders = []
    for prim in Usd.PrimRange(entity):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if enabled_collision_mesh(prim):
            mesh = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh.GetApproximationAttr().Set(UsdPhysics.Tokens.none)
            exact_colliders.append(str(prim.GetPath()))
    if not exact_colliders:
        raise ValueError("The composed support has no enabled triangle collider")
    extent = world_bounds(stage, path + "/entity", local=True)
    metadata["layout"][name] = {"usd": row["id"], "asset": str(row["path"]), "registered": True,
                                "xyz": translation, "xyzw": [0., 0., 0., 1.],
                                "entity_local_bbox": {"min": extent[0].tolist(), "max": extent[1].tolist()}}
    return {"prim": path, "instance": name, "asset_id": row["id"],
            "bounds_xy_m": [(np.asarray(translation[:2]) + lower[:2]).tolist(),
                            (np.asarray(translation[:2]) + upper[:2]).tolist()], "height_m": top,
            "physics": "native_collision_static_triangle_mesh", "collision_meshes": exact_colliders}


def relocate(stage, metadata, source, support, rng, vary, task):
    transforms = {}
    height = support["height_m"] - source["height_m"]
    # Use small, seeded independent offsets so object-to-target transforms genuinely change.
    limits = {"phone_packaging": (.012, .055), "insert_building_block": (.020, .090),
              "put_the_cap_on_the_pen_new": (.014, .070)}
    radius, yaw = limits.get(task,(.012,.055))
    plane = np.asarray(support["bounds_xy_m"])
    occupied = []
    for name, row in metadata["layout"].items():
        if name == support["instance"]:
            continue
        xyz = np.asarray(row["xyz"], dtype=float)
        angles = Rotation.from_quat(row["xyzw"])
        original = xyz.copy()
        xyz[2] += height
        if vary:
            delta = rng.uniform(-radius, radius, 2)
            xyz[:2] += delta
            angles = Rotation.from_rotvec([0, 0, float(rng.uniform(-yaw, yaw))]) * angles
        q = angles.as_quat().tolist()
        prim = stage.GetPrimAtPath("/World/Objects/" + name)
        set_pose(prim, xyz, q)
        low, high = world_bounds(stage, str(prim.GetPath()) + "/entity")
        if np.any(low[:2] < plane[0] + .006) or np.any(high[:2] > plane[1] - .006):
            raise ValueError(f"{name} footprint extends outside support")
        gap = float(low[2] - support["height_m"])
        if gap < -.014 or gap > .065:
            raise ValueError(f"{name} support gap outside preflight envelope: {gap:.4f} m")
        if any(np.min(np.minimum(high, top) - np.maximum(low, bottom)) > .003
               for _, bottom, top in occupied):
            raise ValueError(f"{name} initial bounding overlap")
        occupied.append((name, low, high))
        row.update(xyz=xyz.tolist(), xyzw=q)
        transforms[name] = {"translation_m": (xyz-original).tolist(), "quat_xyzw": q, "initial_box_gap_m": gap}
    return transforms, occupied


def add_distractor(stage, metadata, support, occupied, rng):
    from pxr import Usd, UsdGeom
    paths = sorted(ASSETS.glob("objects/benchmark/desk_decoration/*/Aligned.usda"))
    limits = np.asarray(support["bounds_xy_m"])
    positions = np.array(((.80,.13),(.80,.87),(.21,.13),(.21,.87),(.67,.14),(.67,.86)))
    order = rng.permutation(len(positions))
    for index in rng.permutation(len(paths))[:8]:
        path = paths[index]
        source = Usd.Stage.Open(str(path))
        if not source or not source.GetPrimAtPath("/World/entity"):
            continue
        low, high = world_bounds(source, "/World/entity")
        span = high - low
        if np.any(span[:2] > .105) or not .015 < span[2] < .19:
            continue
        # Sample free space around the outside of the work area, not between mating parts.
        for slot in order:
            center = limits[0] + (limits[1] - limits[0]) * positions[slot]
            xyz = [*(center - (low[:2] + high[:2]) / 2), support["height_m"] - low[2] + .001]
            a, b = low + xyz, high + xyz
            if (a[:2] < limits[0] + .012).any() or (b[:2] > limits[1] - .012).any():
                continue
            if any((np.minimum(b[:2]+.035, hi[:2]) > np.maximum(a[:2]-.035, lo[:2])).all()
                   for _, lo, hi in occupied):
                continue
            name = "collection_distractor_000"
            prim = UsdGeom.Xform.Define(stage, "/World/Objects/" + name).GetPrim()
            prim.GetPayloads().AddPayload(str(path.resolve()))
            set_pose(prim, xyz, [0., 0., 0., 1.])
            metadata["layout"][name] = {"usd": path.parent.name, "registered": True,
                                        "xyz": list(map(float, xyz)), "xyzw": [0., 0., 0., 1.],
                                        "asset": str(path.resolve()),
                                        "entity_local_bbox": {"min": low.tolist(), "max": high.tolist()}}
            return name
    raise ValueError("No compatible distractor fits the sampled tabletop")


def compile_scene(task, source_id, adapter, workspace, *, mode, seed, variation=(), background=None,
                  support_asset=None, goal=None, container_selection=None):
    from pxr import UsdGeom, UsdLux
    base_task = "phone_packaging" if container_selection is not None else task
    stage, metadata, task_info, profile = import_source(base_task, source_id, adapter, workspace)
    if UsdGeom.GetStageUpAxis(stage) != "Z" or abs(UsdGeom.GetStageMetersPerUnit(stage)-1.) > 1e-6:
        raise ValueError("Task scene must use Z up and metres")
    rng = np.random.default_rng(seed)
    source = source_support(stage, metadata)
    if container_selection is not None:
        from container_scene import author_objects
        author_objects(stage, metadata, task_info, container_selection, source, adapter)
    support = source
    background_id = None
    floor_alignment = None
    changes = {}
    distractor = None
    lighting_value = None
    if mode == "compose_from_library":
        background_id = introduce_background(stage, background)
        if support_asset:
            support = replace_support(stage, metadata, source, support_asset)
        if not stage.GetPrimAtPath(support["prim"]):
            raise ValueError("Background replacement requires an independent support asset")
        if background and support_asset and stage.GetPrimAtPath(SOURCE_PHONE_PLANE):
            raise ValueError("Background contains an overlapping implicit support plane")
        if background:
            floor_alignment = align_room_floor(stage,support)
        changes, occupied = relocate(stage, metadata, source, support, rng, "layout" in variation, task)
        if "distractors" in variation:
            distractor = add_distractor(stage, metadata, support, occupied, rng)
        lighting = UsdLux.DomeLight.Get(stage, "/World/CollectionDome")
        if "lighting" in variation:
            if not lighting:
                lighting = UsdLux.DomeLight.Define(stage, "/World/CollectionDome")
            lighting_value = float(rng.uniform(630., 960.))
            lighting.CreateIntensityAttr(lighting_value)
    contents = {name: {"asset_id":row["usd"], "xyz":row["xyz"], "xyzw":row["xyzw"]}
                for name,row in metadata["layout"].items()}
    identity = {"task": task, "source_scene_id": source_id, "mode": mode, "seed": seed,
                "support": support, "background": background_id, "poses": changes,
                "distractor": contents.get(distractor),
                "layout_sha256":content_id({"support":support,"objects":contents}),
                "variation": list(variation)}
    if container_selection is not None:
        identity["container_placement"] = metadata["project_container_placement"]
    if floor_alignment:
        identity["background_floor_alignment"] = floor_alignment
    if lighting_value is not None:
        identity["lighting_intensity"] = lighting_value
    if goal is not None:
        from goal_composition import goal_text, packing_goal
        if task != "phone_packaging":
            raise ValueError("Goal composition requires a compatible asset contract")
        goal = packing_goal(goal)
        metadata["project_packing_goal"] = goal
        metadata["official_instruction"]["instruction"] = goal_text(goal)
        identity["goal"] = goal
    scene_id = "scene-" + content_id(identity)[:16]
    metadata["scene_id"] = scene_id
    metadata["scene_transfer"] = {**identity, "profile_scene_id": profile,
                                  "collision_roots": sorted(set([support["prim"],*background_obstacles(stage,support)])),
                                  "support_height_offset_m": support["height_m"]-source["height_m"],
                                  "source_package": str(source_directory(base_task, source_id, adapter))}
    # Stage-only admission is independent of motion and measured contact acceptance.
    task_info["backend"] = "cpu"
    stage.GetRootLayer().Save()
    save_json(workspace / "scene_info.json", metadata)
    save_json(workspace / "task_info.json", task_info)
    task_spec(metadata, adapter.robot_id)
    dependencies = asset_dependencies(stage,metadata)
    return scene_id, write_manifest(workspace, dependencies, mode=mode, source=identity, asset_root=ASSETS)


def screen_supports(task, adapter, inventory, target):
    with tempfile.TemporaryDirectory(prefix="support-template-", dir=target) as temp:
        stage, metadata, _, _ = import_source(task, "0", adapter, Path(temp))
        reference = source_support(stage, metadata)["height_m"]
    eligible, rejected = [], []
    for row in inventory:
        if row["kind"] != "support":
            continue
        try:
            lo,hi = read_support_asset(row)
            if abs(float(hi[2]-lo[2])-reference)>MAX_SUPPORT_DELTA_M:
                raise ValueError("Support exceeds the initial height transfer range")
        except (ValueError,RuntimeError,OSError) as exc:
            rejected.append({"asset_id":row["id"],"reason":str(exc)})
        else:
            eligible.append(row)
    return eligible,rejected


def prepare_scenes(task, adapter, request, target, api, *, trajectory=None, goal=None, container_selection=None):
    mode, amount = request["mode"], request["scene_count"]
    base_task = "phone_packaging" if container_selection is not None else task
    inventory = list(records())
    backgrounds, background_receipt = [None], None
    supports, support_receipt = [None], None
    support_rejections = []
    variation = request.get("variation", [])
    target.mkdir(parents=True, exist_ok=True)
    if mode == "compose_from_library":
        if "background" in variation or "background_id" in request or "background_query" in request:
            backgrounds, background_receipt = rank(inventory, "background", api,
                query=request.get("background_query", "hdr studio background"), explicit=request.get("background_id"))
        if "support" in variation or "support_id" in request or "support_query" in request or (
                base_task == "phone_packaging" and backgrounds != [None]):
            screened = inventory
            if not request.get("support_id"):
                screened, support_rejections = screen_supports(base_task,adapter,inventory,target)
            supports, support_receipt = rank(screened, "support", api,
                query=request.get("support_query", "benchmark table"), explicit=request.get("support_id"))
    actual_ids = request.get("scene_ids") or existing_ids(base_task, adapter)
    if container_selection is not None:
        actual_ids = [s for s in actual_ids if s in ("0", "1")]
    if mode == "compose_from_library":
        actual_ids = [s for s in actual_ids if s in ("0", "1")]
    if not actual_ids:
        raise UnsupportedRequest("No compatible source scene templates", missing_assets=[task])
    candidates, rejected = [], []
    seed = request.get("seed", 20261124)
    # Explicit scene IDs and fixed layout seeds remain in the requested scope.
    reserve_count = 0 if "scene_ids" in request or "seed" in request else min(2, 12-amount)
    if mode == "existing":
        if "scene_ids" not in request:
            actual_ids = sorted(actual_ids, key=lambda value: value in ("0", "1"))
        attempts = [(value, None, None, seed) for value in actual_ids[:12]]
        if request.get("background_query"):
            raise UnsupportedRequest("Existing-scene background matching needs an indexed task scene",
                                     missing_assets=["matching_complete_scene_description"])
    else:
        pairs = sorted(((b, s) for b in range(len(backgrounds)) for s in range(len(supports))),
                       key=lambda pair: (sum(pair), max(pair), pair))
        attempts = []
        for i in range(12):
            b, s = pairs[i % len(pairs)]
            attempts.append((actual_ids[i % len(actual_ids)], backgrounds[b], supports[s], (seed+i) % 2**31))
    for source_id, background, support, realization_seed in attempts:
        if len(candidates) == amount + reserve_count:
            break
        details = {"source_id": source_id, "seed": realization_seed,
                   "background": background and background["id"], "support": support and support["id"]}
        with tempfile.TemporaryDirectory(prefix="compose-", dir=target) as temp:
            workspace = Path(temp)
            try:
                scene_id, manifest = compile_scene(task, source_id, adapter, workspace, mode=mode,
                    seed=realization_seed, variation=variation, background=background, support_asset=support,
                    goal=goal, container_selection=container_selection)
                folder = target / scene_id
                if folder.exists():
                    raise ValueError("Repeated scene identity")
                descriptor = None
                if trajectory is not None:
                    from trajectory_diversity import scene_conflict, scene_descriptor
                    meta = read_json(workspace / "scene_info.json")
                    descriptor = scene_descriptor(meta, asset_digests=manifest["asset_dependencies"],
                                                  lighting=manifest["source"].get("lighting_intensity"))
                    conflict = scene_conflict(descriptor, [item["diversity_scene"] for item in candidates],
                                              dimensions=trajectory.get("distinct_dimensions", ()))
                    if conflict:
                        raise ValueError("Scene has no requested physical difference from an existing candidate")
                folder.mkdir()
                for filename in (*SCENE_FILES, "scene-manifest.json"):
                    shutil.move(workspace / filename, folder / filename)
            except (ValueError, UnsupportedRequest, OSError, RuntimeError) as exc:
                rejected.append({**details, "reason": str(exc)})
                continue
        candidates.append({"scene_id": scene_id, "directory": str(folder), "sha256": manifest["content_sha256"],
                           "source_scene_id": source_id, "seed": realization_seed})
        if descriptor is not None:
            candidates[-1]["diversity_scene"] = descriptor
    return candidates[:amount], {"requested": amount, "compiled": min(len(candidates),amount),
                                 "reserve_candidates": candidates[amount:], "rejected": rejected,
                                 "support_geometry_rejections":support_rejections,
                                 "background_retrieval": background_receipt, "support_retrieval": support_receipt}
