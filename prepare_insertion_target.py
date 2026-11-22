"""Fit the official star silhouette to the actual top-hole mesh, at native scale."""
import json
from pathlib import Path

import numpy as np
from pxr import UsdGeom
from scipy.optimize import minimize_scalar
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

from preview_scenes import object_meshes
from validate_scenes import open_stage


root = Path(__file__).resolve().parent
data = json.loads((root / 'output/insertion_geometry.json').read_text())
holes = []
for loop in data['loops']:
    points = np.asarray(loop['points'])
    if np.ptp(points[:, :2], axis=0).max() < 0.08:
        polygon = Polygon(points[:, :2])
        holes.append((polygon.area / polygon.convex_hull.area, polygon))
hole = min(holes, key=lambda item: item[0])[1]
hole_center = np.array([hole.centroid.x, hole.centroid.y])
stage = open_stage(root / 'output/finemanip-first/g2crsb90d_finemanip_insert_building_block/0/scene.usda',
                   root / 'assets')
star_name = 'benchmark_building_blocks_35570a3b'
triangles = np.concatenate([mesh for name, mesh in object_meshes(stage) if name == star_name])
points = np.unique(np.round(triangles.reshape(-1, 3), 7), axis=0)
origin = points.mean(axis=0)
cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
areas = np.linalg.norm(cross, axis=1)
normals = cross[areas > 1e-10] / areas[areas > 1e-10, None]
weights = areas[areas > 1e-10]
dominant = max(range(len(normals)), key=lambda i: weights[np.abs(normals @ normals[i]) > 0.999].sum())
normal = normals[dominant]
if normal[2] < 0:
    normal = -normal
u = np.cross(normal, [0., 0., 1.])
if np.linalg.norm(u) < 0.1:
    u = np.cross(normal, [0., 1., 0.])
u /= np.linalg.norm(u)
basis = np.column_stack([u, np.cross(normal, u), normal])
projected = (triangles - origin) @ basis
polygons = [Polygon(triangle[:, :2]) for triangle in projected]
silhouette = unary_union([polygon for polygon in polygons if polygon.area > 1e-12])
if silhouette.geom_type != 'Polygon':
    raise ValueError(f'Unexpected star silhouette {silhouette.geom_type}')
center = np.array([silhouette.centroid.x, silhouette.centroid.y])
silhouette = affinity.translate(silhouette, -center[0], -center[1])


def cost(angle):
    rotated = affinity.rotate(silhouette, angle, origin=(0, 0), use_radians=True)
    moved = affinity.translate(rotated, *hole_center)
    return moved.difference(hole).area / silhouette.area


angles = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
best = float(min(angles, key=cost))
solution = minimize_scalar(cost, bounds=(best - 0.005, best + 0.005), method='bounded')
theta = float(solution.x)
align = np.array([[np.cos(theta), -np.sin(theta), 0.],
                  [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]]) @ basis.T
half_thickness = np.ptp((points - origin) @ normal) / 2
center_world = origin + basis @ np.r_[center, ((points - origin) @ normal).mean()]
cache = UsdGeom.XformCache()


def actor(name):
    prim = stage.GetPrimAtPath(f'/World/Objects/{name}/entity')
    return np.asarray(cache.GetLocalToWorldTransform(prim)).T


star_pose = actor(star_name)
center_local = np.linalg.solve(star_pose[:3, :3], center_world - star_pose[:3, 3])
goal = np.eye(4)
goal[:3, :3] = align @ star_pose[:3, :3]
goal[:3, 3] = np.r_[hole_center, data['top_z'] + half_thickness + 0.003] - goal[:3, :3] @ center_local
relative = np.linalg.inv(actor('benchmark_building_blocks_01e2de23')) @ goal
outside = float(cost(theta))
result = {'star': star_name, 'box': 'benchmark_building_blocks_01e2de23',
          'silhouette_outside_hole_area_fraction': outside,
          'hole_center_reference_world': hole_center.tolist(), 'top_z': data['top_z'],
          'star_half_thickness': float(half_thickness), 'angle_rad': theta,
          'object_goal_in_box_frame': relative.tolist(), 'native_scale_preserved': True}
(root / 'output/insertion_target.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
if outside > 0.001:
    raise ValueError('The native star silhouette does not fit the selected physical hole')
