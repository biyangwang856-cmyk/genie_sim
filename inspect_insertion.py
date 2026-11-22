"""Find the physical boundaries of the sorter's top holes from its official mesh."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from preview_scenes import object_meshes
from validate_scenes import open_stage


base = Path(__file__).resolve().parent
scene = base / 'output/finemanip-first/g2crsb90d_finemanip_insert_building_block/0/scene.usda'
stage = open_stage(scene, base / 'assets')
triangles = np.concatenate([mesh for name, mesh in object_meshes(stage)
                            if name == 'benchmark_building_blocks_01e2de23'])
top = triangles[:, :, 2].max()
flat = triangles[np.all(np.abs(triangles[:, :, 2] - top) < 0.0003, axis=1)]
vertices, indices = np.unique(np.round(flat.reshape(-1, 3), 5), axis=0, return_inverse=True)
faces = indices.reshape(-1, 3)
edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
unique, counts = np.unique(edges, axis=0, return_counts=True)
boundary = unique[counts == 1]
neighbors = defaultdict(set)
for a, b in boundary:
    neighbors[int(a)].add(int(b))
    neighbors[int(b)].add(int(a))
remaining = set(neighbors)
loops = []
while remaining:
    start = min(remaining)
    loop, current, previous = [], start, None
    while current not in loop:
        loop.append(current)
        remaining.discard(current)
        choices = neighbors[current] - {previous}
        if not choices:
            break
        previous, current = current, min(choices)
    points = vertices[loop]
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    loops.append({'vertex_count': len(points), 'center': center.tolist(), 'points': points.tolist(),
                  'radius': np.linalg.norm(points[:, :2] - center[:2], axis=1).tolist()})
result = {'top_z': float(top), 'top_triangles': len(flat), 'loops': loops}
output = base / 'output/insertion_geometry.json'
output.write_text(json.dumps(result, indent=2))
print(json.dumps(result))
