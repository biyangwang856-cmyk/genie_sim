"""Render a static geometry preview of the official task objects through EGL."""
import argparse
import json
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Usd, UsdGeom

from validate_scenes import open_stage


TITLES = {
    'phone_packaging': 'Phone packaging',
    'put_the_cap_on_the_pen_new': 'Pen capping',
    'insert_building_block': 'Block insertion',
}
COLORS = np.array([
    (0.33, 0.63, 0.83), (0.87, 0.53, 0.33), (0.50, 0.74, 0.56),
    (0.69, 0.59, 0.83), (0.86, 0.76, 0.41), (0.72, 0.45, 0.52),
], dtype='f4')
VERTEX = '''
#version 330
in vec3 position;
in vec3 normal;
in vec3 color;
out vec3 shade;
void main() {
    float diffuse = abs(dot(normal, normalize(vec3(-0.35, 0.55, 0.78))));
    shade = color * (0.48 + 0.52 * diffuse);
    gl_Position = vec4(position, 1.0);
}
'''
FRAGMENT = '''
#version 330
in vec3 shade;
out vec4 frag_color;
void main() { frag_color = vec4(shade, 1.0); }
'''


def object_meshes(stage):
    transforms = UsdGeom.XformCache()
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = prim.GetPath().pathString.split('/')
        if len(path) < 4 or path[1:3] != ['World', 'Objects'] or path[3].startswith('table'):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility() == 'invisible' or mesh.ComputePurpose() in ('guide', 'proxy'):
            continue
        if any('collision' in part.casefold() for part in path[4:]):
            continue
        points = np.asarray(mesh.GetPointsAttr().Get())
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        sizes = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not len(sizes):
            continue
        matrix = np.asarray(transforms.GetLocalToWorldTransform(prim))
        world = points @ matrix[:3, :3] + matrix[3, :3]
        if np.all(sizes == 3):
            triangles = faces.reshape(-1, 3)
        else:
            triangles = []
            offset = 0
            for size in sizes:
                face = faces[offset:offset + size]
                triangles.extend((face[0], face[j], face[j + 1]) for j in range(1, size - 1))
                offset += size
            if not triangles:
                continue
        yield path[3], world[np.asarray(triangles)]


def render(stage, ctx, program, width, height):
    meshes = list(object_meshes(stage))
    if not meshes:
        raise ValueError('No visible task object meshes')
    names = sorted({name for name, _ in meshes})
    points = np.concatenate([triangles.reshape(-1, 3) for _, triangles in meshes])
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    forward = np.array([0.65, 0.20, -1.0])
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    basis = np.stack([right, np.cross(right, forward), forward], axis=1)
    span = np.ptp((points - center) @ basis, axis=0)
    half_height = max(span[1], span[0] * height / width, 0.02) * 0.63
    scale = np.array([half_height * width / height, half_height, max(span[2], 0.02)])
    target = ctx.simple_framebuffer((width, height), components=3)
    target.use()
    target.clear(0.91, 0.93, 0.95, 1.)
    ctx.enable(moderngl.DEPTH_TEST)
    for name, triangles in meshes:
        vertices = (triangles - center) @ basis
        normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        normals = np.repeat(normals, 3, axis=0)
        coords = vertices.reshape(-1, 3) / scale
        colors = np.broadcast_to(COLORS[names.index(name) % len(COLORS)], coords.shape)
        data = np.column_stack([coords, normals, colors]).astype('f4')
        buffer = ctx.buffer(data.tobytes())
        geometry = ctx.vertex_array(program, [(buffer, '3f 3f 3f', 'position', 'normal', 'color')])
        geometry.render(mode=moderngl.TRIANGLES)
        geometry.release()
        buffer.release()
    picture = Image.frombytes('RGB', target.size, target.read()).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    target.release()
    return picture, {'object_count': len(names), 'triangle_count': sum(len(mesh) for _, mesh in meshes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('assets', type=Path)
    args = parser.parse_args()
    batch = json.loads((args.batch / 'request.json').read_text(encoding='utf-8'))
    ctx = moderngl.create_standalone_context(backend='egl')
    program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
    width, height, header = 540, 308, 42
    montage = Image.new('RGB', (2 * width, len(batch['tasks']) * (height + header)), '#f7f8fa')
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default(size=21)
    rows = []
    for i, task in enumerate(batch['tasks']):
        task_id = task['task_id']
        for j, scene in enumerate(task['scenes']):
            instance = scene['instance_id']
            folder = args.batch / task_id / str(instance)
            stage = open_stage(folder / 'scene.usda', args.assets.resolve())
            if stage.GetCompositionErrors():
                raise ValueError(f'Cannot render {task_id}/{instance} with composition errors')
            preview, stats = render(stage, ctx, program, width, height)
            preview.save(folder / 'objects.png')
            x, y = j * width, i * (height + header)
            title = TITLES[task_id.split('_finemanip_', 1)[1]]
            draw.text((x + 16, y + 9), f'{title}  /  Instance {instance}', font=font, fill='#283747')
            montage.paste(preview, (x, y + header))
            rows.append(dict(task_id=task_id, instance_id=instance, **stats))
    montage.save(args.batch / 'static-preview.png')
    receipt = {'kind': 'static_task_object_geometry', 'renderer': ctx.info['GL_RENDERER'], 'scenes': rows}
    (args.batch / 'static_preview.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
