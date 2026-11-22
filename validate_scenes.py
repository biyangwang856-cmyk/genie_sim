"""Validate the six official USD layers and report whether payloads are available."""
import argparse
import hashlib
import json
from pathlib import Path
from pxr import Sdf, Usd, UsdGeom, UsdUtils


def resolve_layer(path, assets):
    layer = Sdf.Layer.OpenAsAnonymous(str(path.resolve()))

    def resolve(asset):
        if asset.startswith('/geniesim_assets/'):
            return str(assets / asset.removeprefix('/geniesim_assets/'))
        if asset.startswith('./'):
            return str(path.parent.resolve() / asset)
        return asset

    UsdUtils.ModifyAssetPaths(layer, resolve)
    return layer


def open_stage(path, assets):
    return Usd.Stage.Open(resolve_layer(path, assets), load=Usd.Stage.LoadAll)


def compose(path, assets):
    stage = open_stage(path, assets)
    prims = list(stage.Traverse())
    errors = [str(error) for error in stage.GetCompositionErrors()]
    return {'mesh_count': sum(prim.IsA(UsdGeom.Mesh) for prim in prims),
            'prim_count': len(prims), 'error_count': len(errors), 'errors': errors[:12]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=Path)
    parser.add_argument('assets', type=Path)
    args = parser.parse_args()
    assets = args.assets.resolve()
    request = json.loads((args.batch / 'request.json').read_text(encoding='utf-8'))
    rows = []
    for task in request['tasks']:
        for scene in task['scenes']:
            path = args.batch / task['task_id'] / str(scene['instance_id'])
            for name, digest in scene['sha256'].items():
                if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Hash mismatch in {path / name}')
            layer = Sdf.Layer.FindOrOpen(str((path / 'scene.usda').resolve()))
            if layer is None or not layer.defaultPrim:
                raise ValueError(f'Scene layer cannot be opened: {path}')
            missing = []
            for reference in scene['top_layer_references']:
                if reference.startswith('/geniesim_assets/'):
                    candidate = assets / reference.removeprefix('/geniesim_assets/')
                    if not candidate.is_file():
                        missing.append(reference)
            rows.append({'task_id': task['task_id'], 'instance_id': scene['instance_id'],
                         'default_prim': layer.defaultPrim, 'sha256_verified': True,
                         'missing_top_layer_assets': missing,
                         'usd_composition': None if missing else compose(path / 'scene.usda', assets)})
    receipt = {'layer_parse_passed': True, 'scenes': rows}
    (args.batch / 'scene_check.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'parsed_scenes': len(rows),
                      'with_all_direct_asset_paths': sum(not r['missing_top_layer_assets'] for r in rows),
                      'composed_without_errors': sum(
                          row['usd_composition'] is not None and row['usd_composition']['error_count'] == 0
                          and row['usd_composition']['mesh_count'] > 0 for row in rows)}))


if __name__ == '__main__':
    main()
