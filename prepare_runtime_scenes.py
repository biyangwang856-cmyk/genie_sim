"""Resolve archived USD asset references into a project-local runtime copy."""
import json
from pathlib import Path

from validate_scenes import resolve_layer


base = Path(__file__).resolve().parent
request_root = base / 'output/finemanip-first'
runtime_root = base / 'output/runtime-scenes'
request = json.loads((request_root / 'request.json').read_text(encoding='utf-8'))
receipt = []
for task in request['tasks']:
    short_name = task['task_id'].split('_finemanip_', 1)[1]
    for scene in task['scenes']:
        instance = str(scene['instance_id'])
        source = request_root / task['task_id'] / instance / 'scene.usda'
        target = runtime_root / short_name / instance / 'scene.usda'
        target.parent.mkdir(parents=True, exist_ok=True)
        resolve_layer(source, base / 'assets').Export(str(target))
        receipt.append({'source': str(source), 'runtime_copy': str(target)})
(runtime_root / 'receipt.json').write_text(json.dumps(receipt, indent=2))
print(f'Resolved {len(receipt)} runtime scene layers')
