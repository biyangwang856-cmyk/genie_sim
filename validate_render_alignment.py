"""Compare recorded physics-link snapshots with rendering's USD transformations."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
run=Path(sys.argv[1]);events=[json.loads(x) for x in (run/'evidence/stage-evidence.jsonl').read_text().splitlines()]
rows=[]
for event in events:
 if not event['phase'].startswith('model_snapshot'):continue
 snapshot=event['snapshot']
 for name,link in snapshot['links'].items():
  if name not in snapshot.get('usd_links',{}):continue
  actual=np.eye(4);actual[:3,3]=link['position'];actual[:3,:3]=Rotation.from_quat(np.asarray(link['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
  usd=np.asarray(snapshot['usd_links'][name])
  rows.append({'phase':event['phase'],'link':name,'position_error_m':float(np.linalg.norm(actual[:3,3]-usd[:3,3])),
    'rotation_error_rad':float(Rotation.from_matrix(actual[:3,:3].T@usd[:3,:3]).magnitude())})
passed=bool(rows) and any(r['phase']=='model_snapshot_after_grasp' for r in rows) and all(r['position_error_m']<.002 and r['rotation_error_rad']<.02 for r in rows)
result={'passed':passed,'samples':rows}
(run/'render-state-validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'passed':passed,'samples':len(rows),'max_position_error_m':max([r['position_error_m'] for r in rows],default=None),
 'max_rotation_error_rad':max([r['rotation_error_rad'] for r in rows],default=None)}))
raise SystemExit(0 if passed else 1)
