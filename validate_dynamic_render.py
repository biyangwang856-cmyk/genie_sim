"""Require actual moving-link samples, not two identical rollback endpoints."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
run=Path(sys.argv[1]);path=run/'evidence/render-motion-samples.jsonl'
rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
errors=[];motions=[]
for side in ['l','r']:
    poses=[]
    for row in rows:
        actual=np.asarray(row['links'][side]['physics']);usd=np.asarray(row['links'][side]['usd'])
        poses.append(actual[:3,3])
        errors.append({'time':row['sim_time'],'arm':side,'position_error_m':float(np.linalg.norm(actual[:3,3]-usd[:3,3])),
          'angle_error_rad':float(Rotation.from_matrix(actual[:3,:3].T@usd[:3,:3]).magnitude())})
    motions.append(float(np.max(np.linalg.norm(np.asarray(poses)-poses[0],axis=1))) if poses else 0.)
result={'passed':len(rows)>=10 and max(motions)>.05 and all(x['position_error_m']<.002 and x['angle_error_rad']<.02 for x in errors),
    'samples':len(rows),'max_motion_m':max(motions),'max_position_error_m':max((x['position_error_m'] for x in errors),default=None),
    'max_angle_error_rad':max((x['angle_error_rad'] for x in errors),default=None),'errors':errors}
(run/'dynamic-render-validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='errors'}))
raise SystemExit(0 if result['passed'] else 1)
