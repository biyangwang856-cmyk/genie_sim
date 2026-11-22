"""Regression on real candidate poses: preserve wrist goals across TCP revisions."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--offset',type=float,required=True);args=p.parse_args()
root=Path('/home/vipuser/by/genie_sim')
trace=Path('/data/by/genie_sim/collection-runs/block-scene0-calibrated04/evidence/stage-evidence.jsonl')
row=next(json.loads(line) for line in trace.read_text().splitlines() if json.loads(line)['phase']=='pick_ik_filter')
goals=np.asarray(row['poses']);arm=row['arm']
old=np.eye(4);old[:3,3]=[0,0,.2292200029]
old[:3,:3]=Rotation.from_euler('z',np.pi/2 if arm=='left' else -np.pi/2).as_matrix()
new=np.eye(4);new[:3,3]=[0,0,.185]
measurements=[]
for goal in goals:
    legacy=goal.copy();legacy[:3,:3]=goal[:3,:3]@old[:3,:3]
    expected_wrist=legacy@np.linalg.inv(old)
    converted=goal.copy();converted[:3,3]+=goal[:3,:3]@np.array([0,0,args.offset])
    predicted_wrist=converted@np.linalg.inv(new)
    measurements.append(float(np.linalg.norm(expected_wrist[:3,3]-predicted_wrist[:3,3])))
report={'fixture':str(trace),'candidates':len(goals),'offset_m':args.offset,
        'max_wrist_goal_error_m':max(measurements),'passed':max(measurements)<1e-8}
(root/'jobs'/('grasp-tcp-'+('corrected' if report['passed'] else 'uncompensated')+'.json')).write_text(json.dumps(report,indent=2))
print(json.dumps(report));raise SystemExit(0 if report['passed'] else 1)
