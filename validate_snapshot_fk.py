"""Independent URDF FK using only archived XML and measured joint values."""
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
run=Path(sys.argv[1]);tree=ET.parse(run/'source/robot.urdf')
events=[json.loads(x) for x in (run/'evidence/stage-evidence.jsonl').read_text().splitlines()]
joints=tree.getroot().findall('joint');children={j.find('child').get('link') for j in joints}
roots=[x.get('name') for x in tree.getroot().findall('link') if x.get('name') not in children]
aliases={f'gripper_{s}_{lr}_{p}_link':f'gripper_{s}_{p}_link{n}' for s in ['l','r'] for lr,n in [('left',1),('right',2)] for p in ['inner','outer','support']}
rows=[]
for e in events:
 if not e['phase'].startswith('model_snapshot'):continue
 snap=e['snapshot'];q=dict(zip(snap['joint_names'],snap['joint_positions']))
 base=np.eye(4);base[:3,3]=snap['robot_base_position'];base[:3,:3]=Rotation.from_quat(np.asarray(snap['robot_base_quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
 poses={r:base for r in roots};remaining=list(joints)
 while remaining:
  count=len(remaining)
  for joint in list(remaining):
   parent=joint.find('parent').get('link');child=joint.find('child').get('link')
   if parent not in poses:continue
   origin=joint.find('origin');t=np.eye(4)
   if origin is not None:
    t[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
    t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
   motion=np.eye(4);axis=joint.find('axis');v=np.fromstring(axis.get('xyz','1 0 0'),sep=' ') if axis is not None else np.array([1.,0,0])
   value=q.get(joint.get('name'),0.)
   if joint.get('type') in ['revolute','continuous']:motion[:3,:3]=Rotation.from_rotvec(v*value).as_matrix()
   elif joint.get('type')=='prismatic':motion[:3,3]=v*value
   poses[child]=poses[parent]@t@motion;remaining.remove(joint)
  if len(remaining)==count:raise ValueError('Disconnected joint graph')
 for name,expected in poses.items():
  actual=snap['links'].get(aliases.get(name,name))
  if actual is None:continue
  r=Rotation.from_quat(np.asarray(actual['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
  rows.append({'phase':e['phase'],'link':name,'position_error_m':float(np.linalg.norm(expected[:3,3]-actual['position'])),
   'angle_error_rad':float(Rotation.from_matrix(expected[:3,:3].T@r).magnitude())})
result={'passed':bool(rows) and all(x['position_error_m']<.002 and x['angle_error_rad']<.02 for x in rows),'samples':rows}
(run/'snapshot-fk-validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'passed':result['passed'],'samples':len(rows),'max_position_error_m':max(x['position_error_m'] for x in rows),'max_angle_error_rad':max(x['angle_error_rad'] for x in rows)}))
raise SystemExit(0 if result['passed'] else 1)
