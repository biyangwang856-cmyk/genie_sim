"""Validate planner frames against live PhysX links, including gripper aliases."""
from pathlib import Path
import json,sys,xml.etree.ElementTree as ET
import numpy as np,pinocchio as pin
from scipy.spatial.transform import Rotation
root=Path('/home/vipuser/by/genie_sim');run=Path(sys.argv[1])
events=[json.loads(line) for line in (run/'evidence/stage-evidence.jsonl').read_text().splitlines()]
snap=[r['snapshot'] for r in events if r['phase'].startswith('model_snapshot')][-1]
names=snap['joint_names'];positions=snap['joint_positions']
aliases={f'gripper_{s}_{lr}_{part}_link':f'gripper_{s}_{part}_link{n}'
 for s in ['l','r'] for lr,n in [('left',1),('right',2)] for part in ['inner','outer','support']}
reverse={v:k for k,v in aliases.items()}
base=np.eye(4);base[:3,3]=snap['robot_base_position']
base[:3,:3]=Rotation.from_quat(np.asarray(snap['robot_base_quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
urdf=run/'source/robot.urdf'
if not urdf.exists():urdf=root/'cache/collection/G2_crsB_90D_from_usd_calibrated.urdf'
tree=ET.parse(urdf)
temp=run/'kinematics-diagnostic.urdf'
report={}
for label in ['current_urdf','authored_usd_axes']:
 if label=='authored_usd_axes':
  records=json.loads((root/'jobs/joint-model-validation.json').read_text())['joints']
  by_joint={j['joint']:j for j in records}
  for joint in tree.getroot().findall('joint'):
   if joint.attrib['name'] in by_joint:
    joint.find('axis').set('xyz',' '.join(map(str,by_joint[joint.attrib['name']]['usd_axis'])))
 tree.write(temp,encoding='unicode')
 model=pin.buildModelFromUrdf(str(temp));data=model.createData();q=pin.neutral(model)
 for name,value in zip(names,positions):
  if model.existJointName(name):
   j=model.joints[model.getJointId(name)]
   if j.nq==1:q[j.idx_q]=value
   elif j.nq==2:q[j.idx_q:j.idx_q+2]=[np.cos(value),np.sin(value)]
 pin.framesForwardKinematics(model,data,q);rows=[]
 for name,link in snap['links'].items():
  actual=np.eye(4);actual[:3,3]=link['position'];actual[:3,:3]=Rotation.from_quat(np.asarray(link['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
  expected=base@data.oMf[model.getFrameId(reverse.get(name,name))].homogeneous
  rows.append({'frame':name,'position_error_m':float(np.linalg.norm(expected[:3,3]-actual[:3,3])),
   'angle_error_rad':float(Rotation.from_matrix(expected[:3,:3].T@actual[:3,:3]).magnitude())})
 report[label]=rows
for label,rows in report.items():print(label,'max_position_m',max(r['position_error_m'] for r in rows),'max_angle_rad',max(r['angle_error_rad'] for r in rows))
report['native_gripper_values']={k:v for k,v in zip(names,positions) if 'gripper' in k}
limits=np.asarray(snap['native_limits'])
if limits.shape==(2,len(names)):limits=limits.T
report['native_gripper_limits']={k:limits[i].tolist() for i,k in enumerate(names) if 'gripper' in k}
(run/'live-kinematic-validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'gripper_values':report['native_gripper_values'],'gripper_limits':report['native_gripper_limits']}))
