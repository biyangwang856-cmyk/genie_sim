"""Independent USD Physics joint-frame/axis/limit comparison against the URDF."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdPhysics
ROOT=Path('/home/vipuser/by/genie_sim')
stage=Usd.Stage.Open(str(ROOT/'GenieSimAssets/robot/G2_crsB_90d/robot_fix.usda'))
tree=ET.parse(ROOT/'cache/collection/G2_crsB_90D_from_usd_calibrated.urdf')
def transform(p,q):
 t=np.eye(4);t[:3,3]=p;t[:3,:3]=Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix();return t
by_name={p.GetName():p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
rows=[]
aliases={f'gripper_{s}_{lr}_{part}_link':f'gripper_{s}_{part}_link{n}'
 for s in ['l','r'] for lr,n in [('left',1),('right',2)] for part in ['inner','outer','support']}
for joint in tree.getroot().findall('joint'):
 name=joint.attrib['name']
 if name not in by_name:continue
 u=UsdPhysics.RevoluteJoint(by_name[name])
 b0=u.GetBody0Rel().GetTargets();b1=u.GetBody1Rel().GetTargets()
 if not b0 or not b1:continue
 a=joint.find('parent').attrib['link'];b=joint.find('child').attrib['link']
 ua=aliases.get(a,a);ub=aliases.get(b,b)
 t0=transform(u.GetLocalPos0Attr().Get(),u.GetLocalRot0Attr().Get())
 t1=transform(u.GetLocalPos1Attr().Get(),u.GetLocalRot1Attr().Get())
 if str(b0[0]).split('/')[-1]==ua and str(b1[0]).split('/')[-1]==ub:
  actual=t0@np.linalg.inv(t1)
  axis=t1[:3,:3]@np.eye(3)['XYZ'.index(u.GetAxisAttr().Get())]
 elif str(b1[0]).split('/')[-1]==ua and str(b0[0]).split('/')[-1]==ub:
  actual=t1@np.linalg.inv(t0)
  axis=-t0[:3,:3]@np.eye(3)['XYZ'.index(u.GetAxisAttr().Get())]
 else:
  print('JOINT_BODY_MISMATCH',name,a,b,[str(x) for x in b0],[str(x) for x in b1])
  continue
 origin=joint.find('origin');expected=np.eye(4)
 expected[:3,3]=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ')
 expected[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
 old_axis=np.fromstring(joint.find('axis').attrib['xyz'],sep=' ')
 limit=joint.find('limit');lo=np.deg2rad(u.GetLowerLimitAttr().Get());hi=np.deg2rad(u.GetUpperLimitAttr().Get())
 rows.append({'joint':name,'position_error_m':float(np.linalg.norm(actual[:3,3]-expected[:3,3])),
 'angle_error_rad':float(Rotation.from_matrix(expected[:3,:3].T@actual[:3,:3]).magnitude()),
 'axis_error':float(np.linalg.norm(axis-old_axis)),'usd_axis':axis.tolist(),'urdf_axis':old_axis.tolist(),
 'usd_limits_rad':[float(lo),float(hi)],'urdf_limits_rad':[float(limit.attrib.get('lower','-inf')),float(limit.attrib.get('upper','inf'))],
 'usd_origin':actual.tolist()})
report={'joints':rows}
(ROOT/'jobs/joint-model-validation.json').write_text(json.dumps(report,indent=2))
for r in rows:
 if max(r['position_error_m'],r['angle_error_rad'],r['axis_error'],max(abs(np.asarray(r['usd_limits_rad'])-r['urdf_limits_rad'])))>1e-5:print(json.dumps(r))
print('COMPARED',len(rows))
