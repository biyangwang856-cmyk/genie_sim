"""Compare every authored rigid mount against the actual full robot asset."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom
ROOT=Path('/home/vipuser/by/genie_sim')
stage=Usd.Stage.Open(str(ROOT/'GenieSimAssets/robot/G2_crsB_90d/robot_fix.usda'))
cache=UsdGeom.XformCache()
by_name={p.GetName():p for p in stage.Traverse()}
tree=ET.parse(ROOT/'cache/collection/G2_crsB_90D_from_usd_calibrated.urdf')
rows=[]
for joint in tree.getroot().findall('joint'):
    if joint.attrib['type']!='fixed':continue
    a=joint.find('parent').attrib['link'];b=joint.find('child').attrib['link']
    if a not in by_name or b not in by_name:continue
    wa=np.asarray(cache.GetLocalToWorldTransform(by_name[a])).T
    wb=np.asarray(cache.GetLocalToWorldTransform(by_name[b])).T
    actual=np.linalg.inv(wa)@wb
    origin=joint.find('origin');expected=np.eye(4)
    if origin is not None:
        expected[:3,3]=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ')
        expected[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
    rows.append({'joint':joint.attrib['name'],'parent':a,'child':b,
        'position_error_m':float(np.linalg.norm(actual[:3,3]-expected[:3,3])),
        'angle_error_rad':float(Rotation.from_matrix(expected[:3,:3].T@actual[:3,:3]).magnitude()),
        'usd_relative':actual.tolist(),'urdf_relative':expected.tolist()})
report={'asset':str(stage.GetRootLayer().identifier),'fixed_joints':rows}
(ROOT/'jobs/fixed-frame-validation.json').write_text(json.dumps(report,indent=2))
for r in rows:
    if r['position_error_m']>1e-5 or r['angle_error_rad']>1e-5:
        print(json.dumps(r))
print('COMPARED',len(rows),'MISMATCHED',sum(r['position_error_m']>1e-5 or r['angle_error_rad']>1e-5 for r in rows))
