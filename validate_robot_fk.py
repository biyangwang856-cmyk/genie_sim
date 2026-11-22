"""Compare native recorded USD link poses against planner URDF FK."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation
root=Path('/home/vipuser/by/genie_sim')
path=Path('/workspace/config/robot_cfg/G2/G2_crsB_90D_from_usd.urdf')
target=root/'cache/collection/G2_crsB_90D_from_usd_calibrated.urdf'
tree=ET.parse(path)
for joint in tree.getroot().findall('joint'):
    if joint.attrib['name'] in ('idx52_gripper_l_center_joint','idx92_gripper_r_center_joint'):
        origin=joint.find('origin');origin.set('xyz','0 0 0.185');origin.set('rpy','0 0 0')
tree.write(target,encoding='unicode')
state=json.load(open('/workspace/recording_data/g2crsb90d_finemanip_insert_building_block/state.json'))
info=json.load(open('/workspace/recording_data/g2crsb90d_finemanip_insert_building_block/recording_info.json'))
base=np.eye(4);base[:3,3]=info['robot_init_position'];base[:3,:3]=Rotation.from_quat(np.asarray(info['robot_init_rotation'])[[1,2,3,0]]).as_matrix()
report={}
for label,urdf in [('original',path),('calibrated',target)]:
 model=pin.buildModelFromUrdf(str(urdf));data=model.createData();measurements=[]
 for i in (0,len(state['frames'])//2,len(state['frames'])-1):
  f=state['frames'][i];j=f['robot']['joints'];values=dict(zip(j['joint_name'],j['joint_position']));q=pin.neutral(model)
  for name,value in values.items():
   if not model.existJointName(name):continue
   joint=model.joints[model.getJointId(name)]
   if joint.nq==1:q[joint.idx_q]=value
   elif joint.nq==2:q[joint.idx_q:joint.idx_q+2]=[np.cos(value),np.sin(value)]
  pin.framesForwardKinematics(model,data,q)
  for side,letter in [('left','l'),('right','r')]:
   for key,frame in [('ee',f'arm_{letter}_end_link'),('ee_center',f'gripper_{letter}_center_link')]:
    actual=np.asarray(f[key][side]['pose']);pred=base@data.oMf[model.getFrameId(frame)].homogeneous
    measurements.append({'sample':i,'arm':side,'frame':frame,'position_error_m':float(np.linalg.norm(actual[:3,3]-pred[:3,3])),
     'rotation_error_rad':float(Rotation.from_matrix(actual[:3,:3].T@pred[:3,:3]).magnitude())})
 report[label]=measurements
(root/'jobs/robot-fk-calibration.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
