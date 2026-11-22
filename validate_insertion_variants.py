from pathlib import Path
import json,sys,numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom
root=Path('/home/vipuser/by/genie_sim');sys.path.insert(0,str(root))
from insertion_physics import measure
new=json.loads((root/'output/insertion_target_v2.json').read_text());old=json.loads((root/'output/insertion_target.json').read_text());report=[]
for scene_id in ['0','1']:
 stage=Usd.Stage.Open(str(root/'tasks/g2crsb90d_finemanip_insert_building_block'/scene_id/'scene.usda'))
 box=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Objects/benchmark_building_blocks_01e2de23/entity'))).T
 mesh_center=np.asarray(new['star_center_entity']);target=box@np.asarray(new['object_goal_in_box_frame']);center=(target@np.r_[mesh_center,1])[:3]
 old_target=box@np.asarray(old['object_goal_in_box_frame']);a=[];b=[]
 for angle in np.arange(5)*2*np.pi/5:
  rz=Rotation.from_rotvec([0,0,angle]).as_matrix()
  candidate=old_target.copy();candidate[:3,:3]=rz@old_target[:3,:3]
  a.append(measure(candidate,box)['outside_hole_area_fraction'])
  candidate=target.copy();candidate[:3,:3]=rz@target[:3,:3];candidate[:3,3]=center-candidate[:3,:3]@mesh_center
  b.append(measure(candidate,box)['outside_hole_area_fraction'])
 report.append({'scene':scene_id,'old_entity_origin_rotation_outside':a,'corrected_mesh_center_rotation_outside':b,'passing_corrected_variants':sum(v<=.001 for v in b)})
(root/'jobs/insertion-variants-regression.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2));assert all(r['passing_corrected_variants']>0 for r in report)
