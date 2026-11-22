"""Fast actual target-construction check against official precise geometry rules."""
from pathlib import Path
import contextlib,json,os,sys
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom
root=Path('/home/vipuser/by/genie_sim');out=root/'jobs/static-phone-targets';out.mkdir(exist_ok=True)
os.environ['GENIE_EPISODE_EVIDENCE']=str(out)
sys.path[:0]=[str(root),'/workspace']
with (out/'imports.log').open('w') as log,contextlib.redirect_stdout(log):
 from collection_programs import phone_target,phone_rules
class GeometryFixture:
 def get_object_pose(self,name):
  t=self.transforms[name];q=Rotation.from_matrix(t[:3,:3]).as_quat();return t[:3,3],q[[3,0,1,2]]
report=[]
for scene in ['0','1']:
 folder=root/'tasks/g2crsb90d_finemanip_phone_packaging'/scene
 fixture=GeometryFixture();fixture.scene_info=json.loads((folder/'scene_info.json').read_text())
 stage=Usd.Stage.Open(str(folder/'scene.usda'));cache=UsdGeom.XformCache()
 fixture.transforms={name:np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Objects/'+name+'/entity'))).T
  for name in fixture.scene_info['layout']}
 results=[]
 for step,(name,depth) in enumerate([('benchmark_phone_000',.008),('benchmark_packaging_box_000',.020),('benchmark_packaging_box_002',.026)]):
  goal=phone_target(fixture,name,depth);fixture.transforms[name]=goal
  results.append({'object':name,'target':goal.tolist(),'official_rule_pass':phone_rules(fixture,2*step+1)})
 report.append({'scene':scene,'stages':results,'all_passed':all(r['official_rule_pass'] for r in results)})
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps([{'scene':r['scene'],'passed':[s['official_rule_pass'] for s in r['stages']]} for r in report]))
raise SystemExit(0 if all(r['all_passed'] for r in report) else 1)
