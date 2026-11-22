"""Inspect authored physical collision shapes at the insertion bottleneck."""
from pathlib import Path
import json,numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
root=Path('/home/vipuser/by/genie_sim')
folder=root/'tasks/g2crsb90d_finemanip_insert_building_block/0'
stage=Usd.Stage.Open(str(folder/'scene.usda'))
report=[]
for label in ['benchmark_building_blocks_01e2de23','benchmark_building_blocks_35570a3b']:
 prim=stage.GetPrimAtPath('/World/Objects/'+label+'/entity')
 box=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
 for p in Usd.PrimRange(prim):
  if p.HasAPI(UsdPhysics.CollisionAPI):
   enabled=UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
   approximate=UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() if p.HasAPI(UsdPhysics.MeshCollisionAPI) else None
   bound=box.ComputeWorldBound(p).ComputeAlignedRange()
   report.append({'path':str(p.GetPath()),'enabled':enabled,'approximation':approximate,
     'world_bounds':[list(bound.GetMin()),list(bound.GetMax())],'schemas':p.GetAppliedSchemas()})
print(json.dumps(report,indent=2))
(root/'jobs/insertion-physical-colliders.json').write_text(json.dumps(report,indent=2))
