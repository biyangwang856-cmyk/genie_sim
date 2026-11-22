from pathlib import Path
import json,sys,numpy as np
from pxr import Usd,UsdGeom
root=Path('/home/vipuser/by/genie_sim');sys.path.insert(0,str(root))
from insertion_physics import measure
run=Path('/data/by/genie_sim/collection-runs/block-scene0-calibrated08')
events=[json.loads(x) for x in (run/'evidence/stage-evidence.jsonl').read_text().splitlines()]
scene=Usd.Stage.Open(str(run/'tasks/g2crsb90d_finemanip_insert_building_block/0/scene.usda'))
box=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(scene.GetPrimAtPath('/World/Objects/benchmark_building_blocks_01e2de23/entity'))).T
rows=[e for e in events if e['phase']=='star_measured_entry']
for i,r in enumerate(rows[:3]):print('RECORDED_RELEASE_CANDIDATE',i,measure(np.asarray(r['actual']),box))
