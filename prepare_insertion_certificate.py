"""Bind original hole boundary and full native star surface to object frames."""
from pathlib import Path
import json,numpy as np
from pxr import Usd,UsdGeom
from shapely.geometry import Polygon
root=Path('/home/vipuser/by/genie_sim')
stage=Usd.Stage.Open(str(root/'tasks/g2crsb90d_finemanip_insert_building_block/0/scene.usda'))
cache=UsdGeom.XformCache()
def actor(name):return np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Objects/'+name+'/entity'))).T
box=actor('benchmark_building_blocks_01e2de23');star=actor('benchmark_building_blocks_35570a3b')
data=json.loads((root/'output/insertion_geometry.json').read_text())
loops=[np.asarray(x['points']) for x in data['loops'] if np.ptp(np.asarray(x['points'])[:,:2],axis=0).max()<.08]
loop=min(loops,key=lambda x:Polygon(x[:,:2]).area/Polygon(x[:,:2]).convex_hull.area)
hole=(np.linalg.inv(box)@np.c_[loop,np.ones(len(loop))].T).T[:,:3]
triangles=[]
entity=stage.GetPrimAtPath('/World/Objects/benchmark_building_blocks_35570a3b/entity')
for prim in Usd.PrimRange(entity):
 if not prim.IsA(UsdGeom.Mesh):continue
 mesh=UsdGeom.Mesh(prim)
 if mesh.ComputeVisibility()=='invisible' or 'collision' in prim.GetName().lower():continue
 points=np.asarray(mesh.GetPointsAttr().Get());world=np.asarray(cache.GetLocalToWorldTransform(prim)).T
 points=(np.linalg.inv(star)@world@np.c_[points,np.ones(len(points))].T).T[:,:3]
 indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get());counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get());offset=0
 for count in counts:
  face=indices[offset:offset+count];offset+=count
  for i in range(1,len(face)-1):triangles.append(points[[face[0],face[i],face[i+1]]].tolist())
if not triangles:raise RuntimeError('No full star mesh')
result={'star':'benchmark_building_blocks_35570a3b','box':'benchmark_building_blocks_01e2de23',
 'hole_boundary_box_frame':hole.tolist(),'hole_plane_y_box_frame':float(hole[:,1].mean()),
 'star_triangles_entity_frame':triangles,'native_scale_preserved':True}
(root/'output/insertion-certificate.json').write_text(json.dumps(result))
print(json.dumps({'hole_vertices':len(hole),'star_triangles':len(triangles),'hole_plane_box_y':result['hole_plane_y_box_frame']}))
