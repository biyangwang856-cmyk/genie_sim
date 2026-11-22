"""Fit the complete current asset mesh, without resizing, in the actual aperture."""
from pathlib import Path
import json,numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import affinity
from scipy.optimize import minimize
root=Path('/home/vipuser/by/genie_sim');c=json.loads((root/'output/insertion-certificate.json').read_text())
triangles=np.asarray(c['star_triangles_entity_frame']);hole=Polygon(np.asarray(c['hole_boundary_box_frame'])[:,[0,2]])
polys=[Polygon(t[:,[0,2]]) for t in triangles]
surface=unary_union([p for p in polys if p.area>1e-12]);cent=np.asarray(surface.centroid.coords[0]);hc=np.asarray(hole.centroid.coords[0])
surface=affinity.translate(surface,*(-cent))
def cost(v):
 p=affinity.rotate(surface,float(v[0]),origin=(0,0),use_radians=True)
 p=affinity.translate(p,float(hc[0]+v[1]),float(hc[1]+v[2]))
 return p.difference(hole).area/surface.area
angles=np.linspace(0,2*np.pi,1440,endpoint=False);scans=[cost([a,0,0]) for a in angles]
candidates=[]
for i in np.argsort(scans)[:8]:
 r=minimize(cost,[angles[i],0,0],method='Nelder-Mead',options={'xatol':1e-10,'fatol':1e-10,'maxiter':2000})
 candidates.append(r)
best=min(candidates,key=lambda r:r.fun);angle,dx,dz=best.x
co,si=np.cos(angle),np.sin(angle)
r=np.array([[co,0,-si],[0,1,0],[si,0,co]])
translation=np.r_[hc[0]+dx,0,hc[1]+dz]-r@np.r_[cent[0],0,cent[1]]
translation[1]=c['hole_plane_y_box_frame']+.003-float(triangles[:,:,1].min())
goal=np.eye(4);goal[:3,:3]=r;goal[:3,3]=translation
old=json.loads((root/'output/insertion_target.json').read_text())
report={'native_scale_preserved':True,'surface_area':surface.area,'hole_area':hole.area,
 'min_scan_outside_fraction':float(min(scans)),'best_outside_fraction':float(best.fun),'optimization':best.x.tolist(),
 'old_goal':old['object_goal_in_box_frame'],'new_goal':goal.tolist()}
(root/'jobs/insertion-target-refit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if best.fun>.001:raise RuntimeError('Current full native mesh does not fit: no change deployed')
updated={**old,'object_goal_in_box_frame':goal.tolist(),'silhouette_outside_hole_area_fraction':float(best.fun),
 'geometry_source':'Full downloaded asset surface, fitted in box-local frame without resizing',
 'star_half_thickness':float(np.ptp(triangles[:,:,1])/2),
 'star_center_entity':[float(cent[0]),float((triangles[:,:,1].min()+triangles[:,:,1].max())/2),float(cent[1])]}
(root/'output/insertion_target_v2.json').write_text(json.dumps(updated,indent=2))
