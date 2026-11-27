"""Full mesh projection and physical crossing of the official star aperture."""
from pathlib import Path
import json,os
import numpy as np
from shapely.geometry import Polygon,LineString
from shapely.ops import unary_union
ROOT=Path(os.environ.get('GENIE_PROJECT_ROOT','/home/vipuser/by/genie_sim'))
def certificate():
 source=Path(os.environ.get('GENIE_RUN_SOURCE',ROOT))/'insertion-certificate.json'
 return json.loads((source if source.exists() else ROOT/'output/insertion-certificate.json').read_text())
def measure(star_world,box_world,full_projection=True,contract=None):
 c=contract or certificate();points=np.asarray(c['star_triangles_entity_frame']);shape=points.shape
 relative=np.linalg.inv(box_world)@star_world
 points=(relative@np.c_[points.reshape(-1,3),np.ones(points.size//3)].T).T[:,:3].reshape(shape)
 hole=np.asarray(c['hole_boundary_box_frame'])
 result={'min_plane_distance_m':float(points[:,:,1].min()-c['hole_plane_y_box_frame']),
  'max_plane_distance_m':float(points[:,:,1].max()-c['hole_plane_y_box_frame'])}
 if full_projection:
  aperture=Polygon(hole[:,[0,2]])
  polys=[Polygon(t[:,[0,2]]) for t in points]
  surface=unary_union([p for p in polys if p.area>1e-12])
  result['outside_hole_area_fraction']=float(surface.difference(aperture).area/surface.area)
  result['inside_aperture_margin_m']=float(surface.distance(aperture.boundary)) if aperture.covers(surface) else 0.
 return result
def crossing_report(path,contract=None):
 c=contract or certificate();rows=[];slice_checks=[];tolerance=.0005
 triangles=np.asarray(c['star_triangles_entity_frame']);homogeneous=np.c_[triangles.reshape(-1,3),np.ones(triangles.size//3)]
 aperture=Polygon(np.asarray(c['hole_boundary_box_frame'])[:,[0,2]]).buffer(tolerance)
 for line in Path(path).read_text().splitlines():
  frame=json.loads(line)
  if c['star'] in frame['poses'] and c['box'] in frame['poses']:
   star=np.asarray(frame['poses'][c['star']]);box=np.asarray(frame['poses'][c['box']])
   if not np.isfinite(star).all() or not np.isfinite(box).all():raise ValueError('Nonfinite physical mesh pose')
   value=measure(star,box,False,c)
   segment=rows[-1]['tracking_segment'] if rows and 0<frame['sim_time']-rows[-1]['sim_time']<=.02 else (rows[-1]['tracking_segment']+1 if rows else 0)
   rows.append({'sim_time':frame['sim_time'],'tracking_segment':segment,**value})
   if value['min_plane_distance_m']<=0<=value['max_plane_distance_m']:
    points=(np.linalg.inv(box)@star@homogeneous.T).T[:,:3].reshape(triangles.shape)
    distances=points[:,:,1]-c['hole_plane_y_box_frame'];segments=[]
    for tri,d in zip(points,distances):
     if d.min()>0 or d.max()<0:continue
     cuts=[]
     for a,b in [(0,1),(1,2),(2,0)]:
      if abs(d[a])<1e-9:cuts.append(tri[a,[0,2]])
      if d[a]*d[b]<0:cuts.append((tri[a]+(tri[b]-tri[a])*(-d[a])/(d[b]-d[a]))[[0,2]])
     unique=[]
     for point in cuts:
      if not any(np.linalg.norm(point-other)<1e-9 for other in unique):unique.append(point)
     if len(unique)>=2:segments.append(LineString(unique[:2]))
    total=sum(s.length for s in segments);outside=sum(s.difference(aperture).length for s in segments)
    slice_checks.append({'sim_time':frame['sim_time'],'tracking_segment':segment,'segments':len(segments),'outside_fraction':outside/max(total,1e-12),'passed':bool(segments) and outside/max(total,1e-12)<=.001})
 crossed=[];attempts=[]
 for index,(before,after) in enumerate(zip(rows,rows[1:]),start=1):
  if before['tracking_segment']!=after['tracking_segment'] or not (before['min_plane_distance_m']>=0 and after['min_plane_distance_m']<0):continue
  crossed.append(after['sim_time']);complete=None;reason='tracking_ended_before_complete_passage'
  for row in rows[index:]:
   if row['tracking_segment']!=after['tracking_segment']:break
   if row['min_plane_distance_m']>=0:reason='withdrawn_above_plane';break
   if row['max_plane_distance_m']<-.002:complete=row;reason='fully_below_plane';break
  end=complete['sim_time'] if complete else float('inf')
  checks=[s for s in slice_checks if s['tracking_segment']==after['tracking_segment'] and after['sim_time']<=s['sim_time']<=end]
  passed=complete is not None and len(checks)>=2 and all(s['passed'] for s in checks)
  attempts.append({'tracking_segment':after['tracking_segment'],'start_time_s':after['sim_time'],
    'fully_below_time_s':complete['sim_time'] if complete else None,'slice_checks':len(checks),'passed':passed,'reason':reason,
    'max_outside_fraction':max((s['outside_fraction'] for s in checks),default=None)})
 fully_below=[r for r in rows if r['max_plane_distance_m']<-.002]
 # A continuous episode can physically recover from an incomplete rim contact.
 # Certify a complete passage, retaining every earlier failed contact as evidence.
 # The final tracked pose must still be fully below; no rollback is admitted here.
 slice_pass=any(a['passed'] for a in attempts)
 final_below=bool(rows) and rows[-1]['max_plane_distance_m']<-.002
 return {'physics_samples':len(rows),'first_crossing_times_s':crossed,
  'fully_below_plane':bool(fully_below),'first_fully_below_time_s':fully_below[0]['sim_time'] if fully_below else None,
  'correct_aperture_slice_passed':slice_pass,'slice_tolerance_m':tolerance,'slice_checks':slice_checks,
  'all_contact_slices_passed':bool(slice_checks) and all(s['passed'] for s in slice_checks),
  'passage_attempts':attempts,'final_fully_below_plane':final_below,
  'passed':bool(crossed and fully_below and slice_pass and final_below),'samples':rows}
