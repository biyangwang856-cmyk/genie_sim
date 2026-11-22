"""Recompute final task relations from raw measured poses, independently of flags."""
import importlib.util,json
from itertools import product
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def pose(frame,name):return np.asarray(frame['objects']['World_Objects_'+name+'_entity']['pose'])


def cap_table_support(layout, frames, pen, cap):
    tables = [name for name, asset in layout.items() if asset.get('usd') == 'table_000']
    if len(tables) != 1 or len(frames) < 10:
        return False, {'table_candidates': tables, 'sample_count': len(frames)}

    table = tables[0]
    bounds = layout[table]['entity_local_bbox']
    lower, upper = np.asarray(bounds['min']), np.asarray(bounds['max'])
    corners = {}
    for name in (pen, cap):
        box = layout[name]['entity_local_bbox']
        axes = zip(box['min'], box['max'])
        corners[name] = np.asarray(list(product(*axes)))

    gaps, borders = [], []
    for frame in frames:
        table_pose = np.linalg.inv(pose(frame, table))
        points = []
        for name in (pen, cap):
            relative = table_pose @ pose(frame, name)
            points.append(corners[name] @ relative[:3, :3].T + relative[:3, 3])
        payload = np.concatenate(points)
        gaps.append(float(payload[:, 2].min() - upper[2]))
        borders.append(float(min((payload[:, :2] - lower[:2]).min(),
                                 (upper[:2] - payload[:, :2]).min())))

    # Local box corners conservatively enclose the rounded pen and cap. Their
    # predicted lowest point may lie slightly below the actual contact surface.
    metrics = {'table': table, 'sample_count': len(frames), 'gap_min_m': min(gaps),
               'gap_max_m': max(gaps), 'border_min_m': min(borders),
               'gap_limits_m': [-.005, .003], 'method': 'measured_poses_local_bounds'}
    supported = min(gaps) >= -.005 and max(gaps) <= .003 and min(borders) >= 0.
    return bool(supported), metrics


def audit(run,raw,frames):
    run=Path(run);launch=json.loads((run/'launch.json').read_text())
    scene=json.loads((run/'tasks'/launch['task']/launch['scene']/'scene_info.json').read_text());name=scene['task_template']
    tail=frames[-20:];checks={};metrics={}
    if name=='insert_building_block':
        c=json.loads((run/'source/insertion-certificate.json').read_text());obj,box=c['star'],c['box']
        local=[np.linalg.inv(pose(f,box))@pose(f,obj) for f in tail]
        checks['final_inside_box']=all(abs(t[0,3])<.060 and abs(t[2,3])<.060 and -.075<t[1,3]<.055 for t in local)
        from insertion_physics import measure
        checks['final_full_mesh_below_plane']=all(measure(pose(f,obj),pose(f,box),False,c)['max_plane_distance_m']<-.002 for f in tail)
        points=np.asarray([pose(f,obj)[:3,3] for f in tail]);metrics['final_motion_m']=float(np.linalg.norm(points-points[0],axis=1).max())
        checks['final_stable']=len(tail)>=10 and metrics['final_motion_m']<.003
    elif name=='put_the_cap_on_the_pen_new':
        target=scene['official_instruction']['target'];pen,cap=target['id1'],target['id2']
        contract=json.loads((run/'evidence/task-spec.json').read_text())['assembly']
        expected=contract['pen_tip_x_m']-contract['cap_mouth_x_m']+contract['insertion_depth_m'];relations=[]
        for f in tail:
            t=np.linalg.inv(pose(f,pen))@pose(f,cap);depth=float(t[0,3]-contract['pen_tip_x_m']+contract['cap_mouth_x_m'])
            half_width=contract['tip_half_width_m'] if depth<=contract['transition_start_depth_m'] else contract['transition_half_width_m'] if depth<=contract['barrel_start_depth_m'] else contract['barrel_half_width_m']
            clearance=contract['bore_half_width_m']-half_width
            relations.append({'depth_m':depth,'axial_error_m':abs(float(t[0,3])-expected),
                'radial_error_m':float(np.linalg.norm(t[1:3,3])),'angle_rad':float(Rotation.from_matrix(t[:3,:3]).magnitude()),'radial_clearance_m':clearance})
        checks['cap_seated']=len(tail)>=10 and all(r['axial_error_m']<.004 and r['radial_error_m']<r['radial_clearance_m'] and r['angle_rad']<.035 for r in relations)
        checks['pen_returned_to_table'], metrics['table_support'] = cap_table_support(scene['layout'], tail, pen, cap)
        movements = {}
        for name in (pen, cap):
            points = np.asarray([pose(frame, name)[:3, 3] for frame in tail])
            movements[name] = float(np.linalg.norm(points - points[0], axis=1).max())
        checks['pen_stable'] = all(movement < .003 for movement in movements.values())
        metrics['final_motion_m'] = movements
        gc=json.loads((run/'evidence/gripper-contract.json').read_text())['grippers'];hand_errors={}
        for side,c in gc.items():
            values=[]
            for f in tail:
                j=f['robot']['joints'];mapping=dict(zip(j['joint_name'],j['joint_position']))
                values.extend(abs(mapping[n]-q) for n,q in zip(c['joint_names'],c['opened_positions']))
            hand_errors[side]=max(values)
        checks['both_hands_physically_open']=all(v<.03 for v in hand_errors.values())
        metrics.update(final_relations=relations,gripper_open_errors=hand_errors,official_snap_trigger='relative_pose_threshold')
    elif name=='phone_packaging':
        path=run/'source/precise_geometry.py';module=importlib.util.spec_from_file_location('independent_official_geometry',path)
        geometry=importlib.util.module_from_spec(module);module.loader.exec_module(geometry)
        stages=[]
        for index in [1,3,5]:
            rule_set=scene['official_subtask_steps'][index]['checker']['MixedRules'];verdicts=[]
            for f in tail:
                measurements=[]
                for rule in rule_set['rules']:
                    p=rule['params'];reference=pose(f,p['reference']);relative=np.linalg.inv(reference)@pose(f,p['object'])
                    if rule['name']=='InsidePrecise':
                        result=geometry.containment(geometry.transformed(np.asarray(p['points']),relative),p['lower'],p['upper'],p.get('quantifier','all'))
                    elif rule['name']=='CoverPrecise':
                        target=pose(f,p.get('target_object',p['reference']))
                        result=geometry.coverage(geometry.transformed(np.asarray(p['source_points']),relative),
                            geometry.transformed(np.asarray(p['target_points']),np.linalg.inv(reference)@target),
                            p['up_axis'],p['threshold'],p.get('max_gap'),float(relative[p['up_axis'],p['normal_axis']]) if 'normal_axis' in p else None,p.get('min_normal_dot'))
                    else:raise ValueError('Unknown official relation')
                    measurements.append(result['pass'])
                verdicts.append(all(measurements) if rule_set.get('mode','or')=='and' else any(measurements))
            stages.append({'stage':index,'all_final_frames_pass':all(verdicts)})
        checks['official_three_stage_geometry']=all(s['all_final_frames_pass'] for s in stages);metrics['stages']=stages
        objects=['benchmark_phone_000','benchmark_packaging_box_000','benchmark_packaging_box_002'];motions={}
        for obj in objects:
            p=np.asarray([pose(f,obj)[:3,3] for f in tail]);motions[obj]=float(np.linalg.norm(p-p[0],axis=1).max())
        checks['assembly_stable']=len(tail)>=10 and all(v<.003 for v in motions.values());metrics['final_motion_m']=motions
    else:checks['supported_task']=False
    result={'passed':bool(checks) and all(checks.values()),'checks':checks,'metrics':metrics,'measurement_source':str(Path(raw)/'state.json')}
    (run/'independent-task-geometry.json').write_text(json.dumps(result,indent=2));return result
