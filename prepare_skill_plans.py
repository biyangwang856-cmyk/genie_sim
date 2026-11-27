"""Server-side real Qwen/embedding preflight; creates plans, never simulates success."""
import argparse,json,time,uuid
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from asset_retrieval import load_index
from container_goal import TASK_ID as CONTAINER_TASK, asset_gap, validate_goal
from robot_adapters import ROOT,load_robot,UnsupportedRequest
from task_geometry import task_spec
from structured_planner import StructuredPlanner
from qwen_backend import QwenAPI
from scene_contract import scene_request,count

from collection_scope import DYNAMIC_TASKS, SCENES, TASKS
CURRENT_JOB=None

def prepare_scene_plan(adapter,spec,scene,folder,instruction):
    folder.mkdir(parents=True)
    poses={}
    for obj in spec['manipulated_objects']:
        info=scene['layout'][obj]
        transform=np.eye(4)
        transform[:3,3]=info['xyz']
        transform[:3,:3]=Rotation.from_quat(info['xyzw']).as_matrix()
        poses[obj]=transform.tolist()
    state={'holding_by_role':{},'achieved':[],'object_poses':poses,
           'state_source':'scene_metadata_preflight; runtime revalidation required'}
    planner=StructuredPlanner(adapter,spec,folder,instruction)
    failure=None
    for _ in range(3):
        try:
            plan,digest=planner.plan(state,failure=failure)
            break
        except UnsupportedRequest:
            raise
        except ValueError as exc:
            failure={'type':'plan_contract_violation','details':str(exc)}
    else:
        raise RuntimeError('Plan contract budget exhausted')
    archive=folder/('llm-plan-'+str(planner.calls)+'.json')
    if not archive.is_file() or json.loads(archive.read_text()).get('plan_sha256')!=digest:
        raise RuntimeError('Prepared plan archive does not match the validated model response')
    return {'path':str(archive),'plan_sha256':digest},len(plan['steps'])

def retrieval(api,queries,specs):
    """Rank local USD assets, then enforce each official scene's assembly bindings."""
    records,matrix=load_index()
    by_id={item['asset_id']:i for i,item in enumerate(records)}
    selected={}
    missing=set()
    for spec in specs:
        scene_assets={}
        for instance in sorted(set(spec['manipulated_objects']) | set(spec['bindings'].values())):
            asset=spec['objects'][instance]
            asset_id=asset['asset_id']
            index=by_id.get(asset_id)
            if index is None or not Path(records[index]['usd']).is_file():
                missing.add(asset_id)
                continue
            scene_assets[instance]={'asset_id':asset_id,'bbox':asset['bbox'],
                'grasp_profiles':spec['grasp_profiles'].get(instance,[]),
                'interaction_metadata_available':Path(records[index]['interaction']).is_file()}
        selected[spec['scene_id']]=scene_assets
    if missing:
        raise UnsupportedRequest('Official scene requires assets unavailable in the local catalog',missing_assets=sorted(missing))

    vectors,receipt=api.embed(queries)
    vectors=np.asarray(vectors,dtype=matrix.dtype)
    if vectors.shape!=(len(queries),matrix.shape[1]) or not np.isfinite(vectors).all():
        raise ValueError('Invalid task embedding response')
    norm=np.linalg.norm(vectors,axis=1,keepdims=True)
    if np.any(norm==0):
        raise ValueError('Empty task embedding response')
    scores=(vectors/norm) @ matrix.T
    bound_ids=sorted({asset['asset_id'] for scene in selected.values() for asset in scene.values()})
    results=[]
    for query,row in zip(queries,scores,strict=True):
        ranked=np.argsort(-row,kind='stable')
        ranks=np.empty(len(ranked),dtype=int)
        ranks[ranked]=np.arange(1,len(ranked)+1)
        results.append({'query':query,
            'candidates':[{'asset_id':records[i]['asset_id'],'similarity':float(row[i])} for i in ranked[:10]],
            'bound_asset_ranking':{asset_id:{'rank':int(ranks[by_id[asset_id]]),'similarity':float(row[by_id[asset_id]])}
                                   for asset_id in bound_ids}})
    return {'model':'text-embedding-v4','receipt':receipt,'queries':results,'scene_constrained_selection':selected,
        'policy':'Semantic ranking locates candidates. Official asset bindings and geometry constrain the plan; reachability and collision are verified at execution.'}

def main():
    global CURRENT_JOB
    p=argparse.ArgumentParser();p.add_argument('instruction',nargs='?');p.add_argument('--instruction-file');p.add_argument('--robot-id',default='g2crsb90d')
    p.add_argument('--tasks',nargs='+',default=TASKS)
    p.add_argument('--scenes',nargs='+',choices=SCENES,default=SCENES)
    a=p.parse_args();adapter=load_robot(a.robot_id)
    intent_queries={}
    scene_options=None
    quota_override=None
    trajectory=None
    goals={}
    if a.instruction_file:
        path=Path(a.instruction_file);value=path.read_text(encoding='utf-8')
        if path.suffix=='.json':
            request=json.loads(value)
            a.instruction=request['instruction']
            intent_queries=request.get('asset_queries',{})
            goals=request.get('goals',{})
            if request.get('scene_request') is not None:scene_options=scene_request(request['scene_request'])
            if request.get('successes_per_scene') is not None:
                quota_override=count(request['successes_per_scene'],'successes_per_scene',12)
            if request.get('trajectory_request') is not None:
                from trajectory_diversity import trajectory_request
                trajectory=trajectory_request(request['trajectory_request'])
        else:
            a.instruction=value
    if not a.instruction:p.error('instruction or --instruction-file is required')
    unknown=set(a.tasks)-set(TASKS)-set(DYNAMIC_TASKS)
    if unknown:raise UnsupportedRequest('Task has no supported asset/skill contract',missing_skills=sorted(unknown))
    jobid=time.strftime('%Y%m%d-%H%M%S')+'-skills-'+uuid.uuid4().hex[:16]
    output=ROOT/'jobs'/jobid;output.mkdir()
    CURRENT_JOB=output/'job.json'
    job={'job_id':jobid,'instruction':a.instruction,'robot_id':a.robot_id,'status':'preparing_skill_plans',
        'success_counts':{},'task_bindings':{},'plan':{'robot':a.robot_id,'tasks':[]},'attempts':[],
        'execution_mode':'structured','prepared_skill_plans':{},'asset_retrieval':{},'preflight_only':True}
    (output/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2))
    api=QwenAPI()
    if scene_options:
        job.update(scene_request=scene_options,scene_preflight={},automatic_admission=True,
                   require_distinct_motion=True,status='composing')
        from scene_composer import prepare_scenes
    if trajectory is not None:
        job['trajectory_request']=trajectory
        job['require_distinct_motion']=trajectory['duplicate_policy']=='distinct'
        if trajectory['duplicate_policy']=='replay':
            from collection_control import STORAGE
            from trajectory_replay import populate_replay
            if len(a.tasks)!=1:
                raise ValueError('A replay reference can select only one task')
            name=a.tasks[0]
            populate_replay(job,storage=STORAGE,asset_root=ROOT/'GenieSimAssets',
                            task_name=name,scene_options=scene_options,per_scene=quota_override,goal=goals.get(name))
            CURRENT_JOB.write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'job_id':jobid,'path':str(CURRENT_JOB),'status':job['status']}))
            return
    for name in a.tasks:
        source_id=adapter.config['scene_task_prefix']+name;job['prepared_skill_plans'][source_id]={}
        location=ROOT/'tasks'/source_id
        scene_ids=list(a.scenes)
        scene_sources={}
        reserve_ids=[]
        container_asset_receipt=None
        if name==CONTAINER_TASK and not scene_options:
            raise ValueError("Containment goals require a prepared tabletop scene")
        if scene_options:
            location=output/'scenes'/source_id
            selection=None
            if name==CONTAINER_TASK:
                from container_scene import choose_pair
                goal=validate_goal(goals[name],a.instruction)
                selection,container_asset_receipt=choose_pair(goal,adapter,api,ROOT/'GenieSimAssets')
                job['asset_retrieval'][name]=container_asset_receipt
            additions,report=prepare_scenes(name,adapter,scene_options,location,api,
                                           trajectory=trajectory,goal=goals.get(name) if selection is None else None,
                                           container_selection=selection)
            if selection is not None and not additions:
                raise asset_gap("找到了匹配物体和容器，但无法编译可用于执行的桌面场景",
                                ["compatible_tabletop_layout"], code="ASSET_GEOMETRY_INCOMPATIBLE",
                                candidates=report["rejected"])
            job['scene_preflight'][name]=report
            scene_sources={item['scene_id']:item for item in additions}
            scene_ids=list(scene_sources)
            for item in report['reserve_candidates']:
                scene_sources[item['scene_id']]=item
                reserve_ids.append(item['scene_id'])
        job['success_counts'][source_id]={s:0 for s in scene_ids}
        binding={'template':name,'scenes':scene_ids.copy(),'scene_sources':scene_sources,'reserves':reserve_ids.copy()}
        job['task_bindings'][source_id]=binding
        totals=trajectory.get('total_episodes') if trajectory is not None else None
        if totals is not None:
            from trajectory_diversity import episode_quotas
            quotas=episode_quotas(scene_ids,total_episodes=totals,per_scene=quota_override,
                                  requested_scene_count=scene_options['scene_count'] if scene_options else len(a.scenes))
        elif name==CONTAINER_TASK:
            quotas=[{'scene_id':s,'successes':quota_override or 1} for s in scene_ids]
        else:
            quotas=[{'scene_id':s,'successes':quota_override or
                     (5 if scene_options and scene_options['mode']!='existing' else
                      (3 if scene_sources.get(s,{}).get('source_scene_id',s)=='0' else 2))}
                    for s in scene_ids]
        job['plan']['tasks'].append({'task_id':source_id,'scene_quotas':quotas})
        scenes={}
        for scene_id in [*scene_ids,*reserve_ids]:
            scene=json.loads((location/scene_id/'scene_info.json').read_text(encoding='utf-8'));spec=task_spec(scene,a.robot_id)
            adapter.require(spec['required_capabilities'])
            scenes[scene_id]=(scene,spec)
        if not scenes:
            (output/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
            continue
        queries=intent_queries.get(name,[])
        if not isinstance(queries,list) or any(not isinstance(query,str) for query in queries):
            raise ValueError('Asset queries must be a list of descriptions')
        queries=list(dict.fromkeys(query.strip()[:512] for query in queries if query.strip()))[:8]
        if not queries:
            queries=[next(iter(scenes.values()))[1]['official_instruction']]
        retrieved=container_asset_receipt or retrieval(api,queries,[spec for _,spec in scenes.values()])
        job['asset_retrieval'][name]=retrieved
        (output/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2))
        for scene_id,(scene,spec) in scenes.items():
            if container_asset_receipt is not None:
                spec['resolved_assets']=retrieved['selected']
            else:
                spec['resolved_assets']=retrieved['scene_constrained_selection'][scene_id]
            folder=output/'plans'/name/scene_id
            try:
                prepared,steps=prepare_scene_plan(adapter,spec,scene,folder,a.instruction)
            except (ValueError,RuntimeError,OSError) as exc:
                if scene_id not in reserve_ids:
                    raise
                binding['reserves'].remove(scene_id)
                job['scene_preflight'][name].setdefault('reserve_plan_rejected',[]).append(
                    {'scene_id':scene_id,'reason':str(exc)})
                continue
            job['prepared_skill_plans'][source_id][scene_id]=prepared
            print(json.dumps({'task':name,'scene':scene_id,'validated_plan':True,'steps':steps,'plan_sha256':prepared['plan_sha256']}),flush=True)
            (output/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2))
    job['status']='skill_plans_prepared' if any(task['scene_quotas'] for task in job['plan']['tasks']) else 'scene_search_exhausted'
    (output/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'job_id':jobid,'path':str(output/'job.json'),'status':job['status']}))

if __name__=='__main__':
    import sys
    try:main()
    except Exception as exc:
        unsupported=isinstance(exc,UnsupportedRequest)
        failure=exc.details if unsupported else {'status':'failed','reason':str(exc)}
        if CURRENT_JOB and CURRENT_JOB.exists():
            job=json.loads(CURRENT_JOB.read_text());job['status']='unsupported' if unsupported else 'preparation_failed';job['failure']=failure
            CURRENT_JOB.write_text(json.dumps(job,ensure_ascii=False,indent=2))
        print(json.dumps(failure,ensure_ascii=False));sys.exit(2 if unsupported else 1)
