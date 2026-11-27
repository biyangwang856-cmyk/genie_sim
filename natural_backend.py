"""Callable server entry: request, robot, task and scene are independent inputs."""
import argparse,json,subprocess,uuid
import re
from pathlib import Path
from robot_adapters import ROOT,load_robot,UnsupportedRequest
from collection_scope import DYNAMIC_TASKS, SCENES, TASKS
from qwen_backend import QwenAPI
from scene_contract import count, scene_request

_CHINESE_COUNTS = dict(zip('一二三四五六七八九', range(1, 10)))
_CHINESE_COUNTS.update({'两':2, '十':10, '十一':11, '十二':12})
_ENGLISH_COUNTS = dict(zip('one two three four five six seven eight nine ten eleven twelve'.split(), range(1, 13)))


def requested_distinct_scene_count(instruction):
    """Retain explicit scene cardinality independently of the model's episode count."""
    zh=re.findall(r'(\d+|十[一二]?|[一二两三四五六七八九])\s*(?:个|套|种|条)\s*'
                  r'(?:各不相同|不同|互异)的?\s*(?:现有|已有|原有)?(?:的)?\s*(?:场景|布局|房间|桌子)',
                  instruction)
    en=re.findall(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+'
                  r'(?:different|distinct|unique)\s+(?:existing\s+)?(?:scenes|layouts|rooms|tables)\b',
                  instruction,re.I)
    values={int(token) if token.isdecimal() else _CHINESE_COUNTS[token] for token in zh}
    values.update(int(token) if token.isdecimal() else _ENGLISH_COUNTS[token.lower()] for token in en)
    if len(values)>1:
        raise ValueError('Multiple distinct-scene counts require an unambiguous scene_request')
    return count(values.pop(),'scene_count',12) if values else None


def requested_containment_total(instruction):
    """Keep an explicit total when the containment parser omits it."""
    if re.search(r'每(?:个|种|套)场景|各(?:采集|生成|录制|采)|per scene|each scene', instruction, re.I):
        return None
    zh=re.findall(r'(?:生成|采集|录制|创建|需要|想要|我要|采)\s*(?:总共|合计|共)?\s*'
                  r'(\d+|十[一二]?|[一二两三四五六七八九])\s*条', instruction)
    en=re.findall(r'\b(?:generate|record|collect|produce)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)'
                  r'\s+(?:successful\s+)?(?:trajector(?:y|ies)|episode(?:s)?)\b', instruction,re.I)
    values={int(token) if token.isdecimal() else _CHINESE_COUNTS[token] for token in zh}
    values.update(int(token) if token.isdecimal() else _ENGLISH_COUNTS[token.lower()] for token in en)
    if len(values)>1:
        raise ValueError('Multiple trajectory totals require one explicit total_episodes')
    return count(values.pop(),'total_episodes',144) if values else None


def result_for(job_path):
    result=json.loads(job_path.read_text(encoding='utf-8'))
    return {'job_id':result['job_id'],'job_path':str(job_path),'status':result['status'],
            'dataset':result.get('dataset'),'dataset_failure':result.get('dataset_failure'),
            'partial_dataset':result.get('partial_dataset'),
            'success_counts':result.get('success_counts',{}),'scene_preflight':result.get('scene_preflight'),
            'raw_trajectory_path':result.get('raw_trajectory_path'),
            'trajectory_outcome':result.get('trajectory_outcome')}

def resume(job_id, *, execute=True):
    from scene_contract import require_id
    job_path=ROOT/'jobs'/require_id(job_id)/'job.json'
    job=json.loads(job_path.read_bytes())
    if job.get('job_type')=='skill_onboarding':
        from skill_onboarding import resume as resume_skill
        return resume_skill(job_id,execute=execute)
    if not job.get('automatic_admission') or job.get('job_id')!=job_id:
        raise ValueError('Job is not eligible for automatic restart')
    if execute and job['status'] not in ('complete','stopped_by_user'):
        subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'batch_collect.py'),'--job',str(job_path),
                        '--wait-for-worker'],check=True)
    return result_for(job_path)

def resolve_intent(api,instruction,catalog,robot,explicit,quota):
    from trajectory_diversity import trajectory_request

    receipts=[]
    repair=None
    scene_count=requested_distinct_scene_count(instruction)
    if scene_count is not None and explicit and explicit.get('scene_count')!=scene_count:
        raise ValueError('scene_request conflicts with the explicitly requested number of different scenes')
    for _ in range(2):
        plan,receipt=api.plan(instruction,catalog,robot=robot,resolve_scene=True,
                              scene_constraints=explicit,episode_quota=quota,repair=repair)
        receipts.append(receipt)
        if plan.get('status')=='unsupported':
            return plan,None,receipts
        if plan.get('status')=='skill_request':
            if scene_count is not None:
                repair={'error':f'The user explicitly requested {scene_count} distinct scenes. Preserve the scene count'}
                continue
            goal=plan.get('skill_goal')
            if isinstance(goal,dict) and goal.get('skill')=='open_hinged_lid':
                if goal.get('target_position')!='open' or set(goal)-{'skill','actor','target_position'}:
                    repair={'error':'Use target_position="open" for the physical open endpoint. '
                                    'The trusted asset binding provides the signed joint limit. '
                                    'If the user requests a different endpoint, preserve it as unsupported.',
                            'skill':'open_hinged_lid'}
                    continue
                plan={**plan,'skill_goal':{key:value for key,value in goal.items() if key!='target_position'}}
            return plan,None,receipts
        try:
            proposed=scene_request(plan['scene_request']) if 'scene_request' in plan else None
            if scene_count is not None and (proposed or explicit or {}).get('scene_count')!=scene_count:
                raise ValueError(f'The instruction requires scene_count={scene_count}. This is separate from the episode count')
            if 'successes_per_scene' in plan:
                count(plan['successes_per_scene'],'successes_per_scene',12)
            if 'trajectory_request' in plan:
                plan['trajectory_request'] = trajectory_request(plan['trajectory_request'], instruction=instruction)
            from goal_composition import packing_goal
            for entry in plan.get('tasks', []):
                if isinstance(entry, dict) and 'goal' in entry:
                    if entry.get('task_id') != 'phone_packaging':
                        raise ValueError('Goal must select a compatible asset template')
                    entry['goal'] = packing_goal(entry['goal'])
            change_layout=re.search(r'重新摆放|不同的?摆放|改变物体位置|\brearrange\b|\breposition\b',instruction,re.I)
            preserve_layout=re.search(r'(?:不|无需|不用|不要)重新摆放|(?:保持|保留)原(?:有|来的)?布局|(?:do not|don.t) (?:rearrange|reposition)',instruction,re.I)
            if (not explicit and proposed and proposed["mode"]=="compose_from_library"
                    and change_layout and not preserve_layout):
                proposed["variation"]=list(dict.fromkeys([*proposed["variation"],"layout"]))
            return plan,proposed,receipts
        except UnsupportedRequest:
            raise
        except ValueError as exc:
            repair={'error':str(exc),'scene_request':plan.get('scene_request'),
                    'successes_per_scene':plan.get('successes_per_scene'),
                    'trajectory_request':plan.get('trajectory_request')}
    if 'skill' in repair:
        raise UnsupportedRequest(repair['error'],missing_skills=[repair['skill']],
                                 missing_capabilities=['validated_joint_target'])
    raise UnsupportedRequest('Could not express the scene requirement in the validated scene request: '+repair['error'],
                             missing_capabilities=['supported_scene_layout_constraints'])

def submit(instruction,*,robot_id='g2crsb90d',task_ids=None,scenes=SCENES,execute=False,on_prepared=None,
           scene_options=None,successes_per_scene=None,trajectory_options=None,skill_goal=None):
    from container_goal import TASK_ID as CONTAINER_TASK
    from trajectory_diversity import resolve_trajectory_request

    adapter=load_robot(robot_id)
    if not isinstance(instruction,str) or not instruction.strip():raise ValueError('Nonempty natural-language instruction is required')
    if skill_goal is not None:
        if (task_ids or tuple(scenes)!=tuple(SCENES) or scene_options is not None
                or successes_per_scene is not None or trajectory_options is not None):
            raise ValueError('Skill validation accepts a single trusted goal without trajectory quotas')
        from skill_onboarding import prepare
        return prepare(skill_goal,robot_id,execute=execute,on_prepared=on_prepared,instruction=instruction)
    if set(scenes)-set(SCENES):raise UnsupportedRequest('Requested scene not registered',missing_assets=sorted(set(scenes)-set(SCENES)))
    available_tasks=set(TASKS)|set(DYNAMIC_TASKS)
    if task_ids is not None and (not task_ids or set(task_ids)-available_tasks):
        raise UnsupportedRequest('Explicit task IDs are not registered',missing_skills=sorted(set(task_ids or [])-available_tasks))
    from task_plugins import registered
    plugins=registered()
    catalog=[{'task_id':name,'description':plugins[name]['description'] if name in plugins else
        json.loads((ROOT/'tasks'/(adapter.config['scene_task_prefix']+name)/'0/scene_info.json').read_text(encoding='utf-8'))['official_instruction']['instruction']}
        for name in TASKS]
    for item in catalog:
        official=ROOT/'official-task-scenes'/item['task_id']
        candidates={path.parent.name for path in official.glob('*/scene.usda') if path.parent.name.isdecimal()}
        item['existing_scene_candidates']=sorted(set(SCENES) | candidates)
    if scene_options is not None:
        scene_options=scene_request(scene_options)
    scene_count=requested_distinct_scene_count(instruction)
    explicit_trajectory=resolve_trajectory_request(trajectory_options,instruction=instruction,
                                                   distinct_scene_count=scene_count)
    if explicit_trajectory.get('duplicate_policy')=='replay':
        from collection_control import STORAGE
        from trajectory_replay import reference_task
        name=reference_task(explicit_trajectory['reference_run'],storage=STORAGE,
                            asset_root=ROOT/'GenieSimAssets',robot_id=robot_id)
        plan={'robot':robot_id,'tasks':[{'task_id':name}],'trajectory_request':explicit_trajectory}
        proposed,intent_receipts=None,[]
        if scene_options is None and tuple(scenes)!=tuple(SCENES):
            scene_options=scene_request({'mode':'existing','scene_ids':list(scenes)})
    else:
        api=QwenAPI()
        plan,proposed,intent_receipts=resolve_intent(api,instruction,catalog,adapter.describe(),
                                                    scene_options,successes_per_scene)
        explicit_container = task_ids == [CONTAINER_TASK]
        probe_container = not task_ids and (plan.get('status')=='unsupported' or len(plan.get('tasks', []))==1)
        if explicit_container or probe_container:
            from container_goal import matches_phone_packaging, resolve_goal, skill_requirements
            candidate, receipt = resolve_goal(api, instruction, adapter, scene_options, successes_per_scene)
            intent_receipts.append(receipt)
            if candidate['status']=='container_goal':
                candidate_scene = scene_request(candidate['scene_request']) if 'scene_request' in candidate else None
                if scene_count is not None and (candidate_scene or proposed or scene_options or {}).get('scene_count') != scene_count:
                    raise ValueError(f'The instruction requires scene_count={scene_count}')
                if explicit_container or not matches_phone_packaging(plan, candidate['goal']):
                    skill_requirements(adapter)
                    proposed = candidate_scene or proposed
                    plan = {key: candidate[key] for key in ('scene_request','successes_per_scene','trajectory_request')
                            if key in candidate}
                    total=requested_containment_total(instruction)
                    if total is not None:
                        plan['trajectory_request']={**plan.get('trajectory_request', {}), 'total_episodes':total}
                    if proposed is not None:
                        plan['scene_request']=proposed
                    plan.update(robot=robot_id, tasks=[{'task_id':CONTAINER_TASK, 'goal':candidate['goal']}])
            elif explicit_container or (
                plan.get('status')=='unsupported' and not any(
                    plan.get(key) for key in ('missing_assets','missing_skills','missing_robot_capabilities')
                )
            ):
                missing = candidate.get('missing_skills')
                plan = {'status': 'unsupported', 'reason': candidate.get('reason') or plan.get('reason')}
                if isinstance(missing, list) and missing and all(
                    isinstance(skill, str) and re.fullmatch(r'[a-z0-9_]{1,80}', skill) for skill in missing
                ):
                    plan['missing_skills'] = missing
                else:
                    error=UnsupportedRequest(plan['reason'] or 'No supported goal could be resolved from the request')
                    error.details['code']='TASK_GOAL_UNSUPPORTED'
                    raise error
    if plan.get('status')=='skill_request':
        if plan.get('robot')!=robot_id or not isinstance(plan.get('skill_goal'),dict):
            raise ValueError('Skill request failed robot or schema validation')
        if (task_ids or tuple(scenes)!=tuple(SCENES) or scene_options is not None
                or successes_per_scene is not None or trajectory_options is not None):
            raise ValueError('A standalone skill goal conflicts with specified task or trajectory quota')
        from skill_onboarding import prepare
        return prepare(plan['skill_goal'],robot_id,execute=execute,on_prepared=on_prepared,
                       instruction=instruction,intent_receipts=intent_receipts)
    if plan.get('status')=='unsupported':
        if not any(plan.get(k) for k in ['missing_assets','missing_skills','missing_robot_capabilities']):
            plan['missing_skills']=['validated_task_contract_for_requested_operation']
        error=UnsupportedRequest(plan.get('reason','Request exceeds the supported scope'),
            missing_assets=plan.get('missing_assets',()), missing_skills=plan.get('missing_skills',()),
            missing_capabilities=plan.get('missing_robot_capabilities',()))
        skills=plan.get('missing_skills',[])
        if isinstance(skills,list) and skills and all(isinstance(skill,str) and re.fullmatch(r'[a-z0-9_]{1,80}',skill) for skill in skills):
            from skill_capabilities import discover_skills
            capabilities=discover_skills(skills,adapter)
            for name in ('code','message','candidate_apis','skill_details','missing_verifiers'):
                if name in capabilities:
                    error.details[name]=capabilities[name]
            error.details['missing_robot_capabilities']=sorted(set(error.details['missing_robot_capabilities'])
                                                              | set(capabilities['missing_robot_capabilities']))
        raise error
    if plan.get('robot')!=robot_id or not isinstance(plan.get('tasks'),list):raise ValueError('Intent router identity/schema mismatch')
    routed=[t['task_id'] for t in plan['tasks']]
    if len(set(routed))!=len(routed):raise ValueError('Intent router duplicated tasks')
    if task_ids is not None and not set(task_ids)<=set(routed):raise UnsupportedRequest('Explicit task IDs conflict with the natural-language requirement',missing_skills=sorted(set(task_ids)-set(routed)))
    task_ids=task_ids or routed
    if not task_ids or set(task_ids)-available_tasks:raise UnsupportedRequest('Task lacks an existing asset/skill contract',missing_skills=list(set(task_ids or [])-available_tasks) or ['supported_task'])
    if proposed and scene_options:
        for key in ('mode','scene_count','scene_ids','background_id','support_id','object_policy'):
            if key in plan['scene_request'] and key in scene_options and proposed[key]!=scene_options[key]:
                raise ValueError(f"Scene request conflicts with instruction: {key}")
    scene_options=scene_options or proposed
    if successes_per_scene is not None and plan.get('successes_per_scene',successes_per_scene)!=successes_per_scene:
        raise ValueError('successes_per_scene conflicts with instruction')
    successes_per_scene=successes_per_scene if successes_per_scene is not None else plan.get('successes_per_scene')
    if successes_per_scene is not None:successes_per_scene=count(successes_per_scene,'successes_per_scene',12)
    inferred = None if explicit_trajectory['duplicate_policy']=='replay' else plan.get('trajectory_request')
    trajectory = resolve_trajectory_request(trajectory_options,inferred,instruction=instruction,
                                            distinct_scene_count=scene_count)
    if trajectory.get('total_episodes') and len(task_ids)>1:
        raise ValueError('A total trajectory quota requires one task. Use successes_per_scene for multiple tasks')
    if scene_options is None and trajectory['duplicate_policy']!='replay':
        selected = list(scenes)
        total = trajectory.get('total_episodes')
        if total and total < len(selected):
            selected = selected[:total]
        scene_options = scene_request({'mode': 'existing', 'scene_ids': selected})
    if scene_options and scene_options['mode']=='compose_from_library':
        dimensions=trajectory.get('distinct_dimensions',[])
        if scene_options['scene_count']>1 and any(
            dimension in dimensions and field in scene_options
            for dimension,field in (('background','background_id'),('support','support_id'))):
            raise ValueError('A requested difference cannot be fixed to a single asset')
        scene_options['variation']=list(dict.fromkeys([*scene_options['variation'],*dimensions]))
    request_id=uuid.uuid4().hex
    directory=ROOT/'requests';directory.mkdir(exist_ok=True);request=directory/(request_id+'.json')
    asset_queries={task['task_id']:task.get('asset_queries',[]) for task in plan['tasks'] if task['task_id'] in task_ids}
    goals={task['task_id']:task['goal'] for task in plan['tasks'] if task['task_id'] in task_ids and 'goal' in task}
    request.write_text(json.dumps({'instruction':instruction,'robot_id':robot_id,'tasks':task_ids,'scenes':list(scenes),
        'intent_receipts':intent_receipts,'asset_queries':asset_queries,'scene_request':scene_options,'goals':goals,
        'successes_per_scene':successes_per_scene,'trajectory_request':trajectory},ensure_ascii=False,indent=2),encoding='utf-8')
    command=[str(ROOT/'.venv/bin/python'),str(ROOT/'prepare_skill_plans.py'),'--instruction-file',str(request),'--robot-id',robot_id,'--tasks',*task_ids,'--scenes',*scenes]
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:
        failure_log=request.with_suffix('.preparation.log');failure_log.write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        try:failure=json.loads(result.stdout.splitlines()[-1])
        except (ValueError,IndexError):failure={}
        if failure.get('status') in {'unsupported','validating_skill','capability_gap'}:
            error=UnsupportedRequest(failure.get('reason','Plan unsupported'))
            error.details.update(failure)
            raise error
        raise RuntimeError('Skill-plan preparation failed; diagnostic log: '+str(failure_log))
    receipt=json.loads(result.stdout.splitlines()[-1]);job_path=Path(receipt['path']);job=json.loads(job_path.read_text())
    job.update(original_request_file=str(request),intent_receipt=intent_receipts[-1] if intent_receipts else None,
               intent_receipts=intent_receipts,preflight_only=not execute)
    job_path.write_text(json.dumps(job,ensure_ascii=False,indent=2))
    if on_prepared is not None:on_prepared(job['job_id'])
    if execute:subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'batch_collect.py'),'--job',str(job_path),
                               '--wait-for-worker'],check=True)
    return result_for(job_path)

def main():
    p=argparse.ArgumentParser();p.add_argument('instruction',nargs='?');p.add_argument('--request',type=Path);p.add_argument('--robot-id',default='g2crsb90d')
    p.add_argument('--task',action='append',dest='tasks');p.add_argument('--scene',action='append',choices=SCENES,dest='scenes');p.add_argument('--execute',action='store_true');a=p.parse_args()
    request=json.loads(a.request.read_text(encoding='utf-8')) if a.request else {}
    instruction=request.get('instruction',a.instruction)
    tasks=a.tasks or request.get('task_ids') or ([request['task_id']] if request.get('task_id') else None)
    try:print(json.dumps(submit(instruction,robot_id=request.get('robot_id',a.robot_id),
        task_ids=tasks,scenes=tuple(a.scenes or request.get('scenes',SCENES)),execute=a.execute or request.get('execute',False),
        scene_options=request.get('scene_request'),successes_per_scene=request.get('successes_per_scene'),
        trajectory_options=request.get('trajectory_request'),skill_goal=request.get('skill_goal')),ensure_ascii=False))
    except UnsupportedRequest as exc:print(json.dumps(exc.details,ensure_ascii=False));raise SystemExit(2)
    except (ValueError,RuntimeError) as exc:print(json.dumps({'status':'failed','reason':str(exc)},ensure_ascii=False));raise SystemExit(1)
if __name__=='__main__':main()
