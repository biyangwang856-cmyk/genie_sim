"""Callable server entry: request, robot, task and scene are independent inputs."""
import argparse,json,subprocess,uuid
from pathlib import Path
from robot_adapters import ROOT,load_robot,UnsupportedRequest
from collection_scope import SCENES, TASKS
from qwen_backend import QwenAPI

def submit(instruction,*,robot_id='g2crsb90d',task_ids=None,scenes=SCENES,execute=False,on_prepared=None):
    adapter=load_robot(robot_id)
    if not isinstance(instruction,str) or not instruction.strip():raise ValueError('Nonempty natural-language instruction is required')
    if set(scenes)-set(SCENES):raise UnsupportedRequest('Requested scene not registered',missing_assets=sorted(set(scenes)-set(SCENES)))
    if task_ids is not None and (not task_ids or set(task_ids)-set(TASKS)):
        raise UnsupportedRequest('Explicit task IDs are not registered',missing_skills=sorted(set(task_ids or [])-set(TASKS)))
    catalog=[{'task_id':name,'description':json.loads((ROOT/'tasks'/(adapter.config['scene_task_prefix']+name)/'0/scene_info.json').read_text())['official_instruction']['instruction']} for name in TASKS]
    plan,intent_receipt=QwenAPI().plan(instruction,catalog,robot=adapter.describe())
    if plan.get('status')=='unsupported':
        if not any(plan.get(k) for k in ['missing_assets','missing_skills','missing_robot_capabilities']):
            plan['missing_skills']=['validated_task_contract_for_requested_operation']
        raise UnsupportedRequest(plan.get('reason','Request exceeds the supported scope'),missing_assets=plan.get('missing_assets',()),
            missing_skills=plan.get('missing_skills',()),missing_capabilities=plan.get('missing_robot_capabilities',()))
    if plan.get('robot')!=robot_id or not isinstance(plan.get('tasks'),list):raise ValueError('Intent router identity/schema mismatch')
    routed=[t['task_id'] for t in plan['tasks']]
    if len(set(routed))!=len(routed):raise ValueError('Intent router duplicated tasks')
    if task_ids is not None and not set(task_ids)<=set(routed):raise UnsupportedRequest('Explicit task IDs conflict with the natural-language requirement',missing_skills=sorted(set(task_ids)-set(routed)))
    task_ids=task_ids or routed
    if not task_ids or set(task_ids)-set(TASKS):raise UnsupportedRequest('Task lacks an existing asset/skill contract',missing_skills=list(set(task_ids or [])-set(TASKS)) or ['supported_task'])
    request_id=uuid.uuid4().hex
    directory=ROOT/'requests';directory.mkdir(exist_ok=True);request=directory/(request_id+'.json')
    asset_queries={task['task_id']:task.get('asset_queries',[]) for task in plan['tasks'] if task['task_id'] in task_ids}
    request.write_text(json.dumps({'instruction':instruction,'robot_id':robot_id,'tasks':task_ids,'scenes':list(scenes),
        'intent_receipt':intent_receipt,'asset_queries':asset_queries},ensure_ascii=False,indent=2))
    command=[str(ROOT/'.venv/bin/python'),str(ROOT/'prepare_skill_plans.py'),'--instruction-file',str(request),'--robot-id',robot_id,'--tasks',*task_ids,'--scenes',*scenes]
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:
        failure_log=request.with_suffix('.preparation.log');failure_log.write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        try:failure=json.loads(result.stdout.splitlines()[-1])
        except (ValueError,IndexError):failure={}
        if failure.get('status')=='unsupported':
            raise UnsupportedRequest(failure.get('reason','Plan unsupported'),missing_assets=failure.get('missing_assets',()),
                missing_skills=failure.get('missing_skills',()),missing_capabilities=failure.get('missing_robot_capabilities',()))
        raise RuntimeError('Skill-plan preparation failed; diagnostic log: '+str(failure_log))
    receipt=json.loads(result.stdout.splitlines()[-1]);job_path=Path(receipt['path']);job=json.loads(job_path.read_text())
    job.update(original_request_file=str(request),intent_receipt=intent_receipt,preflight_only=not execute)
    job_path.write_text(json.dumps(job,ensure_ascii=False,indent=2))
    if on_prepared is not None:on_prepared(job['job_id'])
    if execute:subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'batch_collect.py'),'--job',str(job_path)],check=True)
    result=json.loads(job_path.read_text())
    return {'job_id':job['job_id'],'job_path':str(job_path),'status':result['status'],
            'dataset':result.get('dataset'),'dataset_failure':result.get('dataset_failure')}

def main():
    p=argparse.ArgumentParser();p.add_argument('instruction',nargs='?');p.add_argument('--request',type=Path);p.add_argument('--robot-id',default='g2crsb90d')
    p.add_argument('--task',action='append',dest='tasks');p.add_argument('--scene',action='append',choices=SCENES,dest='scenes');p.add_argument('--execute',action='store_true');a=p.parse_args()
    instruction=json.loads(a.request.read_text(encoding='utf-8'))['instruction'] if a.request else a.instruction
    try:print(json.dumps(submit(instruction,robot_id=a.robot_id,task_ids=a.tasks,scenes=tuple(a.scenes or ['0','1']),execute=a.execute),ensure_ascii=False))
    except UnsupportedRequest as exc:print(json.dumps(exc.details,ensure_ascii=False));raise SystemExit(2)
    except (ValueError,RuntimeError) as exc:print(json.dumps({'status':'failed','reason':str(exc)},ensure_ascii=False));raise SystemExit(1)
if __name__=='__main__':main()
