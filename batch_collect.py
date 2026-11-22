"""Server quota worker: fresh episodes, evidence gates, and bottleneck certificates."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,subprocess,time
from audit_episode import audit,recording_completed_after_timeout
from archive_episode import archive
from reextract_recording import ensure_aligned
from collection_control import ROOT,STORAGE,SOCKET,execution_profile,SOURCE_FILES
from robot_adapters import load_robot
FILES=SOURCE_FILES
AUDIT_REVISION=hashlib.sha256(b''.join((ROOT/f).read_bytes() for f in [
    'audit_episode.py','audit_plan_execution.py','audit_task_geometry.py','insertion_physics.py'])).hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(temp,path)

def hashes(robot_id='g2crsb90d',plan_digest=None,slot='primary'):
    result={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES}
    result['execution-profile.json']=hashlib.sha256(json.dumps(execution_profile(slot=slot),sort_keys=True,indent=2).encode()).hexdigest()
    result['insertion-certificate.json']=hashlib.sha256((ROOT/'output/insertion-certificate.json').read_bytes()).hexdigest()
    result['insertion-target.json']=hashlib.sha256((ROOT/'output/insertion_target_v2.json').read_bytes()).hexdigest()
    adapter=load_robot(robot_id)
    result['robot-profile.json']=hashlib.sha256(json.dumps(adapter.describe(),sort_keys=True,indent=2).encode()).hexdigest()
    result['precise_geometry.py']=hashlib.sha256((ROOT/'genie_sim-main/source/geniesim_benchmark/src/geniesim_benchmark/plugins/ader/action/custom/precise_geometry.py').read_bytes()).hexdigest()
    result['initial_plan_sha256']=plan_digest
    calibration=adapter.calibration_path(ROOT)
    if calibration and calibration.exists():
        cfg=json.loads(calibration.read_text())
        result.update({'robot-calibration.json':hashlib.sha256(calibration.read_bytes()).hexdigest(),
            'robot.urdf':hashlib.sha256(Path(cfg['host_calibrated_urdf']).read_bytes()).hexdigest()})
    return result

def recipe_sources(launch):
    result={k:v for k,v in launch['source_sha256'].items() if k not in ['request.json','prepared-plan.json']}
    result['initial_plan_sha256']=launch.get('initial_plan_sha256')
    return result

def verified_recipe(jobdir,task,scene,source):
    local=jobdir/'recipe-validation'/task/(scene+'.json')

    def matches(cert):
        return cert.get('passed') is True and cert.get('source_sha256')==source and bool(cert.get('hardpoint_evidence'))

    if local.is_file():
        cert=json.loads(local.read_text())
        if cert.get('source_sha256')==source:return matches(cert)
    digest=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    job=json.loads((jobdir/'job.json').read_text())
    common=ROOT/'pipeline/validated-recipes'/job.get('robot_id',job['plan']['robot'])/task/scene/(digest+'.json')
    return common.is_file() and matches(json.loads(common.read_text()))

def audit_recording(run):
    """Archive and independently verify a finished run for any collection slot."""
    evidence=run/'evidence'
    evidence.mkdir(exist_ok=True)
    with (evidence/'audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        archived=archive(run).get('archived',False)
        native=json.loads((run/'collection_result.json').read_text())
        task_result=run/'raw/task_result.json'
        late_result=task_result.is_file() and recording_completed_after_timeout(native,json.loads(task_result.read_text()))
        alignment_error=None;aligned=None
        if archived and (native.get('success') is True or late_result):
            try:aligned=ensure_aligned(run)
            except (ValueError,OSError,subprocess.SubprocessError) as exc:alignment_error=str(exc)
        cached=run/'acceptance.json'
        result=json.loads(cached.read_text()) if cached.exists() else {}
        if result.get('audit_revision')!=AUDIT_REVISION or alignment_error or (aligned and str(aligned)!=result.get('raw_directory')):
            result=audit(run)
            if alignment_error:
                result['accepted']=False
                result['physical_sample_accepted']=False
                result['checks']['original_sensor_headers_synchronized']=False
                result['errors'].append(alignment_error)
            result['audit_revision']=AUDIT_REVISION
            write(cached,result)
        return archived,result


def motion_fingerprint(run):
    import numpy as np
    from export_lerobot import native_arrays,recording_path

    state=recording_path(run)/'state.json'
    controls=run/'evidence/native-control-commands.jsonl'
    sources={str(p):{'size':p.stat().st_size,'modified_ns':p.stat().st_mtime_ns} for p in (state,controls)}
    cache=run/'evidence/training-motion-fingerprint.json'
    method='exact_float32_training_motion_between_control_changes'
    if cache.is_file():
        record=json.loads(cache.read_text())
        if record.get('method')==method and record['sources']==sources:return record['sha256']

    arrays,spec=native_arrays(run)
    fields=sorted(key for key in arrays if key not in ('action.command_time','observation.sim_time'))
    control_changes=np.zeros(len(arrays['action'])-1,dtype=bool)
    for key in fields:
        if key.startswith('action'):
            control_changes|=np.any(arrays[key][1:]!=arrays[key][:-1],axis=1)
    changed=np.flatnonzero(control_changes)
    window=slice(int(changed[0]),int(changed[-1])+2) if len(changed) else slice(None)
    motion=hashlib.sha256(json.dumps(spec['joint_names']).encode())
    for key in fields:
        values=arrays[key][window].astype('<f4',copy=False)
        motion.update(json.dumps([key,values.shape]).encode())
        motion.update(np.where(values==0.,0.,values).tobytes())
    fingerprint=motion.hexdigest()
    write(cache,{'method':method,'sources':sources,'sha256':fingerprint,
                 'compared_frame_range':[window.start,window.stop]})
    return fingerprint


def update_success_counts(job,jobdir):
    counts={task:{'0':0,'1':0} for task in job['task_bindings']}
    admitted={}
    preserved=set(job.get('preserved_admissions',[]))
    for attempt in job['attempts']:
        attempt['accepted']=False
        attempt.pop('duplicate_motion_of',None)
        attempt.pop('preserved_duplicate_motion_of',None)
        if not attempt.get('automatic_acceptance') or not attempt.get('raw_archive'):continue
        task,scene=attempt['task'],str(attempt['scene'])
        run=Path(attempt['run_directory'])
        identity=(task,scene,motion_fingerprint(run))
        if identity in admitted and str(run) not in preserved:
            attempt['duplicate_motion_of']=admitted[identity]
        elif verified_recipe(jobdir,task,scene,attempt['source_sha256']):
            if identity in admitted:attempt['preserved_duplicate_motion_of']=admitted[identity]
            admitted.setdefault(identity,str(run))
            attempt['accepted']=True
            counts[task][scene]+=1
    job['success_counts']=counts


def refresh(job,jobdir):
    seen=set()
    for attempt in job['attempts']:
        run=Path(attempt['run_directory'])
        if str(run) in seen:raise ValueError('Duplicate run in job')
        seen.add(str(run))
        launch=json.loads((run/'launch.json').read_text())
        attempt['source_sha256']=recipe_sources(launch)
        attempt['automatic_acceptance']=False
        if (run/'collection_result.json').exists():
            attempt['raw_archive'],result=audit_recording(run)
            attempt['automatic_acceptance']=result['accepted']
    update_success_counts(job,jobdir)

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--adopt-run');p.add_argument('--max-attempts-per-scene',type=int,default=8,help='Limit for the selected code, model and plan in each scene');p.add_argument('--audit-only',action='store_true')
    p.add_argument('--tasks',nargs='+');p.add_argument('--scenes',nargs='+',choices=['0','1']);p.add_argument('--source-run',type=Path)
    p.add_argument('--slot',choices=['primary','secondary'],default='primary')
    p.add_argument('--trial-seed',type=int,help='One controlled episode; requires exactly one task and scene')
    a=p.parse_args()
    if a.trial_seed is not None and (not a.tasks or len(a.tasks)!=1 or not a.scenes or len(a.scenes)!=1):p.error('--trial-seed requires one task and one scene')
    path=Path(a.job).resolve();jobdir=path.parent
    with (jobdir/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        job=json.loads(path.read_text())
        robot=load_robot(job.get('robot_id',job['plan']['robot']))
        if job['plan']['robot']!=robot.robot_id:raise ValueError('Robot identity mismatch')
        if a.adopt_run:
            run=Path(a.adopt_run).resolve()
            if run.parent!=STORAGE/'collection-runs':raise ValueError('Run outside project collection root')
            launch=json.loads((run/'launch.json').read_text())
            if launch['task'] not in job['task_bindings'] or launch['scene'] not in ['0','1']:raise ValueError('Run outside requested task/scene bindings')
            if not any(v['run_directory']==str(run) for v in job['attempts']):
                job['attempts'].append({'run_directory':str(run),'task':launch['task'],'scene':launch['scene'],'adopted':True})
        refresh(job,jobdir);write(path,job)
        if a.audit_only:
            print(json.dumps({'success_counts':job['success_counts']}));return
        env=dict(os.environ,DOCKER_HOST=SOCKET)
        pending_review=False;exhausted=False
        for spec in job['plan']['tasks']:
            task=spec['task_id']
            if a.tasks and job['task_bindings'][task]['template'] not in a.tasks:continue
            for quota in spec['scene_quotas']:
                scene=quota['scene_id'];goal=quota['successes']
                if a.scenes and scene not in a.scenes:continue
                while job['success_counts'][task][scene]<goal:
                    prepared=job.get('prepared_skill_plans',{}).get(task,{}).get(scene)
                    source=recipe_sources(json.loads((a.source_run/'launch.json').read_text())) if a.source_run else hashes(job.get('robot_id','g2crsb90d'),prepared['plan_sha256'] if prepared else None,a.slot)
                    if a.source_run and source['initial_plan_sha256']!=(prepared['plan_sha256'] if prepared else None):raise ValueError('Frozen recipe plan mismatch')
                    certified=verified_recipe(jobdir,task,scene,source)
                    all_previous=[x for x in job['attempts'] if x['task']==task and x['scene']==scene]
                    previous=[x for x in all_previous if x.get('source_sha256')==source]
                    # One first execution discovers the hardpoint. Only a recorded
                    # technical certificate for that recipe enables quota repeats.
                    if previous and not certified:pending_review=True;break
                    if len(previous)>=a.max_attempts_per_scene:exhausted=True;break
                    short=job['task_bindings'][task]['template']
                    label=short.replace('_','-')+'-s'+scene+'-'+time.strftime('%Y%m%d%H%M%S')
                    command=[str(ROOT/'.venv/bin/python'),str(ROOT/'collection_control.py'),short,'--scene',scene,'--label',label,
                        '--robot-id',job.get('robot_id','g2crsb90d'),'--execution-mode',job.get('execution_mode','baseline'),'--instruction',job['instruction'],'--slot',a.slot]
                    if prepared:command+=['--prepared-plan',prepared['path']]
                    if a.source_run:command+=['--source-run',str(a.source_run)]
                    if a.trial_seed is not None:command+=['--seed',str(a.trial_seed)]
                    launched_at=time.time()
                    result=subprocess.run(command,capture_output=True,text=True)
                    if result.returncode:
                        failure={'task':task,'scene':scene,'label':label,'run_directory':str(STORAGE/'collection-runs'/label),
                            'started_unix':launched_at,'finished_unix':time.time(),'returncode':result.returncode,
                            'phase':'launcher_before_physical_episode','stdout':result.stdout,'stderr':result.stderr}
                        log=jobdir/'launch-failures'/(label+'.json');write(log,failure)
                        job.setdefault('launch_failures',[]).append({k:v for k,v in failure.items() if k not in ['stdout','stderr']})
                        job['launch_failures'][-1]['diagnostic_file']=str(log)
                        job['status']='launch_failed';write(path,job)
                        print(json.dumps({'status':'launch_failed','diagnostic_file':str(log),'returncode':result.returncode}))
                        return
                    receipt=json.loads(result.stdout.splitlines()[-1]);run=Path(receipt['output'])
                    attempt={'task':task,'scene':scene,'run_directory':str(run),'started_unix':launched_at}
                    job['attempts'].append(attempt);job['status']='collecting';write(path,job)
                    deadline=time.monotonic()+4500
                    while True:
                        state=subprocess.check_output(['docker','inspect','-f','{{.State.Running}}',receipt['container']],env=env,text=True).strip()
                        if state=='false':break
                        if time.monotonic()>deadline:raise TimeoutError('Collection exceeded outer watchdog; inspect '+str(run))
                        job['worker_heartbeat_unix']=time.time();write(path,job);time.sleep(10)
                    attempt['finished_unix']=time.time();refresh(job,jobdir);write(path,job)
                    if not attempt['automatic_acceptance'] and not certified:break
                    if attempt.get('duplicate_motion_of') and not certified:break
                    if not certified:
                        review = subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'review_hardpoint.py'),str(run)],
                                                capture_output=True,text=True)
                        if review.returncode==0:attempt['hardpoint_review_directory']=str(run/'review')
                        else:attempt['hardpoint_review_error']=review.stderr[-1500:]
                        pending_review=True
                        write(path,job)
                        break
                    if a.trial_seed is not None:break
        refresh(job,jobdir)
        complete=all(job['success_counts'][t['task_id']][q['scene_id']]>=q['successes'] for t in job['plan']['tasks'] for q in t['scene_quotas'])
        job['status']='finalizing_dataset' if complete else 'needs_hardpoint_check' if pending_review else 'attempt_budget_exhausted' if exhausted else 'recipe_failed'
        write(path,job)
        if complete:
            from finalize_job import finalize
            try:job['dataset']=finalize(job,path)
            except Exception as exc:
                job['status']='dataset_finalization_failed';job['dataset_failure']=str(exc)
            else:
                job['status']='complete';job.pop('dataset_failure',None)
            write(path,job)
        print(json.dumps({'status':job['status'],'success_counts':job['success_counts'],
                          'dataset':job.get('dataset'),'dataset_failure':job.get('dataset_failure')}))

if __name__=='__main__':main()
