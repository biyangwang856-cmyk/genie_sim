"""Count actual attempts, independent admissions and model usage separately."""
import argparse,hashlib,json,time
from pathlib import Path
from collections import Counter

ROOT=Path('/home/vipuser/by/genie_sim');STORAGE=Path('/data/by/genie_sim')

def read_json(path,errors):
    path=Path(path)
    if not path.exists():return {}
    try:return json.loads(path.read_text())
    except (ValueError,OSError) as exc:errors.append({'path':str(path),'error':str(exc)});return {}

def summarize_usage(receipts):
    usage={}
    for receipt in receipts.values():
        model=receipt['model'];entry=usage.setdefault(model,{'calls':0,'usage':{},'elapsed_s':0.})
        entry['calls']+=1;entry['elapsed_s']+=receipt.get('elapsed_s',0.)
        for key,value in (receipt.get('usage') or {}).items():
            if isinstance(value,(float,int)):entry['usage'][key]=entry['usage'].get(key,0)+value
    return usage


def receipt_id(receipt):
    return receipt.get('id') or hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()


def report(job_path,*,print_report=True):
    job_path=Path(job_path);job=json.loads(job_path.read_text());attempts=[];receipts={};first={};metadata_errors=[]
    def add_receipt(receipt):
        if receipt:
            receipts[receipt_id(receipt)]=receipt
    for attempt in job['attempts']:
        run=Path(attempt['run_directory']);launch=read_json(run/'launch.json',metadata_errors)
        result=read_json(run/'collection_result.json',metadata_errors)
        acceptance=read_json(run/'acceptance.json',metadata_errors)
        if not launch:launch={'task':attempt['task'],'scene':attempt['scene']}
        row={'run':str(run),'task':launch['task'],'scene':str(launch['scene']),'seed':launch.get('seed'),
            'launch_unix':launch.get('created_unix',attempt.get('started_unix')),
            'physical_task_success':result.get('success') is True or bool(acceptance.get('native_recording_completed_after_runner_timeout')
                and acceptance.get('checks',{}).get('task_physical_gates') and acceptance.get('checks',{}).get('raw_measured_task_geometry')),
            'independent_admission':acceptance.get('accepted') is True,
            'counted_toward_quota':attempt.get('accepted') is True,
            'duplicate_motion_of':attempt.get('duplicate_motion_of'),
            'preserved_duplicate_motion_of':attempt.get('preserved_duplicate_motion_of'),
            'duration_s':(result.get('episodes') or [{}])[0].get('duration_s')}
        failures=[]
        path=run/'evidence/skill-execution.jsonl'
        if path.exists():
            for number,line in enumerate(path.read_text().splitlines(),1):
                try:event=json.loads(line)
                except ValueError as exc:
                    metadata_errors.append({'path':str(path),'line':number,'error':str(exc)})
                    continue
                if event['event']=='step_failed':failures.append(event.get('error_type','unspecified'))
        row['recorded_step_failures']=failures;attempts.append(row)
        scope=(row['task'],row['scene'])
        previous=first.get(scope)
        if previous is None or (row['launch_unix'] is not None and previous['launch_unix'] is not None
                                and row['launch_unix']<previous['launch_unix']):
            first[scope]=row
        for p in (run/'evidence').glob('llm-plan-*.json'):
            archive=json.loads(p.read_text());receipt=archive.get('receipt',{})
            add_receipt(receipt)
    # Plan upgrades have their own intent, retrieval and repair-call receipts.
    # Follow their provenance instead of reporting only the most recent plan.
    pending=[job_path];preparation_jobs=set()
    while pending:
        path=Path(pending.pop()).resolve()
        if str(path) in preparation_jobs:continue
        if not path.is_relative_to((ROOT/'jobs').resolve()):raise ValueError('Preparation provenance outside project jobs')
        preparation_jobs.add(str(path));prepared=read_json(path,metadata_errors)
        for p in path.parent.glob('plans/*/*/llm-plan-*.json'):
            add_receipt(read_json(p,metadata_errors).get('receipt',{}))
        for retrieval in prepared.get('asset_retrieval',{}).values():add_receipt(retrieval.get('receipt',{}))
        add_receipt(prepared.get('intent_receipt') or {})
        pending.extend(h['source_job'] for h in prepared.get('preparation_history',[]) if h.get('source_job'))
    usage=summarize_usage(receipts)
    project_receipts=dict(receipts);api_log=ROOT/'jobs/api-receipts.jsonl'
    if api_log.exists():
        for number,line in enumerate(api_log.read_text().splitlines(),1):
            try:receipt=json.loads(line)
            except ValueError as exc:metadata_errors.append({'path':str(api_log),'line':number,'error':str(exc)});continue
            project_receipts[receipt_id(receipt)]=receipt
    historical=[];incomplete_launches=[]
    production_runs={a['run'] for a in attempts}
    for run in sorted((STORAGE/'collection-runs').iterdir()):
        if str(run) in production_runs:continue
        if not (run/'launch.json').exists():
            incomplete_launches.append({'run':str(run),'collection_result_present':(run/'collection_result.json').exists(),
                'interpretation':'No launch receipt; not counted as a completed physical episode.'})
            continue
        launch=read_json(run/'launch.json',metadata_errors);result=read_json(run/'collection_result.json',metadata_errors)
        historical.append({'run':str(run),'mode':launch.get('execution_mode','baseline'),
            'complete_report':bool(result),'task_success':result.get('success') is True})
    accepted=sum(a['counted_toward_quota'] for a in attempts);completed=[a for a in attempts if a['duration_s'] is not None]
    completed_scopes={(a['task'],a['scene']) for a in completed}
    successful_scopes={(a['task'],a['scene']) for a in completed if a['counted_toward_quota']}
    quotas={(task['task_id'],str(q['scene_id'])):q['successes'] for task in job['plan']['tasks'] for q in task['scene_quotas']}
    outcomes=[]
    for (task,scene),quota in quotas.items():
        trials=[a for a in attempts if (a['task'],a['scene'])==(task,scene)]
        count=job['success_counts'][task][scene]
        outcomes.append({'task':task,'scene':scene,'quota':quota,'attempted':len(trials),
            'completed':sum(a['duration_s'] is not None for a in trials),
            'physical_successes':sum(a['physical_task_success'] for a in trials),
            'independent_passes':sum(a['independent_admission'] for a in trials),
            'quota_admissions':count,'at_least_one_admitted':count>0,'quota_met':count>=quota,
            'first_attempt':first.get((task,scene))})
    data={'generated_unix':time.time(),'job':str(job_path),'success_counts':job['success_counts'],'formal_attempts_total':len(attempts),
        'formal_attempts_completed':len(completed),'formal_accepted':accepted,'attempts':attempts,
        'formal_exact_motion_duplicates':sum(bool(a['duplicate_motion_of']) for a in attempts),
        'previously_admitted_exact_motion_repeats':sum(bool(a['preserved_duplicate_motion_of']) for a in attempts),
        'first_attempt_physical_success_rate':sum(x['physical_task_success'] for x in first.values())/len(first) if first else None,
        'first_attempt_independent_admission_rate':sum(x['independent_admission'] for x in first.values())/len(first) if first else None,
        'task_scene_success_after_retries_rate':len(successful_scopes)/len(completed_scopes) if completed_scopes else None,
        'completed_task_scene_denominator':len(completed_scopes),'requested_task_scene_count':sum(len(t['scene_quotas']) for t in job['plan']['tasks']),
        'accepted_per_completed_attempt':accepted/len(completed) if completed else None,
        'first_attempt_denominator':'earliest production launch in the surviving ledger per task/scene; calibration and unlinked runs reported separately',
        'retry_success_denominator':'completed task/scene groups with at least one admitted episode, divided by all completed task/scene groups',
        'retry_outcomes_by_task_scene':[{**row,'scene_goal_met':job['success_counts'][task][scene]>0} for (task,scene),row in first.items()],
        'task_scene_outcomes':outcomes,
        'step_failure_counts':dict(Counter(f for a in attempts for f in a['recorded_step_failures'])),
        'completed_episode_wall_seconds':sum(a['duration_s'] for a in completed),'model_api_usage':usage,
        'job_api_preparation_provenance':sorted(preparation_jobs),
        'project_model_api_usage':summarize_usage(project_receipts),
        'project_api_scope':'All deduplicated archived API receipts in this project, including indexing, diagnostics, plan preparation and recoveries; not just successful data.',
        'api_usage_limitation':'Older unlogged HTTP failures are not reconstructed as zero-cost calls; receipts without provider usage have unknown billed tokens.',
        'api_billed_cost':None,'api_cost_limitation':'Actual token usage retained; provider billing or agreed per-token rates are required for a currency amount.',
        'historical_debug_runs_total':len(historical),'historical_debug_runs':historical,'metadata_errors':metadata_errors,
        'launch_failures':job.get('launch_failures',[]),'project_incomplete_launch_directories':incomplete_launches,
        'limitations':['A small screened success set does not demonstrate a high general success rate.','Only g2crsb90d execution has been exercised.']}
    if job.get('recovery'):
        recovery=job['recovery']
        data['ledger_recovery']={k:recovery[k] for k in ('original_job','prior_metrics_snapshot','historical_limit')}
        data['historical_unlinked_run_count']=len(recovery.get('recovered_run_inventory',[]))
        data['limitations'].append('Attempt and API totals use the surviving ledger and receipts. Historical run membership is incomplete.')
    output=job_path.parent/'metrics.json';output.write_text(json.dumps(data,indent=2))
    if print_report:print(json.dumps(data,indent=2))
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);report(p.parse_args().job)
