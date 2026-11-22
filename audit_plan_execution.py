"""Check actual dispatch order against archived model responses, not task names."""
import hashlib,json
from pathlib import Path

def audit_plan_execution(run):
    run=Path(run);request=run/'source/request.json';result={'passed':False,'reason':'baseline_or_missing_structured_request'}
    if not request.is_file() or json.loads(request.read_text()).get('execution_mode')!='structured':return result
    events_path=run/'evidence/skill-execution.jsonl'
    if not events_path.is_file():return {'passed':False,'reason':'missing_execution_journal'}
    archives={}
    for p in (run/'evidence').glob('llm-plan-*.json'):
        a=json.loads(p.read_text())
        if not a.get('validated'):continue
        plan=json.loads(a['raw_response']);digest=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        if digest!=a.get('plan_sha256') or a.get('receipt',{}).get('status')!=200:continue
        if a.get('model') not in ['qwen-plus','qwen-max']:continue
        archives[digest]=plan
    segments=[];current=None
    for line in events_path.read_text().splitlines():
        event=json.loads(line);kind=event['event']
        if kind=='validated_plan_started':
            plan=archives.get(event['plan_sha256'])
            if plan is None or plan!=event['plan']:return {'passed':False,'reason':'executed_plan_not_the_archived_model_output'}
            current={'digest':event['plan_sha256'],'steps':plan['steps'],'started':[],'succeeded':[],'failed':[]};segments.append(current)
        elif kind=='step_started':
            if current is None:return {'passed':False,'reason':'dispatch_without_plan'}
            index=len(current['started'])
            if index>=len(current['steps']) or event['step']!=current['steps'][index]:return {'passed':False,'reason':'dispatch_order_mismatch'}
            current['started'].append(event['step']['id'])
        elif kind=='step_succeeded':
            if current is None or event['step_id'] not in current['started']:return {'passed':False,'reason':'completion_without_dispatch'}
            current['succeeded'].append(event['step_id'])
        elif kind=='step_failed' and current is not None:current['failed'].append(event['step']['id'])
    if not segments:return {'passed':False,'reason':'no_validated_plan_dispatch'}
    last=segments[-1];expected=[s['id'] for s in last['steps']]
    passed=last['started']==expected and last['succeeded']==expected and not last['failed'] and last['steps'][-1]['skill']=='verify_task'
    result={'passed':passed,'reason':'verified_model_dispatch' if passed else 'last_plan_incomplete','plan_revisions':len(segments),
        'completed_step_count':sum(len(s['succeeded']) for s in segments),'plan_sha256':[s['digest'] for s in segments]}
    (run/'plan-execution-audit.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    import sys
    print(json.dumps(audit_plan_execution(sys.argv[1])))
