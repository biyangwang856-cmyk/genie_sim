"""Persist one reviewed hardpoint certificate for a fully accepted task recipe."""
import argparse,hashlib,json
from pathlib import Path
from batch_collect import audit_recording,recipe_sources,write
from collection_control import ROOT, STORAGE

def certify(run,job_path,review_path):
    run=Path(run).resolve();job_path=Path(job_path).resolve();review_path=Path(review_path).resolve()
    archived,result=audit_recording(run)
    if not archived or not result['accepted']:raise ValueError('Full independent acceptance is required before certification')
    launch=json.loads((run/'launch.json').read_text());job=json.loads(job_path.read_text());review=json.loads(review_path.read_text())
    if not any(Path(a['run_directory']).resolve()==run for a in job['attempts']):
        task=launch['task'];scene=str(launch['scene'])
        requested={(t['task_id'],str(q['scene_id'])) for t in job['plan']['tasks'] for q in t['scene_quotas']}
        prepared=job.get('prepared_skill_plans',{}).get(task,{}).get(scene,{})
        request=json.loads((run/'source/request.json').read_text())
        if (run.parent!=STORAGE/'collection-runs' or launch.get('collection_slot')!='secondary' or
            (task,scene) not in requested or launch['robot_id']!=job['plan']['robot'] or
            launch['execution_mode']!=job['execution_mode'] or request['instruction']!=job['instruction'] or
            not prepared or launch.get('initial_plan_sha256')!=prepared.get('plan_sha256')):
            raise ValueError('Independent episode does not match the job request and prepared plan')
    if review.get('run_directory')!=str(run) or review.get('passed') is not True or not review.get('observations'):
        raise ValueError('Missing concrete hardpoint review')
    files=review.get('reviewed_image_files',{})
    if len(files)<2:raise ValueError('Review at least two actual camera views at the hardpoint')
    for name,digest in files.items():
        path=(run/name).resolve()
        if not path.is_relative_to(run/'review') or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Reviewed image does not belong to this run or changed')
    source=recipe_sources(launch)
    certificate={'passed':True,'task_id':launch['task'],'scene_id':launch['scene'],'robot_id':launch['robot_id'],
        'source_sha256':source,'hardpoint_evidence':{'run':str(run),'review':str(review_path),'image_sha256':files,
            'review_observations':review['observations'],'acceptance_sha256':hashlib.sha256((run/'acceptance.json').read_bytes()).hexdigest()}}
    local=job_path.parent/'recipe-validation'/launch['task']/(launch['scene']+'.json');write(local,certificate)
    digest=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    common=ROOT/'pipeline/validated-recipes'/launch['robot_id']/launch['task']/launch['scene']/(digest+'.json');write(common,certificate)
    print(json.dumps({'certificate':str(local),'reusable_certificate':str(common),'passed':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--job',required=True);p.add_argument('--review',required=True);a=p.parse_args();certify(a.run,a.job,a.review)
