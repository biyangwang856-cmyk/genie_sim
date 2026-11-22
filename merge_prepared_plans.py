"""Adopt fresh, unedited model plans while preserving attempts and accepted data."""
import argparse,fcntl,json,time
from pathlib import Path
from batch_collect import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--prepared-job',type=Path,required=True);a=p.parse_args()
    source=json.loads(a.prepared_job.read_text())
    if source['status']!='skill_plans_prepared':raise ValueError('New plans are not validated')
    with (a.job.parent/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        job=json.loads(a.job.read_text())
        if job['instruction']!=source['instruction'] or job['robot_id']!=source['robot_id']:raise ValueError('Plan assignment differs from original request')
        for task,scenes in source['prepared_skill_plans'].items():
            if task not in job['task_bindings']:raise ValueError('Prepared task outside original scope')
            old=job['prepared_skill_plans'].get(task,{})
            job.setdefault('preparation_history',[]).append({'updated_unix':time.time(),'task':task,'previous_plans':old,
                'replacement_plans':scenes,'source_job':str(a.prepared_job.resolve())})
            job['prepared_skill_plans'][task]={**old,**scenes}
        job['asset_retrieval'].update(source['asset_retrieval']);write(a.job,job)
        print(json.dumps({'updated_tasks':list(source['prepared_skill_plans']),'attempts_preserved':len(job['attempts']),'success_counts':job['success_counts']}))

if __name__=='__main__':main()
