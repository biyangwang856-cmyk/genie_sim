"""Continue the existing collection job serially after a named project run."""
import argparse,json,os,subprocess,time
from pathlib import Path
from collection_control import ROOT,STORAGE,SOCKET
p=argparse.ArgumentParser();p.add_argument('--after',required=True);a=p.parse_args()
run=Path(a.after).resolve()
if run.parent!=STORAGE/'collection-runs':raise ValueError('Run outside project')
launch=json.loads((run/'launch.json').read_text());name=launch['container']
if not name.startswith('genie-by-'):raise ValueError('Not a project container')
env=dict(os.environ,DOCKER_HOST=SOCKET)
deadline=time.monotonic()+3600
while subprocess.check_output(['docker','inspect','-f','{{.State.Running}}',name],env=env,text=True).strip()=='true':
 if time.monotonic()>deadline:raise TimeoutError('Current run remains active; no second simulator started')
 time.sleep(10)
job=ROOT/'jobs/20261119-185730-ed63df63/job.json'
command=[str(ROOT/'.venv/bin/python'),str(ROOT/'batch_collect.py'),'--job',str(job),'--adopt-run',str(run),
 '--tasks','phone_packaging','put_the_cap_on_the_pen_new','--scenes','0']
print('Starting serialized existing-task preflights',flush=True)
raise SystemExit(subprocess.call(command,env=env))
