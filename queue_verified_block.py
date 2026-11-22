"""Wait for this project's current run, then fill only the reviewed left-arm quota."""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/vipuser/by/genie_sim');JOB=ROOT/'jobs/20261120-160355-skills-ed63df63/job.json'
env=dict(os.environ,DOCKER_HOST='unix:///data/by/genie_sim/dockerhub-sidebuild-20261119/docker.sock')
deadline=time.monotonic()+3600
while time.monotonic()<deadline:
    active=subprocess.check_output(['docker','ps','--filter','name=genie-by-','--format','{{.Names}}'],env=env,text=True).strip()
    available=False
    with (JOB.parent/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);available=True
        except BlockingIOError:pass
    if not active and available:break
    time.sleep(10)
else:raise TimeoutError('Project predecessor did not finish; no container was interrupted')
command=[str(ROOT/'.venv/bin/python'),str(ROOT/'batch_collect.py'),'--job',str(JOB),'--tasks','insert_building_block','--scenes','0',
    '--source-run','/data/by/genie_sim/collection-runs/insert-building-block-s0-20261120160637']
print('START_PINNED_VERIFIED_SCENE0',flush=True)
subprocess.run(command,check=True)
