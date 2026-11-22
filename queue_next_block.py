"""Resume the corrected block probe after the current pen preflight finishes."""
from pathlib import Path
import fcntl,os,subprocess,time
root=Path('/home/vipuser/by/genie_sim');job=root/'jobs/20261119-185730-ed63df63'
env=dict(os.environ,DOCKER_HOST='unix:///data/by/genie_sim/dockerhub-sidebuild-20261119/docker.sock')
with (job/'block09-queue.lock').open('w') as queue:
 fcntl.flock(queue,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for _ in range(210):
  active=subprocess.check_output(['docker','ps','--filter','name=genie-by-','--format','{{.Names}}'],env=env,text=True).strip()
  if not active:
   with (job/'worker.lock').open('a') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:time.sleep(10);continue
    subprocess.run([str(root/'.venv/bin/python'),str(root/'collection_control.py'),'insert_building_block','--scene','0','--label','block-scene0-calibrated09'],check=True,env=env)
    break
  time.sleep(10)
 else:raise TimeoutError('No new simulator started because previous preflight is still active')
