"""Run two distinct first-task probes serially after the current block run."""
from pathlib import Path
import fcntl,json,os,subprocess,time
ROOT=Path('/home/vipuser/by/genie_sim');job=ROOT/'jobs/20261119-185730-ed63df63/job.json'
env=dict(os.environ,DOCKER_HOST='unix:///data/by/genie_sim/dockerhub-sidebuild-20261119/docker.sock')
with (job.parent/'preflight-queue.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for _ in range(180):
  active=subprocess.check_output(['docker','inspect','-f','{{.State.Running}}','genie-by-block-scene0-calibrated08'],env=env,text=True).strip()
  if active=='false':break
  time.sleep(10)
 else:raise TimeoutError('Current block run still active; did not start a second simulator')
 subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'batch_collect.py'),'--job',str(job),
  '--adopt-run','/data/by/genie_sim/collection-runs/block-scene0-calibrated08',
  '--tasks','phone_packaging','put_the_cap_on_the_pen_new','--scenes','0'],env=env,check=True)
