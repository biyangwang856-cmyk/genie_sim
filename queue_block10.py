from pathlib import Path
import fcntl,os,subprocess,time
root=Path('/home/vipuser/by/genie_sim')
env=dict(os.environ,DOCKER_HOST='unix:///data/by/genie_sim/dockerhub-sidebuild-20261119/docker.sock')
with (root/'jobs/block10-queue.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for _ in range(180):
  active=subprocess.check_output(['docker','ps','--filter','name=genie-by-','--format','{{.Names}}'],env=env,text=True).strip()
  if not active:
   subprocess.run([str(root/'.venv/bin/python'),str(root/'collection_control.py'),'insert_building_block','--scene','0','--label','block-scene0-calibrated10'],env=env,check=True)
   break
  time.sleep(10)
 else:raise TimeoutError('No second simulator started; current collection still active')
