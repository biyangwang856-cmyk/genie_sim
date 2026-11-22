"""Queue a bounded follow-up in the primary simulation slot."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from collection_control import ROOT,SOCKET,running_collections

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--task',required=True);p.add_argument('--scene',choices=['0','1'],required=True);p.add_argument('--wait-seconds',type=int,default=7200)
    p.add_argument('--trial-seed',type=int);p.add_argument('--adopt-predecessor',type=Path);a=p.parse_args()
    env=dict(os.environ,DOCKER_HOST=SOCKET);deadline=time.monotonic()+a.wait_seconds
    predecessor=None
    if a.adopt_predecessor:
        predecessor=json.loads((a.adopt_predecessor/'launch.json').read_text())['container']
    while time.monotonic()<deadline:
        active=running_collections(env,'primary')
        if predecessor:
            state=subprocess.run(['docker','inspect','--format','{{.State.Running}}',predecessor],
                                 env=env,capture_output=True,text=True)
            if state.returncode==0 and state.stdout.strip()=='true':active.append(predecessor)
        available=False
        with (a.job.parent/'worker.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);available=True
            except BlockingIOError:pass
        if not active and available:break
        time.sleep(10)
    else:raise TimeoutError('Predecessor did not finish; no other process was stopped')
    print('START_TASK',a.task,a.scene,flush=True)
    if a.adopt_predecessor:
        subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'batch_collect.py'),'--job',str(a.job),'--adopt-run',str(a.adopt_predecessor),'--audit-only'],check=True)
    command=[str(ROOT/'.venv/bin/python'),str(ROOT/'batch_collect.py'),'--job',str(a.job),'--tasks',a.task,'--scenes',a.scene]
    if a.trial_seed is not None:command+=['--trial-seed',str(a.trial_seed)]
    subprocess.run(command,check=True)

if __name__=='__main__':main()
