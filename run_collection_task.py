"""Use upstream collection lifecycle with project GPU wrapper and run directory."""
from pathlib import Path
import json,os,runpy,sys,threading,time
import random,numpy as np
seed=int(os.environ.get('GENIE_EPISODE_SEED','20261120'));random.seed(seed);np.random.seed(seed)
ROOT=Path('/home/vipuser/by/genie_sim')
MODULE=Path(os.environ.get('GENIE_COLLECTION_MODULE','/workspace'))
os.chdir(MODULE)
SOURCE=Path(os.environ.get('GENIE_RUN_SOURCE',ROOT))
sys.path[:0]=[str(SOURCE),str(ROOT),str(MODULE/'scripts'),str(MODULE)]
import collection_runtime

bind_host=os.environ.get('GENIE_SIM_GRPC_HOST','127.0.0.1')
if bind_host not in ('127.0.0.1','127.0.0.2','127.0.0.3','127.0.0.4'):
    raise ValueError('Collection must use a registered host loopback address')
if bind_host!='127.0.0.1':
    check_port=collection_runtime.port_open
    run_episode=collection_runtime.run_episode_subprocess

    def port_open(host=bind_host,port=collection_runtime.SERVER_PORT):
        return check_port(host,port)

    def run_episode_subprocess(task_root,task,scene,log_path,result_path,client_host,record,timeout_s,task_result_timeout_s):
        return run_episode(task_root,task,scene,log_path,result_path,
            f'{bind_host}:{collection_runtime.SERVER_PORT}',record,timeout_s,task_result_timeout_s)

    collection_runtime.port_open=port_open
    collection_runtime.run_episode_subprocess=run_episode_subprocess

old=collection_runtime.ServerConfig.argv
def argv(self):
    args=old(self)
    args[1]=str(SOURCE/'run_collection_server.py')
    return args
collection_runtime.ServerConfig.argv=argv
original_start=collection_runtime.start_server
original_stop=collection_runtime.stop_server
monitors={}

def resource_sample(proc):
    data={'wall_unix':time.time(),'server_pid':proc.pid,'returncode':proc.poll()}
    for path,key in [('/proc/'+str(proc.pid)+'/status','server_process_status'),
                     ('/sys/fs/cgroup/memory.events','cgroup_memory_events'),
                     ('/sys/fs/cgroup/memory.current','cgroup_memory_current')]:
        try:
            text=Path(path).read_text()
            data[key]=[line for line in text.splitlines() if line.startswith(('Name:','State:','VmRSS:','VmHWM:','VmSwap:','Threads:'))] if key=='server_process_status' else text
        except OSError:data[key]=None
    data['host_memory_available']=[line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')]
    return data

def start_server(cfg,log_path,timeout_s):
    proc=original_start(cfg,log_path,timeout_s);stop=threading.Event()
    evidence=Path(os.environ['GENIE_EPISODE_EVIDENCE']);evidence.mkdir(parents=True,exist_ok=True)
    def sample():
        with (evidence/'server-resource-samples.jsonl').open('a',buffering=1) as f:
            while not stop.is_set():
                f.write(json.dumps(resource_sample(proc))+'\n')
                if proc.poll() is not None:break
                stop.wait(10)
    thread=threading.Thread(target=sample,daemon=True);monitors[proc.pid]=(stop,thread,evidence)
    thread.start();return proc

def stop_server(proc,timeout_s=30):
    if proc is None:return original_stop(proc,timeout_s)
    before=resource_sample(proc)
    try:return original_stop(proc,timeout_s)
    finally:
        monitor=monitors.pop(proc.pid,None)
        if monitor:
            stop,thread,evidence=monitor;stop.set();thread.join(timeout=2)
            (evidence/'server-exit-status.json').write_text(json.dumps({
                'before_cleanup':before,'returncode_after_cleanup':proc.poll(),
                'unexpected_exit_before_cleanup':before['returncode'] is not None,
                'note':'A signal alone does not establish OOM; use recorded cgroup events and resource samples.'},indent=2))

collection_runtime.start_server=start_server
collection_runtime.stop_server=stop_server
runpy.run_path(str(MODULE/'scripts/collect_llm_task.py'),run_name='__main__')
