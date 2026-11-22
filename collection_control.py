"""Server-only isolated CDI launcher with immutable task/program snapshots."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,shutil,subprocess,time
import socket
from robot_adapters import load_robot
ROOT=Path('/home/vipuser/by/genie_sim')
STORAGE=Path('/data/by/genie_sim')
SOCKET='unix:///data/by/genie_sim/dockerhub-sidebuild-20261119/docker.sock'
IMAGE='geniesim-by-collection:a100-20261119'
GPU='GPU-e5908cee-00db-8139-60af-09a2f7d872fc'
SECONDARY_GPU='GPU-82e9402e-b6e6-4fe7-9ba6-1e09e4362b41'
SECONDARY_LOOPBACK='127.0.0.2'
SECONDARY_ROS_DOMAIN='78'
EPISODE_TIMEOUT_S=3450
RECORDING_RESULT_TIMEOUT_S=1200
SOURCE_FILES=['run_collection_task.py','run_collection_server.py','collection_programs.py','project_runtime_hooks.py','insertion_physics.py',
 'robot_adapters.py','skill_registry.py','task_geometry.py','skill_executor.py','structured_planner.py','qwen_backend.py']
def execution_profile(physics_backend='cpu',slot='primary'):
 profile={'physics_backend':physics_backend,'planner_backend':'curobo_cuda','render_device':'cuda:0',
  'gpu_uuid':SECONDARY_GPU if slot=='secondary' else GPU,'image':IMAGE,'physics_frequency':120,'recording_fps':30,
  'render_sync':'native_world_render','motion_slowdown_factor':4.0,'joint_settle_timeout_s':2.0}
 if slot=='secondary':profile.update(collection_slot=slot, grpc_host=SECONDARY_LOOPBACK, ros_domain=SECONDARY_ROS_DOMAIN)
 return profile

def running_collections(env,slot):
 output=subprocess.check_output(['docker','ps','--filter','name=genie-by-',
     '--format','{{.Names}}\t{{.Label "geniesim.collection-slot"}}'],env=env,text=True)
 active=[]
 for row in output.splitlines():
  name,_,label=row.partition('\t')
  if (label or 'primary')==slot:active.append(name)
 return active

def main():
 p=argparse.ArgumentParser();p.add_argument('task',choices=['phone_packaging','put_the_cap_on_the_pen_new','insert_building_block']);p.add_argument('--scene',choices=['0','1'],default='0');p.add_argument('--label',required=True)
 p.add_argument('--physics-backend',choices=['cpu','gpu'],default='cpu')
 p.add_argument('--slot',choices=['primary','secondary'],default='primary')
 p.add_argument('--robot-id',default='g2crsb90d');p.add_argument('--execution-mode',choices=['baseline','structured'],default='baseline')
 p.add_argument('--instruction');p.add_argument('--prepared-plan',type=Path);p.add_argument('--seed',type=int)
 p.add_argument('--source-run',type=Path,help='Reuse the immutable code and model of an accepted run');args=p.parse_args()
 locks=STORAGE/'collection-slot-locks';locks.mkdir(exist_ok=True)
 with (locks/(args.slot+'.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  launch(args)

def launch(args):
 frozen=args.source_run.resolve() if args.source_run else None
 if frozen:
  if frozen.parent!=STORAGE/'collection-runs':raise ValueError('Source recipe outside project')
  admission=json.loads((frozen/'acceptance.json').read_text());previous=json.loads((frozen/'launch.json').read_text())
  if admission.get('accepted') is not True or previous['scene']!=args.scene or previous['robot_id']!=args.robot_id or previous['execution_mode']!=args.execution_mode:
   raise ValueError('Frozen recipe does not match accepted robot/scene/execution mode')
  frozen_scene=frozen/'tasks'/previous['task']/previous['scene']/'scene_info.json'
  if json.loads(frozen_scene.read_text())['task_template']!=args.task:raise ValueError('Frozen task mismatch')
  os.environ['GENIE_RUN_SOURCE']=str(frozen/'source')
 adapter=load_robot(args.robot_id)
 if frozen:os.environ.pop('GENIE_RUN_SOURCE',None)
 if any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.label):raise ValueError('Invalid run label')
 task=adapter.config['scene_task_prefix']+args.task
 env=dict(os.environ,DOCKER_HOST=SOCKET)
 image_id=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',IMAGE],env=env,text=True).strip()
 if frozen and previous.get('image_id',image_id)!=image_id:raise ValueError('Execution image differs from the frozen recipe')
 asset_receipt=ROOT/'GenieSimAssets/.geniesim-assets.json'
 asset_identity=None
 if asset_receipt.is_file():
  data=asset_receipt.read_bytes();manifest=json.loads(data)
  asset_identity={k:manifest.get(k) for k in ('dataset','revision','release_id','file_count','status')}
  asset_identity['receipt_sha256']=hashlib.sha256(data).hexdigest()
 active=running_collections(env,args.slot)
 if active:raise RuntimeError('Wait for the active collection slot to finish: '+', '.join(active))
 if args.slot=='secondary':
  with socket.socket() as listener:
   listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
   listener.bind((SECONDARY_LOOPBACK,50051))
 slot_profile=execution_profile(args.physics_backend,args.slot)
 out=STORAGE/'collection-runs'/args.label;out.mkdir(exist_ok=False)
 source=out/'source';source.mkdir()
 files=list(SOURCE_FILES)
 for f in files:shutil.copy2((frozen/'source' if frozen else ROOT)/f,source/f)
 if frozen:
  profile=json.loads((frozen/'source/execution-profile.json').read_text())
  if profile!=slot_profile:raise ValueError('Execution environment differs from frozen recipe')
  shutil.copy2(frozen/'source/execution-profile.json',source/'execution-profile.json')
 else:(source/'execution-profile.json').write_text(json.dumps(slot_profile,sort_keys=True,indent=2))
 files+=['execution-profile.json']
 shutil.copy2(frozen/'source/insertion-certificate.json' if frozen else ROOT/'output/insertion-certificate.json',source/'insertion-certificate.json');files+=['insertion-certificate.json']
 shutil.copy2(frozen/'source/insertion-target.json' if frozen else ROOT/'output/insertion_target_v2.json',source/'insertion-target.json');files+=['insertion-target.json']
 taskroot=out/'tasks';taskroot.mkdir();shutil.copytree((frozen/'tasks' if frozen else ROOT/'tasks')/task,taskroot/task)
 scene=json.loads((taskroot/task/args.scene/'scene_info.json').read_text())
 prepared=None
 if args.prepared_plan:
  from task_geometry import task_spec
  from skill_registry import validate_plan
  prepared=json.loads(args.prepared_plan.read_text())
  if prepared.get('validated') is not True:raise ValueError('Prepared LLM plan did not pass its contract')
  if json.loads(prepared['raw_response'])!=prepared['plan']:raise ValueError('Prepared plan differs from model response')
  validate_plan(prepared['plan'],task_spec(scene,args.robot_id),adapter)
 seed=args.seed if args.seed is not None else int(hashlib.sha256(args.label.encode()).hexdigest()[:8],16)%(2**31)
 request={'instruction':args.instruction or (prepared['instruction'] if prepared else scene['official_instruction']['instruction']),'task_id':args.task,
  'source_task_id':task,'scene_id':args.scene,'robot_id':args.robot_id,'execution_mode':args.execution_mode,'seed':seed}
 (source/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2));files+=['request.json']
 if frozen:shutil.copy2(frozen/'source/robot-profile.json',source/'robot-profile.json')
 else:(source/'robot-profile.json').write_text(json.dumps(adapter.describe(),sort_keys=True,indent=2))
 files+=['robot-profile.json']
 if args.prepared_plan:
  shutil.copy2(args.prepared_plan,source/'prepared-plan.json');files+=['prepared-plan.json']
 precise=frozen/'source/precise_geometry.py' if frozen else ROOT/'genie_sim-main/source/geniesim_benchmark/src/geniesim_benchmark/plugins/ader/action/custom/precise_geometry.py'
 shutil.copy2(precise,source/'precise_geometry.py');files+=['precise_geometry.py']
 for metadata_path in (taskroot/task).rglob('task_info.json'):
  metadata=json.loads(metadata_path.read_text());metadata['backend']=args.physics_backend
  metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
 # Upstream client setup can prepend the mutable project root to sys.path.
 # Load only the archived modules by file path after that setup finishes.
 loader="""import importlib.util,os,sys
from pathlib import Path
source=Path(os.environ['GENIE_RUN_SOURCE'])
for name in ['insertion_physics','robot_adapters','skill_registry','task_geometry','qwen_backend','structured_planner','skill_executor','collection_programs']:
    spec=importlib.util.spec_from_file_location(name,source/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
execute_task=sys.modules['collection_programs'].execute_task
"""
 (taskroot/task/'data_gen.py').write_text(loader)
 name='genie-by-'+args.label
 command=['docker','run','-d','--name',name,'--network=host','--user','1000:1000','--shm-size=8g',
          '--label','geniesim.collection-slot='+args.slot,'--device=nvidia.com/gpu='+slot_profile['gpu_uuid']]
 for a,b,mode in [(STORAGE/'runtime-collection','/workspace','rw'),(STORAGE,str(STORAGE),'rw'),(ROOT,str(ROOT),'rw'),(ROOT/'GenieSimAssets',str(ROOT/'GenieSimAssets'),'ro')]:command+=['-v',f'{a}:{b}:{mode}']
 calibration=frozen/'source/robot-calibration.json' if frozen else adapter.calibration_path(ROOT)
 if calibration and calibration.exists():
  calibrated=json.loads(calibration.read_text())
  shutil.copy2(frozen/'source/robot.urdf' if frozen else calibrated['host_calibrated_urdf'],source/'robot.urdf')
  for target in adapter.calibrated_urdf_targets(calibrated):
   command+=['-v',str(source/'robot.urdf')+':'+target+':ro']
  shutil.copy2(calibration,source/'robot-calibration.json')
  files+=['robot.urdf','robot-calibration.json']
 values={'HOME':str(STORAGE/'collection-home'),'XDG_CACHE_HOME':str(STORAGE/'kit-cache'),
 'OMNI_KIT_ACCEPT_EULA':'YES','ACCEPT_EULA':'Y','RUN_MODE':'trial','GENIE_COLLECTION_MODULE':'/workspace','GENIE_PROJECT_GPU':'0','GENIE_RUN_SOURCE':str(source),'GENIE_EPISODE_SEED':str(seed),
 'GENIE_EPISODE_EVIDENCE':str(out/'evidence'),'GENIESIM_LLM_TASK_ROOT':str(taskroot),'SIM_ASSETS':str(ROOT/'GenieSimAssets'),'GENIESIM_ASSETS_PATH':str(ROOT/'GenieSimAssets'),
 'PYTHONPATH':f'{source}:{ROOT}:/workspace:/workspace/scripts'}
 if args.slot=='secondary':
  isolation=out/'isolated-runtime'
  for directory in ('home','cache','recording_data'):(isolation/directory).mkdir(parents=True)
  command+=['-v',f'{isolation}/recording_data:/workspace/recording_data:rw']
  values.update(HOME=str(isolation/'home'),XDG_CACHE_HOME=str(isolation/'cache'),
                GENIE_SIM_GRPC_HOST=SECONDARY_LOOPBACK,ROS_DOMAIN_ID=SECONDARY_ROS_DOMAIN,ROS_LOCALHOST_ONLY='1')
 for k,v in values.items():command+=['-e',f'{k}={v}']
 command+=['--entrypoint','/isaac-sim/python.sh',image_id,str(source/'run_collection_task.py'),'--task',task,'--scene',args.scene,'--run-times','1','--render','--server-start-timeout','600','--episode-timeout',str(EPISODE_TIMEOUT_S),
           '--task-result-timeout',str(RECORDING_RESULT_TIMEOUT_S),'--report-dir',str(out)]
 receipt={'task':task,'scene':args.scene,'robot_id':args.robot_id,'execution_mode':args.execution_mode,'seed':seed,'physics_backend':args.physics_backend,'planner_backend':'curobo_cuda','renderer_backend':'cuda',
 'container':name,'image':IMAGE,'source_sha256':{f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files},'task_loader_sha256':hashlib.sha256(loader.encode()).hexdigest(),'created_unix':time.time(),'accepted_success':False}
 receipt['collection_slot']=args.slot
 receipt['runner_limits_s']={'episode':EPISODE_TIMEOUT_S,'recording_result':RECORDING_RESULT_TIMEOUT_S}
 receipt['image_id']=image_id
 receipt['asset_release']=asset_identity
 receipt['initial_plan_sha256']=prepared['plan_sha256'] if args.prepared_plan else None
 receipt['source_recipe_run']=str(frozen) if frozen else None
 result=subprocess.run(command,env=env,capture_output=True,text=True)
 receipt.update(launch_returncode=result.returncode,container_id=result.stdout.strip(),error=result.stderr[-2000:])
 (out/'launch.json').write_text(json.dumps(receipt,indent=2))
 print(json.dumps({'output':str(out),'container':name,'returncode':result.returncode}))
 raise SystemExit(result.returncode)

if __name__=='__main__':main()
