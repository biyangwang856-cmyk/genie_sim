"""Server-only isolated CDI launcher with immutable task/program snapshots."""
from dataclasses import dataclass
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,time
import errno
import socket
from robot_adapters import ROOT,load_robot
from scene_contract import require_id,verify_manifest
STORAGE=Path('/data/by/genie_sim')
SOCKET='unix:///data/by/genie_sim/dockerhub-sidebuild-20261119/docker.sock'
IMAGE='geniesim-by-collection:a100-20261119'
GPU='GPU-e5908cee-00db-8139-60af-09a2f7d872fc'
SECONDARY_GPU='GPU-82e9402e-b6e6-4fe7-9ba6-1e09e4362b41'
EPISODE_TIMEOUT_S=3450
RECORDING_RESULT_TIMEOUT_S=1200
SLOT_BUSY_EXIT=75
GIB = 1024**3
EXTRA_MEMORY_BYTES = 16 * GIB
HOST_RESERVE_BYTES = 3 * GIB
EXTRA_GPU_RESERVE_BYTES = 10 * GIB
MAX_FULL_MEMORY_STALL_PCT = 5.0
RUNTIME_ROBOT_CONFIG = 'config/robot_cfg'
SOURCE_FILES=['run_collection_task.py','run_collection_server.py','collection_programs.py','project_runtime_hooks.py','insertion_physics.py',
 'robot_adapters.py','skill_registry.py','task_geometry.py','skill_executor.py','structured_planner.py','qwen_backend.py',
 'goal_composition.py','skill_recovery.py','skill_capabilities.py','scene_contract.py']
SKILL_PROBE_FILES={
 'probe_open_hinged_lid':'hinge_fixture.py',
 'probe_rotate_held_about_tool':'rotation_probe.py',
}


@dataclass(frozen=True)
class CollectionSlot:
    gpu_uuid: str
    grpc_host: str
    ros_domain: str
    extra: bool = False


COLLECTION_SLOTS = {
    'primary': CollectionSlot(GPU, '127.0.0.1', '77'),
    'secondary': CollectionSlot(SECONDARY_GPU, '127.0.0.2', '78'),
    'gpu0-extra': CollectionSlot(SECONDARY_GPU, '127.0.0.3', '79', extra=True),
    'gpu1-extra': CollectionSlot(GPU, '127.0.0.4', '80', extra=True),
}


class CollectionDeferred(RuntimeError):
    def __init__(self, status, **details):
        super().__init__(status)
        self.status = status
        self.details = details


def execution_profile(physics_backend='cpu',slot='primary'):
 config=COLLECTION_SLOTS[slot]
 profile={'physics_backend':physics_backend,'planner_backend':'curobo_cuda','render_device':'cuda:0',
  'gpu_uuid':config.gpu_uuid,'image':IMAGE,'physics_frequency':120,'recording_fps':30,
  'render_sync':'native_world_render','motion_slowdown_factor':4.0,'joint_settle_timeout_s':2.0}
 if slot!='primary':profile.update(collection_slot=slot, grpc_host=config.grpc_host, ros_domain=config.ros_domain)
 if config.extra:profile['container_memory_bytes']=EXTRA_MEMORY_BYTES
 if os.environ.get('GENIE_EXACT_RENDER_CAMERA')=='1':profile['exact_render_camera']=True
 return profile

def running_collections(env,slot):
 output=subprocess.check_output(['docker','ps','--filter','name=genie-by-',
     '--format','{{.Names}}\t{{.Label "geniesim.collection-slot"}}'],env=env,text=True,timeout=15)
 active=[]
 for row in output.splitlines():
  name,_,label=row.partition('\t')
  if (label or 'primary')==slot:active.append(name)
 return active


def read_host_available():
    with Path('/proc/meminfo').open() as info:
        for line in info:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    raise RuntimeError('Linux did not report available host memory')


def read_gpu_available(gpu_uuid):
    output = subprocess.check_output(
        ['nvidia-smi', '--id=' + gpu_uuid, '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        text=True, timeout=10,
    )
    return int(output.strip()) * 1024**2


def read_full_memory_stall():
    try:
        rows = Path('/proc/pressure/memory').read_text().splitlines()
    except FileNotFoundError:
        return None
    for row in rows:
        if row.startswith('full '):
            return float(next(field[6:] for field in row.split() if field.startswith('avg10=')))
    return None


def container_working_set(env, names):
    if not names:
        return {}
    output = subprocess.check_output(
        ['docker', 'stats', '--no-stream', '--format', '{{.Name}}\t{{.MemUsage}}', *names],
        env=env, text=True, timeout=15,
    )
    units = {'B': 1, 'KiB': 1024, 'MiB': 1024**2, 'GiB': GIB}
    measured = {}
    for line in output.splitlines():
        name, usage = line.split('\t', 1)
        used = usage.split('/', 1)[0].strip()
        for unit, scale in sorted(units.items(), key=lambda pair: -len(pair[0])):
            if used.endswith(unit):
                measured[name] = int(float(used[:-len(unit)]) * scale)
                break
    return measured


def check_extra_capacity(env, config):
    # Docker reports memory with inactive file cache excluded, like MemAvailable.
    # Account for the memory an already admitted extra slot can still grow into.
    names = [
        name for slot, item in COLLECTION_SLOTS.items() if item.extra
        for name in running_collections(env, slot)
    ]
    measured = container_working_set(env, names)
    reserve = sum(max(0, EXTRA_MEMORY_BYTES - measured.get(name, 0)) for name in names)
    minimum = EXTRA_MEMORY_BYTES + HOST_RESERVE_BYTES + reserve
    available = read_host_available()
    if available < minimum:
        raise CollectionDeferred('waiting_for_host_memory', available_bytes=available,
                                 minimum_bytes=minimum, starting_worker_reserve_bytes=reserve)
    pressure = read_full_memory_stall()
    if pressure is not None and pressure >= MAX_FULL_MEMORY_STALL_PCT:
        raise CollectionDeferred('waiting_for_host_memory_pressure', full_stall_avg10_pct=pressure,
                                 maximum_full_stall_avg10_pct=MAX_FULL_MEMORY_STALL_PCT,
                                 available_bytes=available)
    gpu_available = read_gpu_available(config.gpu_uuid)
    if gpu_available < EXTRA_GPU_RESERVE_BYTES:
        raise CollectionDeferred('waiting_for_gpu_memory', gpu_uuid=config.gpu_uuid,
                                 available_bytes=gpu_available, minimum_bytes=EXTRA_GPU_RESERVE_BYTES)
    return {'host_available_bytes': available, 'host_minimum_bytes': minimum,
            'host_full_stall_avg10_pct': pressure, 'gpu_available_bytes': gpu_available,
            'gpu_minimum_bytes': EXTRA_GPU_RESERVE_BYTES}


def runtime_mounts(out, *, private_recording, private_storage):
    mode = 'ro' if private_storage else 'rw'
    mounts = [
        (STORAGE / 'runtime-collection', '/workspace', mode), (STORAGE, str(STORAGE), mode),
        (ROOT, str(ROOT), mode), (ROOT / 'GenieSimAssets', str(ROOT / 'GenieSimAssets'), 'ro'),
    ]
    if private_storage:
        mounts.extend([
            (out, str(out), 'rw'), (out / 'source', str(out / 'source'), 'ro'),
            (out / 'isolated-runtime' / RUNTIME_ROBOT_CONFIG, '/workspace/' + RUNTIME_ROBOT_CONFIG, 'rw'),
        ])
    if private_recording:
        recording = out / 'isolated-runtime' / 'recording_data'
        mounts.append((recording, '/workspace/recording_data', 'rw'))
    return mounts


def stage_runtime_robot_config(out):
    # Genie Sim writes both initial and reset-time Lula YAML next to the robot descriptions.
    original = STORAGE / 'runtime-collection' / RUNTIME_ROBOT_CONFIG
    staged = out / 'isolated-runtime' / RUNTIME_ROBOT_CONFIG
    shutil.copytree(original, staged, symlinks=True,
                    ignore=shutil.ignore_patterns('*_tmp.yaml', '*_tmp_rotated.yaml'))
    return {
        'source_directory': str(original),
        'working_directory': str(staged),
        'container_directory': '/workspace/' + RUNTIME_ROBOT_CONFIG,
        'template_file_sha256': {
            file.relative_to(staged).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(staged.rglob('*')) if file.is_file()
        },
    }


def check_port_available(config):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((config.grpc_host, 50051))
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            raise CollectionDeferred('waiting_for_collection_slot',
                                     endpoint=f'{config.grpc_host}:50051') from None


def main():
 from collection_scope import DYNAMIC_TASKS,TASKS
 p=argparse.ArgumentParser();p.add_argument('task',choices=[*TASKS,*DYNAMIC_TASKS,*SKILL_PROBE_FILES]);p.add_argument('--scene',type=require_id,default='0');p.add_argument('--label',required=True)
 p.add_argument('--physics-backend',choices=['cpu','gpu'],default='cpu')
 p.add_argument('--slot',choices=COLLECTION_SLOTS,default='primary')
 p.add_argument('--robot-id',default='g2crsb90d');p.add_argument('--execution-mode',choices=['baseline','structured'],default='baseline')
 p.add_argument('--instruction');p.add_argument('--prepared-plan',type=Path);p.add_argument('--seed',type=int)
 p.add_argument('--source-run',type=Path,help='Reuse the immutable code and model of an accepted run')
 p.add_argument('--scene-directory',type=Path,help='Use a prepared immutable task scene')
 p.add_argument('--scene-sha256')
 p.add_argument('--owner-job')
 args=p.parse_args()
 if bool(args.scene_directory) != bool(args.scene_sha256) or (args.scene_directory and args.source_run):
  p.error('A prepared scene requires its digest and cannot override a frozen run')
 if args.owner_job:require_id(args.owner_job)
 if args.task in SKILL_PROBE_FILES and (not args.scene_directory or args.prepared_plan or args.execution_mode!='structured'):
  p.error('Skill validation requires a prepared validation scene and structured runtime')
 import fcntl

 locks=STORAGE/'collection-slot-locks';locks.mkdir(exist_ok=True)
 try:
  with (locks/(args.slot+'.lock')).open('a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:raise CollectionDeferred('waiting_for_collection_slot') from None
   if COLLECTION_SLOTS[args.slot].extra:
    # Serialize capacity measurement and container registration across GPU slots.
    with (locks/'extra-capacity.lock').open('a') as admission:
     fcntl.flock(admission,fcntl.LOCK_EX)
     launch(args)
   else:launch(args)
 except CollectionDeferred as exc:
  print(json.dumps({'status':exc.status,'slot':args.slot,**exc.details}))
  raise SystemExit(SLOT_BUSY_EXIT) from None

def launch(args):
 frozen=args.source_run.resolve() if args.source_run else None
 dynamic=getattr(args,'scene_directory',None)
 scene_manifest=None
 if frozen:
  if frozen.parent!=STORAGE/'collection-runs':raise ValueError('Source recipe outside project')
  admission=json.loads((frozen/'acceptance.json').read_text());previous=json.loads((frozen/'launch.json').read_text())
  if admission.get('accepted') is not True or previous['scene']!=args.scene or previous['robot_id']!=args.robot_id or previous['execution_mode']!=args.execution_mode:
   raise ValueError('Frozen recipe does not match accepted robot/scene/execution mode')
  frozen_scene=frozen/'tasks'/previous['task']/previous['scene']/'scene_info.json'
  if json.loads(frozen_scene.read_text())['task_template']!=args.task:raise ValueError('Frozen task mismatch')
  if previous.get('scene_sha256'):
   scene_manifest=verify_manifest(frozen_scene.parent,ROOT/'GenieSimAssets')
   if scene_manifest['content_sha256']!=previous['scene_sha256']:
    raise ValueError('Frozen prepared scene differs from the accepted source')
   original_request=json.loads((frozen/'source/request.json').read_bytes())
   if (args.seed!=previous['seed'] or args.instruction!=original_request['instruction'] or
       args.slot!=previous['collection_slot'] or args.physics_backend!=previous['physics_backend'] or
       not args.prepared_plan or args.prepared_plan.read_bytes()!=(frozen/'source/prepared-plan.json').read_bytes()):
    raise ValueError('Frozen prepared-scene replay requires identical seed, plan, slot and runtime')
  os.environ['GENIE_RUN_SOURCE']=str(frozen/'source')
 adapter=load_robot(args.robot_id)
 if frozen:os.environ.pop('GENIE_RUN_SOURCE',None)
 if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.label):raise ValueError('Invalid run label')
 task=adapter.config['scene_task_prefix']+args.task
 if dynamic:
  dynamic=dynamic.resolve()
  if not dynamic.is_relative_to((ROOT/'jobs').resolve()) or dynamic.name!=args.scene:
   raise ValueError('Prepared scene is outside the project job or has a mismatched ID')
  scene_manifest=verify_manifest(dynamic,ROOT/'GenieSimAssets')
  if scene_manifest['content_sha256']!=args.scene_sha256:
   raise ValueError('Prepared scene differs from the planned instance')
 elif args.scene not in ('0','1') and not frozen:
  raise ValueError('Dynamic scene requires a prepared scene directory')
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
 if active:raise CollectionDeferred('waiting_for_collection_slot', containers=active)
 config=COLLECTION_SLOTS[args.slot]
 check_port_available(config)
 capacity=check_extra_capacity(env,config) if config.extra else None
 slot_profile=execution_profile(args.physics_backend,args.slot)
 out=STORAGE/'collection-runs'/args.label;out.mkdir(exist_ok=False)
 source=out/'source';source.mkdir()
 strict_frozen=bool(frozen and scene_manifest)
 files=(list(previous['source_sha256']) if strict_frozen else
        [name for name in SOURCE_FILES if (frozen/'source'/name).is_file()] if frozen else list(SOURCE_FILES))
 if args.task in SKILL_PROBE_FILES:files.append(SKILL_PROBE_FILES[args.task])
 if dynamic and args.task=='probe_rotate_held_about_tool':
  profile=json.loads((dynamic/'scene_info.json').read_bytes())['project_rotation_probe']
  if profile.get('completion')=='pack_phone':files.append('rotation_reuse.py')
 for f in files:
  origin=(frozen/'source' if frozen else ROOT)/f
  if strict_frozen and (Path(f).name!=f or hashlib.sha256(origin.read_bytes()).hexdigest()!=previous['source_sha256'][f]):
   raise ValueError('Source file changed in the accepted recipe')
  shutil.copy2(origin,source/f)
 from task_plugins import source_file
 plugin=source_file(args.task,frozen/'source' if frozen else None)
 if plugin and not strict_frozen:
  filename='task-contract-'+args.task+'.py'
  shutil.copy2(plugin,source/filename)
  shutil.copy2(ROOT/'task_plugins.py',source/'task_plugins.py')
  files.extend([filename,'task_plugins.py'])
 if frozen:
  profile=json.loads((frozen/'source/execution-profile.json').read_text())
  if profile!=slot_profile:raise ValueError('Execution environment differs from frozen recipe')
  shutil.copy2(frozen/'source/execution-profile.json',source/'execution-profile.json')
 else:(source/'execution-profile.json').write_text(json.dumps(slot_profile,sort_keys=True,indent=2))
 files+=['execution-profile.json']
 shutil.copy2(frozen/'source/insertion-certificate.json' if frozen else ROOT/'output/insertion-certificate.json',source/'insertion-certificate.json');files+=['insertion-certificate.json']
 shutil.copy2(frozen/'source/insertion-target.json' if frozen else ROOT/'output/insertion_target_v2.json',source/'insertion-target.json');files+=['insertion-target.json']
 taskroot=out/'tasks';taskroot.mkdir()
 if dynamic:
  shutil.copytree(dynamic,taskroot/task/args.scene)
 else:
  shutil.copytree((frozen/'tasks' if frozen else ROOT/'tasks')/task,taskroot/task)
 scene=json.loads((taskroot/task/args.scene/'scene_info.json').read_text())
 prepared=None
 if args.prepared_plan:
  from task_geometry import task_spec
  from skill_registry import validate_plan
  prepared=json.loads(args.prepared_plan.read_text())
  if prepared.get('validated') is not True:raise ValueError('Prepared LLM plan did not pass its contract')
  if json.loads(prepared['raw_response'])!=prepared['plan']:raise ValueError('Prepared plan differs from model response')
  if not strict_frozen:
   validate_plan(prepared['plan'],task_spec(scene,args.robot_id),adapter)
 seed=args.seed if args.seed is not None else int(hashlib.sha256(args.label.encode()).hexdigest()[:8],16)%(2**31)
 request={'instruction':args.instruction or (prepared['instruction'] if prepared else scene['official_instruction']['instruction']),'task_id':args.task,
  'source_task_id':task,'scene_id':args.scene,'robot_id':args.robot_id,'execution_mode':args.execution_mode,'seed':seed}
 if (os.environ.get('GENIE_RECOVERY_VALIDATION_PROBE')=='1'
     and dynamic and scene.get('project_packing_goal') and args.execution_mode=='structured'):
  request['validation_recovery_probe']=True
 if args.task in SKILL_PROBE_FILES:
  request['image_id']=image_id
 if not strict_frozen:(source/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
 files+=['request.json']
 if frozen:shutil.copy2(frozen/'source/robot-profile.json',source/'robot-profile.json')
 else:(source/'robot-profile.json').write_text(json.dumps(adapter.describe(),sort_keys=True,indent=2))
 files+=['robot-profile.json']
 if args.prepared_plan:
  shutil.copy2(args.prepared_plan,source/'prepared-plan.json');files+=['prepared-plan.json']
 precise=frozen/'source/precise_geometry.py' if frozen else ROOT/'genie_sim-main/source/geniesim_benchmark/src/geniesim_benchmark/plugins/ader/action/custom/precise_geometry.py'
 shutil.copy2(precise,source/'precise_geometry.py');files+=['precise_geometry.py']
 for metadata_path in (taskroot/task).rglob('task_info.json'):
  metadata=json.loads(metadata_path.read_text())
  if scene_manifest:
   if metadata.get('backend')!=args.physics_backend:raise ValueError('Prepared scene physics profile differs')
  else:
   metadata['backend']=args.physics_backend
   metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
 # Upstream client setup can prepend the mutable project root to sys.path.
 # Load only the archived modules by file path after that setup finishes.
 loader="""import importlib.util,os,sys
from pathlib import Path
source=Path(os.environ['GENIE_RUN_SOURCE'])
modules=['insertion_physics','robot_adapters']
modules.extend(name for name in ['scene_contract','skill_recovery','goal_composition','skill_capabilities','hinge_fixture','rotation_probe']
               if (source/(name+'.py')).is_file())
modules.extend(['skill_registry','task_geometry','qwen_backend','structured_planner','skill_executor','collection_programs'])
if (source/'rotation_reuse.py').is_file():modules.append('rotation_reuse')
if (source/'task_plugins.py').is_file():modules.insert(2,'task_plugins')
for name in modules:
    spec=importlib.util.spec_from_file_location(name,source/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
task_module=sys.modules.get('rotation_reuse',sys.modules['collection_programs'])
execute_task=task_module.execute_task
"""
 loader_file=taskroot/task/'data_gen.py'
 if strict_frozen:
  loader_sha256=hashlib.sha256(loader_file.read_bytes()).hexdigest()
  if loader_sha256!=previous['task_loader_sha256']:raise ValueError('Frozen recipe loader changed')
 else:
  loader_file.write_text(loader)
  loader_sha256=hashlib.sha256(loader.encode()).hexdigest()
 name='genie-by-'+args.label
 command=['docker','run','-d','--name',name,'--network=host','--user','1000:1000','--shm-size=8g',
          '--label','geniesim.collection-slot='+args.slot,'--device=nvidia.com/gpu='+slot_profile['gpu_uuid']]
 if getattr(args,'owner_job',None):command+=['--label','geniesim.owner-job='+args.owner_job]
 private_recording=bool(scene_manifest or args.slot!='primary')
 isolation=out/'isolated-runtime'
 if private_recording:(isolation/'recording_data').mkdir(parents=True)
 for a,b,mode in runtime_mounts(out,private_recording=private_recording,private_storage=config.extra):
  command+=['-v',f'{a}:{b}:{mode}']
 if config.extra:
  command+=['--memory',str(EXTRA_MEMORY_BYTES),'--memory-swap',str(EXTRA_MEMORY_BYTES),'--no-healthcheck']
 calibration=frozen/'source/robot-calibration.json' if frozen else adapter.calibration_path(ROOT)
 if calibration and calibration.exists():
  calibrated=json.loads(calibration.read_text())
  shutil.copy2(frozen/'source/robot.urdf' if frozen else calibrated['host_calibrated_urdf'],source/'robot.urdf')
  for target in adapter.calibrated_urdf_targets(calibrated):
   command+=['-v',str(source/'robot.urdf')+':'+target+':ro']
  shutil.copy2(calibration,source/'robot-calibration.json')
  files+=['robot.urdf','robot-calibration.json']
 assets=str((ROOT/'GenieSimAssets').resolve() if scene_manifest else ROOT/'GenieSimAssets')
 values={'HOME':str(STORAGE/'collection-home'),'XDG_CACHE_HOME':str(STORAGE/'kit-cache'),
 'OMNI_KIT_ACCEPT_EULA':'YES','ACCEPT_EULA':'Y','RUN_MODE':'trial','GENIE_COLLECTION_MODULE':'/workspace','GENIE_PROJECT_GPU':'0','GENIE_RUN_SOURCE':str(source),'GENIE_EPISODE_SEED':str(seed),
 'GENIE_PROJECT_ROOT':str(ROOT),
 'GENIE_EPISODE_EVIDENCE':str(out/'evidence'),'GENIESIM_LLM_TASK_ROOT':str(taskroot),'SIM_ASSETS':assets,'GENIESIM_ASSETS_PATH':assets,
 'PYTHONPATH':f'{source}:{ROOT}:/workspace:/workspace/scripts',
 'GENIE_SIM_GRPC_HOST':config.grpc_host,'ROS_DOMAIN_ID':config.ros_domain,'ROS_LOCALHOST_ONLY':'1'}
 if 'GENIE_QWEN_CONFIG' in os.environ:values['GENIE_QWEN_CONFIG']=os.environ['GENIE_QWEN_CONFIG']
 if slot_profile.get('exact_render_camera'):values['GENIE_EXACT_RENDER_CAMERA']='1'
 if scene_manifest:values['GENIE_GUARD_LOCAL_JOINT_BRANCH']='1'
 left_block=(args.task=='insert_building_block' and
             scene['official_instruction']['gripper']['1']['side']=='left')
 strict_wrist=bool(scene_manifest and args.robot_id=='g2crsb90d' and
                   (args.task=='put_the_cap_on_the_pen_new' or left_block))
 if strict_wrist:values['GENIE_ENFORCE_G2_COUPLED_WRIST']='1'
 if private_recording:
  for directory in ('home','cache'):(isolation/directory).mkdir()
  values.update(HOME=str(isolation/'home'),XDG_CACHE_HOME=str(isolation/'cache'))
 if config.extra:
  for directory in ('config','tmp','cuda','ros-logs'):(isolation/directory).mkdir()
  robot_config_template=stage_runtime_robot_config(out)
  values.update(XDG_CONFIG_HOME=str(isolation/'config'),TMPDIR=str(isolation/'tmp'),
                CUDA_CACHE_PATH=str(isolation/'cuda'),ROS_LOG_DIR=str(isolation/'ros-logs'),
                PYTHONDONTWRITEBYTECODE='1',MALLOC_ARENA_MAX='2',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
 for k,v in values.items():command+=['-e',f'{k}={v}']
 command+=['--entrypoint','/isaac-sim/python.sh',image_id,str(source/'run_collection_task.py'),'--task',task,'--scene',args.scene,'--run-times','1','--render','--server-start-timeout','600','--episode-timeout',str(EPISODE_TIMEOUT_S),
           '--task-result-timeout',str(RECORDING_RESULT_TIMEOUT_S),'--report-dir',str(out)]
 receipt={'task':task,'scene':args.scene,'robot_id':args.robot_id,'execution_mode':args.execution_mode,'seed':seed,'physics_backend':args.physics_backend,'planner_backend':'curobo_cuda','renderer_backend':'cuda',
 'container':name,'image':IMAGE,'source_sha256':{f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files},'task_loader_sha256':loader_sha256,'created_unix':time.time(),'accepted_success':False}
 receipt['collection_slot']=args.slot
 receipt['slot_endpoint']={'grpc_host':config.grpc_host,'grpc_port':50051,'ros_domain':config.ros_domain,
                           'gpu_uuid':config.gpu_uuid}
 receipt['strict_g2_coupled_wrist']=strict_wrist
 receipt['isolated_recording']=private_recording
 if config.extra:receipt['isolated_robot_config']=robot_config_template
 if capacity:receipt['capacity_at_launch']=capacity
 receipt['runner_limits_s']={'episode':EPISODE_TIMEOUT_S,'recording_result':RECORDING_RESULT_TIMEOUT_S}
 receipt['image_id']=image_id
 receipt['asset_release']=asset_identity
 receipt['initial_plan_sha256']=prepared['plan_sha256'] if args.prepared_plan else None
 receipt['source_recipe_run']=str(frozen) if frozen else None
 if scene_manifest:
  receipt['scene_sha256']=scene_manifest['content_sha256']
  receipt['scene_mode']=scene_manifest['mode']
 if getattr(args,'owner_job',None):receipt['owner_job']=args.owner_job
 result=subprocess.run(command,env=env,capture_output=True,text=True)
 receipt.update(launch_returncode=result.returncode,container_id=result.stdout.strip(),error=result.stderr[-2000:])
 (out/'launch.json').write_text(json.dumps(receipt,indent=2))
 print(json.dumps({'output':str(out),'container':name,'returncode':result.returncode}))
 raise SystemExit(result.returncode)

if __name__=='__main__':main()
