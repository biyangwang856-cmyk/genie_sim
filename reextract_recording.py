"""Repair extractor chunk-boundary loss from unchanged original ROS messages.

Never synthesize frames, interpolate joints, or replay/set a simulation state.
The immutable original recording and its first extraction are retained.
"""
import argparse,ast,asyncio,hashlib,json,os,shutil,subprocess,sys,types
from pathlib import Path
import numpy as np

ROOT=Path('/home/vipuser/by/genie_sim');STORAGE=Path('/data/by/genie_sim')
IMAGE='geniesim-by-collection:a100-20261119'


class CaptureHeaderMismatch(ValueError):
    pass


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ensure_capture_headers(run,raw):
    receipt=run/'evidence/capture-header-audit.json'
    if receipt.exists():
        result=json.loads(receipt.read_text())
        if result.get('passed') and result.get('state_sha256')==sha(raw/'state.json'):return
    env=dict(os.environ,DOCKER_HOST='unix:///data/by/genie_sim/dockerhub-sidebuild-20261119/docker.sock')
    command=['docker','run','--rm','--name','genie-util-headers-'+run.name,'--network','none','--user','1000:1000',
        '-v',str(ROOT)+':'+str(ROOT)+':ro','-v',str(STORAGE)+':'+str(STORAGE)+':rw',
        '--entrypoint','/isaac-sim/python.sh',IMAGE,str(ROOT/'audit_capture_clock.py'),str(raw),str(receipt)]
    with (run/'capture-header-audit.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    if not json.loads(receipt.read_text()).get('passed'):raise CaptureHeaderMismatch('Original camera/state header synchronization failed')

def clock(frames):
    times=np.asarray([f['time_stamp'] for f in frames]);expected=np.arange(len(times))/30
    return bool(len(times)>2 and np.all(np.diff(times)>0) and np.allclose(times-times[0],expected,atol=.00011,rtol=0)),times


def idle_prefix_proof(run, frames, report):
    counts = report.get('missing_exact_header_counts',{})
    examples = report.get('missing_header_examples',{})
    if counts.get('/joint_states') != 0 or counts.get('/tf') != 0 or report.get('gaps'):
        return None
    if len(counts) != 5 or any(len(examples.get(topic,[])) != count for topic,count in counts.items()):
        return None
    missing = sorted({time for stamps in examples.values() for time in stamps})
    if not missing or len(missing) >= len(frames):return None
    prefix = frames[:len(missing)+1]
    if [frame['time_stamp'] for frame in prefix[:-1]] != missing:return None
    start = prefix[-1]['time_stamp']
    # Allow only the idle recorder startup, before any task control is applied.
    with (run/'evidence/native-control-commands.jsonl').open() as stream:
        first_command = next((row['sim_time'] for line in stream if (row:=json.loads(line))['sim_time']>=missing[0]-.0001),None)
    if first_command is None or first_command <= start+.0001 or start-missing[0]>.20:
        return None
    initial = prefix[0]
    q = np.asarray(initial['robot']['joints']['joint_position'])
    joint_drift = max(float(np.max(np.abs(np.asarray(frame['robot']['joints']['joint_position'])-q))) for frame in prefix)
    pose_drift = max(float(np.max(np.abs(np.asarray(frame['objects'][name]['pose'])-pose['pose'])))
                     for frame in prefix for name,pose in initial['objects'].items())
    if joint_drift > .0001 or pose_drift > .0001:return None
    return {'discarded_idle_frames':len(missing),'discarded_sim_times_s':missing,'first_complete_sim_time_s':start,
            'first_subsequent_native_command_sim_time_s':first_command,'max_joint_drift_rad':joint_drift,
            'max_object_transform_element_drift':pose_drift,'source_state_sha256':report['state_sha256']}


def inside(run):
    script_bytes=Path(__file__).read_bytes()
    original=run/'raw';dest=run/'raw-aligned';partial=run/'raw-aligned.part'
    if dest.exists() or partial.exists():raise FileExistsError('Uncommitted or existing re-extraction')
    partial.mkdir()
    for path in original.iterdir():
        if path.suffix=='.mcap':os.symlink(str(path),partial/path.name)
        elif path.name in ['metadata.yaml','recording_info.json','frame_state.json'] or path.name.startswith('_'):
            if path.is_file():shutil.copy2(path,partial/path.name)
    runtime=Path('/workspace');source=runtime/'server/recording/extract_ros_bag.py';source_bytes=source.read_bytes();text=source_bytes.decode()
    provenance=run/'postprocess-source';provenance.mkdir(exist_ok=True)
    (provenance/'reextract_recording.py').write_bytes(script_bytes)
    (provenance/'extract_ros_bag.py').write_bytes(source_bytes)
    tree=ast.parse(text);count=0
    prefix_path=run/'evidence/idle-capture-prefix.json'
    prefix=json.loads(prefix_path.read_text()) if prefix_path.is_file() else None
    # Keep upstream transformations and extraction semantics intact. A single
    # chunk removes its per-chunk common-start/common-end trim at frame 1000.
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='chunk_size' for t in node.targets):
            legacy=isinstance(node.value,ast.Constant) and node.value.value==1000
            already_fixed=ast.dump(node.value)==ast.dump(ast.parse('len(time_stamps)',mode='eval').body)
            if not legacy and not already_fixed:raise ValueError('Unexpected upstream chunk size')
            node.value=ast.Constant(1000000000);count+=1
    if count!=1:raise ValueError('Could not identify upstream extraction chunk boundary')
    if prefix:
        if prefix['source_state_sha256']!=sha(original/'state.json'):raise ValueError('Idle-prefix source changed')
        extractor=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='RosExtrater')
        extract=next(node for node in extractor.body if isinstance(node,ast.AsyncFunctionDef) and node.name=='extract')
        scope=next(node for node in extract.body if isinstance(node,ast.With))
        boundary=next(i for i,node in enumerate(scope.body) if isinstance(node,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='chunk_size' for t in node.targets))
        start=float(prefix['first_complete_sim_time_s'])
        trim=ast.parse(f'time_stamps = [t for t in time_stamps if t >= {start!r}]').body[0]
        scope.body.insert(boundary,trim)
    ast.fix_missing_locations(tree);sys.path[:0]=[str(runtime),str(runtime/'server'),str(source.parent),str(runtime/'scripts')]
    module=types.ModuleType('project_full_span_extractor');module.__file__=str(source)
    exec(compile(tree,str(source),'exec'),module.__dict__)
    info=json.loads((partial/'recording_info.json').read_text())
    reader=module.RosExtrater(bag_file=str(partial),output_dir=str(partial),task_info=info,with_img=True,with_video=True)
    asyncio.run(reader.extract())
    from recording.sim_data_converter import SimDataConverter
    converter=SimDataConverter(str(partial),str(partial),0,0,0,info['gripper_names'],info['robot_name']);converter.convert()
    old=json.loads((original/'state.json').read_text())['frames'];new=json.loads((partial/'state.json').read_text())['frames']
    passed,times=clock(new)
    if not passed:raise ValueError('Single-span extraction still has a real missing capture; do not invent a frame')
    retained=old[prefix['discarded_idle_frames']:] if prefix else old
    if prefix and [f['time_stamp'] for f in new]!=[f['time_stamp'] for f in retained]:
        raise ValueError('Idle-prefix recovery changed the complete retained time span')
    indexed={f['time_stamp']:f for f in new};missing=[]
    for frame in retained:
        other=indexed.get(frame['time_stamp'])
        if other is None:missing.append(frame['time_stamp']);continue
        if frame['robot']['joints']!=other['robot']['joints'] or frame['objects']!=other['objects']:
            raise ValueError('Retained native measurement changed during re-extraction')
    if missing:raise ValueError('Re-extraction dropped previously retained measurements')
    # Task verdict is unchanged: only raw message extraction was repaired.
    shutil.copy2(original/'task_result.json',partial/'task_result.json')
    receipt={'passed':True,'aligned_recording':str(dest),'original_recording':str(original),
        'method':'unchanged original MCAP; upstream extraction with a single full-span chunk',
        'synthesized_frames':0,'interpolated_states':0,'original_frames':len(old),'aligned_frames':len(new),
        'retained_measurements_exact':True,'regular_30hz_physics_clock':True,
        'original_mcap_sha256':{p.name:sha(p) for p in original.glob('*.mcap')},
        'upstream_extractor_sha256':hashlib.sha256(source_bytes).hexdigest(),'repair_script_sha256':hashlib.sha256(script_bytes).hexdigest()}
    if prefix:
        receipt['method']='Original MCAP, from the first complete idle capture before any task command'
        receipt['idle_prefix_proof']=prefix
    partial.rename(dest)
    (run/'evidence/aligned-archive.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

def ensure_aligned(run):
    run=Path(run).resolve()
    if run.parent!=STORAGE/'collection-runs':raise ValueError('Run outside project')
    receipt=run/'evidence/aligned-archive.json'
    if receipt.exists():
        result=json.loads(receipt.read_text())
        if not result['passed']:raise ValueError('Failed re-extraction receipt')
        raw=Path(result['aligned_recording']);ensure_capture_headers(run,raw);return raw
    frames=json.loads((run/'raw/state.json').read_text())['frames'];passed,_=clock(frames)
    if passed:
        try:
            ensure_capture_headers(run,run/'raw')
            return run/'raw'
        except CaptureHeaderMismatch:
            audit=run/'evidence/capture-header-audit.json'
            prefix=idle_prefix_proof(run,frames,json.loads(audit.read_text()))
            if not prefix:raise
            original_audit=run/'evidence/capture-header-audit-before-idle-trim.json'
            if not original_audit.exists():shutil.copy2(audit,original_audit)
            (run/'evidence/idle-capture-prefix.json').write_text(json.dumps(prefix,indent=2))
    env=dict(os.environ,DOCKER_HOST='unix:///data/by/genie_sim/dockerhub-sidebuild-20261119/docker.sock')
    command=['docker','run','--rm','--name','genie-util-align-'+run.name,'--network','none','--user','1000:1000',
        '-v',str(ROOT)+':'+str(ROOT)+':ro','-v',str(STORAGE)+':'+str(STORAGE)+':rw',
        '-v',str(STORAGE/'runtime-collection')+':/workspace:ro','--entrypoint','/isaac-sim/python.sh',IMAGE,
        str(ROOT/'reextract_recording.py'),str(run),'--inside']
    with (run/'reextract.log').open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    raw=Path(json.loads(receipt.read_text())['aligned_recording']);ensure_capture_headers(run,raw);return raw

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--inside',action='store_true');a=p.parse_args()
    if a.inside:inside(a.run)
    else:print(ensure_aligned(a.run))
