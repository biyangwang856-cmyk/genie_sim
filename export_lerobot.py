"""Server-side native robot export, compatible with the team's LeRobot v2.1 reader.

All actions come from resolved actuator commands. The state vector never supplies
missing action values. Diagnostic physics baselines cannot enter training exports.
"""
import argparse,hashlib,json,shutil,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import av

ROOT=Path('/home/vipuser/by/genie_sim')
STORAGE=Path('/data/by/genie_sim')
FPS=30
CAMERAS={'head':'cam_high','hand_left':'cam_left_wrist','hand_right':'cam_right_wrist'}

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def stats(values):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim==1:a=a[:,None]
    return {**{key:fn(a,axis=0).tolist() for key,fn in [('min',np.min),('max',np.max),('mean',np.mean),('std',np.std)]},'count':[len(a)]}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def recording_path(run):
    receipt=Path(run)/'evidence/aligned-archive.json'
    return Path(json.loads(receipt.read_text())['aligned_recording']) if receipt.exists() else Path(run)/'raw'

def native_arrays(run,*,require_gains=True):
    run=Path(run);raw=recording_path(run)
    frames=json.loads((raw/'state.json').read_text())['frames']
    names=frames[0]['robot']['joints']['joint_name'];width=len(names)
    if len(set(names))!=width:raise ValueError('Duplicate measured joint name')
    state=[]
    for f in frames:
        joints=f['robot']['joints']
        if joints['joint_name']!=names:raise ValueError('Changing measured joint order')
        state.append(joints['joint_position'])
    state=np.asarray(state,dtype=np.float32);times=np.asarray([f['time_stamp'] for f in frames])
    if state.shape!=(len(frames),width) or not np.isfinite(state).all():raise ValueError('Invalid measured state')
    if not np.all(np.diff(times)>0):raise ValueError('Nonmonotone native observations')
    if not np.allclose(times-times[0],np.arange(len(times))/FPS,atol=.00011,rtol=0):raise ValueError('Native frames are not synchronized at 30 Hz')
    commands=[json.loads(line) for line in (run/'evidence/native-control-commands.jsonl').read_text().splitlines()]
    command_times=np.asarray([c['sim_time'] for c in commands])
    if not np.isfinite(command_times).all() or not np.all(np.diff(command_times)>=0):raise ValueError('Invalid command clock')
    # Native JSON frame times are rounded to four decimals. This tolerance only
    # resolves that quantization, never shifts commands by a control timestep.
    selected=np.searchsorted(command_times,times+.000051,side='right')-1
    if np.any(selected<0):raise ValueError('Observation precedes every archived actuator command')
    columns={'observation.state':state,'observation.sim_time':times.astype(np.float32),
        'action.command_time':command_times[selected].astype(np.float32)}
    fields={'action':'applied_joint_positions','action.velocity':'applied_joint_velocities','action.effort':'applied_joint_efforts'}
    for key,field in fields.items():
        values=[];valid=[]
        for index in selected:
            c=commands[index];order=c['joint_names']
            if len(order)!=width or set(order)!=set(names):raise ValueError('Command/observation joints differ')
            vector=c.get(field)
            if vector is None:vector=[None]*width
            if len(vector)!=width:raise ValueError('Resolved command has wrong dimension')
            reorder=[order.index(name) for name in names]
            valid.append([vector[i] is not None for i in reorder])
            values.append([0. if vector[i] is None else vector[i] for i in reorder])
        columns[key]=np.asarray(values,dtype=np.float32)
        columns[key+'.valid']=np.asarray(valid,dtype=np.float32)
    for key,field in [('action.stiffness','drive_stiffness'),('action.damping','drive_damping'),('action.effort_limit','drive_max_efforts')]:
        values=[]
        for index in selected:
            c=commands[index]
            if field not in c:
                if require_gains:raise ValueError('Missing actual control gains: '+field)
                break
            vector=np.asarray(c[field],dtype=np.float32).reshape(-1)
            if len(vector)!=width:raise ValueError('Drive parameter dimension mismatch')
            values.append(vector[[c['joint_names'].index(name) for name in names]])
        if len(values)==len(selected):columns[key]=np.asarray(values)
    if not all(np.isfinite(v).all() for v in columns.values()):raise ValueError('Nonfinite training feature')
    joints={j.get('name'):j.get('type') for j in ET.parse(run/'source/robot.urdf').getroot().findall('joint')}
    units=[]
    for name in names:
        kind=joints.get(name)
        if kind not in ['revolute','continuous','prismatic']:raise ValueError('Unspecified joint units: '+name)
        units.append('m' if kind=='prismatic' else 'rad')
    return columns,{'joint_names':names,'position_units':units,'joint_types':[joints[n] for n in names],
        'action_dimension':width,'action_semantics':'resolved native position targets; velocity/effort targets and validity masks are separate features',
        'feature_units':{'action':'position_units per joint','observation.state':'position_units per joint',
            'action.velocity':[unit+'/s' for unit in units],
            'action.effort':['N' if unit=='m' else 'Nm' for unit in units],
            'action.stiffness':['N/m' if unit=='m' else 'Nm/rad' for unit in units],
            'action.damping':['N*s/m' if unit=='m' else 'Nm*s/rad' for unit in units],
            'action.effort_limit':['N' if unit=='m' else 'Nm' for unit in units],
            'valid_masks':'unitless; 1 means native target present'},
        'control_activation':'Position/velocity target influence is defined by the archived stiffness/damping; passive zero-gain targets are not active commands.',
        'state_semantics':'measured native joint positions, in the same explicitly named joint order',
        'alignment':'last applied actuator command at each observation instant; zero-order hold; equal-time commands resolved by journal order',
        'command_clock_quantization_tolerance_s':.000051,'source_command_count':len(commands),
        'drive_parameters_present':all('action.'+k in columns for k in ['stiffness','damping','effort_limit']),
        'pi_action_adapter_verified':False}

def video_info(path):
    with av.open(str(path)) as container:
        stream=container.streams.video[0];fps=float(stream.average_rate)
        codec=stream.codec_context.name
        if abs(fps-FPS)>1e-6 or codec not in ['h264','hevc']:raise ValueError('Unexpected native video codec/clock')
        height,width,pix_fmt=stream.height,stream.width,stream.pix_fmt
    return {'video.height':height,'video.width':width,'video.codec':codec,'video.pix_fmt':pix_fmt,
        'video.is_depth_map':False,'video.fps':FPS,'video.channels':3,'has_audio':False}

def image_stats(path,expected_frames):
    # Decode the whole video; statistics use all frames at 64x64 nearest samples.
    result=subprocess.run([str(ROOT/'.venv/bin/ffmpeg'),'-v','error','-i',str(path),'-vf','scale=64:64:flags=neighbor','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],capture_output=True,check=True)
    pixels=np.frombuffer(result.stdout,dtype=np.uint8).reshape(-1,64*64,3)
    if len(pixels)!=expected_frames:raise ValueError('Exported camera frame count mismatch')
    a=pixels.astype(np.float64)/255.
    return {**{key:fn(a,axis=(0,1)).reshape(3,1,1).tolist() for key,fn in [('min',np.min),('max',np.max),('mean',np.mean),('std',np.std)]},'count':[len(pixels)]}

def export(runs,output,diagnostic=False):
    output=Path(output).resolve();runs=[Path(r).resolve() for r in runs]
    expected_parent=STORAGE/('diagnostic-exports' if diagnostic else 'exports')
    if output.parent!=expected_parent:raise ValueError('Export must use its designated project output directory')
    if output.exists():raise FileExistsError(output)
    from audit_episode import audit
    from batch_collect import audit_recording
    admissions=[]
    for run in runs:
        if run.parent!=STORAGE/'collection-runs':raise ValueError('Source is outside the project collection directory')
        if (run/'evidence/raw-archive.json').is_file():
            _,verdict=audit_recording(run)
        else:verdict=audit(run)
        if not verdict['accepted'] and not (diagnostic and verdict['physical_sample_accepted']):raise ValueError('Unaccepted source: '+str(run))
        admissions.append(verdict)
    output.mkdir(parents=True);meta=output/'meta';meta.mkdir();features=None;tasks=[];episodes=[];episode_stats=[];offset=0;sources=[]
    robot_id=json.loads((runs[0]/'launch.json').read_text())['robot_id']
    for ep,(run,verdict) in enumerate(zip(runs,admissions)):
        arrays,control_spec=native_arrays(run,require_gains=not diagnostic);length=len(arrays['observation.state'])
        launch=json.loads((run/'launch.json').read_text())
        if launch['robot_id']!=robot_id:raise ValueError('Each exported dataset must use one robot schema')
        scene_path=run/'tasks'/launch['task']/launch['scene']/'scene_info.json'
        scene=json.loads(scene_path.read_text())
        if str(scene['scene_id'])!=launch['scene']:raise ValueError('Executed scene identity mismatch')
        instruction=scene['official_instruction']['instruction']
        if instruction not in tasks:tasks.append(instruction)
        task_index=tasks.index(instruction);chunk=ep//1000
        arrays.update(timestamp=np.arange(length,dtype=np.float32)/FPS,frame_index=np.arange(length,dtype=np.int64),
            episode_index=np.full(length,ep,dtype=np.int64),index=np.arange(offset,offset+length,dtype=np.int64),task_index=np.full(length,task_index,dtype=np.int64))
        current={};hf={};arrow={};statistics={}
        for key,a in arrays.items():
            if a.ndim==2:
                current[key]={'dtype':'float32','shape':[a.shape[1]],'names':control_spec['joint_names']}
                hf[key]={'feature':{'dtype':'float32','_type':'Value'},'length':a.shape[1],'_type':'Sequence'}
                arrow[key]=pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1),type=pa.float32()),a.shape[1])
            else:
                dtype='int64' if a.dtype==np.int64 else 'float32'
                current[key]={'dtype':dtype,'shape':[1],'names':None};hf[key]={'dtype':dtype,'_type':'Value'};arrow[key]=pa.array(a)
            statistics[key]=stats(a)
        raw=recording_path(run);info=json.loads((raw/'recording_info.json').read_text())
        for camera,keyname in CAMERAS.items():
            source=raw/'observations/videos'/info['camera_info'][camera]['output']['video']
            key='observation.images.'+keyname;video_meta=video_info(source)
            current[key]={'dtype':'video','shape':[video_meta['video.height'],video_meta['video.width'],3],'names':['height','width','channels'],'info':video_meta}
            dest=output/f'videos/chunk-{chunk:03d}'/key/f'episode_{ep:06d}.mp4';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            if digest(source)!=digest(dest):raise ValueError('Video transfer mismatch')
            statistics[key]=image_stats(dest,length)
        if features is None:features=current
        elif features!=current:raise ValueError('Robot/control/camera feature schema changed across episodes')
        table=pa.table(arrow).replace_schema_metadata({b'huggingface':json.dumps({'info':{'features':hf}}).encode()})
        parquet=output/f'data/chunk-{chunk:03d}/episode_{ep:06d}.parquet';parquet.parent.mkdir(parents=True,exist_ok=True);pq.write_table(table,parquet,compression='zstd')
        back=pq.read_table(parquet)
        for key,values in arrays.items():np.testing.assert_array_equal(np.asarray(back[key].to_pylist(),dtype=values.dtype),values)
        provenance=output/f'provenance/episode_{ep:06d}';provenance.mkdir(parents=True)
        shutil.copytree(run/'source',provenance/'source');shutil.copytree(run/'evidence',provenance/'evidence')
        shutil.copytree(scene_path.parent,provenance/'executed-scene')
        shutil.copy2(scene_path.parent.parent/'data_gen.py',provenance/'task-loader.py')
        shutil.copy2(raw/'recording_info.json',provenance/'recording-info.json')
        plan_archive=run/'evidence/llm-plan-prepared.json'
        initial={'recorded_observation':{'sim_time_s':float(arrays['observation.sim_time'][0]),
            'joint_names':control_spec['joint_names'],'joint_positions':arrays['observation.state'][0].tolist()},'seed':launch['seed']}
        if plan_archive.is_file():
            initial['measured_at_plan_start']=json.loads(plan_archive.read_text()).get('runtime_initial_state')
        dump(provenance/'initial-state.json',initial)
        for name in ['launch.json','collection_result.json','acceptance.json','independent-hole-crossing.json','independent-task-geometry.json','plan-execution-audit.json','snapshot-fk-validation.json','dynamic-render-validation.json']:
            if (run/name).exists():shutil.copy2(run/name,provenance/name)
        shutil.copy2(raw/'task_result.json',provenance/'native-task-result.json')
        dump(provenance/'control-spec.json',control_spec)
        sources.append({'run':str(run),'episode_index':ep,'task_id':scene['task_template'],'scene_id':scene['scene_id'],
            'frames':length,'training_eligible':not diagnostic,'launch_sha256':digest(run/'launch.json')})
        episodes.append({'episode_index':ep,'tasks':[instruction],'length':length});episode_stats.append({'episode_index':ep,'stats':statistics});offset+=length
    dump(meta/'info.json',{'codebase_version':'v2.1','robot_type':f'geniesim_{robot_id}_native','total_episodes':len(runs),'total_frames':offset,
        'total_tasks':len(tasks),'total_videos':len(runs)*3,'total_chunks':(len(runs)+999)//1000,'chunks_size':1000,'fps':FPS,'splits':{'train':'0:'+str(len(runs))},
        'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet','video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4','features':features})
    for name,rows in [('episodes.jsonl',episodes),('episodes_stats.jsonl',episode_stats),('tasks.jsonl',[{'task_index':i,'task':s} for i,s in enumerate(tasks)])]:
        (meta/name).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    receipt={'format':'LeRobot v2.1','export_passed':True,'official_reader_verified':False,'training_eligible':not diagnostic,
        'frames':offset,'sources':sources,'numeric_readback_exact':True,'all_video_frames_decoded':True,'video_payloads_unchanged':True,
        'image_statistics':'all video frames; 64x64 nearest spatial sampling; RGB normalized to [0,1]',
        'reader_requirement':'Team official LeRobotDataset must load numeric features and first/middle/last multimodal samples before delivery.'}
    dump(output/'export_receipt.json',receipt);print(json.dumps({'output':str(output),**receipt}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--diagnostic-physics-baseline',action='store_true');a=p.parse_args()
    export(a.episode,a.output,a.diagnostic_physics_baseline)
