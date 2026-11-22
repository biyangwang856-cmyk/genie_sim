"""Automatic acceptance gate for continuous successful native collection episodes."""
from pathlib import Path
import hashlib,json,re,subprocess
import numpy as np
ROOT=Path('/home/vipuser/by/genie_sim')
MODULE=Path('/data/by/genie_sim/runtime-collection')


def recording_completed_after_timeout(report, task_result):
    episodes = report.get('episodes', [])
    if report.get('success') is not False or report.get('record') is not True or len(episodes) != 1:
        return False
    episode = episodes[0]
    return bool(episode.get('execute_task_success') is True and episode.get('task') == report.get('task')
                and re.fullmatch(r'task_result\.json not found within [1-9]\d*s', episode.get('error') or '')
                and task_result.get('task_status') is True and task_result.get('task_name') == episode.get('task'))


def audit(run):
    run=Path(run);checks={};errors=[]
    report=json.load(open(run/'collection_result.json'))
    checks['task_success']=report.get('success') is True
    episodes=report.get('episodes',[])
    if not episodes:return {'accepted':False,'errors':['No episode report'],'checks':checks}
    episode=episodes[0]
    raw=Path(episode.get('recording_dir',''))
    if str(raw).startswith('/workspace/'):raw=MODULE/raw.relative_to('/workspace')
    archive=run/'evidence/raw-archive.json'
    if archive.exists():raw=Path(json.loads(archive.read_text())['immutable_recording'])
    aligned=run/'evidence/aligned-archive.json'
    idle_prefix=None
    if aligned.exists():
        receipt=json.loads(aligned.read_text())
        if receipt.get('passed') and receipt.get('synthesized_frames')==0 and receipt.get('retained_measurements_exact'):
            raw=Path(receipt['aligned_recording'])
            idle_prefix=receipt.get('idle_prefix_proof')
    checks['raw_exists']=raw.is_dir()
    if not checks['raw_exists']:return {'accepted':False,'errors':['Missing recording directory'],'checks':checks}
    required=['state.json','metadata.yaml','recording_info.json','aligned_joints.h5','task_result.json']
    checks['required_files']=all((raw/p).is_file() for p in required)
    if not checks['required_files']:return {'accepted':False,'errors':['Missing required artifacts'],'checks':checks}
    state=json.load(open(raw/'state.json'));info=json.load(open(raw/'recording_info.json'));task=json.load(open(raw/'task_result.json'))
    late_recording = archive.is_file() and recording_completed_after_timeout(report, task)
    checks['task_success'] = checks['task_success'] or late_recording
    frames=state.get('frames',[]);ts=np.asarray([f['time_stamp'] for f in frames])
    checks['continuous_timestamps']=len(ts)>2 and bool(np.isfinite(ts).all() and np.all(np.diff(ts)>0))
    checks['regular_physical_capture_clock']=len(ts)>2 and bool(np.allclose(ts-ts[0],np.arange(len(ts))/30,atol=.00011,rtol=0))
    header_path=run/'evidence/capture-header-audit.json'
    header=json.loads(header_path.read_text()) if header_path.exists() else {}
    checks['original_sensor_headers_synchronized']=bool(header.get('passed') and header.get('native_frame_count')==len(frames)
        and header.get('state_sha256')==hashlib.sha256((raw/'state.json').read_bytes()).hexdigest())
    if idle_prefix:
        from reextract_recording import idle_prefix_proof
        original_state=run/'raw/state.json'
        original_frames=json.loads(original_state.read_text())['frames']
        original_headers=json.loads((run/'evidence/capture-header-audit-before-idle-trim.json').read_text())
        expected=original_frames[idle_prefix['discarded_idle_frames']:]
        measured=('time_stamp','robot','objects','articulated_object','cameras','ee','ee_center')
        checks['idle_prefix_preserved_all_commands_and_measurements']=bool(
            idle_prefix['source_state_sha256']==hashlib.sha256(original_state.read_bytes()).hexdigest()
            and idle_prefix_proof(run,original_frames,original_headers)==idle_prefix
            and len(expected)==len(frames)
            and all(old[key]==new[key] for old,new in zip(expected,frames) for key in measured))
    client_log=run/'client_1.log'
    reported_playback=False
    if client_log.is_file():
        with client_log.open(errors='replace') as log:
            reported_playback=any('Playback state ' in line for line in log)
    checks['no_state_rollback']=not info.get('playback_timerange') and not reported_playback
    checks['native_task_success']=task.get('task_status') is True
    checks['measured_states_finite']=all(np.isfinite(np.asarray(f['robot']['joints']['joint_position'])).all() for f in frames)
    controls=run/'evidence/native-control-commands.jsonl'
    checks['actual_controls_saved']=controls.is_file() and controls.stat().st_size>0
    if checks['actual_controls_saved']:
        commands=[json.loads(line) for line in controls.read_text().splitlines()]
        # None is the native sparse-action sentinel for an uncommanded DOF.
        # Validate every commanded number; a sentinel is not NaN or a zero command.
        checks['controls_finite']=all(np.isfinite(float(v))
            for c in commands for key in ['joint_positions','joint_velocities','joint_efforts']
            if c.get(key) is not None for v in c[key] if v is not None)
        resolved=[c for c in commands if 'applied_joint_positions' in c]
        if resolved:
            checks['resolved_controls_finite']=len(resolved)==len(commands) and all(
                np.isfinite(np.asarray(c[key],dtype=float)).all() for c in resolved
                for key in ['applied_joint_positions','applied_joint_velocities','applied_joint_efforts'] if c.get(key) is not None)
    evidence=run/'evidence/stage-evidence.jsonl'
    events=[json.loads(line) for line in evidence.read_text().splitlines()] if evidence.exists() else []
    # Recompute validation from this run's evidence; stale receipts cannot admit a run.
    if events:
        for validator in ['validate_render_alignment.py','validate_dynamic_render.py','validate_snapshot_fk.py']:
            validation=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/validator),str(run)],capture_output=True,text=True)
            if validation.returncode not in (0,1):errors.append(validator+': '+validation.stderr[-500:])
    launch=json.loads((run/'launch.json').read_text());expected=launch['source_sha256']
    loaded=[e for e in events if e['phase']=='program_loaded'];server_source=run/'evidence/runtime-source.json'
    server=json.loads(server_source.read_text()) if server_source.exists() else {}
    checks['executed_archived_source']=bool(loaded and loaded[0].get('path')==str(run/'source/collection_programs.py')
        and loaded[0].get('sha256')==expected['collection_programs.py']
        and server.get('server_wrapper_sha256')==expected['run_collection_server.py']
        and server.get('server_hooks_sha256')==expected['project_runtime_hooks.py'])
    from audit_plan_execution import audit_plan_execution
    checks['structured_model_plan_executed']=audit_plan_execution(run)['passed']
    request=json.loads((run/'source/request.json').read_text()) if (run/'source/request.json').exists() else {}
    if request.get('execution_mode')=='structured':
        modules=[e for e in events if e['phase']=='skill_modules_loaded']
        names=['robot_adapters','skill_registry','task_geometry','skill_executor','structured_planner','qwen_backend']
        checks['executed_archived_skill_modules']=bool(modules) and all(
            modules[0].get('modules',{}).get(name,{}).get('path')==str(run/'source'/(name+'.py'))
            and modules[0].get('modules',{}).get(name,{}).get('sha256')==expected[name+'.py'] for name in names)
    task_scene=run/'tasks'/report['task']/str(episode['scene'])/'scene_info.json'
    task_name=request.get('task_id') or json.loads(task_scene.read_text())['task_template']
    picks=[e for e in events if e['phase']=='pick_result']
    successful_picks={e['object'] for e in picks if e.get('api_success') is True and e.get('lift_m',0)>.055}
    scene=json.loads(task_scene.read_text());target=scene['official_instruction'].get('target') or {}
    required_picks={'insert_building_block':{target.get('id1')},
        'phone_packaging':{'benchmark_phone_000','benchmark_packaging_box_000','benchmark_packaging_box_002'},
        'put_the_cap_on_the_pen_new':{target.get('id1'),target.get('id2')}}.get(task_name,set())
    checks['physical_lift_evidence']=bool(required_picks) and required_picks<=successful_picks
    if task_name=='insert_building_block':
        entries=[e for e in events if e['phase']=='star_precise_entry_gate']
        finals=[e for e in events if e['phase']=='star_final_gate']
        checks['task_physical_gates']=bool(entries and finals and entries[-1]['position_error_m']<.004
            and entries[-1]['rotation_error_rad']<.035 and finals[-1].get('passed') is True)
        crossing=run/'evidence/hole-crossing-audit.json'
        from insertion_physics import crossing_report
        physics=run/'evidence/object-physics-120hz.jsonl'
        independent=crossing_report(physics,contract=json.loads((run/'source/insertion-certificate.json').read_text())) if physics.exists() else {'passed':False,'reason':'Task failed before release tracking began'}
        (run/'independent-hole-crossing.json').write_text(json.dumps(independent,indent=2))
        checks['native_hole_crossing']=crossing.exists() and json.loads(crossing.read_text()).get('passed') is True and independent['passed']
    elif task_name=='phone_packaging':
        rules=[e for e in events if e['phase']=='official_phone_rules']
        last_rule_by_stage={e['subtask_index']:e for e in rules}
        checks['task_physical_gates']=required_picks<=successful_picks and len(rules)>=6 and all(
            last_rule_by_stage.get(i,{}).get('success') is True for i in [1,3,5])
        environment=[e for e in events if e['phase']=='environment_collision_gate']
        checks['table_in_planner_world']=bool(environment and environment[0].get('passed') is True)
    elif task_name=='put_the_cap_on_the_pen_new':
        connections=[e for e in events if e['phase']=='cap_connection_gate']
        finals=[e for e in events if e['phase']=='pen_final_gate']
        checks['task_physical_gates']=required_picks<=successful_picks and bool(connections and finals and finals[-1].get('on_table')
            and any(finals[-1].get('snapshot',{}).get('joint_enabled',{}).values()))
    else:checks['task_physical_gates']=False
    if checks['task_success']:
        from audit_task_geometry import audit as audit_geometry
        checks['raw_measured_task_geometry']=audit_geometry(run,raw,frames)['passed']
    checks['not_diagnostically_aborted']=not (run/'diagnostic-stop.json').exists()
    checks['camera_tracking_no_errors']=not (run/'evidence/camera-follow-error.json').exists()
    alignment=run/'render-state-validation.json'
    checks['render_tracks_physics']=alignment.exists() and json.loads(alignment.read_text()).get('passed') is True
    dynamic=run/'dynamic-render-validation.json'
    checks['moving_render_tracks_physics']=dynamic.exists() and json.loads(dynamic.read_text()).get('passed') is True
    fk=run/'snapshot-fk-validation.json'
    checks['measured_robot_matches_urdf']=fk.exists() and json.loads(fk.read_text()).get('passed') is True
    attachments=run/'evidence/planner-attachment-events.jsonl'
    checks['held_collision_geometry_saved']=attachments.exists() and all(json.loads(line).get('success') is True for line in attachments.read_text().splitlines())
    videos={}
    for name,camera in info.get('camera_info',{}).items():
        video=raw/'observations/videos'/camera['output']['video']
        result=subprocess.run([str(ROOT/'.venv/bin/ffmpeg'),'-v','error','-nostats','-progress','pipe:1','-i',str(video),'-f','null','-'],capture_output=True,text=True)
        decoded=[int(v.split('=',1)[1]) for v in result.stdout.splitlines() if v.startswith('frame=')]
        videos[name]={'path':str(video),'decode_ok':result.returncode==0,'decoded_frames':decoded[-1] if decoded else 0}
        if result.returncode==0:
            pixels=subprocess.run([str(ROOT/'.venv/bin/ffmpeg'),'-v','error','-i',str(video),
                '-vf','scale=16:16,format=gray','-f','rawvideo','pipe:1'],capture_output=True,check=True).stdout
            pixels=np.frombuffer(pixels,dtype=np.uint8).reshape(-1,256)
            std=pixels.std(axis=1)
            videos[name]['noninformative_fraction']=float(np.mean(std<1.0))
            videos[name]['median_spatial_std']=float(np.median(std))
    checks['three_cameras_decodable']=len(videos)>=3 and all(v['decode_ok'] and v['decoded_frames']==len(frames) for v in videos.values())
    checks['three_cameras_informative']=len(videos)>=3 and all(v.get('noninformative_fraction',1)<.05 for v in videos.values())
    result={'accepted':bool(all(checks.values())),
        'physical_sample_accepted':bool(all(v for k,v in checks.items() if k!='structured_model_plan_executed')),
        'checks':checks,'raw_directory':str(raw),'frames':len(frames),'videos':videos,'errors':errors}
    if late_recording:
        result['native_recording_completed_after_runner_timeout'] = {
            'runner_report_sha256': hashlib.sha256((run/'collection_result.json').read_bytes()).hexdigest(),
            'native_task_result_sha256': hashlib.sha256((raw/'task_result.json').read_bytes()).hexdigest(),
            'runner_error': episode['error']}
    (run/'acceptance.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    import sys
    print(json.dumps(audit(sys.argv[1]),indent=2))
