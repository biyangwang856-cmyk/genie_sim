"""Automatic acceptance gate for continuous successful native collection episodes."""
from pathlib import Path
import hashlib,json,re,subprocess
from xml.etree import ElementTree
import numpy as np
from robot_adapters import ROOT
MODULE=Path('/data/by/genie_sim/runtime-collection')


def arm_command_rate_check(times, targets, active, joint_names, velocity_limits, min_dt):
    """Check active position command changes against the archived robot limits."""
    times = np.asarray(times, dtype=float)
    targets = np.asarray(targets, dtype=float)
    active = np.asarray(active, dtype=bool)
    limits = np.asarray(velocity_limits, dtype=float)
    if (len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0)
            or targets.shape != active.shape or targets.shape != (len(times), len(limits))
            or not len(limits) or not np.all(np.isfinite(limits) & (limits > 0))):
        raise ValueError('Invalid arm command clock, dimensions or URDF limits')
    dt = np.maximum(np.diff(times), min_dt)
    observed = active[1:] & active[:-1]
    if not observed.any():
        raise ValueError('No active arm position commands to verify')
    if not np.isfinite(targets[active]).all():
        raise ValueError('Nonfinite active arm position command')
    jumps = np.abs(np.diff(targets, axis=0))
    allowable = dt[:, None] * limits + .002
    invalid = observed & (jumps > allowable)
    ratios = np.where(observed, jumps / allowable, 0.)
    row, col = np.unravel_index(np.argmax(ratios), ratios.shape)
    return {
        'passed': not bool(invalid.any()), 'violations': int(invalid.sum()),
        'active_transitions': int(observed.sum()),
        'worst_joint': joint_names[col], 'worst_sim_time_s': float(times[row + 1]),
        'worst_step_rad': float(jumps[row, col]), 'step_limit_rad': float(allowable[row, col]),
        'urdf_velocity_limit_rad_s': float(limits[col]),
    }


def g2_arm_command_continuity(run, commands, observation_times, *, enforce_coupled_wrist=False):
    """Verify original drives and the commands selected into the training clock."""
    joints = ElementTree.parse(run/'source/robot.urdf').getroot().findall('joint')
    limits = {}
    for joint in joints:
        name = joint.attrib['name']
        if joint.get('type') == 'revolute' and ('_arm_l_joint' in name or '_arm_r_joint' in name):
            limits[name] = float(joint.find('limit').attrib['velocity'])
    names = list(limits)
    if len(names) != 14 or not commands:
        raise ValueError('Expected fourteen G2 arm limits and recorded commands')
    times, targets, active = [], [], []
    for command in commands:
        order = command['joint_names']
        cols = [order.index(name) for name in names]
        applied = command['applied_joint_positions']
        if applied is None:
            applied = [None] * len(order)
        gain = command['drive_stiffness']
        times.append(command['sim_time'])
        targets.append([0. if applied[i] is None else applied[i] for i in cols])
        active.append([applied[i] is not None and gain[i] > 0 for i in cols])
    native_times = np.asarray(times, dtype=float)
    targets = np.asarray(targets, dtype=float)
    active = np.asarray(active, dtype=bool)
    profile = json.loads((run/'source/execution-profile.json').read_bytes())
    native = arm_command_rate_check(native_times, targets, active, names, list(limits.values()),
                                    1. / profile['physics_frequency'])
    observation_times = np.asarray(observation_times, dtype=float)
    selected = np.searchsorted(native_times, observation_times + .000051, side='right') - 1
    if (selected < 0).any() or np.any(np.diff(observation_times) <= 0):
        raise ValueError('Training clock does not align to live arm controls')
    sampled = arm_command_rate_check(observation_times, targets[selected], active[selected], names,
                                     list(limits.values()), 1e-9)
    verdict = {'passed': native['passed'] and sampled['passed'], 'native_commands': native,
               'sampled_training_commands': sampled, 'model': 'source/robot.urdf',
               'position_margin_rad': .002}
    if enforce_coupled_wrist:
        from project_runtime_hooks import G2_COUPLED_WRIST_LIMIT_RAD, G2_COUPLED_WRIST_PAIRS
        pairs = {}
        for pair in G2_COUPLED_WRIST_PAIRS:
            columns = [names.index(name) for name in pair]
            indices = np.flatnonzero(active[:, columns].all(axis=1))
            if not len(indices):
                raise ValueError('No active position commands for the G2 coupled wrist: ' + pair[0])
            total = np.abs(targets[np.ix_(indices, columns)]).sum(axis=1)
            violations = int(np.count_nonzero(total > G2_COUPLED_WRIST_LIMIT_RAD + .002))
            worst = int(np.argmax(total))
            pairs[pair[0]] = {'active_samples': len(indices), 'violations': violations,
                              'max_combined_rad': float(total[worst]),
                              'worst_sim_time_s': float(native_times[indices[worst]])}
        passed = all(pair['violations'] == 0 for pair in pairs.values())
        verdict['coupled_wrist'] = {'passed': passed, 'limit_rad': G2_COUPLED_WRIST_LIMIT_RAD, 'pairs': pairs}
        verdict['passed'] = verdict['passed'] and passed
    return verdict


def recording_completed_after_timeout(report, task_result):
    episodes = report.get('episodes', [])
    if report.get('success') is not False or report.get('record') is not True or len(episodes) != 1:
        return False
    episode = episodes[0]
    return bool(episode.get('execute_task_success') is True and episode.get('task') == report.get('task')
                and re.fullmatch(r'task_result\.json not found within [1-9]\d*s', episode.get('error') or '')
                and task_result.get('task_status') is True and task_result.get('task_name') == episode.get('task'))


def camera_render_clock_passed(rows, stamps, topics):
    from project_runtime_hooks import camera_stamp_for_render

    captured = {topic: set() for topic in topics}
    expected = {round(float(t), 4) for t in stamps}
    for row in rows:
        if row['topic'] not in captured:
            return False
        try:
            stamp = camera_stamp_for_render(row['header_sim_time_s'], row['world_sim_time_after_render_s'],
                                            row['physics_dt_s'])
        except (TypeError, ValueError):
            return False
        captured[row['topic']].add(round(stamp, 4))
    return len(captured) == 3 and bool(expected) and all(expected <= times for times in captured.values())


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
    runtime_path = run/'evidence/runtime-source.json'
    runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else {}
    if runtime.get('camera_recording', {}).get('sampling') == 'after_native_render_of_capture_step':
        boundary = run/'evidence/camera-render-boundary.jsonl'
        topics = [topic for topic in header.get('topic_counts', {}) if topic.endswith('_compressed')]
        if boundary.is_file():
            with boundary.open() as stream:
                checks['atomic_camera_render_clock'] = camera_render_clock_passed(
                    (json.loads(line) for line in stream), ts, topics)
        else:
            checks['atomic_camera_render_clock'] = False
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
    if launch.get('scene_sha256'):
        from scene_contract import content_id, digest, read_json
        folder=run/'tasks'/launch['task']/launch['scene']
        manifest=read_json(folder/'scene-manifest.json')
        checks['exact_scene_snapshot']=(
            launch['scene_sha256']==manifest['content_sha256']==content_id(
                {key:value for key,value in manifest.items() if key!='content_sha256'})
            and all(digest(folder/name)==sha for name,sha in manifest['files'].items()))
        preflight=run/'evidence/scene-physics-preflight.json'
        checks['scene_initial_support_and_stability']=preflight.is_file() and read_json(preflight).get('passed') is True
        if launch['robot_id'] == 'g2crsb90d' and checks['task_success']:
            try:
                continuity = g2_arm_command_continuity(run, commands if checks['actual_controls_saved'] else [], ts,
                    enforce_coupled_wrist=bool(launch.get('strict_g2_coupled_wrist')))
            except (ValueError, KeyError, IndexError, TypeError, OSError, AttributeError, ElementTree.ParseError) as exc:
                continuity = {'passed': False, 'error': str(exc)}
            (run/'evidence/training-arm-command-continuity.json').write_text(json.dumps(continuity, indent=2))
            checks['training_arm_command_continuity'] = continuity['passed']
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
        names.extend(name for name in ['skill_recovery','goal_composition','skill_capabilities','hinge_fixture']
                     if name+'.py' in expected)
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
    plugin=(run/'source'/('task-contract-'+task_name+'.py')).is_file()
    if plugin:
        spec_path=run/'evidence/task-spec.json'
        if spec_path.is_file():required_picks=set(json.loads(spec_path.read_bytes())['manipulated_objects'])
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
        from goal_composition import packing_rule_indices
        rules=[e for e in events if e['phase']=='official_phone_rules']
        last_rule_by_stage={e['subtask_index']:e for e in rules}
        indices=packing_rule_indices(scene)
        if 'project_packing_goal' in scene:
            selected=['benchmark_phone_000','benchmark_packaging_box_000','benchmark_packaging_box_002'][:len(indices)]
            required_picks=set(selected)
            checks['physical_lift_evidence']=required_picks<=successful_picks
        checks['task_physical_gates']=required_picks<=successful_picks and len(rules)>=2*len(indices) and all(
            last_rule_by_stage.get(i,{}).get('success') is True for i in indices)
        environment=[e for e in events if e['phase']=='environment_collision_gate']
        checks['table_in_planner_world']=bool(environment and environment[0].get('passed') is True)
    elif task_name=='put_the_cap_on_the_pen_new':
        connections=[e for e in events if e['phase']=='cap_connection_gate']
        finals=[e for e in events if e['phase']=='pen_final_gate']
        checks['task_physical_gates']=required_picks<=successful_picks and bool(connections and finals and finals[-1].get('on_table')
            and any(finals[-1].get('snapshot',{}).get('joint_enabled',{}).values()))
    else:checks['task_physical_gates']=plugin and checks['physical_lift_evidence']
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
        'physical_sample_accepted':bool(all(v for k,v in checks.items()
            if k not in {'structured_model_plan_executed', 'training_arm_command_continuity'})),
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
