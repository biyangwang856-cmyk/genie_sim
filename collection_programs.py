"""Task-specific RobotAPI programs with physical stage evidence and explicit failure."""
import json,os,time,importlib.util,signal
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from client.llm_gen.robot_api import GraspConfig,MoveConfig,PlaceConfig,FollowTrajectoryConfig

ROOT=Path('/home/vipuser/by/genie_sim')
def _abort_episode(signum,frame):
    raise RuntimeError('Project requested a clean diagnostic stop; episode is not a success')
signal.signal(signal.SIGUSR1,_abort_episode)

def top_down_grasps(poses,object_pose,widths):
    mask=np.asarray(poses)[:,2,2]<-.90
    return poses[mask],widths[mask]

def install_grasp_contact_compatibility():
    from client.planner.action.grasp import PickStage
    if getattr(PickStage,'_project_contact_compat',False):return
    original=PickStage.generate_action_sequence
    def generate(self,*args,**kwargs):
        sequence=original(self,*args,**kwargs)
        if sequence is not None and self.use_pre_grasp and len(sequence.actions)>=3:
            # Use a full 6D cuRobo goal; the inherited partial-pose cost can
            # reject the corrected wrist frame despite valid endpoint IK.
            sequence.actions[1].extra_params=dict(sequence.actions[1].extra_params,path_constraint=[])
        return sequence
    PickStage.generate_action_sequence=generate
    PickStage._project_contact_compat=True

def pose(robot,name):
    p,q=robot.get_object_pose(name)
    if p is None or q is None:raise RuntimeError(f'No live pose for {name}')
    t=np.eye(4);t[:3,:3]=Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix();t[:3,3]=p
    return t

def note(robot,phase,**data):
    folder=Path(os.environ.get('GENIE_EPISODE_EVIDENCE',ROOT/'output/collection-evidence'))
    folder.mkdir(parents=True,exist_ok=True)
    row={'phase':phase,'wall_time':time.time(),'task':robot.scene_info['task_template'],**data}
    with (folder/'stage-evidence.jsonl').open('a') as f:f.write(json.dumps(row,default=lambda v:np.asarray(v).tolist())+'\n')
    print('COLLECTION_PHASE',phase,flush=True)

def description(text):return {'zh':text,'en':text}

def local_box(robot,name):
    b=robot.scene_info['layout'][name]['entity_local_bbox']
    return np.asarray(b['min']),np.asarray(b['max'])

def center(robot,name):
    low,high=local_box(robot,name)
    return (pose(robot,name)@np.r_[(low+high)/2,1])[:3]

def custom_grasps(robot,name,axis=0,offset=None,vertical_offsets_m=(0.,.002,-.002),
                  yaw_offsets_rad=(0.,np.pi,-.12,.12),longitudinal_offsets_m=(0.,)):
    t=pose(robot,name); low,high=local_box(robot,name)
    c=(low+high)/2 if offset is None else np.asarray(offset)
    c=(t@np.r_[c,1])[:3]
    long=t[:3,axis].copy();long[2]=0
    if np.linalg.norm(long)<0.1:long=np.array([1.,0,0])
    long/=np.linalg.norm(long)
    poses=[]
    for angle in yaw_offsets_rad:
        x=Rotation.from_rotvec(np.array([0,0,angle])).apply(long)
        z=np.array([0.,0,-1.]);y=np.cross(z,x)
        for dz in vertical_offsets_m:
            for dx in longitudinal_offsets_m:
                target=np.eye(4);target[:3,:3]=np.column_stack([x,y,z]);target[:3,3]=c+dx*long+[0,0,dz];poses.append(target)
    return np.asarray(poses)

def packing_lid_grasps(robot,name):
    upright=custom_grasps(robot,name,axis=2,vertical_offsets_m=(0.,),yaw_offsets_rad=(0.,np.pi))
    long_axis=upright[0,:3,0]
    low,high=local_box(robot,name)
    width,height,length=high-low
    # These offset and tilt combinations passed the scene-0 calibrated URDF
    # screen with joint-limit margin. Native IK, collision and lift still apply.
    patterns=((1,.045,6.),(0,.055,6.),(1,.055,6.),(1,.025,10.),(1,.045,10.),(1,.055,10.))
    candidates=[];details=[]
    for direction,offset,tilt in patterns:
        angle=np.deg2rad(tilt)
        span=width/np.cos(angle)
        if span+.002>.085 or width*np.tan(angle)+.004>height or offset+.020>length/2:continue
        target=upright[direction].copy()
        target[:3,3]+=offset*long_axis
        target[:3,:3]=Rotation.from_rotvec(long_axis*angle).as_matrix()@target[:3,:3]
        candidates.append(target)
        details.append({'yaw_rad':direction*np.pi,'long_axis_offset_m':offset,'tilt_deg':tilt,'jaw_span_m':span})
    if not candidates:raise RuntimeError('No reach-screened grasp fits the original lid and gripper opening')
    note(robot,'packing_lid_grasp_candidates',object=name,scene_id=str(robot.scene_info['scene_id']),candidates=details,
         source='logs/phone-lid-kinematic-screen-20261121162326.json')
    return np.asarray(candidates)

def lift_packing_lid(robot,name,arm,initial_height):
    probe=pose(robot,name).copy()
    probe[2,3]+=.012
    if not move(robot,name,arm,probe,'packing_lid_world_z_probe'):return False
    probe_lift=float(center(robot,name)[2]-initial_height)
    retained=.006<=probe_lift<=.022
    note(robot,'packing_lid_probe_following',object=name,commanded_lift_m=.012,measured_lift_m=probe_lift,passed=retained)
    if not retained:return False

    current=pose(robot,name)
    inward=pose(robot,'benchmark_packaging_box_001')[:3,3]-current[:3,3]
    inward[2]=0.
    inward/=max(float(np.linalg.norm(inward)),1e-9)
    offsets=(0.,.025,.055)
    lift_goals=np.repeat(current[None],len(offsets),axis=0)
    lift_goals[:,:3,3]+=np.array(offsets)[:,None]*inward
    lift_goals[:,2,3]+=.10-probe_lift
    object_to_hand=np.linalg.inv(current)@robot.get_ee_pose(arm)
    mask,_=robot.robot.solve_ik(lift_goals@object_to_hand,ee_type='gripper',arm=arm,type='AvoidObs')
    feasible=np.asarray(mask,dtype=bool).reshape(-1)
    if len(feasible)!=len(lift_goals):raise RuntimeError('Invalid lid lift avoidance result')
    note(robot,'packing_lid_lift_candidates',object=name,inward_offsets_m=offsets,avoidance_passed=feasible)
    return bool(feasible.any() and move(robot,name,arm,lift_goals[feasible],'packing_lid_world_lift'))

def detach_planner_payload(robot,arm):
    receipt=robot.client.command_any({'command':'ProjectDetachPlannerArm','arm':arm})
    link='attached_object' if arm=='right' else 'left_attached_object'
    if not isinstance(receipt,dict) or receipt.get('success') is not True or link in receipt.get('planner_attachments',{}):
        raise RuntimeError('Held object did not detach from the planner: '+str(receipt))
    robot.attached_obj_id=None
    phase='packing_payload_detached' if robot.scene_info['task_template']=='phone_packaging' else 'planner_payload_detached'
    note(robot,phase,arm=arm,receipt=receipt)

def require_payload_proxy(robot, name, arm, phase):
    prim = robot.get_object_prim_path(name)
    link = 'attached_object' if arm == 'right' else 'left_attached_object'
    paths = robot.client.command_any({'command': 'ProjectSnapshot'}).get('planner_attachments', {}).get(link, [])
    passed = bool(paths) and all(path == prim or path.startswith(prim + '/') for path in paths)
    note(robot, phase, object=name, arm=arm, paths=paths, passed=passed)
    if not passed:
        raise RuntimeError('Measured payload is absent from the active planner link: ' + name)


def pick(robot,name,arm,axis=0,offset=None,goal_topdown=False,goal_lookahead=False,
         pinch_axis_offset_m=0.,release_tcp_x_points_up=False,pinch_height_offsets_m=(0.,.002,-.002),
         custom_pose_options=None,lift_probe_distance_m=None,lift_recovery=None,lift_tracking_recovery=None):
    kinematic=robot.client.command_any({'command':'ProjectKinematicSnapshot'})
    note(robot,'model_snapshot',snapshot=kinematic)
    info=robot.scene_info['layout'][name];interaction=robot.interactions.get(info['usd'],{})
    annotations=interaction.get('passive',{}).get('grasp',{})
    packing_grasp = (robot.scene_info['task_template'] == 'phone_packaging' and not annotations
                     and offset is None and name in {'benchmark_phone_000','benchmark_packaging_box_000','benchmark_packaging_box_002'})
    phone_grasp = packing_grasp and name == 'benchmark_phone_000'
    insert_grasp = packing_grasp and name == 'benchmark_packaging_box_000'
    lid_grasp = packing_grasp and info['usd'] == 'benchmark_packaging_box_002' and arm == 'right'
    if packing_grasp:
        # A guarded failure leaves the live gripper at the failed pose. Open
        # the same arm before the next pick, then clear its old planner proxy.
        robot.robot.open_gripper(id=arm, width=1.0, detach=False)
        detach_planner_payload(robot,arm)
        note(robot, 'packing_grasp_opened', object=name, arm=arm)
    before=center(robot,name)
    if pinch_axis_offset_m and (not goal_topdown or not annotations or offset is not None):
        raise ValueError('An axial pinch offset requires a release-compatible grasp on an annotated asset')
    # Native loading can synthesize a default annotation. Honor the caller's
    # measured geometry when it explicitly selects a procedural grasp.
    use_annotated_grasp = bool(annotations) and offset is None and (goal_topdown or custom_pose_options is None)
    # Asset TCP is now the wrist-aligned 185 mm frame, not the legacy rotated TCP.
    legacy=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    robot.robot.robot_gripper_2_grasp_gripper=legacy@Rotation.from_rotvec([0,0,-np.pi/2 if arm=='left' else np.pi/2]).as_matrix()
    # Preserve the wrist goal of grasp annotations authored for the 229.22 mm
    # contact frame while commanding the real 185 mm USD marker. Rotation-only
    # conversion put the fingers 44.22 mm too deep into the object/table.
    contact_frame_offset=.185-.2292200029
    # The native grasp prefilter applies lift offsets in the tool frame, where
    # the calibrated lid approach points down. Command the lift in world space.
    lift_probe = .012 if phone_grasp else lift_probe_distance_m
    if phone_grasp:
        lift_recovery = None
    if lift_probe is not None and (not np.isfinite(lift_probe) or not 0. < lift_probe < .10):
        raise ValueError('Grasp lift probe must be between zero and the full lift distance')
    pick_up_distance=0. if lid_grasp else (lift_probe if lift_probe is not None else .10)
    config=GraspConfig(grasp_offset=contact_frame_offset,pre_grasp_offset=.09,pick_up_distance=pick_up_distance,
        # Legacy up-side filtering tests the old rotated TCP's Y axis.
        # Reachability, collision checks and native humanlike ranking remain on.
        flip_grasp=False,disable_upside_down=False,action_description=description('Pick up '+name))
    if use_annotated_grasp:
        if goal_topdown:
            geometry=json.loads((Path(os.environ.get('GENIE_RUN_SOURCE',ROOT))/'insertion-target.json').read_text())
            from insertion_physics import measure,certificate
            target=np.asarray(geometry['object_goal_in_box_frame']);box=pose(robot,geometry['box']);goal=box@target;initial=pose(robot,name)
            if measure(goal,box)['outside_hole_area_fraction']>.001:raise RuntimeError('Native mesh does not fit aperture')
            robot._project_shape_goal_local=target
            # Existing thin-face grasps rotate their closing axis vertical at
            # the aperture. Pinch across the star plane instead, while keeping
            # the initial approach top-down and clear of the table.
            mesh=np.asarray(certificate()['star_triangles_entity_frame']).reshape(-1,3)
            world=(initial@np.c_[mesh,np.ones(len(mesh))].T).T[:,:3]
            contact_center_entity=np.asarray(geometry['star_center_entity'])
            if pinch_axis_offset_m:
                insertion_up=target[:3,:3].T@np.array([0.,1.,0.])
                upper_extent=float(np.max((mesh-contact_center_entity)@insertion_up))
                if not 0.<pinch_axis_offset_m<upper_extent:
                    raise ValueError('Axial pinch offset places the grasp center outside the original star')
                contact_center_entity=contact_center_entity+pinch_axis_offset_m*insertion_up
            contact_center=(initial@np.r_[contact_center_entity,1])[:3]
            closing=np.cross([0.,0.,1.],initial[:3,1]);closing/=np.linalg.norm(closing)
            candidates=[];widths=[];release_x_z=[]
            for angle in [0.,np.pi,-.04,.04]:
                y=Rotation.from_rotvec([0,0,angle]).apply(closing);z=np.array([0.,0.,-1.]);x=np.cross(y,z)
                width=float(np.ptp(world@y))+.008
                if width>.085:continue
                for dz in pinch_height_offsets_m:
                    t=np.eye(4);t[:3,:3]=np.column_stack([x,y,z]);t[:3,3]=contact_center+[0,0,dz]
                    final=goal[:3,:3]@initial[:3,:3].T@t[:3,:3]
                    if abs(final[2,1])>=.15:continue
                    release_x_z.append(float(final[2,0]))
                    # A half-turn of the TCP keeps the closing axis horizontal
                    # but changes the actual clearance of the asymmetric G2 hand.
                    if release_tcp_x_points_up and final[2,0]<.95:continue
                    candidates.append(t);widths.append(width)
            if release_tcp_x_points_up:
                note(robot,'shape_release_wrist_roll_candidates',tcp_x_world_z=release_x_z,retained=len(candidates))
            if not candidates:raise RuntimeError('No release-compatible pinch fits the native gripper aperture')
            config.grasp_pose=np.asarray(candidates);config.grasp_pose_frame='world';config.grasp_width=np.asarray(widths)
            note(robot,'release_compatible_grasp_candidates',source='analytic_original_mesh_cross_section',poses=config.grasp_pose,widths_m=config.grasp_width,
                pinch_axis_offset_m=pinch_axis_offset_m,pinch_height_offsets_m=pinch_height_offsets_m,contact_center_entity_m=contact_center_entity,
                criterion='initial top-down approach; horizontal closing axis at release; native IK and collision checks retained')
            if goal_lookahead:
                lookahead_ik=robot.robot.solve_ik
                def downstream_filter(poses,object_pose,widths):
                    base=pose(robot,geometry['box'])@target;mesh_center=np.asarray(geometry['star_center_entity'])
                    point=(base@np.r_[mesh_center,1])[:3];goals=[]
                    for yaw in np.arange(5)*2*np.pi/5:
                        g=base.copy();g[:3,:3]=Rotation.from_rotvec([0,0,yaw]).as_matrix()@base[:3,:3]
                        g[:3,3]=point-g[:3,:3]@mesh_center;g[2,3]+=.07;goals.append(g)
                    tcp=np.asarray(poses).copy();tcp[:,:3,3]+=tcp[:,:3,2]*contact_frame_offset
                    prospective=np.asarray([g@np.linalg.inv(object_pose)@t for t in tcp for g in goals])
                    simple=np.asarray(lookahead_ik(prospective,ee_type='gripper',arm=arm,type='Simple')[0]).reshape(len(tcp),len(goals))
                    avoid=np.asarray(lookahead_ik(prospective,ee_type='gripper',arm=arm,type='AvoidObs')[0]).reshape(len(tcp),len(goals))
                    collision_mask=avoid.any(axis=1);mask=collision_mask if collision_mask.any() else simple.any(axis=1)
                    note(robot,'downstream_grasp_ik_filter',candidate_count=len(tcp),simple_goal_counts=simple.sum(axis=1),
                        avoid_goal_counts=avoid.sum(axis=1),retained=int(mask.sum()),
                        collision_precheck='passed_empty_hand_proxy' if collision_mask.any() else 'inconclusive_empty_hand_proxy; actual held-object avoidance remains mandatory')
                    return np.asarray(poses)[mask],np.asarray(widths)[mask]
                config.grasp_pose_filter=downstream_filter
        else:config.grasp_pose_filter=top_down_grasps
    if not use_annotated_grasp:
        config.disable_upside_down=False
        if phone_grasp:
            # The native contact traces show the straight jaws lift from both
            # ends. The off-axis jaws pushed the thin phone sideways.
            vertical_offsets_m=(-.002,)
            yaw_offsets_rad=(0.,np.pi)
            note(robot,'phone_grasp_contact_height',object=name,vertical_offsets_m=vertical_offsets_m,
                 yaw_offsets_rad=yaw_offsets_rad,
                 reference_runs=['phone-packaging-s0-20261121104804','phone-packaging-s0-20261121112952'])
            config.grasp_pose=custom_grasps(robot,name,axis,offset,vertical_offsets_m,yaw_offsets_rad)
        elif insert_grasp:
            # This original candidate lifted the tray 100.24 mm. The lower
            # opposite wrist orientation passed IK but failed to plan departure.
            vertical_offsets_m=(0.,)
            yaw_offsets_rad=(-.12,)
            note(robot,'packing_insert_grasp_candidate',object=name,vertical_offsets_m=vertical_offsets_m,
                 yaw_offsets_rad=yaw_offsets_rad,reference_run='phone-packaging-s0-20261120212557',
                 reference_candidate_index=6)
            config.grasp_pose=custom_grasps(robot,name,axis,offset,vertical_offsets_m,yaw_offsets_rad)
        elif lid_grasp:
            config.grasp_pose=packing_lid_grasps(robot,name)
        else:
            config.grasp_pose=custom_grasps(robot,name,axis,offset,**(custom_pose_options or {}))
        config.grasp_pose_frame='world'
        low,high=local_box(robot,name)
        # Width is a candidate opening, never a success label or physical pin.
        config.grasp_width=np.full(len(config.grasp_pose),min(.085,max(.018,np.sort(high-low)[1]+.01)))
    note(robot,'pick_start',object=name,arm=arm,annotation_labels=list(annotations),before_pose=pose(robot,name),
         grasp_candidate_source='native_annotation' if use_annotated_grasp and not goal_topdown else 'computed_world_pose',
         contact_frame_offset_m=contact_frame_offset)
    # The grasp target must be contactable. Only its planner proxy is excluded;
    # PhysX collision/contact remains active for the actual grasp and lift.
    original_ik=robot.robot.solve_ik
    def traced_ik(poses,*args,**kwargs):
        result=original_ik(poses,*args,**kwargs)
        flags=np.asarray(result[0])
        note(robot,'pick_ik_filter',object=name,arm=arm,solver=kwargs.get('type','Simple'),
             count=int(flags.size),passed=int(flags.sum()),poses=np.asarray(poses),success=flags)
        if kwargs.get('type')=='AvoidObs' and flags.size and not flags.any():
            current=np.asarray(robot.get_ee_pose(arm))
            current_result=original_ik(current[None],*args,**kwargs)
            current_flags=np.asarray(current_result[0])
            note(robot,'pick_current_pose_ik_diagnostic',object=name,arm=arm,passed=int(current_flags.sum()),count=int(current_flags.size),
                interpretation='A rejected current pose suggests a globally inconsistent collision world or held inactive arm, not just an unreachable grasp target.')
            # Diagnostic-only height sweep. Never executes these alternative
            # poses or changes a collision model to force a success verdict.
            for height in (.04,.08,.12):
                shifted=np.asarray(poses).copy();shifted[:,2,3]+=height
                probe=original_ik(shifted,*args,**kwargs)
                mask=np.asarray(probe[0])
                note(robot,'pick_ik_height_diagnostic',object=name,arm=arm,height_m=height,
                     count=int(mask.size),passed=int(mask.sum()))
        return result
    robot.robot.solve_ik=traced_ik
    target_path=robot.get_object_prim_path(name)
    robot.remove_objs_from_obstacle([target_path])
    if packing_grasp:
        robot.client.command_any({'command': 'ProjectTrackObjects', 'paths': {name: target_path}})
    try:
        ok=robot.pick_actor(name,arm=arm,grasp_config=config)
    finally:
        robot.robot.solve_ik=original_ik
        robot.add_objs_to_obstacle([target_path])
        if packing_grasp:
            robot.client.command_any({'command': 'ProjectTrackObjects', 'paths': {}})
    if lift_probe is not None and ok:
        probe_lift = float(center(robot, name)[2] - before[2])
        retained = .5 * lift_probe <= probe_lift <= lift_probe + .01
        probe_phase = 'phone_grasp_probe' if phone_grasp else 'grasp_lift_probe'
        note(robot, probe_phase, measured_lift_m=probe_lift, commanded_lift_m=lift_probe, passed=retained)
        if retained:
            lift_goal = pose(robot, name).copy()
            if not phone_grasp:
                require_payload_proxy(robot, name, arm, 'grasp_probe_payload_attachment')
                held = np.linalg.inv(robot.get_ee_pose(arm)) @ lift_goal
            lift_goal[2, 3] += .10 - probe_lift
            phase = 'phone_full_lift_after_probe' if phone_grasp else 'grasp_full_lift_after_probe'
            from skill_executor import SkillFailure
            try:
                ok = move(robot, name, arm, lift_goal, phase)
            except SkillFailure as exc:
                if lift_recovery is None or exc.kind != 'native_action_failed_before_rollback':
                    raise
                ok = False
            if not ok and lift_recovery is not None:
                lift_recovery(lift_goal, held)
                ok = True
            if ok and not phone_grasp:
                expected = robot.get_ee_pose(arm) @ held
                actual = pose(robot, name)
                slip = float(np.linalg.norm(expected[:3, 3] - actual[:3, 3]))
                tilt = float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude())
                ok = slip <= .008 and tilt <= .08
                note(robot, 'grasp_probe_full_lift_tracking', measured_slip_m=slip, measured_slip_rad=tilt, passed=ok)
                if not ok:
                    if lift_tracking_recovery is None:
                        raise SkillFailure('probed_grasp_lift_exhausted', {'slip_m': slip, 'slip_rad': tilt})
                    lift_tracking_recovery(actual, {'position_error_m': slip, 'rotation_error_rad': tilt})
                    ok = True
        else:
            ok = False
    elif lid_grasp and ok:
        robot.client.command_any({'command':'ProjectTrackObjects','paths':{name:target_path}})
        try:ok=lift_packing_lid(robot,name,arm,before[2])
        finally:robot.client.command_any({'command':'ProjectTrackObjects','paths':{}})
    after=center(robot,name)
    lifted=after[2]-before[2]>.055
    note(robot,'pick_result',object=name,arm=arm,api_success=bool(ok),lift_m=float(after[2]-before[2]),object_pose=pose(robot,name),ee_pose=robot.get_ee_pose(arm))
    if packing_grasp and ok and lifted:
        require_payload_proxy(robot, name, arm, 'packing_payload_attachment_gate')
    note(robot,'model_snapshot_after_grasp',snapshot=robot.client.command_any({'command':'ProjectKinematicSnapshot'}))
    return bool(ok and lifted)

def move(robot,name,arm,target,text,*,slowdown_factor=1.):
    """Plan a native collision-checked motion at the requested execution speed."""
    if not np.isfinite(slowdown_factor) or slowdown_factor < 1.:
        raise ValueError('Native motion slowdown must be finite and at least one')
    candidates=np.asarray(target)
    if candidates.ndim==2:candidates=candidates[None]
    ok=robot.move_actor_to_target_pose(name,candidates,arm=arm,
        move_config=MoveConfig(action_description=description(text),
                               extra_params={'slowdown_factor': float(slowdown_factor)}))
    note(robot,text,object=name,arm=arm,api_success=bool(ok),target=target,
         slowdown_factor=float(slowdown_factor),
         actual=robot.get_ee_pose(arm) if name=='gripper' else pose(robot,name))
    return bool(ok)

def unload_gripper(robot, arm, profile):
    command = {'command': 'ProjectGripUnload', 'arm': arm}
    start = robot.client.command_any({**command, 'effort_fraction': 1.})
    if not isinstance(start, dict) or start.get('success') is not True:
        raise RuntimeError('Native closing drive is unavailable for effort-limited release')

    dt = float(robot.client.get_physics_dt())
    if not np.isfinite(dt) or dt <= 0.:
        raise RuntimeError('Invalid physics time step for gripper unloading')
    ticks = int(np.ceil(profile['unload_seconds'] / dt))
    floor = float(profile['unload_min_effort_Nm']) / start['max_effort_Nm']
    if ticks < 2 or not 0. < floor < 1.:
        raise RuntimeError('Invalid duration or effort limit for gripper unloading')
    for fraction in np.linspace(1., floor, ticks + 1)[1:]:
        result = robot.client.command_any({**command, 'effort_fraction': float(fraction)})
        if not isinstance(result, dict) or result.get('success') is not True:
            raise RuntimeError('Native gripper rejected its decreasing effort target')
        robot.client.run_step(1)
    final = robot.client.command_any({'command': 'ProjectSnapshot'})['sim_time']
    elapsed = float(final) - start['sim_time_s']
    note(robot, 'gentle_release_effort_unload', arm=arm, start=start, end=result,
         physics_ticks=ticks, elapsed_s=elapsed, end_sim_time_s=float(final))
    if not np.isclose(elapsed, ticks * dt, atol=.00001, rtol=0.):
        raise RuntimeError('Grip unloading did not advance the requested physical ticks')


def separate_gripper_at_low_effort(robot, arm, check):
    """Give a fragile payload time to clear before fully opening the jaws."""
    contract = robot.client.command_any({'command': 'ProjectGripperConfiguration'})['grippers'][arm]
    name = contract['actuator']
    index = contract['joint_names'].index(name)
    closed = float(contract['closed_positions'][index])
    span = float(contract['opened_positions'][index]) - closed
    if abs(span) < .08:
        raise ValueError('Native actuator has insufficient separation travel')
    opening = (float(robot.get_joint_positions()[name]) - closed) / span
    for increment in (.035, .075):
        target = float(np.clip(opening + increment / abs(span), 0., 1.))
        result = robot.control_gripper(arm=arm, opening_ratio=target, speed=.075, max_effort=.12,
                                       mode='velocity', tolerance=.002, timeout=3.,
                                       minimum_duration=.05, completion='target')
        robot.client.run_step(24)
        passed, witness = check()
        note(robot, 'low_effort_gripper_separation', arm=arm, opening_ratio=target,
             result=result, passed=passed, **witness)
        if passed:
            return
    from skill_executor import SkillFailure
    raise SkillFailure('payload_not_cleared_during_release', witness)


def release(robot,arm,gentle=False,profile=None,separation_check=None):
    native_modes={'native_velocity':'velocity','native_position':'position'}
    method=profile.get('method') if profile else None
    if gentle and method == 'native_effort_unload':
        unload_gripper(robot,arm,profile)
        if separation_check is not None:
            separate_gripper_at_low_effort(robot, arm, separation_check)
        method = 'native_velocity'
    mode=native_modes.get(method)
    if gentle and mode:
        # The native close uses velocity control and leaves its actuator stiffness at zero.
        gains={'stiffness':profile['stiffness'],'damping':profile['damping']} if mode=='position' else {}
        result=robot.control_gripper(arm=arm,opening_ratio=1.,speed=profile['speed_rad_s'],max_effort=profile['max_effort_Nm'],
            mode=mode,tolerance=.002,timeout=8.,minimum_duration=.15,completion='target',**gains)
        note(robot,'native_gripper_release',arm=arm,result=result,profile=profile,passive_joints_unchanged=True)
        if result.get('success') is not True or result.get('reason')!='target':raise RuntimeError('Native gripper release did not reach its target')
        robot.client.run_step(30)
        measured=robot.get_joint_positions();error=max(abs(measured[name]-target) for name,target in profile['coupled_open_targets_rad'].items())
        note(robot,'gentle_release_end',arm=arm,max_joint_error_rad=float(error),method=profile['method'])
        if error>.02:raise RuntimeError('Native gripper follower did not open')
    elif gentle:
        prepare={'command':'ProjectPrepareFingerRelease','arm':arm}
        session=robot.client.command_any(prepare)
        if not isinstance(session,dict) or session.get('success') is not True:raise RuntimeError('Finger release setup failed: '+str(session))
        speed=.15;dt=.05
        ticks_per_target=6
        hold_ticks=profile.get('neutral_hold_ticks',0) if profile else 0
        if hold_ticks:
            start_time=float(robot.client.command_any({'command':'ProjectSnapshot'})['sim_time'])
            start_fingers=session['initial_rad']
            robot.client.run_step(hold_ticks)
            end_time=float(robot.client.command_any({'command':'ProjectSnapshot'})['sim_time'])
            dt=float(robot.client.get_physics_dt())
            if dt<=0 or not np.isclose(end_time-start_time,hold_ticks*dt,atol=.00001,rtol=0.):
                raise RuntimeError('Neutral finger hold did not advance the requested physics ticks')
            # Rebase the next position targets on the finger positions after
            # the measured neutral hold. Object physics tracking stays active.
            session=robot.client.command_any(prepare)
            if not isinstance(session,dict) or session.get('success') is not True:
                raise RuntimeError('Finger release rebase failed: '+str(session))
            note(robot,'gentle_release_neutral_hold',arm=arm,physics_ticks=hold_ticks,
                sim_time_started_s=start_time,sim_time_finished_s=end_time,
                start_finger_positions_rad=start_fingers,end_finger_positions_rad=session['initial_rad'])
            speed=float(profile['speed_rad_s']);ticks_per_target=1
        count=max(1,int(np.ceil(np.max(np.abs(np.asarray(session['target_rad'])-session['initial_rad']))/(speed*dt))))
        note(robot,'gentle_release_start',arm=arm,session=session,speed_rad_s=speed,steps=count,
            physics_ticks_per_target=ticks_per_target)
        for i in range(1,count+1):
            response=robot.client.command_any({'command':'ProjectFingerReleaseTarget','fraction':i/count})
            if not isinstance(response,dict) or response.get('success') is not True:raise RuntimeError('Finger release command rejected: '+str(response))
            robot.client.run_step(ticks_per_target)
        robot.client.run_step(60)
        final=robot.client.command_any({'command':'ProjectFingerReleaseTarget','fraction':1.})
        error=float(np.max(np.abs(np.asarray(final['actual_rad'])-session['target_rad'])))
        note(robot,'gentle_release_end',arm=arm,actual=final,max_joint_error_rad=error)
        if error>.02:raise RuntimeError('Finger release failed to reach open target')
    else:
        robot.robot.open_gripper(id=arm,width=1.0,detach=False)
    if robot.scene_info['task_template'] in ('phone_packaging','put_the_cap_on_the_pen_new'):
        detach_planner_payload(robot,arm)
    else:
        robot.client.command_any({'command':'ProjectDetachPlannerArm','arm':arm})
    robot.client.run_step(60)

def retreat(robot,arm,distance=.12):
    current=robot.get_ee_pose(arm).copy();candidates=[]
    up=current.copy();up[2,3]+=distance;candidates.append(up)
    snapshot=robot.client.command_any({'command':'ProjectKinematicSnapshot'})
    direction=np.asarray(snapshot['robot_base_position'])-current[:3,3];direction[2]=0.
    if np.linalg.norm(direction)>.01:
        direction/=np.linalg.norm(direction)
        for height in [.04,.08]:
            goal=current.copy();goal[:3,3]+=direction*.10+[0,0,height];candidates.append(goal)
    away=-current[:3,2];away[2]=max(away[2],0.)
    if np.linalg.norm(away)>.1:
        goal=current.copy();goal[:3,3]+=distance*away/np.linalg.norm(away);candidates.append(goal)
    return move(robot,'gripper',arm,np.asarray(candidates),'empty_hand_withdrawal')

def phone_rules(robot,index):
    path=Path(os.environ.get('GENIE_RUN_SOURCE',ROOT))/'precise_geometry.py'
    if not path.exists():path=ROOT/'genie_sim-main/source/geniesim_benchmark/src/geniesim_benchmark/plugins/ader/action/custom/precise_geometry.py'
    spec=importlib.util.spec_from_file_location('official_precise_geometry',path)
    geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
    rules=robot.scene_info['official_subtask_steps'][index]['checker']['MixedRules']
    measurements=[]
    for rule in rules['rules']:
        p=rule['params'];frame=pose(robot,p['reference'])
        relative=np.linalg.inv(frame)@pose(robot,p['object'])
        if rule['name']=='InsidePrecise':
            result=geometry.containment(geometry.transformed(np.asarray(p['points']),relative),p['lower'],p['upper'],p.get('quantifier','all'))
        elif rule['name']=='CoverPrecise':
            target=pose(robot,p.get('target_object',p['reference']))
            result=geometry.coverage(geometry.transformed(np.asarray(p['source_points']),relative),
                geometry.transformed(np.asarray(p['target_points']),np.linalg.inv(frame)@target),
                p['up_axis'],p['threshold'],p.get('max_gap'),
                float(relative[p['up_axis'],p['normal_axis']]) if 'normal_axis' in p else None,p.get('min_normal_dot'))
        else:raise RuntimeError('Unsupported official precision rule '+rule['name'])
        measurements.append({'name':rule['name'],**result})
    success=all(v['pass'] for v in measurements) if rules.get('mode','or')=='and' else any(v['pass'] for v in measurements)
    note(robot,'official_phone_rules',subtask_index=index,success=success,measurements=measurements)
    return success

def phone_target(robot,name,depth):
    base='benchmark_packaging_box_001'
    low,high=local_box(robot,base)
    tb=pose(robot,base);target=tb.copy()
    if name.endswith('002'):target[:3,:3]=tb[:3,:3]@Rotation.from_rotvec([0,np.pi,0]).as_matrix()
    ownlow,ownhigh=local_box(robot,name)
    c=(low+high)/2;c[1]=low[1]+depth
    target[:3,3]=(tb@np.r_[c,1])[:3]-target[:3,:3]@((ownlow+ownhigh)/2)
    return target

def phone(robot):
    base='benchmark_packaging_box_001'
    table='/World/background/benchmark_table_019'
    robot.add_objs_to_obstacle([table])
    environment=robot.client.command_any({'command':'ProjectSnapshot'})
    table_registered=any(p.startswith(table+'/') for p in environment.get('planner_obstacles',[]))
    note(robot,'environment_collision_gate',passed=table_registered,snapshot=environment)
    if not table_registered:raise RuntimeError('Official phone table is missing from planner collision world')
    for step,(name,depth) in enumerate([('benchmark_phone_000',.008),('benchmark_packaging_box_000',.020),('benchmark_packaging_box_002',.026)]):
        if not pick(robot,name,'right',axis=2):return False
        # The base frame and object frame derive from the official asset transforms.
        # Place the object center above the cavity floor, then descend with contact.
        target=phone_target(robot,name,depth)
        approach=target.copy();approach[2,3]+=.10
        if not move(robot,name,'right',approach,'phone_pre_place'):return False
        receiver=robot.get_object_prim_path(base)
        robot.remove_objs_from_obstacle([receiver])
        try:
            if not move(robot,name,'right',target,'phone_contact_place'):return False
            release(robot,'right')
            if not retreat(robot,'right'):return False
        finally:
            robot.add_objs_to_obstacle([receiver])
        now=pose(robot,name)
        err=float(np.linalg.norm(now[:3,3]-target[:3,3]))
        note(robot,'phone_place_settle',object=name,error_m=err)
        if err>.025 or not phone_rules(robot,2*step+1):return False
    robot.client.run_step(120)
    return all(phone_rules(robot,i) for i in (1,3,5))

def block(robot):
    from insertion_physics import measure,crossing_report
    instruction=robot.scene_info['official_instruction'];name=instruction['target']['id1'];arm=instruction['gripper']['1']['side']
    if not pick(robot,name,arm,goal_topdown=True):return False
    target_file=Path(os.environ.get('GENIE_RUN_SOURCE',ROOT))/'insertion-target.json'
    if not target_file.exists():target_file=ROOT/'output/insertion_target_v2.json'
    geometry=json.loads(target_file.read_text())
    if geometry['silhouette_outside_hole_area_fraction']>.001:raise RuntimeError('Star silhouette certificate failed')
    box=geometry['box'];target=pose(robot,box)@getattr(robot,'_project_shape_goal_local',np.asarray(geometry['object_goal_in_box_frame']))
    # Five valid in-plane symmetries preserve the star-hole fit and vary wrist reach.
    variants=[];mesh_center=np.asarray(geometry['star_center_entity'])
    center_target=(target@np.r_[mesh_center,1])[:3]
    for angle in np.arange(5)*2*np.pi/5:
        candidate=target.copy()
        candidate[:3,:3]=Rotation.from_rotvec([0,0,angle]).as_matrix()@target[:3,:3]
        # Rotate about the mesh centre, not the asset's offset entity origin.
        candidate[:3,3]=center_target-candidate[:3,:3]@mesh_center
        if measure(candidate,pose(robot,box))['outside_hole_area_fraction']<=.001:variants.append(candidate)
    if not variants:raise RuntimeError('No full-mesh orientation fits the native aperture')
    above=np.asarray(variants);above[:,2,3]+=.07
    if not move(robot,name,arm,above,'star_align_above_matching_hole'):return False
    # Keep receiver physics active; exclude only the top contact mesh from planning.
    prim=robot.get_object_prim_path(box)+'/body'
    robot.remove_objs_from_obstacle([prim])
    try:
        actual=pose(robot,name)
        selected=min(variants,key=lambda v:Rotation.from_matrix(v[:3,:3].T@actual[:3,:3]).magnitude()).copy()
        # The finger pads extend below the held star. Retain safe aperture
        # clearance, release accurately above it, and verify gravity passage.
        # Pushing the fingers into the plate is neither feasible nor required.
        release_clearance=.05
        selected_local=np.linalg.inv(pose(robot,box))@selected
        for dz in (.06,release_clearance):
            step=pose(robot,box)@selected_local;step[2,3]+=dz
            if not move(robot,name,arm,step,'star_measured_entry'):return False
        release_goal=pose(robot,box)@selected_local;release_goal[2,3]+=release_clearance
        for correction_index in range(4):
            entry=pose(robot,name);projection=measure(entry,pose(robot,box))
            position_delta=release_goal[:3,3]-entry[:3,3]
            rotation_delta=Rotation.from_matrix(release_goal[:3,:3]@entry[:3,:3].T)
            if np.linalg.norm(position_delta)<.0015 and rotation_delta.magnitude()<.012 and projection['outside_hole_area_fraction']<=.001:break
            if correction_index==3:break
            command=release_goal.copy();delta=.8*position_delta
            delta*=min(1.,.005/max(np.linalg.norm(delta),1e-9))
            command[:3,3]+=delta
            command[:3,:3]=Rotation.from_rotvec(.8*rotation_delta.as_rotvec()).as_matrix()@command[:3,:3]
            note(robot,'star_aperture_feedback',iteration=correction_index,position_error_m=float(np.linalg.norm(position_delta)),
                outside_hole_area_fraction=projection['outside_hole_area_fraction'])
            if not move(robot,name,arm,command,'star_aperture_feedback_move'):return False
            robot.client.run_step(30)
        entry=pose(robot,name)
        entry_error=float(np.linalg.norm(entry[:3,3]-release_goal[:3,3]))
        angle_error=float(Rotation.from_matrix(selected[:3,:3].T@entry[:3,:3]).magnitude())
        projection=measure(entry,pose(robot,box))
        note(robot,'star_precise_entry_gate',position_error_m=entry_error,rotation_error_rad=angle_error,
            projection=projection,method='aligned_gravity_insertion',release_clearance_m=release_clearance)
        if entry_error>.004 or angle_error>.035 or projection['outside_hole_area_fraction']>.001:return False
        robot.client.command_any({'command':'ProjectTrackObjects','paths':{name:robot.get_object_prim_path(name),box:robot.get_object_prim_path(box)}})
        release(robot,arm,gentle=True)
        if not retreat(robot,arm):return False
        robot.client.run_step(120)
        after=pose(robot,name)
        robot.client.command_any({'command':'ProjectTrackObjects','paths':{}})
    finally:robot.add_objs_to_obstacle([prim])
    note(robot,'star_release_settle',before=entry,after=after,drop_m=float(entry[2,3]-after[2,3]))
    local=np.linalg.inv(pose(robot,box))@after
    hole_plane_z=target[2,3]-geometry['star_half_thickness']-.003
    # Entity origin changes with normal reversal; native mesh crossing is the
    # invariant physical depth test for both orientations.
    contained=abs(local[0,3])<.060 and abs(local[2,3])<.060 and -.075<local[1,3]<.055
    robot.client.run_step(60);stable=float(np.linalg.norm(pose(robot,name)[:3,3]-after[:3,3]))<.003
    crossing=crossing_report(Path(os.environ['GENIE_EPISODE_EVIDENCE'])/'object-physics-120hz.jsonl')
    descended=bool(crossing['fully_below_plane'])
    (Path(os.environ['GENIE_EPISODE_EVIDENCE'])/'hole-crossing-audit.json').write_text(json.dumps(crossing,indent=2))
    passed=bool(descended and contained and stable and crossing['passed'])
    note(robot,'star_final_gate',passed=passed,descended=bool(descended),contained=bool(contained),stable=stable,
        local_pose=local,correct_hole_entry_recorded=bool(crossing['passed']),physics_samples=crossing['physics_samples'])
    return passed

def pen(robot):
    instruction=robot.scene_info['official_instruction'];pen=instruction['target']['id1'];cap=instruction['target']['id2']
    pa=instruction['gripper']['1']['side'];ca=instruction['gripper']['2']['side']
    return_pose=pose(robot,pen).copy()
    joint_paths=[robot.get_object_prim_path(cap)+'/PenAttachJoint',robot.get_object_prim_path(cap)+'/body/PenAttachJoint']
    initial=robot.client.command_any({'command':'ProjectSnapshot','joint_paths':joint_paths})
    if any(initial['joint_enabled'].values()):raise RuntimeError('Stale PenAttachJoint at episode start')
    if not pick(robot,pen,pa,offset=[.025,0,0]):return False
    if not pick(robot,cap,ca,offset=[-.004,0,0]):return False
    direction=np.array([0.,-1. if pa=='right' else 1.,0.])
    tp=np.eye(4);tp[:3,:3]=np.column_stack([direction,[0,0,1],np.cross(direction,[0,0,1])]);tp[:3,3]=[-.08,-.06 if pa=='right' else .06,.98]
    tc=tp.copy();tc[:3,3]+=direction*(-.07802-.02127-.03)
    if not move(robot,pen,pa,tp,'pen_present'):return False
    if not move(robot,cap,ca,tc,'cap_align'):return False
    prim=robot.get_object_prim_path(pen)
    robot.remove_objs_from_obstacle([prim])
    try:
        for depth in np.linspace(0,.03+.03152,16)[1:]:
            target=tc.copy();target[:3,3]+=direction*depth
            if not move(robot,cap,ca,target,'cap_insertion_measure'):return False
        note(robot,'cap_final_relative_pose',relative=np.linalg.inv(pose(robot,pen))@pose(robot,cap))
        connected=robot.client.command_any({'command':'ProjectSnapshot','joint_paths':joint_paths})
        note(robot,'cap_connection_gate',snapshot=connected)
        if not any(connected['joint_enabled'].values()):return False
        # Open only the cap hand, preserving the pen-hand planner representation.
        robot.robot.open_gripper(id=ca,width=1.0,detach=False)
        robot.client.command_any({'command':'ProjectDetachPlannerArm','arm':ca})
        if not retreat(robot,ca):return False
        robot.client.attach_obj(prim_paths=[robot.get_object_prim_path(pen),robot.get_object_prim_path(cap)],is_right=pa=='right')
        return_pose[2,3]+=.004
        if not move(robot,pen,pa,return_pose,'capped_pen_return_to_table'):return False
        release(robot,pa)
        if not retreat(robot,pa):return False
        robot.client.run_step(120)
        final=robot.client.command_any({'command':'ProjectSnapshot','joint_paths':joint_paths})
        on_table=abs(pose(robot,pen)[2,3]-return_pose[2,3])<.025
        note(robot,'pen_final_gate',on_table=on_table,snapshot=final,official_snap_model=True)
        return bool(on_table and any(final['joint_enabled'].values()))
    finally:robot.add_objs_to_obstacle([prim])

def execute_task(robot):
    import hashlib
    source=Path(__file__)
    note(robot,'program_loaded',path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    snapshot=robot.client.command_any({'command':'ProjectKinematicSnapshot'})
    deltas=[float(np.linalg.norm(np.asarray(snapshot['usd_links'][name])[:3,3]-link['position']))
        for name,link in snapshot['links'].items()]
    note(robot,'render_baseline_gate',passed=max(deltas)<.002,max_position_error_m=max(deltas))
    if max(deltas)>=.002:raise RuntimeError('Renderer transform baseline differs from native physics; task not executed')
    install_grasp_contact_compatibility()
    task=robot.scene_info['task_template']
    instruction=robot.scene_info['official_instruction']['instruction']
    robot.client.set_task_task_basic_info({'task_name':instruction,'english_task_name':instruction,'init_scene_text':''})
    request_path=Path(os.environ.get('GENIE_RUN_SOURCE',ROOT))/'request.json'
    if request_path.exists():
        request=json.loads(request_path.read_text())
        if request.get('execution_mode')=='structured':
            import sys
            modules={}
            names=['robot_adapters','skill_registry','task_geometry','skill_executor','structured_planner','qwen_backend']
            names.extend(name for name in ['skill_recovery','goal_composition','skill_capabilities','hinge_fixture','rotation_probe']
                         if name in sys.modules and (request_path.parent/(name+'.py')).exists())
            for name in names:
                module=sys.modules[name];path=Path(module.__file__)
                modules[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            note(robot,'skill_modules_loaded',modules=modules)
            from skill_executor import SkillFailure,run_structured
            client=robot.robot.client
            original_playback=client.playback

            def reject_rollback(stored_state,*args,**kwargs):
                note(robot,'native_rollback_request_blocked',stored_state=stored_state)
                raise SkillFailure('native_action_failed_before_rollback',{'stored_state':stored_state})

            client.playback=reject_rollback
            try:
                if task=='probe_open_hinged_lid' and robot.scene_info.get('project_hinge_probe'):
                    from hinge_fixture import execute_probe
                    return execute_probe(robot,sys.modules[__name__],request)
                if request['task_id']=='probe_rotate_held_about_tool' and robot.scene_info.get('project_rotation_probe'):
                    from rotation_probe import execute_probe
                    return execute_probe(robot,sys.modules[__name__],request)
                return run_structured(robot,sys.modules[__name__],request)
            finally:client.playback=original_playback
    return {'phone_packaging':phone,'insert_building_block':block,'put_the_cap_on_the_pen_new':pen}[task](robot)
