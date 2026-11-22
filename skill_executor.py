"""Execute validated LLM steps; task names never dispatch whole task programs."""
import contextlib,json,os,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from robot_adapters import load_robot,UnsupportedRequest
from skill_registry import SKILLS,validate_plan
from task_geometry import Geometry,bbox_corners,task_spec,transformed_points

def accumulate_pose_correction(goal, measured, correction):
    """Carry bounded task-space compensation between measured alignment moves."""
    position_delta = goal[:3, 3] - measured[:3, 3]
    position_bias = correction[:3, 3] + .7 * position_delta
    distance = np.linalg.norm(position_bias)
    correction[:3, 3] = position_bias * min(1., .008 / max(distance, 1e-9))

    rotation_delta = Rotation.from_matrix(goal[:3, :3] @ measured[:3, :3].T)
    rotation_bias = Rotation.from_rotvec(.7 * rotation_delta.as_rotvec()) * Rotation.from_matrix(correction[:3, :3])
    rotation_vector = rotation_bias.as_rotvec()
    angle = np.linalg.norm(rotation_vector)
    correction[:3, :3] = Rotation.from_rotvec(rotation_vector * min(1., .08 / max(angle, 1e-9))).as_matrix()

    command = goal.copy()
    command[:3, 3] += correction[:3, 3]
    command[:3, :3] = correction[:3, :3] @ goal[:3, :3]
    return command

def simulator_unavailable(exc):
    code = getattr(exc, 'code', None)
    return callable(code) and getattr(code(), 'name', None) == 'UNAVAILABLE'

class SkillFailure(RuntimeError):
    def __init__(self,kind,details):super().__init__(kind);self.kind=kind;self.details=details

class SkillExecutor:
    def __init__(self,robot,adapter,spec,ops,folder):
        self.robot,self.adapter,self.spec,self.ops,self.folder=robot,adapter,spec,ops,Path(folder)
        self.geometry=Geometry(robot,spec,ops,adapter);self.holding={};self.facts=set();self.started=time.monotonic()
        self.held_transforms={}
        self.events=self.folder/'skill-execution.jsonl';self.executed=[]
        self.max_wall_seconds=int(adapter.config.get('execution_budgets',{}).get('max_episode_execution_wall_s',2100))
        (self.folder/'budget-profile.json').write_text(json.dumps({'max_episode_execution_wall_s':self.max_wall_seconds,'max_model_plans':3,'api_wall_time_budget_s':900},indent=2))
        self.contact_exclusions=set()
        environment=spec.get('environment_collision_paths',[])
        if environment:
            robot.add_objs_to_obstacle(environment)
            snapshot=robot.client.command_any({'command':'ProjectSnapshot'})
            registered=all(any(path.startswith(parent+'/') for path in snapshot.get('planner_obstacles',[])) for parent in environment)
            ops.note(robot,'environment_collision_gate',passed=registered,snapshot=snapshot)
            if not registered:raise SkillFailure('missing_environment_collision_geometry',{'paths':environment})

    def event(self,name,**values):
        row={'event':name,'wall_unix':time.time(),'plan_sha256':getattr(self,'plan_sha256',None),**values}
        with self.events.open('a') as f:f.write(json.dumps(row,default=lambda x:np.asarray(x).tolist())+'\n')

    @contextlib.contextmanager
    def contact_planning(self,paths):
        new=set(paths)-self.contact_exclusions
        if new:self.robot.remove_objs_from_obstacle(sorted(new));self.contact_exclusions.update(new)
        try:yield
        except Exception:
            self.restore_contact_planning();raise

    def restore_contact_planning(self):
        if self.contact_exclusions:self.robot.add_objs_to_obstacle(sorted(self.contact_exclusions));self.contact_exclusions.clear()

    def arm(self,role):return self.adapter.resolve_role(role,self.spec['role_bindings'])

    def state(self):
        poses={o:self.ops.pose(self.robot,o) for o in self.spec['manipulated_objects']};evidence={}
        for role,obj in list(self.holding.items()):
            expected=self.robot.get_ee_pose(self.arm(role))@self.held_transforms[role]
            error=float(np.linalg.norm(expected[:3,3]-poses[obj][:3,3]));evidence[role]={'object':obj,'measured_following_error_m':error,'holding_is_estimate':True}
            if error>.08:
                self.holding.pop(role);self.held_transforms.pop(role);self.facts.add('lost_grasp:'+obj)
        return {'holding_by_role':self.holding.copy(),'holding_evidence':evidence,'achieved':sorted(self.facts),
            'object_poses':{o:t.tolist() for o,t in poses.items()}}

    def grasp(self,p):
        obj,arm=p['object'],self.arm(p['role']);profile=p['grasp_profile']
        options={'annotated_topdown':{},'release_compatible':{'goal_topdown':True,'goal_lookahead':bool(self.adapter.config.get('shape_grasp_lookahead',False))},'topdown_box':{'axis':2},'pen_barrel':{'offset':[.025,0,0]},'cap_body':{'offset':[-.004,0,0]}}[profile]
        calibration=self.adapter.config.get('custom_grasp_candidates',{}).get(profile,{})
        if calibration.get('scene_id')==self.spec['scene_id'] and calibration.get('arm')==arm:
            options['custom_pose_options']=calibration['pose_options']
            self.event('calibrated_grasp_candidates',object=obj,role=p['role'],profile=profile,**calibration)
        if profile=='release_compatible':
            pinch=self.adapter.config.get('shape_grasp_axial_offset_by_arm',{}).get(arm,{})
            if pinch.get('scene_id')==self.spec['scene_id']:
                options['pinch_axis_offset_m']=pinch['offset_m']
                options['release_tcp_x_points_up']=bool(pinch.get('release_tcp_x_points_up',False))
                if 'contact_height_offsets_m' in pinch:
                    options['pinch_height_offsets_m']=pinch['contact_height_offsets_m']
        if not self.ops.pick(self.robot,obj,arm,**options):raise SkillFailure('grasp_unreachable_or_failed',{'object':obj,'profile':profile})
        self.holding[p['role']]=obj
        self.facts.discard('released:'+obj)
        for ref,owner in self.spec['target_objects'].items():
            if owner==obj:self.facts.discard('target:'+ref)
        if self.spec['template']=='insert_building_block':self.facts.discard('shape_aligned')
        self.held_transforms[p['role']]=np.linalg.inv(self.robot.get_ee_pose(arm))@self.ops.pose(self.robot,obj)

    def shape_entry(self,p):
        arm=self.arm(p['role'])
        if arm in self.adapter.config.get('shape_release_clearance_by_arm_m',{}):
            return self.geometry.shape_release_geometry(p['object'],arm)
        goal,excluded=self.geometry.target('shape_entry',p['object'])
        return goal,excluded,{}

    def cap_ik_mask(self,role,obj,candidates):
        arm=self.arm(role)
        object_to_hand=np.linalg.inv(self.ops.pose(self.robot,obj))@self.robot.get_ee_pose(arm)
        flags,_=self.robot.robot.solve_ik(candidates@object_to_hand,ee_type='gripper',arm=arm,type='AvoidObs')
        mask=np.asarray(flags,dtype=bool).reshape(-1)
        if len(mask)!=len(candidates):raise ValueError('Planner returned an invalid cap candidate mask')
        return mask

    def pen_presentation_candidates(self,p,goal):
        cap=self.spec['bindings']['cap']
        if self.holding.get('assistant')!=cap:return goal
        # Center the next preinsert near the elevated outboard pose reached by
        # the cap hand in the previous trial. Test both hands before moving.
        candidates=np.repeat(goal[None],3,axis=0)
        candidates[0,:3,3]-=.04*goal[:3,0]
        candidates[:2,2,3]+=.035
        preinsert=np.array([self.geometry.cap_preinsert_pose(candidate) for candidate in candidates])
        pen_ik=self.cap_ik_mask(p['role'],p['object'],candidates)
        cap_ik=self.cap_ik_mask('assistant',cap,preinsert)
        eligible=pen_ik&cap_ik
        self.event('cap_presentation_candidates',pen_targets=candidates,preinsert_targets=preinsert,
                   pen_collision_ik=pen_ik,cap_collision_ik=cap_ik,eligible=eligible)
        if not eligible.any():raise SkillFailure('no_paired_presentation_candidate',{'evaluated':len(candidates)})
        selected=int(np.flatnonzero(eligible)[0])
        self.event('cap_presentation_selected',candidate=selected)
        return candidates[selected]

    def follow_local_pose(self, params, goal, phase, *, free_space_rotation=False):
        obj = params['object']
        arm = self.arm(params['role'])
        start = self.ops.pose(self.robot, obj)
        delta = goal[:3, 3] - start[:3, 3]
        rotations = Rotation.from_matrix(np.stack([start[:3, :3], goal[:3, :3]]))
        distance = float(np.linalg.norm(delta))
        angle = float((rotations[0].inv() * rotations[1]).magnitude())
        if distance > .09 or (angle > .25 and not free_space_rotation):
            raise SkillFailure('local_waypoint_segment_out_of_range', {'distance_m': distance, 'angle_rad': angle})

        angle_step = .02 if free_space_rotation else .005
        segments = max(4, int(np.ceil(distance / .0008)), int(np.ceil(angle / angle_step)))
        fractions = np.linspace(0, 1, segments + 1)
        fractions = fractions**2 * (3 - 2*fractions)
        positions = start[:3, 3] + fractions[:, None] * delta
        orientations = Slerp([0, 1], rotations)(fractions).as_quat()[:, [3, 0, 1, 2]]
        waypoints = np.column_stack([positions, orientations])
        config = self.ops.FollowTrajectoryConfig(action_description=self.ops.description(phase))
        self.event('local_waypoints_submitted', phase=phase, count=len(waypoints), start=start, goal=goal)
        # RobotAPI checks every waypoint with AvoidObs. The native follower
        # seeds each solve from the previous joint solution and steps PhysX.
        moved = self.robot.follow_trajectory(obj, waypoints, arm=arm, trajectory_config=config)
        if moved:
            self.robot.client.run_step(30)
        measured = self.ops.pose(self.robot, obj)
        error = float(np.linalg.norm(measured[:3, 3] - goal[:3, 3]))
        rotation_error = float(Rotation.from_matrix(goal[:3, :3].T @ measured[:3, :3]).magnitude())
        self.ops.note(self.robot, phase, api_success=bool(moved), object=obj, arm=arm,
                      waypoint_count=len(waypoints), target=goal, actual=measured,
                      position_error_m=error, rotation_error_rad=rotation_error)
        if not moved or error > .008 or rotation_error > .05:
            raise SkillFailure('local_waypoint_plan_or_tracking_failed', {
                'phase': phase, 'planner_accepted': bool(moved), 'position_error_m': error,
                'rotation_error_rad': rotation_error, 'waypoint_count': len(waypoints)})

    def move_cap_preinsert(self, params, goal):
        arm = self.arm(params['role'])
        cap = params['object']
        current = self.ops.pose(self.robot, cap)
        gap = np.linalg.norm(current[:3, 3] - goal[:3, 3])
        angle = Rotation.from_matrix(goal[:3, :3].T @ current[:3, :3]).magnitude()
        if gap > .065 or angle > .20:
            stage = goal.copy()
            stage[:3, 3] -= .04 * goal[:3, 0]
            stage[2, 3] += .035
            self.event('cap_preinsert_staging_candidate', target=stage)
            # Keep the friction grasp steady during the long translation. Rotate
            # over the outboard clearance with a sampled, collision-checked path.
            hand_to_cap = np.linalg.inv(self.robot.get_ee_pose(arm)) @ current
            translation = stage.copy()
            translation[:3, :3] = current[:3, :3]
            if not self.ops.move(self.robot, cap, arm, translation, 'plan_cap_outboard_translation'):
                raise SkillFailure('cap_staging_motion_failed', params)
            turn = self.ops.pose(self.robot, cap).copy()
            turn[:3, :3] = stage[:3, :3]
            self.follow_local_pose(params, turn, 'cap_outboard_staging_rotation', free_space_rotation=True)
            measured = self.ops.pose(self.robot, cap)
            expected = self.robot.get_ee_pose(arm) @ hand_to_cap
            slip = float(np.linalg.norm(expected[:3, 3] - measured[:3, 3]))
            slip_angle = float(Rotation.from_matrix(expected[:3, :3].T @ measured[:3, :3]).magnitude())
            self.event('cap_staging_grasp_stability', measured_slip_m=slip, measured_slip_rad=slip_angle)
            state = self.state()
            if self.holding.get(params['role']) != cap or slip > .008 or slip_angle > .05:
                raise SkillFailure('cap_grasp_lost_before_alignment', state)
            goal, _ = self.geometry.target(params['target'], cap)
        self.follow_local_pose(params, goal, 'cap_preinsert_waypoint_motion')

    def cap_mouth_measurement(self, cap_pose):
        config = self.spec['assembly']
        pen_pose = self.ops.pose(self.robot, self.spec['bindings']['pen'])
        relative = np.linalg.inv(pen_pose) @ cap_pose
        mouth = relative @ np.array([config['cap_mouth_x_m'], 0., 0., 1.])
        return {
            'depth_m': float(mouth[0] - config['pen_tip_x_m']),
            'radial_m': float(np.linalg.norm(mouth[1:3])),
            'axis_sine': float(np.linalg.norm(relative[1:3, 0])),
            'rotation_rad': float(Rotation.from_matrix(relative[:3, :3]).magnitude()),
        }

    def cap_radial_clearance(self, measured, next_depth):
        config = self.spec['assembly']
        if next_depth < config['transition_start_depth_m'] - 1e-9:
            half_width = config['tip_half_width_m']
        elif next_depth < config['barrel_start_depth_m'] - 1e-9:
            half_width = config['transition_half_width_m']
        else:
            half_width = config['barrel_half_width_m']
        tilt_offset = max(0., next_depth) * measured['axis_sine']
        return float(config['bore_half_width_m'] - half_width - measured['radial_m'] - tilt_offset)

    def move_shape_approach(self, p, goals, route):
        from hashlib import sha256
        arm = self.arm(p['role'])
        obj = p['object']
        grasp = self.held_transforms[p['role']]

        def checked_pose(phase):
            actual = self.ops.pose(self.robot, obj)
            predicted = self.robot.get_ee_pose(arm) @ grasp
            translation = float(np.linalg.norm(actual[:3, 3] - predicted[:3, 3]))
            angle = float(Rotation.from_matrix(predicted[:3, :3].T @ actual[:3, :3]).magnitude())
            evidence = {'phase': phase, 'position_error_m': translation, 'rotation_error_rad': angle}
            self.event('shape_transport_grasp_tracking', **evidence)
            if translation > .01 or angle > .08:
                raise SkillFailure('shape_transport_grasp_slipped', evidence)
            return actual

        receiver = self.spec['bindings']['receiver']
        if (route['receiver'], route['workpiece']) != (receiver, obj):
            raise SkillFailure('shape_transport_route_asset_mismatch', {'receiver': receiver, 'object': obj})
        certificate_path = Path(os.environ['GENIE_RUN_SOURCE']) / 'insertion-certificate.json'
        if sha256(certificate_path.read_bytes()).hexdigest() != route['insertion_certificate_sha256']:
            raise SkillFailure('shape_transport_route_geometry_changed', {'receiver': receiver})
        checked_pose('initial')
        values = np.asarray(route['waypoints_box_xyz_wxyz'], dtype=float)
        waypoints = np.repeat(np.eye(4)[None], len(values), axis=0)
        waypoints[:, :3, 3] = values[:, :3]
        waypoints[:, :3, :3] = Rotation.from_quat(values[:, [4, 5, 6, 3]]).as_matrix()
        waypoints = self.ops.pose(self.robot, receiver) @ waypoints
        angles = Rotation.from_matrix(waypoints[-1, :3, :3].T @ goals[:, :3, :3]).magnitude()
        index = int(np.argmin(angles))
        if angles[index] > .035:
            raise SkillFailure('shape_transport_route_goal_mismatch', {'rotation_error_rad': float(angles[index])})
        waypoints[-1] = goals[index]
        self.event('shape_transport_selected', candidate=index, waypoint_count=len(waypoints),
                   source_run=route['source_run'], target=goals[index])
        # The outboard turn is continuous within the calibrated joint limits.
        # Plan every segment again from the measured pose and actual grasp.
        for step, target in enumerate(waypoints, start=1):
            self.event('shape_transport_waypoint', step=step, target=target)
            self.follow_local_pose(p, target, 'shape_transport_waypoint', free_space_rotation=True)
            checked_pose(f'waypoint_{step}')
        actual = checked_pose('approach_complete')
        self.geometry.cache.pop('shape_goal_local', None)
        self.ops.note(self.robot, 'plan_shape_approach', object=obj, arm=arm, api_success=True,
                      target=goals, actual=actual)

    def move_held(self,p):
        if self.holding.get(p['role'])!=p['object']:raise SkillFailure('not_held',p)
        if p['target'].startswith('packing_') and p['target'].endswith('_contact'):
            self.align_packaging(p);return
        geometry={}
        if p['target']=='shape_entry':goal,excluded,geometry=self.shape_entry(p)
        else:goal,excluded=self.geometry.target(p['target'],p['object'])
        if p['target']=='pen_present':goal=self.pen_presentation_candidates(p,goal)
        if geometry:self.event('shape_entry_clearance_target',**geometry)
        arm=self.arm(p['role'])
        pinch=self.adapter.config.get('shape_grasp_axial_offset_by_arm',{}).get(arm,{})
        with self.contact_planning(excluded):
            if p['target']=='cap_preinsert':self.move_cap_preinsert(p,goal)
            elif p['target']=='pen_return':self.place_capped_pen(p,goal)
            elif p['target']=='shape_approach' and pinch.get('scene_id')==self.spec['scene_id'] and 'transport_approach_route' in pinch:
                self.move_shape_approach(p,goal,pinch['transport_approach_route'])
            elif not self.ops.move(self.robot,p['object'],arm,goal,'plan_'+p['target']):
                raise SkillFailure('motion_or_tracking_failed',{'target':p['target']})
        self.facts.add('target:'+p['target'])

    def place_capped_pen(self,params,nominal):
        try:
            self.return_capped_pen(params,nominal)
            return
        except SkillFailure as failure:
            details=failure.details
            slipped_during_motion=(failure.kind=='local_waypoint_plan_or_tracking_failed' and
                                   details.get('planner_accepted') is True)
            reached_table=(failure.kind=='capped_pen_not_ready_to_release' and
                           details.get('inside_table') and details.get('hand_clearance_m',0.)>=.001 and
                           -.002<=details.get('gap_m',.1)<.001)
            if not (slipped_during_motion or reached_table):raise
            if self.state()['holding_by_role'].get(params['role'])!=params['object']:raise
            self.verify_relation({'object':self.spec['bindings']['cap'],'relation':'cap_seated'})
            self.event('capped_pen_return_local_replan',cause=failure.kind,details=details)
        self.return_capped_pen(params,nominal)

    def return_capped_pen(self,params,nominal):
        arm=self.arm(params['role']);pen=params['object'];cap=self.spec['bindings']['cap']
        table=[name for name,item in self.robot.scene_info['layout'].items() if item.get('usd')=='table_000']
        if len(table)!=1:raise SkillFailure('pen_support_geometry_unavailable',{'table_assets':table})
        table=table[0]
        boxes={obj:bbox_corners(*self.ops.local_box(self.robot,obj)) for obj in (pen,cap,table)}
        supported=transformed_points(self.ops.pose(self.robot,table),boxes[table])
        tabletop=float(supported[:,2].max())

        def witness():
            pen_pose=self.ops.pose(self.robot,pen)
            cap_pose=self.ops.pose(self.robot,cap)
            payload=np.concatenate([transformed_points(pen_pose,boxes[pen]),
                                    transformed_points(cap_pose,boxes[cap])])
            hand,_=self.geometry.gripper_collision_points(arm)
            return pen_pose,payload,hand

        current,payload,hand=witness()
        # The task permits any valid tabletop location for the assembled pen.
        # Keep the grasp orientation and test a seeded offset before the original XY.
        goal=current.copy()
        lowest=float(min(payload[:,2].min(),hand[:,2].min()))
        goal[2,3]=current[2,3]+tabletop+.004-lowest
        original_gap=float(payload[:,2].min()+goal[2,3]-current[2,3]-tabletop)
        offset=self.geometry.pen_return_offset_m
        offsets=(offset,-offset,np.zeros(2)) if offset.any() else (offset,)
        table_min,table_max=supported[:,:2].min(axis=0),supported[:,:2].max(axis=0)
        tabletop_candidates=[]
        for candidate_offset in offsets:
            xy=nominal[:2,3]+candidate_offset
            footprint=payload[:,:2]+xy-current[:2,3]
            if (footprint.min(axis=0)>=table_min).all() and (footprint.max(axis=0)<=table_max).all():
                tabletop_candidates.append(xy)
        if not tabletop_candidates:
            raise SkillFailure('pen_return_outside_table',{'table':table,'nominal_xy':nominal[:2,3].tolist()})
        heights=[h for h in (0.,.005,.010) if original_gap+h<=.025]
        if not heights:raise SkillFailure('pen_return_tool_clearance_too_small',{'lowest_feasible_release_gap_m':original_gap})
        releases=np.repeat(goal[None],len(tabletop_candidates)*len(heights),axis=0)
        releases[:,:2,3]=np.repeat(tabletop_candidates,len(heights),axis=0)
        releases[:,2,3]+=np.tile(heights,len(tabletop_candidates))
        approaches=releases.copy();approaches[:,2,3]+=.050
        object_to_hand=np.linalg.inv(current)@self.robot.get_ee_pose(arm)
        candidates=np.concatenate([releases,approaches])
        valid,_=self.robot.robot.solve_ik(candidates@object_to_hand,ee_type='gripper',arm=arm,type='AvoidObs')
        valid=np.asarray(valid,dtype=bool).reshape(-1)
        if len(valid)!=len(candidates):raise ValueError('Planner returned an invalid capped pen target mask')
        release_ik,approach_ik=np.split(valid,2)
        shared=release_ik&approach_ik
        self.event('capped_pen_table_ik',table=table,drop_candidates_m=np.tile(heights,len(tabletop_candidates))+original_gap,
                   release_xy_candidates_m=releases[:,:2,3],
                   release_ik=release_ik,approach_ik=approach_ik)
        if not shared.any():raise SkillFailure('capped_pen_table_pose_unreachable',{'table':table})
        index=int(np.flatnonzero(shared)[0])
        self.event('capped_pen_selected_table_target',requested_offset_m=offset,selected_xy_m=releases[index,:2,3])
        lateral=approaches[index].copy()
        lateral[2,3]=max(current[2,3],lateral[2,3])
        for target,phase in ((lateral,'capped_pen_lateral_waypoints'),
                             (approaches[index],'capped_pen_preplace_waypoints')):
            start=self.ops.pose(self.robot,pen)[:3,3]
            length=float(np.linalg.norm(target[:3,3]-start))
            segments=max(1,int(np.ceil(length/.075)))
            for fraction in np.linspace(0.,1.,segments+1)[1:]:
                waypoint=target.copy()
                waypoint[:3,3]=(1.-fraction)*start+fraction*target[:3,3]
                self.follow_local_pose(params,waypoint,phase)
        self.follow_local_pose(params,releases[index],'capped_pen_descent_waypoints')
        _,end_payload,end_hand=witness()
        gap=float(end_payload[:,2].min()-tabletop)
        hand_clearance=float(end_hand[:,2].min()-tabletop)
        inside=bool(np.all(end_payload[:,:2]>=supported[:,:2].min(axis=0)) and
                    np.all(end_payload[:,:2]<=supported[:,:2].max(axis=0)))
        self.event('capped_pen_table_release_gate',drop_m=gap,hand_clearance_m=hand_clearance,inside_table=inside)
        if not inside or not .001<=gap<=.025 or hand_clearance<.001:
            raise SkillFailure('capped_pen_not_ready_to_release',{'gap_m':gap,'hand_clearance_m':hand_clearance,'inside_table':inside})
        self.verify_relation({'object':cap,'relation':'cap_seated'})

    @staticmethod
    def packing_clear(metrics):
        return metrics['rim_clearance_m']>=.002 and metrics['projected_fit_margin_m']>=.0004

    def align_packaging(self,p):
        arm=self.arm(p['role']);obj=p['object'];target=p['target']
        self.geometry.cache.pop('release_clearance:'+target,None)
        self.geometry.cache.pop('release_yaw:'+target,None)
        base=self.ops.pose(self.robot,self.spec['bindings']['receiver'])
        up=base[:3,1]
        current=self.ops.pose(self.robot,obj)
        current_hand=self.robot.get_ee_pose(arm)
        object_to_hand=np.linalg.inv(current)@current_hand
        # The native insert's two in-plane directions fit the original case.
        # Phone and lid keep their original target orientation.
        yaws=(0.,np.pi) if obj==self.spec['bindings']['insert'] else (0.,)
        goals=[];clearances=[];turns=[];excluded=[]
        for yaw in yaws:
            goal,excluded,metrics=self.geometry.packing_release_geometry(target,obj,arm,yaw=yaw)
            if metrics['projected_goal_fit_margin_m']<.0004:continue
            for offset in (0.,.007,.016,.025):
                candidate=goal.copy();candidate[:3,3]+=offset*up
                goals.append(candidate);clearances.append(metrics['target_clearance_m']+offset);turns.append(yaw)
        if not goals:raise SkillFailure('no_packing_geometry_fits',{'target':target})
        hand_goals=np.asarray(goals)@object_to_hand
        with self.contact_planning(excluded):
            feasible,_=self.robot.robot.solve_ik(hand_goals,ee_type='gripper',arm=arm,type='AvoidObs')
            feasible=np.asarray(feasible,dtype=bool).reshape(-1)
            if len(feasible)!=len(goals):raise ValueError('Planner returned an invalid candidate mask')
            self.event('packing_release_candidate_ik',target=target,clearances_m=clearances,yaw_rad=turns,avoidance_passed=feasible)
            if not feasible.any():
                kinematic,_=self.robot.robot.solve_ik(hand_goals,ee_type='gripper',arm=arm,type='Simple')
                current_ok,_=self.robot.robot.solve_ik(current_hand[None],ee_type='gripper',arm=arm,type='AvoidObs')
                self.event('packing_release_ik_diagnostic',target=target,kinematic_passed=kinematic,current_pose_avoidance=current_ok)
                raise SkillFailure('packing_release_ik_infeasible',{'target':target,'clearances_m':clearances,'yaw_rad':turns})
        selected=int(np.flatnonzero(feasible)[0])
        self.geometry.cache['release_clearance:'+target]=float(clearances[selected])
        self.geometry.cache['release_yaw:'+target]=float(turns[selected])
        if turns[selected]:
            # Turn in free space before descending, with the receiver restored
            # in the planner. Measured object and hand poses drive final alignment.
            self.restore_contact_planning()
            staging=goals[selected].copy()
            lift=max(.05,float(up@(current[:3,3]-staging[:3,3])))
            staging[:3,3]+=lift*up
            if not self.ops.move(self.robot,obj,arm,staging,'plan_packing_turned_approach'):
                raise SkillFailure('packing_turn_path_failed',{'target':target,'yaw_rad':turns[selected]})
        self.align_pose(p,.0015,.018,packing=True)
        self.facts.add('target:'+p['target'])

    def align_pose(self,p,position_tolerance,rotation_tolerance,*,packing=False):
        arm=self.arm(p['role'])
        correction_pose=np.eye(4)
        for attempt in range(5):
            metrics={}
            if packing:goal,excluded,metrics=self.geometry.packing_release_geometry(p['target'],p['object'],arm)
            else:goal,excluded=self.geometry.target(p['target'],p['object'])
            actual=self.ops.pose(self.robot,p['object'])
            delta=goal[:3,3]-actual[:3,3];dr=Rotation.from_matrix(goal[:3,:3]@actual[:3,:3].T)
            self.event('pose_alignment_measurement',target=p['target'],attempt=attempt,position_error_m=float(np.linalg.norm(delta)),rotation_error_rad=float(dr.magnitude()),**metrics)
            if p['target']=='cap_preinsert':
                mouth=self.cap_mouth_measurement(actual)
                gap=-mouth['depth_m']
                initial_clearance=self.cap_radial_clearance(mouth,0.)-gap*mouth['axis_sine']
                nominal_gap=self.spec['assembly']['preinsert_gap_m']
                # Initial entry is governed by the tip clearance. The larger
                # taper and barrel are checked at their own insertion depths.
                passed=(abs(gap-nominal_gap)<.005 and mouth['rotation_rad']<rotation_tolerance and initial_clearance>.001)
                self.event('cap_preinsert_geometry_gate',attempt=attempt,gap_m=gap,tip_clearance_m=initial_clearance,passed=passed,**mouth)
                if passed:return
            elif np.linalg.norm(delta)<position_tolerance and dr.magnitude()<rotation_tolerance and (not packing or self.packing_clear(metrics)):return
            if attempt==4:break
            command=goal.copy()
            if p['target']=='cap_preinsert':
                command=accumulate_pose_correction(goal,actual,correction_pose)
            elif np.linalg.norm(delta)<.03:
                correction=.8*delta;correction*=min(1.,.005/max(np.linalg.norm(correction),1e-9));command[:3,3]+=correction
                command[:3,:3]=Rotation.from_rotvec(.8*dr.as_rotvec()).as_matrix()@command[:3,:3]
            with self.contact_planning(excluded):
                if p['target']=='cap_preinsert':
                    self.follow_local_pose(p,command,'cap_alignment_waypoint_motion')
                else:
                    if not self.ops.move(self.robot,p['object'],arm,command,'plan_pose_alignment'):raise SkillFailure('pose_alignment_motion_failed',p)
                    self.robot.client.run_step(30)
        raise SkillFailure('pose_alignment_not_reached',{'target':p['target'],'position_error_m':float(np.linalg.norm(delta)),'rotation_error_rad':float(dr.magnitude()),**metrics})

    def align_precision(self,p):
        from insertion_physics import measure
        if p['target']=='cap_preinsert':
            self.align_pose(p,.001,.012);self.facts.add('cap_aligned');return
        if p['target'].startswith('packing_') and p['target'].endswith('_contact'):
            self.align_packaging(p);return
        if p['target']!='shape_entry':raise UnsupportedRequest('Precision profile unavailable',missing_skills=[p['target']])
        goal,excluded,clearance=self.shape_entry(p);box=self.spec['bindings']['receiver']
        arm=self.arm(p['role'])
        grip=self.adapter.config.get('shape_grasp_axial_offset_by_arm',{}).get(arm,{})
        local_alignment=grip.get('scene_id')==self.spec['scene_id'] and grip.get('release_tcp_x_points_up')
        correction_pose=np.eye(4)
        with self.contact_planning(excluded):
            for attempt in range(4):
                if local_alignment:goal,_,clearance=self.shape_entry(p)
                actual=self.ops.pose(self.robot,p['object']);projection=measure(actual,self.ops.pose(self.robot,box))
                delta=goal[:3,3]-actual[:3,3];dr=Rotation.from_matrix(goal[:3,:3]@actual[:3,:3].T)
                self.event('precision_measurement',attempt=attempt,position_error_m=float(np.linalg.norm(delta)),rotation_error_rad=float(dr.magnitude()),projection=projection)
                if np.linalg.norm(delta)<.0015 and dr.magnitude()<.012 and projection['outside_hole_area_fraction']<=.001:break
                if attempt==3:break
                if local_alignment:
                    command=accumulate_pose_correction(goal,actual,correction_pose)
                    self.follow_local_pose(p,command,'shape_alignment_waypoint_motion')
                else:
                    command=goal.copy();correction=.8*delta;correction*=min(1.,.005/max(np.linalg.norm(correction),1e-9))
                    command[:3,3]+=correction;command[:3,:3]=Rotation.from_rotvec(.8*dr.as_rotvec()).as_matrix()@command[:3,:3]
                    if not self.ops.move(self.robot,p['object'],arm,command,'plan_precision_feedback'):raise SkillFailure('precision_motion_failed',p)
                    self.robot.client.run_step(30)
        actual=self.ops.pose(self.robot,p['object']);projection=measure(actual,self.ops.pose(self.robot,box))
        err=float(np.linalg.norm(actual[:3,3]-goal[:3,3]));angle=float(Rotation.from_matrix(goal[:3,:3].T@actual[:3,:3]).magnitude())
        if clearance:
            _,_,clearance=self.shape_entry(p)
            if min(clearance['actual_hand_clearance_m'],clearance['actual_workpiece_clearance_m'])<.002:
                raise SkillFailure('insufficient_release_clearance',clearance)
        self.ops.note(self.robot,'star_precise_entry_gate',position_error_m=err,rotation_error_rad=angle,projection=projection,method='aligned_gravity_insertion',
            release_clearance_m=.05 if not clearance else None,measured_release_clearance=clearance)
        if err>.004 or angle>.035 or projection['outside_hole_area_fraction']>.001:raise SkillFailure('precision_not_reached',{'position_error_m':err,'rotation_error_rad':angle,'projection':projection})
        self.facts.add('shape_aligned')

    def release(self,p):
        if self.holding.get(p['role'])!=p['object']:raise SkillFailure('not_held',p)
        arm=self.arm(p['role'])
        release_override={}
        if self.spec['template']=='insert_building_block':
            if 'shape_aligned' not in self.facts:raise SkillFailure('release_precondition_failed',p)
            b=self.spec['bindings']
            grip=self.adapter.config.get('shape_grasp_axial_offset_by_arm',{}).get(arm,{})
            if grip.get('scene_id')==self.spec['scene_id']:
                release_override=grip.get('release_override',{})
            if release_override:
                from insertion_physics import measure
                start=self.ops.pose(self.robot,p['object'])
                self.robot.client.run_step(30)
                settled=self.ops.pose(self.robot,p['object'])
                motion=float(np.linalg.norm(settled[:3,3]-start[:3,3]))
                rotation=float(Rotation.from_matrix(start[:3,:3].T@settled[:3,:3]).magnitude())
                projection=measure(settled,self.ops.pose(self.robot,b['receiver']))
                self.event('shape_grasp_stillness_before_release',position_change_m=motion,rotation_change_rad=rotation,projection=projection)
                if motion>.001 or rotation>.012 or projection['outside_hole_area_fraction']>.001:
                    raise SkillFailure('shape_grasp_not_stable_over_aperture',
                                       {'position_change_m':motion,'rotation_change_rad':rotation,'projection':projection})
            _,_,clearance=self.shape_entry(p)
            if clearance:
                self.event('shape_release_clearance',object=p['object'],**clearance)
                if min(clearance['actual_hand_clearance_m'],clearance['actual_workpiece_clearance_m'])<.002:
                    raise SkillFailure('insufficient_release_clearance',clearance)
                if grip.get('scene_id')==self.spec['scene_id'] and grip.get('release_tcp_x_points_up'):
                    tcp_x_up=float(self.robot.get_ee_pose(arm)[2,0])
                    drop=clearance['actual_workpiece_clearance_m']
                    limit=float(grip['maximum_release_gap_m'])
                    self.event('shape_release_wrist_roll',tcp_x_world_z=tcp_x_up,drop_m=drop,maximum_drop_m=limit)
                    if tcp_x_up<.95 or drop>limit:
                        raise SkillFailure('shape_release_outside_verified_geometry',
                                           {'tcp_x_world_z':tcp_x_up,'drop_m':drop,'maximum_drop_m':limit})
            self.geometry.cache['shape_before_release']=self.ops.pose(self.robot,p['object'])
            self.robot.client.command_any({'command':'ProjectTrackObjects','paths':{o:self.robot.get_object_prim_path(o) for o in [b['workpiece'],b['receiver']]}})
        gripper=self.adapter.config['gripper']
        profile={**gripper.get('release_profile',{}),**gripper.get('release_profiles',{}).get(arm,{}),**release_override}
        packing=self.spec['template']=='phone_packaging'
        if packing:
            index=self.spec['manipulated_objects'].index(p['object']);target=f'packing_{index}_contact'
            binding = next(name for name, obj in self.spec['bindings'].items() if obj == p['object'])
            profile.update(gripper.get('release_overrides_by_task', {}).get(self.spec['template'], {}).get(binding, {}))
            goal,_,metrics=self.geometry.packing_release_geometry(target,p['object'],arm)
            actual=self.ops.pose(self.robot,p['object'])
            angle=float(Rotation.from_matrix(goal[:3,:3].T@actual[:3,:3]).magnitude())
            err=float(np.linalg.norm(actual[:3,3]-goal[:3,3]))
            passed=self.packing_clear(metrics) and err<.0015 and angle<.018
            self.event('packing_release_gate',object=p['object'],passed=passed,position_error_m=err,rotation_error_rad=angle,**metrics)
            if not passed:raise SkillFailure('packing_release_geometry_not_ready',metrics)
            self.robot.client.command_any({'command':'ProjectTrackObjects','paths':{p['object']:self.robot.get_object_prim_path(p['object'])}})
        try:self.ops.release(self.robot,arm,gentle=p['mode']=='gentle',profile=profile)
        finally:
            if packing:self.robot.client.command_any({'command':'ProjectTrackObjects','paths':{}})
        self.holding.pop(p['role']);self.facts.add('released:'+p['object'])
        self.held_transforms.pop(p['role'],None)

    def withdraw_seated_cap_hand(self,arm):
        current=self.robot.get_ee_pose(arm)
        pen=self.ops.pose(self.robot,self.spec['bindings']['pen'])
        up=np.array([0.,0.,1.])
        outward=-pen[:3,0].copy()
        outward[2]=0.
        horizontal_norm=np.linalg.norm(outward)
        directions=[up]
        if horizontal_norm>.1:
            outward/=horizontal_norm
            directions.extend([(outward+up)/np.sqrt(2),outward])
        directions=np.asarray(directions)
        distances=np.linspace(.001,.045,45)
        points=current[:3,3]+distances[None,:,None]*directions[:,None,:]
        rotation=Rotation.from_matrix(current[:3,:3]).as_quat()[[3,0,1,2]]
        orientations=np.broadcast_to(rotation,(*points.shape[:2],4))
        candidates=np.concatenate([points,orientations],axis=2)
        config=self.ops.FollowTrajectoryConfig(action_description=self.ops.description('cap_empty_hand_waypoint_withdrawal'))
        moved=self.robot.follow_trajectory('gripper',candidates,arm=arm,trajectory_config=config)
        actual=self.robot.get_ee_pose(arm)
        end_error=float(np.linalg.norm(points[:,-1]-actual[:3,3],axis=1).min())
        angle_error=float(Rotation.from_matrix(current[:3,:3].T@actual[:3,:3]).magnitude())
        self.event('cap_empty_hand_waypoint_withdrawal',api_success=bool(moved),directions=directions,
                   start=current,endpoint_candidates=points[:,-1],actual=actual,
                   end_error_m=end_error,rotation_error_rad=angle_error)
        if not moved or end_error>.008 or angle_error>.05:
            diagnostic=np.repeat(current[None],len(directions)+1,axis=0)
            diagnostic[1:,:3,3]=points[:,-1]
            collision_ik,_=self.robot.robot.solve_ik(diagnostic,ee_type='gripper',arm=arm,type='AvoidObs')
            kinematic_ik,_=self.robot.robot.solve_ik(diagnostic,ee_type='gripper',arm=arm,type='Simple')
            raise SkillFailure('cap_empty_hand_withdrawal_failed',{'end_error_m':end_error,'rotation_error_rad':angle_error,
                'current_and_endpoint_collision_ik':collision_ik,'current_and_endpoint_kinematic_ik':kinematic_ik})
        self.verify_relation({'object':self.spec['bindings']['cap'],'relation':'cap_seated'})

    def retreat(self,p):
        role=p['role']
        if role in self.holding:raise SkillFailure('hand_not_empty',p)
        arm=self.arm(role)
        cap_seated=self.spec['template']=='put_the_cap_on_the_pen_new' and 'relation:cap_seated' in self.facts
        if cap_seated and role=='assistant':
            self.withdraw_seated_cap_hand(arm)
        elif not self.ops.retreat(self.robot,arm):
            raise SkillFailure('retreat_failed',p)
        if cap_seated:self.facts.add('cap_'+role+'_retreated')
        self.restore_contact_planning()

    def wait_stable(self,p):
        a=self.ops.pose(self.robot,p['object']);self.robot.client.run_step(120);b=self.ops.pose(self.robot,p['object'])
        movement=float(np.linalg.norm(a[:3,3]-b[:3,3]));self.event('settling',object=p['object'],movement_m=movement)
        if movement>.003:raise SkillFailure('unstable',{'movement_m':movement})
        if self.spec['template']=='put_the_cap_on_the_pen_new' and p['object']==self.spec['bindings']['pen']:
            self.facts.add('cap_pen_stable')

    def insert_axial(self,p):
        if self.spec['template']!='put_the_cap_on_the_pen_new':raise UnsupportedRequest('No axial assembly contract',missing_skills=['insert_axial'])
        if 'cap_aligned' not in self.facts:raise SkillFailure('assembly_alignment_required',p)
        config=self.spec['assembly']
        self.robot.client.command_any({'command':'ProjectTrackObjects','paths':{obj:self.robot.get_object_prim_path(obj) for obj in [p['object'],p['reference']]}})
        with self.contact_planning([self.robot.get_object_prim_path(p['reference'])]):
            for depth in np.linspace(0,config['preinsert_gap_m']+config['insertion_depth_m'],16)[1:]:
                requested_depth=float(depth-config['preinsert_gap_m'])
                self.trim_cap_for_insertion(p,requested_depth)
                goal,_=self.geometry.target(p['target'],p['object']);t=goal.copy();t[:3,3]+=goal[:3,0]*depth
                self.follow_cap_axial_step(p,t,requested_depth)
                try:self.verify_relation({'object':p['object'],'relation':'cap_seated'})
                except SkillFailure:pass
                else:self.facts.add('assembled');return
        self.verify_relation({'object':p['object'],'relation':'cap_seated'})
        self.facts.add('assembled')

    def follow_cap_axial_step(self, params, target, depth):
        try:
            self.follow_local_pose(params,target,'cap_axial_waypoint_motion')
            return
        except SkillFailure as failure:
            if failure.kind!='native_action_failed_before_rollback' or depth>=-.003:
                raise
            cap=self.ops.pose(self.robot,params['object'])
            mouth=self.cap_mouth_measurement(cap)
            advance=depth-mouth['depth_m']
            clearance=self.cap_radial_clearance(mouth,depth)-abs(depth)*mouth['axis_sine']
            if not .0005<advance<.008 or mouth['rotation_rad']>=.012 or clearance<=.001:
                raise
            arm=self.arm(params['role'])
            actual_grasp=self.robot.get_ee_pose(arm)@self.held_transforms[params['role']]
            slip_m=float(np.linalg.norm(actual_grasp[:3,3]-cap[:3,3]))
            slip_rad=float(Rotation.from_matrix(actual_grasp[:3,:3].T@cap[:3,:3]).magnitude())
            if slip_m>.008 or slip_rad>.05:
                raise

        # Preserve the measured tilt and radial position across the free gap.
        # This candidate still uses the full native joint and collision filters.
        alternate=cap.copy()
        pen_axis=self.ops.pose(self.robot,params['reference'])[:3,0]
        alternate[:3,3]+=advance*pen_axis
        self.event('cap_axis_only_approach_candidate',requested_depth_m=depth,advance_m=advance,
                   radial_margin_m=clearance,grasp_slip_m=slip_m,grasp_slip_rad=slip_rad)
        self.follow_local_pose(params,alternate,'cap_axis_only_approach_waypoints')

    def trim_cap_for_insertion(self, params, requested_depth):
        obj=params['object']
        correction=np.eye(4)
        for attempt in range(3):
            cap=self.ops.pose(self.robot,obj)
            measured=self.cap_mouth_measurement(cap)
            clearance=self.cap_radial_clearance(measured,requested_depth)
            self.event('cap_insertion_clearance',requested_depth_m=requested_depth,
                       clearance_m=clearance,attempt=attempt,**measured)
            if clearance>.00015 and measured['rotation_rad']<.02:return
            if attempt==2:break
            target,_=self.geometry.target(params['target'],obj)
            axis=target[:3,0]
            # Preserve the axial position while correcting the radial error.
            target[:3,3]+=axis*float(axis@(cap[:3,3]-target[:3,3]))
            command=accumulate_pose_correction(target,cap,correction)
            command[:3,3]-=axis*float(axis@(command[:3,3]-target[:3,3]))
            self.follow_local_pose(params,command,'cap_radial_waypoint_correction')
        raise SkillFailure('insufficient_cap_insertion_clearance',{
            'requested_depth_m':requested_depth,'clearance_m':clearance,**measured})

    def verify_relation(self,p):
        relation=p['relation'];ok=False
        if relation.startswith('packing_stage_'):ok=self.ops.phone_rules(self.robot,2*int(relation.rsplit('_',1)[1])+1)
        elif relation=='cap_seated':
            b=self.spec['bindings'];c=self.spec['assembly'];relative=np.linalg.inv(self.ops.pose(self.robot,b['pen']))@self.ops.pose(self.robot,b['cap'])
            expected_x=c['pen_tip_x_m']-c['cap_mouth_x_m']+c['insertion_depth_m']
            axial=abs(relative[0,3]-expected_x);radial=float(np.linalg.norm(relative[1:3,3]));angle=float(Rotation.from_matrix(relative[:3,:3]).magnitude())
            joint_paths=[self.robot.get_object_prim_path(b['cap'])+suffix for suffix in ['/PenAttachJoint','/body/PenAttachJoint']]
            snapshot=self.robot.client.command_any({'command':'ProjectSnapshot','joint_paths':joint_paths})
            depth=float(relative[0,3]-c['pen_tip_x_m']+c['cap_mouth_x_m'])
            half_width=c['tip_half_width_m'] if depth<=c['transition_start_depth_m'] else c['transition_half_width_m'] if depth<=c['barrel_start_depth_m'] else c['barrel_half_width_m']
            clearance=c['bore_half_width_m']-half_width
            ok=axial<.004 and radial<clearance and angle<.035 and any(snapshot['joint_enabled'].values())
            self.ops.note(self.robot,'cap_connection_gate',snapshot=snapshot,axial_error_m=float(axial),radial_error_m=radial,
                measured_insertion_depth_m=depth,radial_clearance_m=clearance,rotation_error_rad=angle,passed=bool(ok),official_snap_model=True)
        elif relation=='correct_hole_and_stable':ok=self.verify_task({})
        if not ok:raise SkillFailure('relation_not_satisfied',{'relation':relation})
        self.facts.add('relation:'+relation)

    def sync_assembly_proxy(self,p):
        arm=self.arm(p['role'])
        expected=[self.robot.get_object_prim_path(p[x]) for x in ('object','reference')]
        self.ops.detach_planner_payload(self.robot,arm)
        snapshot=self.robot.client.attach_obj(prim_paths=expected,is_right=arm=='right')
        measured=self.robot.client.command_any({'command':'ProjectSnapshot'})
        link='attached_object' if arm=='right' else 'left_attached_object'
        actual=measured.get('planner_attachments',{}).get(link,[])
        if not all(any(path==root or path.startswith(root.rstrip('/')+'/') for path in actual)
                   for root in expected):
            raise SkillFailure('planner_attachment_failed',{'expected_roots':expected,'planner_colliders':actual})
        self.facts.add('assembly_proxy_synced')
        self.event('assembly_planner_proxy',result=str(snapshot),physical_weld_created=False,snapshot=measured)

    def verify_task(self,p):
        if self.holding:raise SkillFailure('objects_still_held',self.holding)
        name=self.spec['template'];b=self.spec['bindings'];ok=False
        if name=='phone_packaging':ok=all(self.ops.phone_rules(self.robot,i) for i in [1,3,5])
        elif name=='insert_building_block':
            from insertion_physics import crossing_report
            crossing=crossing_report(self.folder/'object-physics-120hz.jsonl')
            (self.folder/'hole-crossing-audit.json').write_text(json.dumps(crossing,indent=2))
            self.robot.client.command_any({'command':'ProjectTrackObjects','paths':{}})
            actual=self.ops.pose(self.robot,b['workpiece']);local=np.linalg.inv(self.ops.pose(self.robot,b['receiver']))@actual
            contained=abs(local[0,3])<.060 and abs(local[2,3])<.060 and -.075<local[1,3]<.055
            a=actual.copy();self.robot.client.run_step(60);stable=float(np.linalg.norm(self.ops.pose(self.robot,b['workpiece'])[:3,3]-a[:3,3]))<.003
            ok=bool(crossing['passed'] and contained and stable)
            self.ops.note(self.robot,'star_final_gate',passed=ok,descended=crossing['fully_below_plane'],contained=bool(contained),stable=stable,correct_hole_entry_recorded=crossing['passed'],physics_samples=crossing['physics_samples'])
            if contained and stable and crossing['final_fully_below_plane'] and not crossing['passed']:
                self.event('independent_task_verdict',passed=False,task_id=self.spec['task_id'])
                raise SkillFailure('failed_passage_evidence_after_containment',{'passage_attempts':crossing['passage_attempts'],
                    'reason':'Workpiece is now inside the receiver. Starting a new controlled trial preserves the actual failed passage.'})
        elif name=='put_the_cap_on_the_pen_new':
            self.verify_relation({'object':b['cap'],'relation':'cap_seated'})
            on_table=abs(self.ops.pose(self.robot,b['pen'])[2,3]-self.geometry.initial[b['pen']][2,3])<.025
            paths=[self.robot.get_object_prim_path(b['cap'])+s for s in ['/PenAttachJoint','/body/PenAttachJoint']]
            snapshot=self.robot.client.command_any({'command':'ProjectSnapshot','joint_paths':paths})
            ok=on_table;self.ops.note(self.robot,'pen_final_gate',on_table=bool(on_table),snapshot=snapshot,official_snap_model=True)
        self.event('independent_task_verdict',passed=bool(ok),task_id=self.spec['task_id'])
        if not ok:raise SkillFailure('physical_task_incomplete',{'task_id':self.spec['task_id']})
        return True

    def execute(self,plan):
        validate_plan(plan,self.spec,self.adapter,held=self.holding,achieved=self.facts)
        for step in plan['steps']:
            if time.monotonic()-self.started>self.max_wall_seconds:raise SkillFailure('wall_time_budget',{})
            self.event('step_started',step=step,state=self.state())
            try:getattr(self,step['skill'])(step['parameters'])
            except Exception as exc:
                failed_state=None if simulator_unavailable(exc) else self.state()
                self.event('step_failed',step=step,error_type=exc.kind if isinstance(exc,SkillFailure) else type(exc).__name__,error=str(exc),
                    details=exc.details if isinstance(exc,SkillFailure) else str(exc),state=failed_state)
                raise
            self.executed.append(step['id']);self.event('step_succeeded',step_id=step['id'],state=self.state())
        return True

def run_structured(robot,ops,request):
    from structured_planner import StructuredPlanner,PlannerBudgetExceeded
    folder=Path(os.environ['GENIE_EPISODE_EVIDENCE']);adapter=load_robot(request.get('robot_id','g2crsb90d'))
    spec=task_spec(robot.scene_info,adapter.robot_id);adapter.require(spec['required_capabilities'])
    executor=SkillExecutor(robot,adapter,spec,ops,folder)
    (folder/'task-spec.json').write_text(json.dumps(spec,indent=2))
    contract=robot.client.command_any({'command':'ProjectGripperConfiguration'})
    if not isinstance(contract,dict) or contract.get('success') is not True:raise RuntimeError('Native gripper contract unavailable')
    (folder/'gripper-contract.json').write_text(json.dumps(contract,indent=2))
    planner=StructuredPlanner(adapter,spec,folder,request['instruction'])
    failure=None
    for revision in range(3):
        try:
            if failure and spec['template'] == 'phone_packaging':
                for index in range(len(spec['manipulated_objects'])):
                    relation = f'relation:packing_stage_{index}'
                    if relation in executor.facts and not ops.phone_rules(robot, 2*index+1):
                        executor.facts.discard(relation)
            if failure and spec['template']=='put_the_cap_on_the_pen_new' and 'relation:cap_seated' in executor.facts:
                try:executor.verify_relation({'object':spec['bindings']['cap'],'relation':'cap_seated'})
                except SkillFailure:
                    executor.facts.difference_update({'relation:cap_seated','assembled','assembly_proxy_synced',
                                                      'cap_aligned','cap_assistant_retreated','cap_operator_retreated',
                                                      'cap_pen_stable'})
            plan,digest=planner.plan(executor.state(),failure=failure);executor.plan_sha256=digest
            executor.event('validated_plan_started',plan=plan,robot_profile=adapter.describe(),revision=revision)
            return executor.execute(plan)
        except UnsupportedRequest as exc:
            executor.event('unsupported_request',**exc.details);return False
        except PlannerBudgetExceeded as exc:
            executor.event('budget_terminated',reason=exc.reason,calls=planner.calls,tokens=planner.tokens);return False
        except Exception as exc:
            if simulator_unavailable(exc):
                executor.event('infrastructure_terminated',reason='isaac_rpc_unavailable',error=str(exc))
                return False
            executor.restore_contact_planning()
            if isinstance(exc,SkillFailure) and exc.kind=='failed_passage_evidence_after_containment':
                executor.event('physical_verification_terminated',reason=exc.kind,details=exc.details);return False
            if isinstance(exc,SkillFailure) and exc.kind=='wall_time_budget':
                executor.event('budget_terminated',reason='execution_wall_time',calls=planner.calls,tokens=planner.tokens);return False
            failure={'type':exc.kind if isinstance(exc,SkillFailure) else type(exc).__name__,
                'details':exc.details if isinstance(exc,SkillFailure) else str(exc)}
            executor.event('bounded_recovery_requested',failure=failure,revision=revision)
    executor.event('budget_terminated',reason='maximum_three_llm_plans',calls=planner.calls,tokens=planner.tokens)
    return False
