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
        self.profile_scene_id=spec.get('profile_scene_id',spec['scene_id'])
        self.held_transforms={}
        self.pen_presentation_pose=None
        self.pen_workspace_recoveries=0
        self.shape_transport_waypoint_index=0
        self.shape_grasp_probe_used=False
        self.events=self.folder/'skill-execution.jsonl';self.executed=[]
        self.max_wall_seconds=int(adapter.config.get('execution_budgets',{}).get('max_episode_execution_wall_s',2100))
        (self.folder/'budget-profile.json').write_text(json.dumps({'max_episode_execution_wall_s':self.max_wall_seconds,'max_model_plans':3,'api_wall_time_budget_s':900},indent=2))
        self.contact_exclusions=set()
        environment=spec.get('environment_collision_paths',[])
        if environment:
            robot.add_objs_to_obstacle(environment)
            snapshot=robot.client.command_any({'command':'ProjectSnapshot'})
            registered=all(any(path==parent or path.startswith(parent+'/') for path in snapshot.get('planner_obstacles',[])) for parent in environment)
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

    def shape_grasp_profile(self, arm):
        profile = self.adapter.config.get('shape_grasp_axial_offset_by_arm', {}).get(arm, {})
        if profile.get('scene_id') != self.profile_scene_id:
            return {}
        if profile.get('requires_composed_scene') and 'support' not in self.spec:
            return {}
        return profile

    def state(self):
        poses={o:self.ops.pose(self.robot,o) for o in self.spec['manipulated_objects']};evidence={}
        for role,obj in list(self.holding.items()):
            expected=self.robot.get_ee_pose(self.arm(role))@self.held_transforms[role]
            error=float(np.linalg.norm(expected[:3,3]-poses[obj][:3,3]));evidence[role]={'object':obj,'measured_following_error_m':error,'holding_is_estimate':True}
            if error>.08:
                self.holding.pop(role);self.held_transforms.pop(role);self.facts.add('lost_grasp:'+obj)
        clock=self.robot.client.command_any({'command':'ProjectSnapshot'})
        stamp=clock.get('sim_time') if isinstance(clock,dict) else None
        return {'holding_by_role':self.holding.copy(),'holding_evidence':evidence,'achieved':sorted(self.facts),
            'object_poses':{o:t.tolist() for o,t in poses.items()},
            'sim_time':float(stamp) if isinstance(stamp,(int,float)) and np.isfinite(stamp) else None}

    def grasp(self,p):
        obj,arm=p['object'],self.arm(p['role']);profile=p['grasp_profile']
        options={'annotated_topdown':{},'release_compatible':{'goal_topdown':True,'goal_lookahead':bool(self.adapter.config.get('shape_grasp_lookahead',False))},'topdown_box':{'axis':2},'pen_barrel':{'offset':[.025,0,0]},'cap_body':{'offset':[-.004,0,0]}}[profile]
        if obj in self.spec.get('grasp_axes', {}):
            options['axis'] = self.spec['grasp_axes'][obj]
        if 'placement_geometry' in self.spec:
            from task_plugins import load
            attempts = self.__dict__.setdefault('placement_grasp_attempts', {})
            attempt = attempts.get(obj, 0)
            attempts[obj] = attempt + 1
            options.update(load(self.spec['task_plugin']).grasp_options(self, obj, arm, attempt))
        if profile=='pen_barrel' and 'support' in self.spec:
            # Shorten the lever arm on the held pen before moving the other arm.
            options['offset']=[.005,0,0]
            self.pen_presentation_pose=None
            self.pen_workspace_recoveries=0
        calibration=self.adapter.config.get('custom_grasp_candidates',{}).get(profile,{})
        exact_scene_id = calibration.get('exact_scene_id')
        scene_matches = (self.spec.get('scene_id') == exact_scene_id if exact_scene_id is not None
                         else calibration.get('scene_id') == self.profile_scene_id)
        use_calibrated_grasp = scene_matches and calibration.get('arm') == arm
        if use_calibrated_grasp:
            options['custom_pose_options']=calibration['pose_options']
            self.event('calibrated_grasp_candidates',object=obj,role=p['role'],profile=profile,**calibration)
        if profile=='release_compatible':
            self.shape_grasp_probe_used = False
            pinch=self.shape_grasp_profile(arm)
            if pinch:
                options['pinch_axis_offset_m']=pinch['offset_m']
                options['release_tcp_x_points_up']=bool(pinch.get('release_tcp_x_points_up',False))
                if 'contact_height_offsets_m' in pinch:
                    options['pinch_height_offsets_m']=pinch['contact_height_offsets_m']
                if 'lift_probe_distance_m' in pinch:
                    options['lift_probe_distance_m']=pinch['lift_probe_distance_m']
                    options['lift_recovery']=lambda target, held: self.recover_probed_lift(p, target, held)
                    options['lift_tracking_recovery']=lambda actual, slip: self.recover_shape_grasp(p, actual, slip)
        alternatives = self.adapter.config.get('grasp_candidate_recovery', {}).get(profile, [])
        if use_calibrated_grasp and exact_scene_id is not None:
            # A measured scene candidate gets the first attempt; retain the native default batch.
            alternatives = [None, *alternatives]
        if 'support' in self.spec and alternatives:
            self.try_grasp_candidates(p, options, alternatives)
        elif not self.ops.pick(self.robot,obj,arm,**options):
            raise SkillFailure('grasp_unreachable_or_failed',{'object':obj,'profile':profile})
        self.holding[p['role']]=obj
        self.facts.discard('released:'+obj)
        if 'placement_geometry' in self.spec:
            self.facts.difference_update({'retreated','stable','relation:inside_container_and_stable'})
        for ref,owner in self.spec['target_objects'].items():
            if owner==obj:self.facts.discard('target:'+ref)
        if self.spec['template']=='insert_building_block':
            self.facts.discard('shape_aligned')
            self.geometry.cache.pop('shape_release_clearance_m', None)
            self.shape_transport_waypoint_index=0
        self.held_transforms[p['role']]=np.linalg.inv(self.robot.get_ee_pose(arm))@self.ops.pose(self.robot,obj)

    def recover_probed_lift(self, params, target, held_transform):
        """Retry a verified friction grasp through short absolute lift targets."""
        obj, arm = params['object'], self.arm(params['role'])

        def measure():
            actual = self.ops.pose(self.robot, obj)
            expected = self.robot.get_ee_pose(arm) @ held_transform
            slip = float(np.linalg.norm(expected[:3, 3] - actual[:3, 3]))
            tilt = float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude())
            if slip > .008 or tilt > .08:
                raise SkillFailure('probed_grasp_lift_exhausted', {'slip_m': slip, 'slip_rad': tilt})
            return actual

        start = measure()
        rise = float(target[2, 3] - start[2, 3])
        if rise <= .004:
            if np.linalg.norm(target[:3, 3] - start[:3, 3]) <= .008:
                return
            raise SkillFailure('probed_grasp_lift_exhausted', {'remaining_lift_m': rise})
        segments = int(np.ceil(rise / .031))
        if segments > 4:
            raise SkillFailure('probed_grasp_lift_exhausted', {'remaining_lift_m': rise})
        step = target.copy()
        for height in np.linspace(start[2, 3], target[2, 3], segments + 1)[1:]:
            step[2, 3] = height
            try:
                self.follow_local_pose(params, step, 'grasp_lift_short_segment')
            except SkillFailure as exc:
                if exc.kind not in {'native_action_failed_before_rollback', 'local_waypoint_plan_or_tracking_failed'}:
                    raise
                bypass = self.try_probed_lift_bypass(params, step, measure)
                if bypass is None:
                    raise SkillFailure('probed_grasp_lift_exhausted', {'target_height_m': float(height),
                                                                      'failure': exc.kind}) from exc
                # Subsequent vertical segments use the feasible column.
                step = bypass
            actual = measure()
            if abs(actual[2, 3] - height) > .004:
                raise SkillFailure('probed_grasp_lift_exhausted', {'target_height_m': float(height),
                                                                  'observed_height_m': float(actual[2, 3])})
            self.event('grasp_lift_short_segment_completed', target=step, measured=actual)

    def try_probed_lift_bypass(self, params, goal, measure):
        """Find a nearby lift column after a straight segment is rejected."""
        routes = [('measured_vertical', np.zeros(2))]
        support = self.spec['support']
        corners = bbox_corners(*self.ops.local_box(self.robot, params['object']))
        start = measure()
        if transformed_points(start, corners)[:, 2].min() - support['height_m'] >= .012:
            base = self.robot.client.command_any({'command': 'ProjectKinematicSnapshot'})
            receiver = self.ops.pose(self.robot, self.spec['bindings']['receiver'])
            for name, direction in (
                ('inboard', np.asarray(base['robot_base_position'][:2]) - start[:2, 3]),
                ('away_from_fixture', start[:2, 3] - receiver[:2, 3]),
            ):
                length = np.linalg.norm(direction)
                if length > .02:
                    routes.append((name, .02 * direction / length))

        for name, offset in routes:
            live = measure()
            if goal[2, 3] <= live[2, 3] + .004:
                break
            departure = live.copy()
            departure[:2, 3] += offset
            candidate = departure.copy()
            candidate[2, 3] = goal[2, 3]
            if np.allclose(candidate, goal, atol=1e-5, rtol=0.0):
                continue
            via = () if name == 'measured_vertical' else (departure,)
            if via and transformed_points(live, corners)[:, 2].min() - support['height_m'] < .012:
                break
            try:
                self.follow_local_pose(params, candidate, 'grasp_lift_bypass', via=via)
            except SkillFailure as exc:
                if exc.kind not in {'native_action_failed_before_rollback', 'local_waypoint_plan_or_tracking_failed'}:
                    raise
                self.event('grasp_lift_bypass_rejected', route=name, reason=exc.kind, target=candidate)
                continue
            self.event('grasp_lift_bypass_selected', route=name, target=candidate, measured=measure())
            return candidate
        return None

    def try_grasp_candidates(self, params, options, alternatives):
        """Try calibrated alternatives only after a measured pre-command refusal."""
        from functools import partial
        from skill_recovery import FailureKind, PhysicalState, RecoveryCandidate, RecoveryFailure
        from skill_recovery import native_command_offset, run_recoverable_skill

        obj, arm = params['object'], self.arm(params['role'])
        profile = params['grasp_profile']
        command_path = self.folder / 'native-control-commands.jsonl'
        low, high = self.ops.local_box(self.robot, obj)
        center = np.r_[(np.asarray(low) + np.asarray(high)) / 2., 1.]
        attempts = [options, *({**options, 'custom_pose_options': candidate} for candidate in alternatives)]

        def observe():
            poses = {'payload': self.ops.pose(self.robot, obj), 'hand': self.robot.get_ee_pose(arm)}
            return PhysicalState(poses, command_offset=native_command_offset(command_path), holding=self.holding)

        def invoke(_before, *, arguments):
            if not self.ops.pick(self.robot, obj, arm, **arguments):
                raise SkillFailure('grasp_unreachable_or_failed', {'object': obj, 'profile': profile})

        def lifted(before, after, _result):
            rise = float(((after.poses['payload'] - before.poses['payload']) @ center)[2])
            hand_rise = float(after.poses['hand'][2, 3] - before.poses['hand'][2, 3])
            if rise > .055:
                return True
            raise RecoveryFailure(FailureKind.GRASP_FAILED, {'measured_lift_m': rise, 'hand_lift_m': hand_rise})

        def record(event, **fields):
            self.event(event, object=obj, role=params['role'], profile=profile, **fields)
            index = int(fields['candidate'])
            if event in {'skill_attempt_succeeded', 'skill_attempt_recovered'}:
                self.event('grasp_candidate_used', object=obj, arm=arm, candidate=index,
                           options=attempts[index].get('custom_pose_options'))
            elif event == 'skill_attempt_failed':
                details = {**fields.get('details', {}), 'object': obj, 'profile': profile,
                           'candidate': index, 'reason': fields['failure_kind']}
                self.event('grasp_candidate_rejected', **details)

        candidates = [RecoveryCandidate(str(index), partial(invoke, arguments=arguments))
                      for index, arguments in enumerate(attempts)]
        if not hasattr(self, 'grasp_recovery_history'):
            self.grasp_recovery_history = {}
        try:
            run_recoverable_skill(f'grasp:{arm}:{obj}:{profile}', candidates, observe=observe, verify=lifted,
                                  event=record, deadline=self.started + self.max_wall_seconds,
                                  max_attempts=len(candidates), history=self.grasp_recovery_history)
        except RecoveryFailure as exc:
            kind = {
                FailureKind.PLANNING_REFUSED: 'grasp_unreachable_or_failed',
                FailureKind.GRASP_FAILED: 'grasp_candidate_changed_world',
                FailureKind.PHYSICAL_FAILURE: 'grasp_candidate_changed_world',
                FailureKind.BUDGET_EXHAUSTED: 'wall_time_budget',
            }.get(exc.kind, exc.kind.value)
            raise SkillFailure(kind, {**exc.details, 'object': obj, 'profile': profile,
                                      'recovery_failure_kind': exc.kind.value}) from exc

    def shape_entry(self,p):
        arm=self.arm(p['role'])
        gaps=self.adapter.config.get('shape_release_clearance_by_arm_m',{})
        if arm in gaps:
            return self.geometry.shape_release_geometry(p['object'],arm,gaps[arm])
        if 'support' in self.spec and self.spec['template']=='insert_building_block':
            clearance = self.geometry.cache.get('shape_release_clearance_m', .008)
            return self.geometry.shape_release_geometry(p['object'], arm, clearance)
        goal,excluded=self.geometry.target('shape_entry',p['object'])
        return goal,excluded,{}

    def held_goal_mask(self,role,obj,candidates):
        arm=self.arm(role)
        object_to_hand=np.linalg.inv(self.ops.pose(self.robot,obj))@self.robot.get_ee_pose(arm)
        flags,_=self.robot.robot.solve_ik(candidates@object_to_hand,ee_type='gripper',arm=arm,type='AvoidObs')
        mask=np.asarray(flags,dtype=bool).reshape(-1)
        if len(mask)!=len(candidates):raise ValueError('Planner returned an invalid held-object candidate mask')
        return mask

    def pen_presentation_candidates(self,p,goal):
        cap=self.spec['bindings']['cap']
        if self.holding.get('assistant')!=cap:return goal[None]
        # Center the next preinsert near the elevated outboard pose reached by
        # the cap hand in the previous trial. Test both hands before moving.
        candidates=np.repeat(goal[None],5,axis=0)
        candidates[0,:3,3]-=.04*goal[:3,0]
        candidates[:2,2,3]+=.035
        candidates[3:,:3,:3]=Rotation.from_rotvec([[0.,0.,.24],[0.,0.,-.24]]).as_matrix()@goal[:3,:3]
        candidates[3:,:3,3]-=.025*goal[:3,0]
        preinsert=np.array([self.geometry.cap_preinsert_pose(candidate) for candidate in candidates])
        pen_ik=self.held_goal_mask(p['role'],p['object'],candidates)
        cap_ik=self.held_goal_mask('assistant',cap,preinsert)
        eligible=pen_ik&cap_ik
        self.event('cap_presentation_candidates',pen_targets=candidates,preinsert_targets=preinsert,
                   pen_collision_ik=pen_ik,cap_collision_ik=cap_ik,eligible=eligible)
        if not eligible.any():raise SkillFailure('no_paired_presentation_candidate',{'evaluated':len(candidates)})
        indices=np.flatnonzero(eligible)
        return candidates[indices]

    def move_pen_presentation(self, params, goal):
        arm = self.arm(params['role'])
        pen = params['object']
        try:
            candidates = self.pen_presentation_candidates(params, goal)
        except SkillFailure as exc:
            if exc.kind != 'no_paired_presentation_candidate' or 'support' not in self.spec:
                raise
            candidates = ()
        for index, candidate in enumerate(candidates):
            self.stable_pen_for_alignment()
            try:
                if self.ops.move(self.robot, pen, arm, candidate, 'plan_pen_present'):
                    self.event('cap_presentation_selected', candidate=index, target=candidate)
                    return
            except SkillFailure as exc:
                if exc.kind != 'native_action_failed_before_rollback':
                    raise
            self.event('cap_presentation_candidate_rejected', candidate=index, target=candidate)
        if 'support' in self.spec and self.try_rolled_presentation(params, goal):
            return
        raise SkillFailure('no_paired_presentation_path', {'role': params['role']})

    def try_rolled_presentation(self, params, goal):
        """Find a different wrist orientation around the same assembly axis."""
        cap = self.spec['bindings']['cap']
        if self.holding.get('assistant') != cap:
            return False
        self.stable_pen_for_alignment()
        pen, role = params['object'], params['role']
        rolls = np.deg2rad([-30, 30, -60, 60, -90, 90])
        candidates = np.repeat(goal[None], len(rolls), axis=0)
        candidates[:, :3, :3] = goal[:3, :3] @ Rotation.from_rotvec(np.c_[rolls, np.zeros((len(rolls), 2))]).as_matrix()
        preinsert = np.array([self.geometry.cap_preinsert_pose(candidate) for candidate in candidates])
        pen_ik = self.held_goal_mask(role, pen, candidates)
        cap_ik = self.held_goal_mask('assistant', cap, preinsert)
        eligible = pen_ik & cap_ik
        self.event('cap_presentation_roll_candidates', rolls_rad=rolls, pen_collision_ik=pen_ik,
                   cap_collision_ik=cap_ik, eligible=eligible)
        if not eligible.any():
            return False
        try:
            if not self.ops.move(self.robot, pen, self.arm(role), candidates[eligible], 'plan_pen_present_roll'):
                return False
        except SkillFailure as exc:
            if exc.kind == 'native_action_failed_before_rollback':
                return False
            raise
        measured = self.stable_pen_for_alignment()
        position = np.linalg.norm(candidates[:, :3, 3] - measured[:3, 3], axis=1)
        angle = Rotation.from_matrix(candidates[:, :3, :3].transpose(0, 2, 1) @ measured[:3, :3]).magnitude()
        matches = np.flatnonzero(eligible & (position <= .008) & (angle <= .05))
        if len(matches):
            index = int(matches[0])
            self.event('cap_presentation_roll_selected', roll_rad=float(rolls[index]),
                       position_error_m=float(position[index]), rotation_error_rad=float(angle[index]))
        return bool(len(matches))

    def follow_local_pose(self, params, goal, phase, *, free_space_rotation=False, via=()):
        obj = params['object']
        arm = self.arm(params['role'])
        measure_pose = (lambda: self.robot.get_ee_pose(arm)) if obj == 'gripper' else (
            lambda: self.ops.pose(self.robot, obj))
        start = measure_pose()
        angle_step = .02 if free_space_rotation else .005
        legs = []
        previous = start
        for target in (*via, goal):
            delta = target[:3, 3] - previous[:3, 3]
            rotations = Rotation.from_matrix(np.stack([previous[:3, :3], target[:3, :3]]))
            distance = float(np.linalg.norm(delta))
            angle = float((rotations[0].inv() * rotations[1]).magnitude())
            if distance > .09 or (angle > .25 and not free_space_rotation):
                raise SkillFailure('local_waypoint_segment_out_of_range', {
                    'phase': phase, 'distance_m': distance, 'angle_rad': angle})
            segments = max(4, int(np.ceil(distance / .0008)), int(np.ceil(angle / angle_step)))
            fractions = np.linspace(0, 1, segments + 1)
            fractions = fractions**2 * (3 - 2*fractions)
            positions = previous[:3, 3] + fractions[:, None] * delta
            orientations = Slerp([0, 1], rotations)(fractions).as_quat()[:, [3, 0, 1, 2]]
            leg = np.column_stack([positions, orientations])
            legs.append(leg[1:] if legs else leg)
            previous = target
        waypoints = np.concatenate(legs)
        config = self.ops.FollowTrajectoryConfig(action_description=self.ops.description(phase))
        self.event('local_waypoints_submitted', phase=phase, count=len(waypoints), start=start, goal=goal)
        # RobotAPI checks every waypoint with AvoidObs. The native follower
        # seeds each solve from the previous joint solution and steps PhysX.
        moved = self.robot.follow_trajectory(obj, waypoints, arm=arm, trajectory_config=config)
        if moved:
            self.robot.client.run_step(30)
        measured = measure_pose()
        error = float(np.linalg.norm(measured[:3, 3] - goal[:3, 3]))
        rotation_error = float(Rotation.from_matrix(goal[:3, :3].T @ measured[:3, :3]).magnitude())
        self.ops.note(self.robot, phase, api_success=bool(moved), object=obj, arm=arm,
                      waypoint_count=len(waypoints), target=goal, actual=measured,
                      position_error_m=error, rotation_error_rad=rotation_error)
        if not moved or error > .008 or rotation_error > .05:
            raise SkillFailure('local_waypoint_plan_or_tracking_failed', {
                'phase': phase, 'planner_accepted': bool(moved), 'position_error_m': error,
                'rotation_error_rad': rotation_error, 'waypoint_count': len(waypoints)})

    def move_cap_staging_candidates(self, params, goal, nominal, hand_to_cap):
        """Screen free-space staging poses while retaining the live friction grasp."""
        cap = params['object']
        arm = self.arm(params['role'])
        axial = goal[:3, 0]
        lateral = np.cross([0., 0., 1.], axial)
        start = self.ops.pose(self.robot, cap)
        if (start[:3, 3] - nominal[:3, 3]) @ lateral < 0:
            lateral = -lateral
        candidates = np.repeat(nominal[None], 5, axis=0)
        candidates[1:, 2, 3] += [-.023, -.016, -.016, -.015]
        candidates[2, :3, 3] += .037 * lateral
        candidates[3, :3, 3] -= .037 * lateral
        candidates[4, :3, 3] -= .026 * axial
        available = self.held_goal_mask(params['role'], cap, candidates)
        self.event('cap_staging_candidates', targets=candidates, collision_ik=available)
        for index in np.flatnonzero(available):
            target = candidates[index]
            try:
                if self.ops.move(self.robot, cap, arm, target, 'plan_cap_outboard_translation'):
                    self.event('cap_staging_candidate_selected', candidate=int(index), target=target)
                    return
            except SkillFailure as exc:
                if exc.kind != 'native_action_failed_before_rollback':
                    raise
            actual = self.ops.pose(self.robot, cap)
            expected = self.robot.get_ee_pose(arm) @ hand_to_cap
            slip = float(np.linalg.norm(expected[:3, 3] - actual[:3, 3]))
            tilt = float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude())
            self.event('cap_staging_candidate_rejected', candidate=int(index), slip_m=slip, slip_rad=tilt)
            if slip > .008 or tilt > .05:
                raise SkillFailure('cap_grasp_lost_before_alignment', {'slip_m': slip, 'slip_rad': tilt})
        raise SkillFailure('cap_staging_motion_failed', {'candidate_count': int(available.sum())})

    def move_cap_preinsert(self, params, goal):
        arm = self.arm(params['role'])
        cap = params['object']
        current = self.ops.pose(self.robot, cap)
        gap = np.linalg.norm(current[:3, 3] - goal[:3, 3])
        angle = Rotation.from_matrix(goal[:3, :3].T @ current[:3, :3]).magnitude()
        adaptive = self.pen_presentation_pose is not None
        if gap > (.085 if adaptive else .065) or (not adaptive and angle > .20):
            stage = goal.copy()
            stage[:3, 3] -= .04 * goal[:3, 0]
            stage[2, 3] += .035
            self.event('cap_preinsert_staging_candidate', target=stage)
            # Keep the friction grasp steady during the long translation. Rotate
            # over the outboard clearance with a sampled, collision-checked path.
            hand_to_cap = np.linalg.inv(self.robot.get_ee_pose(arm)) @ current
            translation = stage.copy()
            translation[:3, :3] = current[:3, :3]
            if adaptive:
                self.move_cap_staging_candidates(params, goal, translation, hand_to_cap)
            elif not self.ops.move(self.robot, cap, arm, translation, 'plan_cap_outboard_translation'):
                raise SkillFailure('cap_staging_motion_failed', params)
            if not adaptive:
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
        if adaptive:
            self.recover_cap_preinsert(params)
        else:
            goal, _ = self.geometry.target(params['target'], cap)
            self.follow_local_pose(params, goal, 'cap_preinsert_waypoint_motion')

    def stable_pen_for_alignment(self):
        pen = self.spec['bindings']['pen']
        if self.holding.get('operator') != pen:
            raise SkillFailure('pen_grasp_lost_before_alignment', {'object': pen})
        before = self.ops.pose(self.robot, pen)
        for _ in range(3):
            self.robot.client.run_step(24)
            actual = self.ops.pose(self.robot, pen)
            predicted = self.robot.get_ee_pose(self.arm('operator')) @ self.held_transforms['operator']
            following_error = float(np.linalg.norm(predicted[:3, 3] - actual[:3, 3]))
            drift = float(np.linalg.norm(before[:3, 3] - actual[:3, 3]))
            tilt = float(Rotation.from_matrix(before[:3, :3].T @ actual[:3, :3]).magnitude())
            if following_error > .008:
                raise SkillFailure('pen_grasp_lost_before_alignment', {
                    'object': pen, 'held_following_error_m': following_error})
            if drift <= .001 and tilt <= .015:
                self.event('pen_alignment_reference', pose=actual, drift_m=drift, drift_rad=tilt,
                           held_following_error_m=following_error)
                return actual
            before = actual
        raise SkillFailure('pen_not_stable_before_alignment', {'drift_m': drift, 'drift_rad': tilt})

    def recover_cap_preinsert(self, params):
        cap = params['object']
        arm = self.arm(params['role'])
        pen = self.spec['bindings']['pen']
        transport_margin = self.adapter.config.get('held_transport_fixture_margin_m')
        failures = []
        routes = ('rotate_first', 'direct', 'outboard_axial', 'joint_space_connector')
        if transport_margin is not None:
            routes += ('guarded_joint_space',)
        for round_index in range(3):
            for route in routes:
                if self.holding.get(params['role']) != cap:
                    raise SkillFailure('cap_grasp_lost_before_alignment', {'object': cap})
                pen_pose = self.stable_pen_for_alignment()
                goal = self.geometry.cap_preinsert_pose(pen_pose)
                actual = self.ops.pose(self.robot, cap)
                predicted = self.robot.get_ee_pose(arm) @ self.held_transforms[params['role']]
                slip = float(np.linalg.norm(predicted[:3, 3] - actual[:3, 3]))
                slip_angle = float(Rotation.from_matrix(predicted[:3, :3].T @ actual[:3, :3]).magnitude())
                if slip > .008 or slip_angle > .08:
                    raise SkillFailure('cap_grasp_lost_before_alignment', {
                        'object': cap, 'slip_m': slip, 'slip_rad': slip_angle})
                via = ()
                if route == 'rotate_first':
                    pivot = actual.copy()
                    pivot[:3, :3] = goal[:3, :3]
                    via = (pivot,)
                elif route == 'outboard_axial':
                    outboard = goal.copy()
                    outboard[:3, 3] -= .024 * goal[:3, 0]
                    via = (outboard,)
                phase = 'cap_preinsert_' + route
                try:
                    if route == 'guarded_joint_space':
                        # This target still has the full preinsertion gap. Screen
                        # the whole native path against the held pen before moving.
                        if not self.move_held_clear_of_fixture(params, goal, phase, pen, transport_margin, 2.0):
                            raise SkillFailure('cap_guarded_approach_infeasible', {'fixture': pen})
                    else:
                        if route == 'joint_space_connector':
                            self.connect_cap_outboard(params, goal)
                            goal = self.geometry.cap_preinsert_pose(self.stable_pen_for_alignment())
                        self.follow_local_pose(params, goal, phase, via=via, free_space_rotation=True)
                except SkillFailure as exc:
                    if exc.kind not in {'native_action_failed_before_rollback',
                                        'local_waypoint_plan_or_tracking_failed',
                                        'local_waypoint_segment_out_of_range',
                                        'cap_outboard_connector_infeasible', 'cap_guarded_approach_infeasible'}:
                        raise
                    failures.append({'route': route, 'reason': exc.kind, 'details': exc.details})
                    self.event('cap_preinsert_route_rejected', **failures[-1])
                    continue
                # Both arms can deflect while the cap moves. Check the new
                # relative geometry before handing off to precision alignment.
                latest = self.geometry.cap_preinsert_pose(self.stable_pen_for_alignment())
                achieved = self.ops.pose(self.robot, cap)
                position = float(np.linalg.norm(latest[:3, 3] - achieved[:3, 3]))
                angle = float(Rotation.from_matrix(latest[:3, :3].T @ achieved[:3, :3]).magnitude())
                if position <= .008 and angle <= .05:
                    self.event('cap_preinsert_recovery_applied', route=route, workspace_round=round_index,
                               position_error_m=position, rotation_error_rad=angle)
                    return
                failures.append({'route': route, 'reason': 'reference_drifted',
                                 'position_error_m': position, 'rotation_error_rad': angle})
            if round_index < 2:
                self.reposition_pen_for_alignment()
        raise SkillFailure('cap_preinsert_recovery_exhausted', {'attempts': failures})

    def connect_cap_outboard(self, params, goal):
        """Connect in joint space before the final short axial approach."""
        cap = params['object']
        arm = self.arm(params['role'])
        candidates = np.repeat(goal[None], 4, axis=0)
        candidates[:, :3, 3] -= np.array([.024, .04, .031, .031])[:, None] * goal[:3, 0]
        candidates[1:, 2, 3] += .018
        candidates[2:, 0, 3] += [-.024, .024]
        available = self.held_goal_mask(params['role'], cap, candidates)
        self.event('cap_outboard_connector_candidates', targets=candidates, available=available)
        # This uses native cuRobo joint-space search, including the G2
        # coupled wrist limit. A straight Cartesian path can lose that branch.
        if not available.any() or not self.ops.move(
                self.robot, cap, arm, candidates[available], 'plan_cap_outboard_joint_space'):
            raise SkillFailure('cap_outboard_connector_infeasible', {'candidates': int(available.sum())})
        self.event('cap_outboard_connector_executed', actual=self.ops.pose(self.robot, cap))

    def reposition_pen_for_alignment(self):
        pen, cap = (self.spec['bindings'][part] for part in ('pen', 'cap'))
        current = self.stable_pen_for_alignment()
        mouth = self.cap_mouth_measurement(self.ops.pose(self.robot, cap))
        if mouth['depth_m'] > -.01 and mouth['radial_m'] < .02:
            raise SkillFailure('cap_preinsert_recovery_exhausted', {'reason': 'parts_too_close', **mouth})
        recovery_number = self.pen_workspace_recoveries
        self.pen_workspace_recoveries += 1
        if recovery_number >= 2:
            raise SkillFailure('cap_preinsert_recovery_exhausted', {'reason': 'workspace_budget_exhausted'})
        goals = np.repeat(self.pen_presentation_pose[None], 6, axis=0)
        goals[:, :3, 3] += [[0., 0., -.036], [-.025, 0., -.025], [.025, 0., -.025],
                            [0., 0., -.018], [-.025, 0., -.043], [.025, 0., -.043]]
        yaw = Rotation.from_rotvec([[0., 0., .14], [0., 0., -.14]]).as_matrix()
        goals[4:, :3, :3] = yaw @ goals[4:, :3, :3]
        if recovery_number:
            goals = goals[[4, 5, 1, 2, 3, 0]]
        cap_goals = np.asarray([self.geometry.cap_preinsert_pose(goal) for goal in goals])
        available = self.held_goal_mask('operator', pen, goals) & self.held_goal_mask('assistant', cap, cap_goals)
        change = np.linalg.norm(goals[:, :3, 3] - current[:3, 3], axis=1)
        angle = Rotation.from_matrix(current[:3, :3].T @ goals[:, :3, :3]).magnitude()
        available &= (change > .006) | (angle > .06)
        self.event('paired_alignment_workspace_candidates', pen_targets=goals, cap_targets=cap_goals,
                   available=available)
        for goal in goals[available][:2]:
            try:
                self.follow_local_pose({'role': 'operator', 'object': pen}, goal,
                                       'pen_workspace_recovery', free_space_rotation=True)
                return
            except SkillFailure as exc:
                if exc.kind not in {'native_action_failed_before_rollback', 'local_waypoint_plan_or_tracking_failed'}:
                    raise
                self.event('pen_workspace_candidate_rejected', reason=exc.kind, details=exc.details)
        raise SkillFailure('pen_presentation_recovery_unreachable', {'targets': int(available.sum())})

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

    def recover_shape_grasp(self, params, before, slip):
        support = self.spec.get('support')
        if (not support or self.shape_grasp_probe_used or slip['position_error_m'] > .008
                or slip['rotation_error_rad'] > .14):
            raise SkillFailure('shape_transport_grasp_slipped', slip)
        self.shape_grasp_probe_used = True
        obj = params['object']
        role = params['role']
        arm = self.arm(role)
        self.robot.client.run_step(36)
        settled = self.ops.pose(self.robot, obj)
        corners = transformed_points(settled, bbox_corners(*self.ops.local_box(self.robot, obj)))
        height = float(corners[:, 2].min() - support['height_m'])
        drift = float(np.linalg.norm(settled[:3, 3] - before[:3, 3]))
        tilt = float(Rotation.from_matrix(before[:3, :3].T @ settled[:3, :3]).magnitude())
        details = {**slip, 'settled_drift_m': drift, 'settled_rotation_rad': tilt, 'support_gap_m': height}
        if height < .035 or drift > .001 or tilt > .015:
            raise SkillFailure('shape_transport_grasp_slipped', details)
        hand = self.robot.get_ee_pose(arm)
        measured_grasp = np.linalg.inv(hand) @ settled
        # Register the measured payload geometry with cuRobo before probing.
        # PhysX still has to lift the object using the original friction grasp.
        self.sync_planner_payload(arm, [obj])
        target = settled.copy()
        target[2, 3] += .006
        try:
            self.follow_local_pose(params, target, 'shape_grasp_recovery_probe')
        except SkillFailure as exc:
            if exc.kind not in {'native_action_failed_before_rollback', 'local_waypoint_plan_or_tracking_failed'}:
                raise
            raise SkillFailure('shape_transport_grasp_slipped', {**details, 'probe_failure': exc.kind}) from exc
        actual = self.ops.pose(self.robot, obj)
        expected = self.robot.get_ee_pose(arm) @ measured_grasp
        lift = float(actual[2, 3] - settled[2, 3])
        following = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
        rotation = float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude())
        passed = lift >= .003 and following <= .0025 and rotation <= .025
        details.update(lift_m=lift, following_error_m=following, following_error_rad=rotation)
        self.event('shape_grasp_recovery_probe', passed=passed, **details)
        if not passed:
            raise SkillFailure('shape_transport_grasp_slipped', details)
        self.held_transforms[role] = measured_grasp
        return actual

    def correct_shape_waypoint(self, params, target, measured):
        command = accumulate_pose_correction(target, measured, np.eye(4))
        self.event('shape_transport_tracking_correction', target=target, measured=measured, command=command)
        try:
            self.follow_local_pose(params, command, 'shape_transport_tracking_recovery', free_space_rotation=True)
        except SkillFailure as exc:
            if exc.kind != 'local_waypoint_plan_or_tracking_failed' or not exc.details['planner_accepted']:
                raise
        actual = self.ops.pose(self.robot, params['object'])
        position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        angle = float(Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude())
        if position > .008 or angle > .05:
            raise SkillFailure('local_waypoint_plan_or_tracking_failed', {
                'phase': 'shape_transport_tracking_recovery', 'planner_accepted': True,
                'position_error_m': position, 'rotation_error_rad': angle})

    def try_free_space_waypoint_detours(self, params, target, check_grasp):
        """Try nearby complete paths before falling back to a large joint-space move."""
        start = check_grasp('before_detour')
        vertical = np.array([0., 0., .025])
        lateral = np.cross([0., 0., 1.], target[:3, 3] - start[:3, 3])
        lateral *= .021 / max(np.linalg.norm(lateral), 1e-9)
        for offset in (vertical, vertical + lateral):
            start = check_grasp('before_detour_candidate')
            departure, arrival = start.copy(), target.copy()
            departure[:3, 3] += offset
            arrival[:3, 3] += offset
            try:
                self.follow_local_pose(params, target, 'held_waypoint_detour', via=(departure, arrival),
                                       free_space_rotation=True)
            except SkillFailure as exc:
                if exc.kind not in {'native_action_failed_before_rollback',
                                    'local_waypoint_plan_or_tracking_failed', 'local_waypoint_segment_out_of_range'}:
                    raise
                self.event('held_waypoint_detour_rejected', offset_m=offset, reason=exc.kind, details=exc.details)
                continue
            check_grasp('after_detour')
            self.event('held_waypoint_detour_applied', offset_m=offset, target=target)
            return True
        return False

    def move_held_clear_of_fixture(self, params, target, phase, fixture, margin, slowdown):
        obj = params['object']
        arm = self.arm(params['role'])
        hand = self.robot.get_ee_pose(arm)
        payload = self.ops.pose(self.robot, obj)
        guard = {'command': 'ProjectTransportClearance', 'enabled': True, 'arm': arm,
                 'live_tool_world': hand.tolist(), 'fixture_world': self.ops.pose(self.robot, fixture).tolist(),
                 'payload_in_tool': (np.linalg.inv(hand) @ payload).tolist(), 'margin_m': margin,
                 'payload_bounds': np.asarray(self.ops.local_box(self.robot, obj)).tolist(),
                 'fixture_bounds': np.asarray(self.ops.local_box(self.robot, fixture)).tolist()}
        try:
            response = self.robot.client.command_any(guard)
            if not isinstance(response, dict) or response.get('enabled') is not True:
                raise SkillFailure('held_transport_guard_unavailable', {'phase': phase})
            return self.ops.move(self.robot, obj, arm, target, phase, slowdown_factor=slowdown)
        finally:
            self.robot.client.command_any({'command': 'ProjectTransportClearance', 'enabled': False})

    def reconnect_shape_waypoint(self, params, target, step, receiver, slowdown, check_grasp):
        phase = 'shape_transport_joint_space_reconnect'
        margin = self.adapter.config.get('held_transport_fixture_margin_m')
        if 'support' not in self.spec or step != 1 or margin is None:
            return self.ops.move(self.robot, params['object'], self.arm(params['role']), target, phase,
                                 slowdown_factor=slowdown)
        try:
            if self.move_held_clear_of_fixture(params, target, phase, receiver, margin, slowdown):
                return True
        except SkillFailure as exc:
            if exc.kind != 'native_action_failed_before_rollback':
                raise
        start = check_grasp('before_guarded_waypoint_detour')
        fixture = self.ops.pose(self.robot, receiver)
        away = start[:2, 3] - fixture[:2, 3]
        if np.linalg.norm(away) < .01:
            return False
        outboard = start.copy()
        outboard[:2, 3] += .046 * away / np.linalg.norm(away)
        outboard[2, 3] += .044
        self.event('guarded_transport_detour', target=outboard)
        if not self.move_held_clear_of_fixture(params, outboard, 'held_transport_detour', receiver, margin, slowdown):
            return False
        check_grasp('after_guarded_waypoint_detour')
        return self.move_held_clear_of_fixture(params, target, phase, receiver, margin, slowdown)

    def move_shape_approach(self, p, goals, route):
        from hashlib import sha256
        arm = self.arm(p['role'])
        obj = p['object']

        def checked_pose(phase):
            actual = self.ops.pose(self.robot, obj)
            predicted = self.robot.get_ee_pose(arm) @ self.held_transforms[p['role']]
            translation = float(np.linalg.norm(actual[:3, 3] - predicted[:3, 3]))
            angle = float(Rotation.from_matrix(predicted[:3, :3].T @ actual[:3, :3]).magnitude())
            evidence = {'phase': phase, 'position_error_m': translation, 'rotation_error_rad': angle}
            self.event('shape_transport_grasp_tracking', **evidence)
            if translation > .01 or angle > .08:
                return self.recover_shape_grasp(p, actual, evidence)
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
        if len(waypoints) > 1 and self.geometry.shape_transport_lift_m:
            waypoints[0, 2, 3] += self.geometry.shape_transport_lift_m
            self.event('shape_transport_seeded_lift', height_offset_m=self.geometry.shape_transport_lift_m,
                       target=waypoints[0])
        angles = Rotation.from_matrix(waypoints[-1, :3, :3].T @ goals[:, :3, :3]).magnitude()
        index = int(np.argmin(angles))
        if angles[index] > .035:
            raise SkillFailure('shape_transport_route_goal_mismatch', {'rotation_error_rad': float(angles[index])})
        waypoints[-1] = goals[index]
        start_index = min(self.shape_transport_waypoint_index, len(waypoints)-1)
        slowdown = float(self.adapter.config.get('held_transport_reconnect_slowdown_factor', 1.))
        guard_all_moves = route.get('check_payload_clearance', False)
        margin = self.adapter.config.get('held_transport_fixture_margin_m')
        if guard_all_moves and margin is None:
            raise SkillFailure('held_transport_guard_unavailable', {'fixture': receiver})

        def move_guarded(targets, phase):
            try:
                moved = self.move_held_clear_of_fixture(p, targets, phase, receiver, margin, slowdown)
            except SkillFailure as exc:
                if exc.kind != 'native_action_failed_before_rollback':
                    raise
                moved = False
            measured = checked_pose(phase)
            candidates = np.asarray(targets).reshape(-1, 4, 4)
            position = np.linalg.norm(candidates[:, :3, 3] - measured[:3, 3], axis=1)
            angle = Rotation.from_matrix(candidates[:, :3, :3].transpose(0, 2, 1) @ measured[:3, :3]).magnitude()
            passed = bool(moved and np.any((position <= .008) & (angle <= .05)))
            self.event('shape_transport_guarded_motion', phase=phase, passed=passed,
                       position_errors_m=position, rotation_errors_rad=angle)
            return passed

        travel = waypoints[start_index:]
        if guard_all_moves and start_index == 0 and move_guarded(goals, 'shape_transport_guarded_approach'):
            travel = ()
        if len(travel):
            self.event('shape_transport_selected', candidate=index, waypoint_count=len(waypoints), start_index=start_index,
                       source_run=route['source_run'], target=goals[index])
        # The outboard turn is continuous within the calibrated joint limits.
        # Plan every segment again from the measured pose and actual grasp.
        for step, target in enumerate(travel, start=start_index+1):
            self.event('shape_transport_waypoint', step=step, target=target)
            actual = self.ops.pose(self.robot,obj)
            if guard_all_moves:
                if not move_guarded(target, 'shape_transport_guarded_waypoint'):
                    raise SkillFailure('shape_transport_waypoint_failed', {'waypoint':step})
            elif 'support' in self.spec and np.linalg.norm(actual[:3,3]-target[:3,3])>.09:
                if not self.ops.move(self.robot, obj, arm, target, 'shape_transport_reposition',
                                     slowdown_factor=slowdown):
                    raise SkillFailure('shape_transport_reposition_failed',{'waypoint':step})
            else:
                try:
                    self.follow_local_pose(p, target, 'shape_transport_waypoint', free_space_rotation=True)
                except SkillFailure as exc:
                    details = exc.details
                    if 'support' not in self.spec:
                        raise
                    tracking_error = exc.kind == 'local_waypoint_plan_or_tracking_failed'
                    if (exc.kind == 'native_action_failed_before_rollback'
                            or (tracking_error and not details['planner_accepted'])):
                        if not self.try_free_space_waypoint_detours(p, target, checked_pose):
                            checked_pose(f'waypoint_{step}_before_reconnect')
                            self.event('shape_transport_joint_space_reconnect', waypoint=step,
                                       reason=exc.kind, slowdown_factor=slowdown)
                            if not self.reconnect_shape_waypoint(p, target, step, receiver, slowdown, checked_pose):
                                raise
                    elif (tracking_error and details['position_error_m'] <= .025
                          and details['rotation_error_rad'] <= .12):
                        measured = checked_pose(f'waypoint_{step}_before_correction')
                        self.correct_shape_waypoint(p, target, measured)
                    else:
                        raise
            checked_pose(f'waypoint_{step}')
            self.shape_transport_waypoint_index=step
        actual = checked_pose('approach_complete')
        self.geometry.cache.pop('shape_goal_local', None)
        self.ops.note(self.robot, 'plan_shape_approach', object=obj, arm=arm, api_success=True,
                      target=goals, actual=actual)

    def move_shape_entry(self, params, goal, metrics):
        obj, role = params['object'], params['role']
        arm = self.arm(role)
        try:
            if self.ops.move(self.robot, obj, arm, goal, 'plan_shape_entry'):
                return
        except SkillFailure as exc:
            if exc.kind != 'native_action_failed_before_rollback':
                raise
        maximum_drop = self.shape_grasp_profile(arm).get('maximum_release_gap_m', 0.)
        room = maximum_drop - metrics['target_workpiece_clearance_m'] - .003
        lifts = np.array([.006, .012, .018])
        lifts = lifts[lifts <= room]
        if len(lifts):
            candidates = np.repeat(goal[None], len(lifts), axis=0)
            candidates[:, 2, 3] += lifts
            eligible = self.held_goal_mask(role, obj, candidates)
            self.event('shape_entry_clearance_candidates', height_offsets_m=lifts, eligible=eligible)
            if eligible.any() and self.ops.move(self.robot, obj, arm, candidates[eligible],
                                                'plan_shape_raised_entry'):
                actual = self.ops.pose(self.robot, obj)
                distance = np.linalg.norm(candidates[:, :3, 3] - actual[:3, 3], axis=1)
                angle = Rotation.from_matrix(goal[:3, :3].T @ actual[:3, :3]).magnitude()
                index = int(np.argmin(np.where(eligible, distance, np.inf)))
                if distance[index] <= .005 and angle <= .05:
                    selected = metrics['target_minimum_clearance_m'] + float(lifts[index])
                    self.geometry.cache['shape_release_clearance_m'] = selected
                    self.event('shape_entry_raised_clearance_selected', clearance_m=selected,
                               max_drop_m=maximum_drop, height_offset_m=float(lifts[index]))
                    return
        raise SkillFailure('shape_entry_motion_failed', {'object': obj, 'available_height_m': room})

    def move_held(self,p):
        if self.holding.get(p['role'])!=p['object']:raise SkillFailure('not_held',p)
        if p['target'].startswith('packing_') and p['target'].endswith('_contact'):
            self.align_packaging(p);return
        geometry={}
        if p['target']=='shape_entry':goal,excluded,geometry=self.shape_entry(p)
        else:goal,excluded=self.geometry.target(p['target'],p['object'])
        if geometry:self.event('shape_entry_clearance_target',**geometry)
        arm=self.arm(p['role'])
        pinch=self.shape_grasp_profile(arm)
        transport_route = pinch.get('transport_approach_route') or pinch.get(
            'transport_approach_routes_by_scene', {}).get(self.spec.get('scene_id'))
        with self.contact_planning(excluded):
            if p['target']=='cap_preinsert':self.move_cap_preinsert(p,goal)
            elif p['target']=='pen_present':self.move_pen_presentation(p,goal)
            elif p['target']=='pen_return':self.place_capped_pen(p,goal)
            elif p['target']=='shape_approach' and transport_route:
                self.move_shape_approach(p,goal,transport_route)
            elif p['target']=='shape_entry' and arm=='left' and 'support' in self.spec:
                self.move_shape_entry(p, goal, geometry)
            elif not excluded and 'support' in self.spec:
                self.recoverable_held_travel(p, goal)
            elif not self.ops.move(self.robot,p['object'],arm,goal,'plan_'+p['target']):
                raise SkillFailure('motion_or_tracking_failed',{'target':p['target']})
        if p['target']=='pen_present' and 'support' in self.spec:
            self.pen_presentation_pose=self.ops.pose(self.robot,p['object']).copy()
        self.facts.add('target:'+p['target'])

    def recoverable_held_travel(self, params, goal):
        """Retry a free-space approach after checking the live grasp and native command log."""
        from skill_recovery import (FailureKind, PhysicalState, RecoveryCandidate, RecoveryFailure,
                                    native_command_offset, run_recoverable_skill)

        obj, role, target = params['object'], params['role'], params['target']
        arm = self.arm(role)
        goals = np.asarray(goal).reshape(-1, 4, 4)
        command_log = self.folder/'native-control-commands.jsonl'

        def observe():
            return PhysicalState(
                {'object': self.ops.pose(self.robot,obj), 'tcp': self.robot.get_ee_pose(arm)},
                command_offset=native_command_offset(command_log), holding=self.holding.copy())

        def retained(state):
            pose = state.poses['object']
            expected = state.poses['tcp'] @ self.held_transforms[role]
            return (state.holding.get(role)==obj and np.linalg.norm(pose[:3,3]-expected[:3,3])<.008
                    and Rotation.from_matrix(expected[:3,:3].T@pose[:3,:3]).magnitude()<.05)

        def verified(before, after, _result):
            distances=np.linalg.norm(goals[:,:3,3]-after.poses['object'][:3,3],axis=1)
            angles=Rotation.from_matrix(goals[:,:3,:3].transpose(0,2,1)@after.poses['object'][:3,:3]).magnitude()
            return bool(retained(after) and ((distances<.008) & (angles<.05)).any())

        def via_above(state):
            # The actual two segments are both planned by RobotAPI in the unchanged scene.
            waypoint = state.poses['object'].copy()
            waypoint[2,3] += .035
            if not self.ops.move(self.robot,obj,arm,waypoint,'held_motion_clearance_approach'):
                return False
            if not retained(observe()):
                raise RecoveryFailure(FailureKind.GRASP_LOST, {'phase':'held_motion_clearance_approach'})
            return self.ops.move(self.robot,obj,arm,goals,'plan_'+target)

        candidates = [
            RecoveryCandidate('native_collision_checked_target',
                              lambda _: self.ops.move(self.robot,obj,arm,goals,'plan_'+target),
                              applicable=retained),
            RecoveryCandidate('raise_then_replan', via_above, applicable=retained),
        ]
        if getattr(self, 'validation_recovery_probe', False) and target == 'packing_0_approach':
            def rejected_candidate(_state):
                unreachable = goals[:1].copy()
                unreachable[0,0,3] += 2.5
                survivors = int(self.held_goal_mask(role,obj,unreachable).sum())
                if survivors:
                    raise RecoveryFailure(FailureKind.PRECONDITION_FAILED, {'native_ik_survivors':survivors})
                raise RecoveryFailure(FailureKind.PLANNING_REFUSED,
                                      {'reason':'native_held_goal_ik_rejected','native_ik_survivors':survivors})
            candidates.insert(0, RecoveryCandidate('validation_outside_workspace', rejected_candidate,
                                                   applicable=retained))
        try:
            run_recoverable_skill(f'move_held:{role}:{obj}:{target}', candidates, observe=observe,
                                  verify=verified, event=self.event, deadline=self.started+self.max_wall_seconds,
                                  history=self.__dict__.setdefault('held_motion_history', {}))
        except RecoveryFailure as exc:
            raise SkillFailure('motion_or_tracking_failed', {
                **exc.details, 'target':target, 'recovery_failure_kind':exc.kind.value}) from exc

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
        boxes={obj:bbox_corners(*self.ops.local_box(self.robot,obj)) for obj in (pen,cap)}
        support=self.spec.get('support')
        if support:
            table=support['instance']
            tabletop=support['height_m']
            lower,upper=support['bounds_xy_m']
            supported=bbox_corners([*lower,tabletop],[*upper,tabletop])
        else:
            tables=[name for name,item in self.robot.scene_info['layout'].items() if item.get('usd')=='table_000']
            if len(tables)!=1:raise SkillFailure('pen_support_geometry_unavailable',{'table_assets':tables})
            table=tables[0]
            supported=transformed_points(self.ops.pose(self.robot,table),bbox_corners(*self.ops.local_box(self.robot,table)))
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
        limit=3 if 'support' in self.spec else 1
        last_failure=None
        for selected in np.flatnonzero(feasible)[:limit]:
            self.geometry.cache['release_clearance:'+target]=float(clearances[selected])
            self.geometry.cache['release_yaw:'+target]=float(turns[selected])
            try:
                if turns[selected]:
                    self.restore_contact_planning()
                    staging=goals[selected].copy()
                    current=self.ops.pose(self.robot,obj)
                    lift=max(.05,float(up@(current[:3,3]-staging[:3,3])))
                    staging[:3,3]+=lift*up
                    if not self.ops.move(self.robot,obj,arm,staging,'plan_packing_turned_approach'):
                        raise SkillFailure('packing_turn_path_failed',{'target':target,'yaw_rad':turns[selected]})
                self.align_pose(p,.0015,.018,packing=True)
            except SkillFailure as exc:
                if limit==1 or self.state()['holding_by_role'].get(p['role'])!=obj:
                    raise
                last_failure=exc
                self.event('packing_release_candidate_retry',target=target,candidate=int(selected),
                           cause=exc.kind,clearance_m=clearances[selected])
            else:
                self.facts.add('target:'+p['target'])
                return
        raise last_failure

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
            if p['target']=='cap_preinsert' or (packing and 'support' in self.spec and np.linalg.norm(delta)<.03):
                command=accumulate_pose_correction(goal,actual,correction_pose)
            elif np.linalg.norm(delta)<.03:
                correction=.8*delta;correction*=min(1.,.005/max(np.linalg.norm(correction),1e-9));command[:3,3]+=correction
                command[:3,:3]=Rotation.from_rotvec(.8*dr.as_rotvec()).as_matrix()@command[:3,:3]
            with self.contact_planning(excluded):
                if p['target']=='cap_preinsert':
                    self.follow_local_pose(p,command,'cap_alignment_waypoint_motion')
                elif packing and 'support' in self.spec and np.linalg.norm(delta)<.015 and dr.magnitude()<.09:
                    self.follow_local_pose(p,command,'packing_alignment_waypoint_motion')
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
        grip=self.shape_grasp_profile(arm)
        local_alignment='transport_approach_route' in grip and grip.get('release_tcp_x_points_up')
        transferred='support' in self.spec
        correction_pose=np.eye(4)
        with self.contact_planning(excluded):
            for attempt in range(4):
                if local_alignment:goal,_,clearance=self.shape_entry(p)
                actual=self.ops.pose(self.robot,p['object']);projection=measure(actual,self.ops.pose(self.robot,box))
                delta=goal[:3,3]-actual[:3,3];dr=Rotation.from_matrix(goal[:3,:3]@actual[:3,:3].T)
                error=float(np.linalg.norm(delta));angle=float(dr.magnitude())
                self.event('precision_measurement',attempt=attempt,position_error_m=error,rotation_error_rad=angle,projection=projection)
                if error<.0015 and angle<.012 and projection['outside_hole_area_fraction']<=.001:break
                if transferred and error<.003 and angle<.02 and projection['inside_aperture_margin_m']>.00020:
                    # Another waypoint can disturb a load that already fits.
                    # Confirm the fit persists while held before keeping it.
                    self.robot.client.run_step(30)
                    settled=self.ops.pose(self.robot,p['object'])
                    after=measure(settled,self.ops.pose(self.robot,box))
                    drift=float(np.linalg.norm(settled[:3,3]-actual[:3,3]))
                    residual=float(np.linalg.norm(settled[:3,3]-goal[:3,3]))
                    residual_angle=float(Rotation.from_matrix(settled[:3,:3].T@goal[:3,:3]).magnitude())
                    passed=(drift<.001 and residual<.0035 and residual_angle<.025
                            and after['inside_aperture_margin_m']>.00020)
                    self.event('shape_alignment_settle',passed=passed,drift_m=drift,position_error_m=residual,
                               rotation_error_rad=residual_angle,projection=after)
                    if passed:break
                    actual=settled
                    delta=goal[:3,3]-actual[:3,3]
                    dr=Rotation.from_matrix(goal[:3,:3]@actual[:3,:3].T)
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
        if self.spec.get('task_plugin'):
            from task_plugins import load
            check_release = getattr(load(self.spec['task_plugin']), 'check_release', None)
            if check_release is not None:
                check_release(self,p)
        if self.holding.get(p['role'])!=p['object']:raise SkillFailure('not_held',p)
        arm=self.arm(p['role'])
        release_override={}
        separation_check=None
        if self.spec['template']=='insert_building_block':
            if 'shape_aligned' not in self.facts:raise SkillFailure('release_precondition_failed',p)
            b=self.spec['bindings']
            grip=self.shape_grasp_profile(arm)
            release_override=grip.get('release_override',{})
            if 'support' in self.spec and not release_override:
                native=json.loads((self.folder/'gripper-contract.json').read_text())['grippers'][arm]
                release_override={'method':'native_effort_unload','speed_rad_s':.20,
                                  'unload_seconds':.30,'unload_min_effort_Nm':.20,
                                  'max_effort_Nm':.45,'coupled_open_targets_rad':
                                  dict(zip(native['joint_names'],native['opened_positions'],strict=True))}
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
                if grip.get('release_tcp_x_points_up'):
                    tcp_x_up=float(self.robot.get_ee_pose(arm)[2,0])
                    drop=clearance['actual_workpiece_clearance_m']
                    limit=float(grip['maximum_release_gap_m'])
                    self.event('shape_release_wrist_roll',tcp_x_world_z=tcp_x_up,drop_m=drop,maximum_drop_m=limit)
                    if tcp_x_up<.95 or drop>limit:
                        raise SkillFailure('shape_release_outside_verified_geometry',
                                           {'tcp_x_world_z':tcp_x_up,'drop_m':drop,'maximum_drop_m':limit})
            before = self.ops.pose(self.robot,p['object'])
            self.geometry.cache['shape_before_release']=before
            if 'support' in self.spec and release_override.get('method') == 'native_effort_unload':
                def separation_check():
                    actual = self.ops.pose(self.robot, p['object'])
                    drop = float(before[2, 3] - actual[2, 3])
                    return drop >= .020, {'measured_payload_drop_m': drop, 'required_drop_m': .020}
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
        try:self.ops.release(self.robot,arm,gentle=p['mode']=='gentle',profile=profile,separation_check=separation_check)
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

    def retreat_empty_hand(self, role):
        arm = self.arm(role)
        try:
            if self.ops.retreat(self.robot, arm):
                return
        except SkillFailure as exc:
            if exc.kind != 'native_action_failed_before_rollback' or 'support' not in self.spec:
                raise
        if 'support' not in self.spec:
            raise SkillFailure('retreat_failed', {'role': role})

        start = self.robot.get_ee_pose(arm)
        snapshot = self.robot.client.command_any({'command': 'ProjectKinematicSnapshot'})
        back = np.asarray(snapshot['robot_base_position'][:2]) - start[:2, 3]
        back *= .032 / max(float(np.linalg.norm(back)), .032)
        offsets = ([0., 0., .055], [back[0], back[1], .046])
        for offset in offsets:
            target = self.robot.get_ee_pose(arm).copy()
            target[:3, 3] += offset
            try:
                self.follow_local_pose({'role': role, 'object': 'gripper'}, target, 'empty_hand_local_withdrawal')
                self.event('empty_hand_local_withdrawal_completed', role=role, target=target)
                return
            except SkillFailure as exc:
                if exc.kind not in {'native_action_failed_before_rollback', 'local_waypoint_plan_or_tracking_failed'}:
                    raise
                self.event('empty_hand_local_withdrawal_rejected', role=role, reason=exc.kind, details=exc.details)
        raise SkillFailure('retreat_failed', {'role': role})

    def retreat(self,p):
        role=p['role']
        if role in self.holding:raise SkillFailure('hand_not_empty',p)
        arm=self.arm(role)
        cap_seated=self.spec['template']=='put_the_cap_on_the_pen_new' and 'relation:cap_seated' in self.facts
        if cap_seated and role=='assistant':
            self.withdraw_seated_cap_hand(arm)
        else:
            self.retreat_empty_hand(role)
        if cap_seated:self.facts.add('cap_'+role+'_retreated')
        self.restore_contact_planning()
        if 'placement_geometry' in self.spec:
            self.facts.add('retreated')

    def wait_stable(self,p):
        a=self.ops.pose(self.robot,p['object']);self.robot.client.run_step(120);b=self.ops.pose(self.robot,p['object'])
        movement=float(np.linalg.norm(a[:3,3]-b[:3,3]));self.event('settling',object=p['object'],movement_m=movement)
        if movement>.003:raise SkillFailure('unstable',{'movement_m':movement})
        if 'placement_geometry' in self.spec:
            self.facts.add('stable')
        if self.spec['template']=='put_the_cap_on_the_pen_new' and p['object']==self.spec['bindings']['pen']:
            self.facts.add('cap_pen_stable')

    def insert_axial(self,p):
        if self.spec['template']!='put_the_cap_on_the_pen_new':raise UnsupportedRequest('No axial assembly contract',missing_skills=['insert_axial'])
        if 'cap_aligned' not in self.facts:raise SkillFailure('assembly_alignment_required',p)
        config=self.spec['assembly']
        self.robot.client.command_any({'command':'ProjectTrackObjects','paths':{obj:self.robot.get_object_prim_path(obj) for obj in [p['object'],p['reference']]}})
        depths=np.linspace(-config['preinsert_gap_m'],config['insertion_depth_m'],16)[1:]
        if 'support' in self.spec:
            # Start smaller advances at the shoulder, where the clearance drops.
            coarse=depths[depths<config['transition_start_depth_m']-.003]
            fine_steps=int(np.ceil((config['insertion_depth_m']-coarse[-1])/.0016))
            depths=np.r_[coarse,np.linspace(coarse[-1],config['insertion_depth_m'],fine_steps+1)[1:]]
        stalled=0
        with self.contact_planning([self.robot.get_object_prim_path(p['reference'])]):
            for depth in depths:
                requested_depth=float(depth)
                fine='support' in self.spec and depth>=config['transition_start_depth_m']-.003
                before=None
                if fine:
                    before=self.cap_mouth_measurement(self.ops.pose(self.robot,p['object']))['depth_m']
                    if before>=requested_depth-.00015:
                        continue
                    requested_depth=min(requested_depth,before+.0022)
                self.trim_cap_for_insertion(p,requested_depth)
                goal,_=self.geometry.target(p['target'],p['object']);t=goal.copy()
                t[:3,3]+=goal[:3,0]*(requested_depth+config['preinsert_gap_m'])
                self.follow_cap_axial_step(p,t,requested_depth)
                try:self.verify_relation({'object':p['object'],'relation':'cap_seated'})
                except SkillFailure:pass
                else:self.facts.add('assembled');return
                if fine:
                    measured=self.cap_mouth_measurement(self.ops.pose(self.robot,p['object']))['depth_m']
                    self.event('cap_fine_advance_measured',requested_depth_m=requested_depth,before_depth_m=before,
                               after_depth_m=measured)
                    stalled=stalled+1 if measured-before<.00015 else 0
                    if stalled>=3:
                        raise SkillFailure('axial_insertion_stalled',{'depth_m':measured,'attempts':stalled})
        self.verify_relation({'object':p['object'],'relation':'cap_seated'})
        self.facts.add('assembled')

    def follow_cap_axial_step(self, params, target, depth):
        from skill_recovery import FailureKind, PhysicalState, classify_failure, materially_changed
        from skill_recovery import native_command_offset, physical_change

        log_root = os.environ.get('GENIE_EPISODE_EVIDENCE')
        command_path = Path(log_root) / 'native-control-commands.jsonl' if log_root else None

        def observe():
            if not command_path or native_command_offset(command_path) is None:
                return None
            poses = {
                'payload': self.ops.pose(self.robot, params['object']),
                'reference': self.ops.pose(self.robot, params['reference']),
                'hand': self.robot.get_ee_pose(self.arm(params['role'])),
            }
            return PhysicalState(poses, command_offset=native_command_offset(command_path))

        def precommand_refusal(failure, before, phase):
            if failure.kind == 'native_action_failed_before_rollback':
                return True  # The established measured contact recovery handles this native exception.
            if before is None or failure.kind != 'local_waypoint_plan_or_tracking_failed':
                return False
            after = observe()
            if after is None:
                return False
            change = physical_change(before, after)
            kind = classify_failure(failure.kind, failure.details,
                                    world_changed=materially_changed(change, .001, .02, .01),
                                    commands_written=change['native_command_bytes'])
            self.event('cap_axial_motion_classified', phase=phase, failure_kind=kind.value,
                       initial=before.evidence(), observed=after.evidence(), change=change)
            return kind == FailureKind.PLANNING_REFUSED

        phase = 'cap_axial_waypoint_motion'
        before = observe()
        try:
            self.follow_local_pose(params, target, phase)
            return
        except SkillFailure as failure:
            contact_retry = 'support' in self.spec and depth >= 0.
            if (depth >= -.003 and 'support' not in self.spec) or not precommand_refusal(failure, before, phase):
                raise
            candidate = self.cap_axis_only_candidate(params, depth, contact_retry=contact_retry)
            if candidate is None:
                raise

        alternate, evidence = candidate
        self.event('cap_axis_only_approach_candidate', **evidence)
        phase = 'cap_axis_only_approach_waypoints'
        before = observe()
        try:
            self.follow_local_pose(params, alternate, phase)
        except SkillFailure as failure:
            if not precommand_refusal(failure, before, phase):
                raise
            if not (self.advance_cap_before_contact(params, depth) or self.advance_pen_at_contact(params, depth)):
                raise

    def cap_axis_only_candidate(self, params, depth, *, contact_retry=False):
        """Project a short advance along the live pen axis with the measured grasp."""
        cap = self.ops.pose(self.robot, params['object'])
        mouth = self.cap_mouth_measurement(cap)
        advance = depth - mouth['depth_m']
        clearance = self.cap_radial_clearance(mouth, depth)
        minimum_margin = .00025 if contact_retry else .001
        minimum_advance = .00025 if contact_retry else .0005
        limit = .0012 if contact_retry else .008
        if not contact_retry:
            clearance -= abs(depth) * mouth['axis_sine']
        # Use the existing alignment tolerance while the entire advance stays clear of the pen tip.
        clear_approach = not contact_retry and 'support' in self.spec and mouth['depth_m'] < depth < -.003
        rotation_limit = .02 if clear_approach else .012
        if not minimum_advance < advance < .008 or mouth['rotation_rad'] >= rotation_limit or clearance <= minimum_margin:
            return None
        arm = self.arm(params['role'])
        actual_grasp = self.robot.get_ee_pose(arm) @ self.held_transforms[params['role']]
        slip_m = float(np.linalg.norm(actual_grasp[:3, 3] - cap[:3, 3]))
        slip_rad = float(Rotation.from_matrix(actual_grasp[:3, :3].T @ cap[:3, :3]).magnitude())
        if slip_m > .008 or slip_rad > .05:
            return None
        pen_axis = self.ops.pose(self.robot, params['reference'])[:3, 0]
        cap[:3, 3] += min(advance, limit) * pen_axis
        return cap, {'requested_depth_m': depth, 'before_depth_m': mouth['depth_m'], 'advance_m': advance,
                     'radial_margin_m': clearance, 'grasp_slip_m': slip_m, 'grasp_slip_rad': slip_rad}

    def advance_cap_before_contact(self, params, depth):
        margin = self.adapter.config.get('held_transport_fixture_margin_m')
        if 'support' not in self.spec or margin is None or depth >= -.003:
            return False
        cap, pen = params['object'], params['reference']
        for part in (cap, pen):
            pen_pose = self.stable_pen_for_alignment()
            candidate = self.cap_axis_only_candidate(params, depth)
            if candidate is None:
                return False
            goal, evidence = candidate
            moving, stationary = params, pen
            if part == pen:
                if self.holding.get('operator') != pen:
                    return False
                # Translating the held pen in the opposite direction produces
                # the same relative advance while the cap remains stationary.
                moving, stationary = {'object': pen, 'role': 'operator'}, cap
                goal = pen_pose.copy()
                goal[:3, 3] -= evidence['advance_m'] * pen_pose[:3, 0]
            try:
                moved = self.move_held_clear_of_fixture(
                    moving, goal, 'cap_axial_guarded_advance', stationary, margin, 2.0)
            except SkillFailure as exc:
                if exc.kind != 'native_action_failed_before_rollback':
                    raise
                moved = False
            if not moved:
                self.event('cap_axial_guarded_advance_rejected', moving_part=part, **evidence)
                continue
            measured = self.cap_mouth_measurement(self.ops.pose(self.robot, cap))
            progress = measured['depth_m'] - evidence['before_depth_m']
            passed = (measured['depth_m'] < -.002 and abs(measured['depth_m'] - depth) <= .002
                      and progress >= min(.001, .5 * evidence['advance_m'])
                      and measured['rotation_rad'] < .02 and self.cap_radial_clearance(measured, depth) > .00015)
            self.event('cap_axial_guarded_advance_result', moving_part=part, **evidence,
                       after_depth_m=measured['depth_m'], passed=passed)
            if passed or measured['depth_m'] >= -.002:
                return passed
        return False

    def advance_pen_at_contact(self, params, depth):
        """Try the same measured insertion with a short motion of the pen hand."""
        pen, cap = params['reference'], params['object']
        if ('support' not in self.spec or depth < 0. or self.holding.get('operator') != pen
                or self.holding.get(params['role']) != cap):
            return False
        pen_pose = self.stable_pen_for_alignment()
        candidate = self.cap_axis_only_candidate(params, depth, contact_retry=True)
        if candidate is None:
            return False
        _, evidence = candidate
        if evidence['before_depth_m'] < -.003:
            return False
        goal = pen_pose.copy()
        goal[:3, 3] -= min(evidence['advance_m'], .0012) * pen_pose[:3, 0]
        self.follow_local_pose({'object': pen, 'role': 'operator'}, goal, 'pen_reciprocal_contact_waypoints')
        self.stable_pen_for_alignment()
        measured = self.cap_mouth_measurement(self.ops.pose(self.robot, cap))
        advance = measured['depth_m'] - evidence['before_depth_m']
        passed = (advance >= .00015 and measured['rotation_rad'] < .02
                  and self.cap_radial_clearance(measured, measured['depth_m']) > .00015)
        self.event('pen_reciprocal_contact_measurement', **evidence,
                   after_depth_m=measured['depth_m'], achieved_advance_m=advance, passed=passed)
        return passed

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
        if self.spec.get('task_plugin'):
            from task_plugins import load
            ok=load(self.spec['task_plugin']).verify_relation(self,p)
        elif relation.startswith('packing_stage_'):ok=self.ops.phone_rules(self.robot,2*int(relation.rsplit('_',1)[1])+1)
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

    def sync_planner_payload(self, arm, objects):
        expected = [self.robot.get_object_prim_path(obj) for obj in objects]
        attached = self.robot.attached_obj_id
        self.ops.detach_planner_payload(self.robot,arm)
        snapshot=self.robot.client.attach_obj(prim_paths=expected,is_right=arm=='right')
        measured=self.robot.client.command_any({'command':'ProjectSnapshot'})
        link='attached_object' if arm=='right' else 'left_attached_object'
        actual=measured.get('planner_attachments',{}).get(link,[])
        if not all(any(path==root or path.startswith(root.rstrip('/')+'/') for path in actual)
                   for root in expected):
            raise SkillFailure('planner_attachment_failed',{'expected_roots':expected,'planner_colliders':actual})
        self.robot.attached_obj_id = attached
        return snapshot, measured

    def sync_assembly_proxy(self,p):
        arm = self.arm(p['role'])
        snapshot, measured = self.sync_planner_payload(arm, [p['object'], p['reference']])
        self.facts.add('assembly_proxy_synced')
        self.event('assembly_planner_proxy',result=str(snapshot),physical_weld_created=False,snapshot=measured)

    def verify_task(self,p):
        if self.holding:raise SkillFailure('objects_still_held',self.holding)
        name=self.spec['template'];b=self.spec['bindings'];ok=False
        if name=='phone_packaging':
            indices=range(1, 2*len(self.spec['manipulated_objects']), 2)
            preserved=all(
                np.linalg.norm(self.ops.pose(self.robot,obj)[:3,3]-self.geometry.initial[obj][:3,3])<.004
                for obj in self.spec.get('preserve_objects', []))
            ok=preserved and all(self.ops.phone_rules(self.robot,i) for i in indices)
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
        elif self.spec.get('task_plugin'):
            from task_plugins import load
            ok=load(self.spec['task_plugin']).verify_task(self)
        self.event('independent_task_verdict',passed=bool(ok),task_id=self.spec['task_id'])
        if not ok:raise SkillFailure('physical_task_incomplete',{'task_id':self.spec['task_id']})
        return True

    def execute(self,plan):
        validate_plan(plan,self.spec,self.adapter,held=self.holding,achieved=self.facts)
        for step in plan['steps']:
            if time.monotonic()-self.started>self.max_wall_seconds:raise SkillFailure('wall_time_budget',{})
            before=self.state()
            self.event('step_started',step=step,state=before,sim_time=before.get('sim_time'))
            try:getattr(self,step['skill'])(step['parameters'])
            except Exception as exc:
                missing_observation=(isinstance(exc,SkillFailure) and
                                     exc.details.get('recovery_failure_kind')=='observation_missing')
                failed_state=None if simulator_unavailable(exc) or missing_observation else self.state()
                self.event('step_failed',step=step,error_type=exc.kind if isinstance(exc,SkillFailure) else type(exc).__name__,error=str(exc),
                    details=exc.details if isinstance(exc,SkillFailure) else str(exc),state=failed_state,
                    sim_time=failed_state.get('sim_time') if failed_state else None)
                raise
            self.executed.append(step['id'])
            after=self.state()
            self.event('step_succeeded',step_id=step['id'],state=after,sim_time=after.get('sim_time'))
        return True

def run_structured(robot,ops,request,*,before_plan=None):
    """Execute a validated plan from live state, optionally after a trusted skill."""
    from structured_planner import StructuredPlanner,PlannerBudgetExceeded
    folder=Path(os.environ['GENIE_EPISODE_EVIDENCE']);adapter=load_robot(request.get('robot_id','g2crsb90d'))
    spec=task_spec(robot.scene_info,adapter.robot_id);adapter.require(spec['required_capabilities'])
    if 'support' in spec:
        # The raw record includes the stationary initialization and every correction.
        # Measure the actual physics state rather than trusting the authored poses.
        names=sorted(set(spec['manipulated_objects']) | set(spec['bindings'].values()))
        robot.client.run_step(30)
        before={name:ops.pose(robot,name) for name in names}
        robot.client.run_step(30)
        settle={}
        table=spec['support']
        lo,hi=np.asarray(table['bounds_xy_m'])
        for name in names:
            current=ops.pose(robot,name)
            points=transformed_points(current,bbox_corners(*ops.local_box(robot,name)))
            settle[name]={'drift_m':float(np.linalg.norm(current[:3,3]-before[name][:3,3])),
                          'height_gap_m':float(points[:,2].min()-table['height_m']),
                          'inside_support':bool((points[:,:2]>=lo-.002).all() and (points[:,:2]<=hi+.002).all())}
        gap_limits=(-.005,.003) if spec['template']=='put_the_cap_on_the_pen_new' else (-.025,.022)
        passed=all(row['inside_support'] and row['drift_m']<=.006 and gap_limits[0]<=row['height_gap_m']<=gap_limits[1]
                   for row in settle.values())
        receipt={'passed':passed,'settle':settle,'height_gap_limits_m':gap_limits}
        static_support = table.get('physics') == 'native_collision_static_triangle_mesh'
        if spec['template']=='put_the_cap_on_the_pen_new' or static_support:
            probe = robot.client.command_any({'command':'ProjectProbeSupport',
                'path':table['prim'],'height_m':table['height_m'],'bounds_xy_m':table['bounds_xy_m']})
            receipt['support_probe'] = probe
            if static_support:
                hits = probe.get('hits', []) if isinstance(probe, dict) else []
                probe_passed = len(hits) == 9 and all(
                    row.get('requested_support') and abs(float(row['position'][2])-table['height_m']) < .002
                    for row in hits)
                receipt['static_collision_plane_matches_mesh'] = probe_passed
                passed = passed and probe_passed
                receipt['passed'] = passed
        (folder/'scene-physics-preflight.json').write_text(json.dumps(receipt,indent=2))
        if not passed:return False
    executor=SkillExecutor(robot,adapter,spec,ops,folder)
    executor.validation_recovery_probe = bool(request.get('validation_recovery_probe'))
    (folder/'task-spec.json').write_text(json.dumps(spec,indent=2))
    contract=robot.client.command_any({'command':'ProjectGripperConfiguration'})
    if not isinstance(contract,dict) or contract.get('success') is not True:raise RuntimeError('Native gripper contract unavailable')
    (folder/'gripper-contract.json').write_text(json.dumps(contract,indent=2))
    if before_plan is not None and before_plan(executor) is not True:
        return False
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
            if isinstance(exc,SkillFailure) and exc.details.get('recovery_failure_kind')=='observation_missing':
                executor.event('infrastructure_terminated',reason='physical_observation_unavailable',details=exc.details)
                return False
            executor.restore_contact_planning()
            if isinstance(exc,SkillFailure) and exc.kind in {
                    'failed_passage_evidence_after_containment', 'shape_transport_grasp_slipped',
                    'cap_preinsert_recovery_exhausted', 'payload_not_cleared_during_release',
                    'probed_grasp_lift_exhausted'}:
                executor.event('physical_verification_terminated',reason=exc.kind,details=exc.details);return False
            if isinstance(exc,SkillFailure) and exc.kind=='wall_time_budget':
                executor.event('budget_terminated',reason='execution_wall_time',calls=planner.calls,tokens=planner.tokens);return False
            failure={'type':exc.kind if isinstance(exc,SkillFailure) else type(exc).__name__,
                'details':exc.details if isinstance(exc,SkillFailure) else str(exc)}
            executor.event('bounded_recovery_requested',failure=failure,revision=revision)
    executor.event('budget_terminated',reason='maximum_three_llm_plans',calls=planner.calls,tokens=planner.tokens)
    return False
