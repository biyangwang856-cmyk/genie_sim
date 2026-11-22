"""Project-local read-only diagnostics and arm-scoped planner attachment cleanup."""
import json,os
from itertools import product
from pathlib import Path

_controller=None

def record_render_sample(world):
    """Observe live and rendered wrists after rendering; never set a body pose."""
    if _controller is None or not world.is_playing():return
    now=float(world.current_time)
    if now-getattr(_controller,'_project_render_sample_time',-1)<.1-1e-5:return
    import numpy as np
    from pxr import UsdGeom
    import omni.usd
    from sim_backend import physics,to_numpy
    from scipy.spatial.transform import Rotation
    cache=UsdGeom.XformCache();stage=omni.usd.get_context().get_stage();links={}
    for side in ['l','r']:
        path='/G2/arm_'+side+'_end_link'
        p,q=_controller._get_object_pose(path)
        actual=np.eye(4);actual[:3,3]=to_numpy(p)
        actual[:3,:3]=Rotation.from_quat(to_numpy(q)[[1,2,3,0]]).as_matrix()
        usd=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T
        links[side]={'physics':actual.tolist(),'usd':usd.tolist()}
    row={'sim_time':now,'links':links,'joint_positions':to_numpy(physics.get_joint_positions(_controller.ui_builder.articulation,as_torch=False)).tolist()}
    if not hasattr(_controller,'_project_render_samples'):
        _controller._project_render_samples=(Path(os.environ['GENIE_EPISODE_EVIDENCE'])/'render-motion-samples.jsonl').open('a',buffering=1)
    _controller._project_render_samples.write(json.dumps(row)+'\n')
    _controller._project_render_sample_time=now

def install_direct_camera_recording(record_handler):
    """Publish the renderer's exact RGB sample as JPEG without a relay queue."""
    import inspect
    import cv2
    import numpy as np
    from sensor_msgs.msg import CompressedImage
    from common.base_utils.ros_nodes.sim_ros_node import SimPubRosNode,QOS_PROFILE_VOLATILE
    from server.ros_publisher import camera

    if 'GENIE_DIRECT_COMPRESSED_CAMERA' not in inspect.getsource(record_handler):
        raise RuntimeError('The native recorder needs the project direct-camera relay guard')

    class DirectCameraPublisher(SimPubRosNode):
        project_direct_camera = True

        def __init__(self, topic, get_msg_callback, frame_id, robot_name='G1_120s',
                     node_name='image_pub_node', step_size=1):
            super().__init__(CompressedImage, topic+'_compressed', get_msg_callback,
                             frame_id, robot_name, node_name, step_size)

        def create_publisher(self, message_type, topic, qos_profile, **kwargs):
            return super().create_publisher(message_type, topic, QOS_PROFILE_VOLATILE, **kwargs)

        def handle_msg(self, rgb):
            bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            ok, image = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError('Could not encode the live rendered frame')
            msg = CompressedImage()
            msg.header = self._header
            msg.format = 'rgb8; jpeg compressed bgr8'
            msg.data = image.reshape(-1)
            return msg

    camera.ImagePubRosNode = DirectCameraPublisher
    os.environ['GENIE_DIRECT_COMPRESSED_CAMERA'] = '1'
    return {'mode':'single_hop_jpeg','quality':95,'qos_history':'keep_last',
            'qos_reliability':'reliable','qos_depth':QOS_PROFILE_VOLATILE.depth,
            'controller_source':inspect.getsourcefile(record_handler)}


def wait_for_recording_subscribers(nodes, timeout_s=15.):
    import time

    sensors = {node._topic:node._publisher for node in nodes
               if getattr(node,'project_direct_camera',False) or getattr(node,'_topic',None) in ('/joint_states','/tf')}
    if len(sensors) != 5:
        raise RuntimeError('Expected three cameras, joint states and TF for recorded collection')
    started = time.monotonic()
    while True:
        peers = {topic:publisher.get_subscription_count() for topic,publisher in sensors.items()}
        elapsed = time.monotonic()-started
        if all(peers.values()) or elapsed >= timeout_s:
            return {'passed':all(peers.values()),'wait_s':elapsed,'matched_subscribers':peers}
        time.sleep(.05)


def install():
    from server.motion_generator.motion_gen_reacher import CuroboMotion
    from server.command_controller import CommandController
    from pxr import Usd,UsdGeom,UsdPhysics
    import omni.usd
    from isaacsim.core.api.controllers.articulation_controller import ArticulationController
    from isaacsim.core.api import World
    from sim_backend import to_numpy
    import numpy as np
    from scipy.spatial.transform import Rotation
    original_extract=CuroboMotion._extract_cached_obstacles
    def extract_physical_world(self,*args,**kwargs):
        result=original_extract(self,*args,**kwargs)
        # The phone scene contains a 100 m photographic HDR enclosure. It has
        # no PhysX CollisionAPI, but the USD mesh parser registers it as a solid
        # enclosing every robot pose. Exclude only this proven render backdrop;
        # keep the real table, task geometry and all enabled physical colliders.
        stage=omni.usd.get_context().get_stage();excluded=[]
        for path in list(CuroboMotion.cached_obstacle_info):
            if not path.startswith('/World/hdr_backdrop/'):continue
            prim=stage.GetPrimAtPath(path)
            if not prim:raise RuntimeError('Missing HDR prim during collision audit')
            enabled=prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
            if enabled:continue
            excluded.append({'path':path,'reason':'render_only_hdr_backdrop_without_enabled_physical_collision'})
            CuroboMotion.cached_obstacle_info.pop(path)
        evidence=Path(os.environ['GENIE_EPISODE_EVIDENCE']);evidence.mkdir(parents=True,exist_ok=True)
        policy=evidence/'planner-geometry-policy.json'
        previous=json.loads(policy.read_text()).get('excluded_render_only',[]) if policy.exists() else []
        excluded=list({r['path']:r for r in previous+excluded}.values())
        policy.write_text(json.dumps({'excluded_render_only':excluded,
            'retained_obstacle_paths':sorted(CuroboMotion.cached_obstacle_info),'physical_geometry_modified':False},indent=2))
        return result
    CuroboMotion._extract_cached_obstacles=extract_physical_world
    original_motion_init=CuroboMotion.__init__
    def motion_init(self,*args,**kwargs):
        original_motion_init(self,*args,**kwargs)
        self.max_attempts=int(os.environ.get('GENIE_PLANNER_MAX_ATTEMPTS','8'))
        self.plan_config.max_attempts=self.max_attempts
        self._reach_settle_max_delay_s=2.0
    CuroboMotion.__init__=motion_init
    original_slowdown=CommandController._resolve_slowdown_factor
    def bounded_speed(self,user_factor):
        # Native 1.34 s paths requested multi-radian wrist changes that the
        # physical drives could not follow. Resample commands without setting q.
        return original_slowdown(self,user_factor)*4.0
    CommandController._resolve_slowdown_factor=bounded_speed
    original_apply=ArticulationController.apply_action
    journal_path=Path(os.environ['GENIE_EPISODE_EVIDENCE'])/'native-control-commands.jsonl'
    journal_path.parent.mkdir(parents=True,exist_ok=True)
    journal=journal_path.open('a',buffering=1)
    def apply(self,*args,**kwargs):
        action=args[0] if args else kwargs.get('control_actions',kwargs.get('action'))
        view=getattr(self,'_articulation_view',None)
        measured=to_numpy(view.get_joint_positions())[0].tolist()
        result=original_apply(self,*args,**kwargs)
        row={'sim_time':float(World.instance().current_time)}
        row['measured_joint_positions_before']=measured
        row['joint_names']=list(getattr(view,'dof_names',[]))
        for field in ('joint_indices','joint_positions','joint_velocities','joint_efforts'):
            value=getattr(action,field,None)
            row[field]=None if value is None else to_numpy(value).tolist()
        applied=self.get_applied_action()
        for field in ('joint_positions','joint_velocities','joint_efforts'):
            value=getattr(applied,field,None)
            row['applied_'+field]=None if value is None else to_numpy(value).tolist()
        kp,kd=view.get_gains()
        row['drive_stiffness']=to_numpy(kp).reshape(-1).tolist()
        row['drive_damping']=to_numpy(kd).reshape(-1).tolist()
        row['drive_max_efforts']=to_numpy(view.get_max_efforts()).reshape(-1).tolist()
        journal.write(json.dumps(row)+'\n')
        return result
    ArticulationController.apply_action=apply
    original_callback=CommandController.callback_physics
    def track_objects(self,*args,**kwargs):
        paths=getattr(self,'_project_track_paths',{})
        if paths:
            poses={}
            for key,path in paths.items():
                p,q=self._get_object_pose(path);t=np.eye(4)
                t[:3,3]=to_numpy(p);t[:3,:3]=Rotation.from_quat(to_numpy(q)[[1,2,3,0]]).as_matrix();poses[key]=t.tolist()
            self._project_track_file.write(json.dumps({'sim_time':float(self.ui_builder.my_world.current_time),'poses':poses})+'\n')
        return original_callback(self,*args,**kwargs)
    CommandController.callback_physics=track_objects
    original_attach=CuroboMotion.attach_obj
    original_add=CuroboMotion.add_obstacle_from_prim_path
    def add_obstacle(self,prim_path,usd_path):
        if not prim_path.startswith(('/World/background/','/World/Background/')):
            return original_add(self,prim_path,usd_path)
        saved=self.ignore_substring_list
        self.ignore_substring_list=[x for x in saved if x not in ['/World/background','/World/Background']]
        try:return original_add(self,prim_path,usd_path)
        finally:self.ignore_substring_list=saved
    CuroboMotion.add_obstacle_from_prim_path=add_obstacle
    original_obstacles=CuroboMotion.set_obstacles
    original_pose_update=CuroboMotion._update_obstacle_poses_fast
    def shared_mapping(self):return getattr(self.planner,'_project_attachments',{})
    def update_obstacle_poses(self):
        scene,updated=original_pose_update(self)
        return scene,updated or getattr(self,'_project_force_world_refresh',False)
    CuroboMotion._update_obstacle_poses_fast=update_obstacle_poses
    def obstacles(self, *, force=False):
        mapping=shared_mapping(self)
        self._project_attachments=mapping
        self.attached_objects=list(dict.fromkeys(p for paths in mapping.values() for p in paths))
        if not force:return original_obstacles(self)
        self._project_force_world_refresh=True
        try:return original_obstacles(self)
        finally:self._project_force_world_refresh=False
    CuroboMotion.set_obstacles=obstacles
    def detach(self):
        selected=getattr(self,'_project_detach_link',None)
        mapping=shared_mapping(self).copy()
        links=[selected] if selected else ['attached_object','left_attached_object']
        released={link:mapping[link] for link in links if link in mapping}
        if not released:return []
        restored=sorted({path for paths in released.values() for path in paths})
        remaining={link:paths for link,paths in mapping.items() if link not in released}
        # cuRobo's manager enables the exact world obstacles during detach.
        # Held geometry was omitted from the last world rebuild, so restore its
        # real current pose and collider names before the native manager looks them up.
        self.planner._project_attachments=remaining
        try:
            for path in restored:
                if path not in CuroboMotion.cached_obstacle_info:self.add_obstacle_from_prim_path(path,None)
            self.set_obstacles(force=True)
            world=set(self.planner.scene_collision_checker.get_obstacle_names(env_idx=0) or [])
            if not set(restored)<=world:raise RuntimeError('Missing native obstacles for planner detach: '+str(sorted(set(restored)-world)))
            for link,paths in released.items():
                for manager in self._attachment_managers():
                    manager.detach(link_name=link,enable_obstacle_names=paths)
        except Exception:
            self.planner._project_attachments=mapping
            self.set_obstacles(force=True)
            raise
        with (journal_path.parent/'planner-detachment-events.jsonl').open('a') as f:
            f.write(json.dumps({'removed_links':sorted(released),'restored_world_obstacles':restored,
                'remaining_attachments':remaining,'success':True})+'\n')
        return restored
    def attach(self,prim_paths,link_name='attached_object',ee_position=[0,0,0],ee_rotation=[1,0,0,0]):
        # Contact planning may exclude this mesh. Reintroduce its geometry for
        # the attachment manager to fit; the manager then disables its world
        # slot and carries it with the arm. No physical attachment is created.
        for path in prim_paths:
            if path not in CuroboMotion.cached_obstacle_info:self.add_obstacle_from_prim_path(path,None)
        self.set_obstacles()
        self._project_detach_link=link_name
        try:success=original_attach(self,prim_paths,link_name,ee_position,ee_rotation)
        finally:self._project_detach_link=None
        # The native attach only appends names whose meshes are in its world.
        # Unregistered render meshes must stay outside the active link mapping.
        attached_paths=[path for path in prim_paths if success and path in self.attached_objects]
        success=bool(success and attached_paths)
        if success:
            mapping=shared_mapping(self)
            mapping[link_name]=attached_paths;self._project_attachments=mapping
            self.planner._project_attachments=mapping
            self.attached_objects=list(dict.fromkeys(p for paths in mapping.values() for p in paths))
        with (journal_path.parent/'planner-attachment-events.jsonl').open('a') as f:
            f.write(json.dumps({'link':link_name,'paths':attached_paths,'requested_paths':list(prim_paths),'success':success})+'\n')
        return success
    CuroboMotion.detach_obj=detach
    CuroboMotion.attach_obj=attach
    old_command=CommandController.handle_command_any
    def command(self):
        global _controller
        _controller=self
        cfg=self.data['command_config'];name=cfg.get('command')
        if name=='ProjectGripperConfiguration':
            self._init_grippers();contracts={}
            for side,gripper in [('left',self.gripper_L),('right',self.gripper_R)]:
                contracts[side]={'joint_names':list(gripper._joint_prim_names),
                    'opened_positions':np.asarray(gripper.joint_opened_positions).tolist(),
                    'closed_positions':np.asarray(gripper.joint_closed_positions).tolist(),
                    'actuator':gripper._joint_control_prim.rsplit('/',1)[-1],
                    'type':gripper.gripper_type}
            self.data_to_send={'success':True,'grippers':contracts};return
        if name == 'ProjectGripUnload':
            from isaacsim.core.utils.types import ArticulationAction
            from sim_backend import physics

            arm = cfg['arm']
            fraction = float(cfg['effort_fraction'])
            if arm not in ('left', 'right') or not 0. < fraction <= 1.:
                raise ValueError('Invalid grip unloading command')
            robot = self._init_grippers()
            view = robot._articulation_view
            gripper = self.gripper_R if arm == 'right' else self.gripper_L
            if fraction == 1.:
                actuator = gripper._joint_control_prim.rsplit('/', 1)[-1]
                names = physics.get_dof_names(robot)
                index = names.index(actuator)
                finger = gripper._joint_prim_names.index(actuator)
                positions = to_numpy(physics.get_joint_positions(robot, as_torch=False)).reshape(-1)
                gains = [float(to_numpy(g).reshape(-1)[index]) for g in view.get_gains()]
                effort = float(to_numpy(view.get_max_efforts()).reshape(-1)[index])
                velocity = float(gripper._joint_closed_velocities[finger])
                gap = float(gripper.joint_opened_positions[finger] - positions[index])
                if gains[0] != 0. or gains[1] <= 0. or not np.isfinite(effort) or effort <= 0. or not velocity * gap < 0.:
                    raise ValueError('Unload requires an active native closing velocity drive')
                self._project_unload = {'arm': arm, 'index': index, 'actuator': actuator,
                                        'effort': effort, 'velocity': velocity, 'last_fraction': 1.}
            unload = self._project_unload
            if unload['arm'] != arm or fraction > unload['last_fraction']:
                raise ValueError('Unload must continue on the same actuator and decrease effort')
            index = unload['index']
            indices = np.asarray([index])
            target_effort = unload['effort'] * fraction
            if fraction < 1.:
                physics.set_max_efforts(robot, np.asarray([target_effort]), joint_indices=indices)
                position = to_numpy(physics.get_joint_positions(robot, as_torch=False)).reshape(-1)[index]
                physics.apply_action(robot, ArticulationAction(
                    joint_positions=np.asarray([position]),
                    joint_velocities=np.asarray([unload['velocity']]), joint_indices=indices))
            unload['last_fraction'] = fraction
            actual_effort = float(to_numpy(view.get_max_efforts()).reshape(-1)[index])
            self.data_to_send = {'success': bool(np.isclose(actual_effort, target_effort, atol=.0001)),
                                 'sim_time_s': float(World.instance().current_time),
                                 'actuator': unload['actuator'], 'closing_velocity_rad_s': unload['velocity'],
                                 'max_effort_Nm': actual_effort}
            return
        if name=='ProjectPrepareFingerRelease':
            from sim_backend import physics
            self.robot=self._init_grippers();side=cfg['arm']
            if side not in ['left','right']:raise ValueError('Invalid arm')
            gripper=self.gripper_R if side=='right' else self.gripper_L
            ids=np.asarray(gripper._joint_dof_indicies[:2],dtype=int)
            current=to_numpy(physics.get_joint_positions(self.robot,as_torch=False)).reshape(-1)[ids]
            lo,hi=physics.get_dof_limits(self.robot,as_torch=False)
            target=np.clip(np.asarray(gripper.joint_opened_positions[:2],dtype=float),to_numpy(lo)[ids],to_numpy(hi)[ids])
            kp,kd=self.robot._articulation_view.get_gains()
            original={'stiffness':to_numpy(kp).reshape(-1)[ids].tolist(),'damping':to_numpy(kd).reshape(-1)[ids].tolist()}
            hold_command=None
            if side=='right':
                from isaacsim.core.utils.types import ArticulationAction
                velocity=np.zeros_like(current)
                # Clear the old close targets while the existing gains are in
                # effect. The next physics step uses both new position drives.
                physics.apply_action(self.robot,ArticulationAction(
                    joint_positions=current.copy(),joint_velocities=velocity,joint_indices=ids))
                hold_command={'position_rad':current.tolist(),'velocity_rad_s':velocity.tolist(),
                              'joint_indices':ids.tolist()}
            physics.set_gains(self.robot,kps=np.full(2,100.),kds=np.full(2,5.),joint_indices=ids)
            physics.set_max_efforts(self.robot,np.full(2,5.),joint_indices=ids)
            self._project_finger_release={'arm':side,'indices':ids,'initial':current.copy(),'target':target,'fraction':0.}
            self.data_to_send={'success':True,'arm':side,'initial_rad':current.tolist(),'target_rad':target.tolist(),
                'original_gains':original,'hold_command_before_gain_switch':hold_command,
                'release_gains':{'stiffness':100.,'damping':5.,'max_effort_Nm':5.}}
            return
        if name=='ProjectFingerReleaseTarget':
            from sim_backend import physics
            from isaacsim.core.utils.types import ArticulationAction
            session=self._project_finger_release;fraction=float(cfg['fraction'])
            if not session['fraction']<=fraction<=1.:raise ValueError('Opening must advance monotonically')
            target=session['initial']+fraction*(session['target']-session['initial'])
            physics.apply_action(self.robot,ArticulationAction(joint_positions=target,joint_velocities=np.zeros(2),joint_indices=session['indices']))
            session['fraction']=fraction
            current=to_numpy(physics.get_joint_positions(self.robot,as_torch=False)).reshape(-1)[session['indices']]
            self.data_to_send={'success':True,'fraction':fraction,'actual_rad':current.tolist(),'target_rad':target.tolist()}
            return
        if name=='ProjectTrackObjects':
            paths=cfg.get('paths',{})
            if any(not p.startswith('/World/Objects/') for p in paths.values()):raise ValueError('Track outside task objects')
            self._project_track_paths=paths
            if not hasattr(self,'_project_track_file'):
                self._project_track_file=(journal_path.parent/'object-physics-120hz.jsonl').open('a',buffering=1)
            self.data_to_send={'success':True,'tracked':list(paths)};return
        if name=='ProjectGripperCollisionBounds':
            side={'left':'l','right':'r'}[cfg['arm']]
            stage=omni.usd.get_context().get_stage()
            xforms=UsdGeom.XformCache()
            bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy])
            shapes=[]
            for link in stage.GetPrimAtPath('/G2').GetChildren():
                if not link.GetName().startswith('gripper_'+side+'_'):continue
                colliders=[prim for prim in Usd.PrimRange(link,Usd.TraverseInstanceProxies())
                           if prim.IsA(UsdGeom.Boundable) and prim.HasAPI(UsdPhysics.CollisionAPI)
                           and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]
                if not colliders:continue
                authored_link=np.asarray(xforms.GetLocalToWorldTransform(link)).T
                p,q=self._get_object_pose(str(link.GetPath()))
                live_link=np.eye(4);live_link[:3,3]=to_numpy(p)
                live_link[:3,:3]=Rotation.from_quat(to_numpy(q)[[1,2,3,0]]).as_matrix()
                for prim in colliders:
                    extents=bounds.ComputeUntransformedBound(prim).ComputeAlignedRange()
                    if extents.IsEmpty():raise RuntimeError('Enabled gripper collider has no bounds: '+str(prim.GetPath()))
                    mount=np.linalg.inv(authored_link)@np.asarray(xforms.GetLocalToWorldTransform(prim)).T
                    world=live_link@mount
                    points=np.asarray(list(product(*zip(extents.GetMin(),extents.GetMax()))))
                    corners=points@world[:3,:3].T+world[:3,3]
                    shapes.append({'path':str(prim.GetPath()),'corners_world_m':corners.tolist()})
            self.data_to_send={'success':bool(shapes),'shapes':shapes,'sim_time':float(self.ui_builder.my_world.current_time)};return
        if name=='ProjectKinematicSnapshot':
            from sim_backend import physics
            names=physics.get_dof_names(self.ui_builder.articulation)
            joints=to_numpy(physics.get_joint_positions(self.ui_builder.articulation,as_torch=False)).reshape(-1)
            lo,hi=physics.get_dof_limits(self.ui_builder.articulation,as_torch=False)
            limits=np.stack([to_numpy(lo),to_numpy(hi)],axis=-1)
            links={};usd_links={}
            cache=UsdGeom.XformCache()
            stage=omni.usd.get_context().get_stage()
            for side in ['l','r']:
                for suffix in ['arm_'+side+'_end_link','gripper_'+side+'_base_link',
                    *['gripper_'+side+'_'+part+'_link'+str(n) for part in ['inner','outer','support'] for n in [1,2]]]:
                    p,q=self._get_object_pose('/G2/'+suffix)
                    links[suffix]={'position':to_numpy(p).tolist(),'quaternion_wxyz':to_numpy(q).tolist()}
                    usd_links[suffix]=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/G2/'+suffix))).T.tolist()
            self.data_to_send={'success':True,'joint_names':names,'joint_positions':joints.tolist(),
                'native_limits':limits.tolist(),'native_limits_layout':'joint_min_max','links':links,'usd_links':usd_links,'robot_base_position':to_numpy(self.robot_init_position).tolist(),
                'robot_base_quaternion_wxyz':to_numpy(self.robot_init_rotation).tolist()}
            kp,kd=self.ui_builder.articulation._articulation_view.get_gains()
            self.data_to_send['drive_stiffness']=to_numpy(kp).tolist()
            self.data_to_send['drive_damping']=to_numpy(kd).tolist()
            return
        if name=='ProjectSnapshot':
            stage=omni.usd.get_context().get_stage();attributes={}
            for path in cfg.get('joint_paths',[]):
                if not path.startswith('/World/Objects/'):raise ValueError('Diagnostic path outside task objects')
                prim=stage.GetPrimAtPath(path)
                attr=prim.GetAttribute('physics:jointEnabled') if prim else None
                attributes[path]=bool(attr.Get()) if attr else False
            mappings={}
            obstacles=set()
            for motion in self.ui_builder.curoboMotion.values():
                mappings.update(shared_mapping(motion));motion.set_obstacles()
                obstacles.update(motion.planner.scene_collision_checker.get_obstacle_names(env_idx=0) or [])
            self.data_to_send={'success':True,'attach_states':self.attach_states.copy(),'planner_attachments':mappings,
                'planner_obstacles':sorted(obstacles),'joint_enabled':attributes,'sim_time':self.get_current_simulation_time()}
            return
        if name=='ProjectDetachPlannerArm':
            right=cfg['arm']=='right';link='attached_object' if right else 'left_attached_object'
            seen=set();restored=set();remaining={}
            for motion in self.ui_builder.curoboMotion.values():
                if id(motion) in seen:continue
                seen.add(id(motion));motion._project_detach_link=link
                try:restored.update(motion.detach_obj())
                finally:motion._project_detach_link=None
                remaining.update(shared_mapping(motion))
            self.attach_states={p:r for p,r in self.attach_states.items() if r!=right}
            self.data_to_send={'success':True,'attach_states':self.attach_states.copy(),
                'restored_world_obstacles':sorted(restored),'planner_attachments':remaining};return
        return old_command(self)
    CommandController.handle_command_any=command
    original_restore=CommandController.restore_state
    def restore(self,state):
        if self.playback_waited_frame_num==0 and state and not state.get('attach_states') and self.attach_states:
            self.ui_builder.detach_objs();self.attach_states={}
        return original_restore(self,state)
    CommandController.restore_state=restore
