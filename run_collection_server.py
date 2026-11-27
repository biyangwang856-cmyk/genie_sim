"""Run the upstream collection server on the selected physical GPU."""
from pathlib import Path
import dataclasses,faulthandler,os,runpy,sys
import random,numpy as np
seed=int(os.environ.get('GENIE_EPISODE_SEED','20261120'));random.seed(seed);np.random.seed(seed)
ROOT=Path('/home/vipuser/by/genie_sim')
MODULE=Path(os.environ.get('GENIE_COLLECTION_MODULE',ROOT/'genie_sim-main/source/data_collection'))
sys.path.insert(0,str(MODULE))
faulthandler.enable()
import sim_backend
Backend=sim_backend.BackendConfig
def config(*args,**kwargs):
    kwargs['cuda_index']=int(os.environ.get('GENIE_PROJECT_GPU','1'))
    return Backend(*args,**kwargs)
sim_backend.BackendConfig=config
import isaacsim
App=isaacsim.SimulationApp
def app(configuration=None,*args,**kwargs):
    cfg=dict(configuration or {})
    gpu=int(os.environ.get('GENIE_PROJECT_GPU','1'))
    cfg.update(active_gpu=gpu,physics_gpu=gpu,multi_gpu=False,renderer='RaytracedLighting')
    result=App(cfg,*args,**kwargs)
    import torch
    torch.manual_seed(seed)
    from isaacsim.core.utils import extensions
    extensions.enable_extension('omni.behavior.scripting.core')
    import omni.behavior.scripting.core as scripting
    sys.modules.setdefault('omni.kit.scripting',scripting)
    install_project_hooks()
    return result
isaacsim.SimulationApp=app

def install_project_hooks():
    """Calibrate camera optical axes using live scene targets; preserve robot physics."""
    import json
    import numpy as np
    from pxr import UsdGeom,Gf,Usd
    from scipy.spatial.transform import Rotation
    import omni.usd
    from server.command_controller import CommandController
    import importlib.util,hashlib
    path=Path(__file__).with_name('project_runtime_hooks.py')
    spec=importlib.util.spec_from_file_location('project_runtime_hooks',path)
    hooks=importlib.util.module_from_spec(spec);sys.modules['project_runtime_hooks']=hooks;spec.loader.exec_module(hooks)
    hooks.install()
    evidence=Path(os.environ['GENIE_EPISODE_EVIDENCE']);evidence.mkdir(parents=True,exist_ok=True)
    sources={
        'server_wrapper_path':str(Path(__file__)),
        'server_wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'server_hooks_path':str(path),'server_hooks_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (evidence/'runtime-source.json').write_text(json.dumps(sources))
    from isaacsim.core.api import World
    original_render=World.render
    original_step=World.step
    exact_camera_render = os.environ.get('GENIE_EXACT_RENDER_CAMERA') == '1'
    def synchronized_render(world,*args,**kwargs):
        # Keep the engine's native synchronization lifecycle. Switching Fabric
        # or fetching PhysX results from a render callback is unsafe in 6.2.
        result=original_render(world,*args,**kwargs)
        hooks.record_render_sample(world)
        if not exact_camera_render:
            hooks.publish_pending_camera_samples(world)
        return result
    World.render=synchronized_render
    def capture_step(world,*args,**kwargs):
        # ROS joint and TF publishers sample before this upcoming PhysX step.
        # Capture the same state outside the PhysX callback while its clock is fixed.
        prepared = None
        if exact_camera_render:
            prepared = hooks.prepare_exact_camera_samples(world, lambda: original_render(world), update_wrist_cameras)
        result=original_step(world,*args,**kwargs)
        if exact_camera_render:
            hooks.publish_exact_camera_samples(world, prepared)
        elif hooks.pending_direct_cameras():
            world.render()
        return result
    World.step=capture_step
    from server.ros_publisher import base as ros_base
    original_tf=ros_base._world_pose_for_tf
    def camera_tf(prim):
        xf=UsdGeom.Xformable(prim)
        if prim.GetName() in ['Left_Camera','Right_Camera'] and xf.GetResetXformStack():
            # These cameras already carry a world transform sampled from the
            # live wrist. Reapplying an ancestor pose would move TF twice.
            matrix=xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            return matrix.ExtractTranslation(),matrix.ExtractRotationQuat()
        return original_tf(prim)
    ros_base._world_pose_for_tf=camera_tf
    original_physics=CommandController.callback_physics
    def update_wrist_cameras(self, *, force=False):
        mounts=getattr(self,'_project_wrist_cameras',{})
        now=float(self.ui_builder.my_world.current_time)
        if mounts and (force or now-getattr(self,'_project_camera_update_time',-1)>=1/30-1e-5):
            for name,(side,mount) in mounts.items():
                p,q=self._get_object_pose('/G2/arm_'+side+'_end_link')
                wrist=np.eye(4);wrist[:3,:3]=Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix();wrist[:3,3]=p
                prim=omni.usd.get_context().get_stage().GetPrimAtPath(name)
                # Sensor initialization may normalize/remove the earlier
                # attribute. Reacquire it from the current stage.
                xf=UsdGeom.Xformable(prim);op=xf.MakeMatrixXform();xf.SetResetXformStack(True)
                op.Set(Gf.Matrix4d((wrist@mount).T.tolist()))
            self._project_camera_update_time=now
    def follow_wrist_cameras(self,*args,**kwargs):
        try:
            update_wrist_cameras(self)
        except Exception as exc:
            if not getattr(self,'_project_camera_failure_saved',False):
                folder=Path(os.environ['GENIE_EPISODE_EVIDENCE']);folder.mkdir(parents=True,exist_ok=True)
                (folder/'camera-follow-error.json').write_text(json.dumps({'error':repr(exc)}))
                self._project_camera_failure_saved=True
            # Camera diagnostics cannot suppress native control/ROS callbacks.
        return original_physics(self,*args,**kwargs)
    CommandController.callback_physics=follow_wrist_cameras
    original=CommandController.handle_get_observation
    def calibrate_recording_cameras(self):
        stage=omni.usd.get_context().get_stage();cache=UsdGeom.XformCache()
        objects=stage.GetPrimAtPath('/World/Objects');targets=[]
        if objects:
            for p in objects.GetChildren():
                if 'table' not in p.GetName().lower():
                    entity=stage.GetPrimAtPath(str(p.GetPath())+'/entity')
                    if entity:targets.append(np.asarray(cache.GetLocalToWorldTransform(entity).ExtractTranslation()))
        if not targets:return
        center=np.median(np.asarray(targets),axis=0)
        self._project_wrist_cameras={}
        records=[]
        for name in self.cameras:
            prim=stage.GetPrimAtPath(name)
            if not prim:continue
            current=cache.GetLocalToWorldTransform(prim)
            eye=np.asarray(current.ExtractTranslation())
            side='l' if 'Left_Camera' in name else 'r' if 'Right_Camera' in name else None
            if side:
                # Fabric advances the articulation without updating authored USD
                # link transforms. Both eye and target must use the SAME frame
                # source when deriving the camera's rigid local mount transform.
                tcp=stage.GetPrimAtPath('/G2/gripper_'+side+'_center_link')
                if not tcp:raise RuntimeError('Missing wrist camera TCP frame')
                target=np.asarray(cache.GetLocalToWorldTransform(tcp).ExtractTranslation())
            else:target=center
            if np.linalg.norm(target-eye)<.01:continue
            desired=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse()
            parent=cache.GetLocalToWorldTransform(prim.GetParent())
            local=desired*parent.GetInverse()
            xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder();op=xf.AddTransformOp()
            if side:
                # Camera-link markers in the asset can retain spawn transforms
                # under Fabric. Explicitly follow the live rigid wrist, keeping
                # the sensor-to-wrist mount constant and recording its pose.
                wrist=cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/G2/arm_'+side+'_end_link'))
                mount=(desired*wrist.GetInverse())
                self._project_wrist_cameras[name]=(side,np.asarray(mount).T.copy())
                xf.SetResetXformStack(True)
                p,q=self._get_object_pose('/G2/arm_'+side+'_end_link')
                live=np.eye(4);live[:3,:3]=Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix();live[:3,3]=p
                op.Set(Gf.Matrix4d((live@np.asarray(mount).T).T.tolist()))
            else:op.Set(local)
            records.append({'camera':name,'eye':eye.tolist(),'target':target.tolist(),'local_matrix':np.asarray(local).tolist(),
                'tracking':'live_rigid_wrist' if side else 'authored_parent',
                'wrist_to_camera':self._project_wrist_cameras[name][1].tolist() if side else None})
        folder=Path(os.environ['GENIE_EPISODE_EVIDENCE']);folder.mkdir(parents=True,exist_ok=True)
        (folder/'camera-calibration.json').write_text(json.dumps(records,indent=2))

    def init_robot(self,*args,**kwargs):
        if not self.data.get('startRecording'):return original(self,*args,**kwargs)
        if 'camera_recording' not in sources:
            camera=hooks.install_direct_camera_recording(original, self, exact_render=exact_camera_render)
            sources['camera_recording']=camera
            controller=Path(camera['controller_source'])
            sources['native_controller_sha256']=hashlib.sha256(controller.read_bytes()).hexdigest()
            (evidence/'runtime-source.json').write_text(json.dumps(sources))
        if os.environ.get('GENIE_CAMERA_LOOKAT','1')=='1':calibrate_recording_cameras(self)
        result=original(self,*args,**kwargs)
        peers=hooks.wait_for_recording_subscribers(self.ros_publishers_persistent+self.ros_publishers_temporary)
        (evidence/'recorder-subscriptions.json').write_text(json.dumps(peers,indent=2))
        if not peers['passed']:raise RuntimeError('Recording did not match all camera and state publishers')
        return result
    CommandController.handle_get_observation=init_robot

os.chdir(MODULE)
runpy.run_path(str(MODULE/'scripts/data_collector_server.py'),run_name='__main__')
