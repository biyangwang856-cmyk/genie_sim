"""Robot capability boundary. Only G2 has an executable adapter at present."""
from dataclasses import dataclass
import json
import re
import os
from pathlib import Path

ROOT=Path('/home/vipuser/by/genie_sim')

class UnsupportedRequest(ValueError):
    def __init__(self,message,*,missing_assets=(),missing_skills=(),missing_capabilities=()):
        super().__init__(message)
        self.details={'status':'unsupported','reason':message,'missing_assets':list(missing_assets),
            'missing_skills':list(missing_skills),'missing_robot_capabilities':list(missing_capabilities)}

@dataclass(frozen=True)
class RobotAdapter:
    robot_id:str
    arms:tuple
    capabilities:frozenset
    config:dict

    def describe(self):
        return {'robot_id':self.robot_id,'arms':list(self.arms),'capabilities':sorted(self.capabilities),**self.config}

    def resolve_role(self,role,bindings):
        arm=bindings.get(role)
        if arm not in self.arms:raise UnsupportedRequest('Role has no supported arm',missing_capabilities=[str(role)])
        return arm

    def require(self,requirements):
        missing=set(requirements)-self.capabilities
        if missing:raise UnsupportedRequest('Robot lacks required capabilities',missing_capabilities=sorted(missing))

    def calibration_path(self,root):
        calibration=self.config.get('calibration')
        return Path(root)/calibration if calibration else None

    def calibrated_urdf_targets(self,calibration):
        return (calibration['container_planner_urdf'],)


class G2ParallelAdapter(RobotAdapter):
    robot_ids=frozenset({'g2crsb90d'})

    def calibrated_urdf_targets(self,calibration):
        return (*super().calibrated_urdf_targets(calibration),'/workspace/config/robot_cfg/G2/G2_crsB_90D_from_usd.urdf')


ADAPTERS={'g2_parallel_v1':G2ParallelAdapter}


def load_robot(robot_id='g2crsb90d',root=ROOT):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',robot_id):raise UnsupportedRequest('Invalid robot ID')
    directory=Path(root)/'pipeline/robots'
    snapshot=Path(os.environ.get('GENIE_RUN_SOURCE',root))/'robot-profile.json'
    path=snapshot if snapshot.is_file() else directory/(robot_id+'.json')
    if not path.is_file():
        raise UnsupportedRequest('Robot profile is not registered: '+robot_id,missing_capabilities=['validated_robot_adapter'])
    data=json.loads(path.read_text())
    if data['robot_id']!=robot_id:raise UnsupportedRequest('Robot snapshot identity mismatch')
    implementation=ADAPTERS.get(data['adapter_class'])
    if implementation is None or robot_id not in implementation.robot_ids:
        raise UnsupportedRequest('Robot configuration exists but its execution adapter is not implemented',missing_capabilities=['execution_adapter'])
    return implementation(robot_id,tuple(data.pop('arms')),frozenset(data.pop('capabilities')),data)

def initial_g2_profile():
    return {'robot_id':'g2crsb90d','adapter_class':'g2_parallel_v1','arms':['left','right'],'scene_task_prefix':'g2crsb90d_finemanip_',
        'capabilities':['bimanual','friction_grasp','held_object_motion','joint_position_velocity_control','angular_parallel_gripper','gripper_opening_control','curobo'],
        'validation_status':'kinematics_and_three_camera_recording_verified; complete_task_success_pending',
        'calibration':'configs/robot-calibration.json','planner':'curobo_cuda',
        'tcp':{'frame':'gripper_{side}_center_link','wrist_offset_m':.185,'rotation_rpy_rad':[0,0,0]},
        'cooperation_workspace':{'center_world_m':[-.08,0.,.98],'operator_side_offset_m':.06},
        'gripper':{'type':'angular_parallel','units':'rad','actuator_groups':{
            'left':['idx31_gripper_l_inner_joint1','idx32_gripper_l_outer_joint1'],
            'right':['idx71_gripper_r_inner_joint1','idx72_gripper_r_outer_joint1']},
            'limits_source':'native_physics_snapshot','opening_target_source':'native_robot_configuration',
            'native_control_gripper_scope':'one configured actuator; coupled-drive release requires explicit adapter validation',
            'release_profile':{'speed_rad_s':.15,'stiffness':100.,'damping':5.,'max_effort_Nm':5.,'validation_status':'experimental'}},
        'cameras':{'head':'/G2/head_link3/head_front_Camera','left_wrist':'/G2/gripper_l_camera_link/Left_Camera','right_wrist':'/G2/gripper_r_camera_link/Right_Camera'},
        'data':{'joint_order_source':'native_snapshot.joint_names','state_units':'rad for revolute joints; m for prismatic joints',
            'actions':'actual resolved native joint position, velocity and effort targets; sparse commands retained separately',
            'fixed_dimension':False},
        'second_robot_execution_verified':False}

if __name__=='__main__':
    directory=ROOT/'pipeline/robots';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'g2crsb90d.json'
    if not path.exists():path.write_text(json.dumps(initial_g2_profile(),indent=2))
    print(json.dumps(load_robot().describe(),indent=2))
