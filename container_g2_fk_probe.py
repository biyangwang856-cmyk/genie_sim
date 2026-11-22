"""Compile and run the pinned cuRobo G2 CRS-B kinematics on the selected A100."""
import json
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
import curobo
from curobo.kinematics import Kinematics,KinematicsCfg
from curobo.types import JointState

assert torch.cuda.is_available()
assert torch.cuda.device_count()==1, 'Probe must expose one selected GPU only'
assert 'A100' in torch.cuda.get_device_name(0)
assert version('nvidia-curobo')=='0.8.1'
path=Path('/workspace/config/curobo/v2/configs/robot/G2_crsB_90D_dual.yml')
robot=Kinematics(KinematicsCfg.from_robot_yaml_file(str(path)))
q=torch.zeros((2,robot.get_dof()),device='cuda',dtype=torch.float32)
state=robot.compute_kinematics(JointState.from_position(q,joint_names=robot.joint_names))
torch.cuda.synchronize()
positions=state.tool_poses.position.detach().cpu().numpy()
assert np.isfinite(positions).all()
assert len(robot.tool_frames)>=2, 'Both G2 tool frames must be present'
report={'ok':True,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
    'curobo':curobo.__version__,'robot':'g2crsb90d','dof':robot.get_dof(),'tool_frames':list(robot.tool_frames),
    'positions':positions.tolist(),'nvrtc_kernel_execution':True}
Path('/probe-output/g2-fk.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
