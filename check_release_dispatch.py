"""Execute the real extracted release request handler without starting Isaac."""
import ast,sys,types
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

path=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name('project_runtime_hooks.py')
tree=ast.parse(path.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='command')
physics=NS(get_joint_positions=lambda *a,**k:np.array([-.53,-.53]),
    get_dof_limits=lambda *a,**k:(np.full(2,-1.2),np.full(2,.8)),
    set_gains=lambda *a,**k:None,set_max_efforts=lambda *a,**k:None,apply_action=lambda *a,**k:None)
sim=types.ModuleType('sim_backend');sim.physics=physics;sys.modules['sim_backend']=sim
gripper=NS(_joint_dof_indicies=[0,1],joint_opened_positions=np.zeros(2))
robot=NS(_articulation_view=NS(get_gains=lambda:(np.full((1,2),100.),np.full((1,2),20.))))
controller=NS(data={'command_config':{'command':'ProjectPrepareFingerRelease','arm':'left'}},
    _init_grippers=lambda:robot,gripper_L=gripper,gripper_R=gripper)
scope={'np':np,'to_numpy':np.asarray,'old_command':lambda _:None}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
scope['command'](controller)
assert controller.data_to_send['success'] is True
assert len(controller.data_to_send['initial_rad'])==2
print('RELEASE_DISPATCH_PASS: real handler prepared two physical finger drives')
