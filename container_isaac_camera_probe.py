"""Verify headless Kit rendering and physical stepping inside the built image."""
from pathlib import Path
import json
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':0,'physics_gpu':0,'multi_gpu':False,
    'renderer':'RaytracedLighting','extra_args':['--/rtx/hydra/supportMultiTickRate=false']})
import numpy as np
from PIL import Image
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.viewports import set_camera_view
from pxr import UsdLux

world=World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
cube=world.scene.add(DynamicCuboid(prim_path='/World/ProbeCube',name='cube',position=np.array([0.,0.,0.5]),size=0.05,color=np.array([0.8,0.1,0.1])))
UsdLux.DomeLight.Define(world.stage,'/World/ProbeLight').CreateIntensityAttr(900.)
camera=Camera(prim_path='/World/ProbeCamera',position=np.array([1.,1.,1.]),resolution=(160,120))
set_camera_view(eye=np.array([1.,1.,1.]),target=np.array([0.,0.,0.1]),camera_prim_path='/World/ProbeCamera')
world.reset();camera.initialize()
start=float(cube.get_world_pose()[0][2])
for _ in range(100):world.step(render=True)
pixels=np.asarray(camera.get_rgba())
end=float(cube.get_world_pose()[0][2])
assert pixels.shape==(120,160,4),str(pixels.shape)
assert np.std(pixels[...,:3])>1., 'Camera image is empty/uniform'
assert 0.0<end<start-0.1, f'Gravity probe did not fall: {start}->{end}'
output=Path('/probe-output')
Image.fromarray(pixels.astype(np.uint8)).save(output/'isaac-camera.png')
report={'ok':True,'headless':True,'camera_shape':list(pixels.shape),'rgb_std':float(np.std(pixels[...,:3])),
    'initial_cube_z':start,'final_cube_z':end,'physics_steps':100}
(output/'isaac-camera.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
app.close()
