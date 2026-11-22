"""Read native compressed messages and authored lens settings without simulation."""
from pathlib import Path
import json,sys
import cv2,numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores,get_typestore
from pxr import Usd,UsdGeom
root=Path('/home/vipuser/by/genie_sim')
stage=Usd.Stage.Open(str(root/'GenieSimAssets/robot/G2_crsB_90d/robot_fix.usda'))
for prim in stage.Traverse():
 if prim.IsA(UsdGeom.Camera):
  cam=UsdGeom.Camera(prim)
  print('LENS',str(prim.GetPath()),'clipping',cam.GetClippingRangeAttr().Get(),'focal',cam.GetFocalLengthAttr().Get(),
   'aperture',cam.GetHorizontalApertureAttr().Get(),cam.GetVerticalApertureAttr().Get())
run=Path(sys.argv[1]);result=json.loads((run/'collection_result.json').read_text())
raw=Path('/workspace')/Path(result['episodes'][0]['recording_dir']).relative_to('/workspace')
stats={};images={}
with AnyReader([raw],default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
 conns=[c for c in reader.connections if 'rgb_compressed' in c.topic]
 print('CONNECTIONS',[(c.topic,c.msgcount) for c in conns])
 for c,t,b in reader.messages(connections=conns):
  msg=reader.deserialize(b,c.msgtype);frame=cv2.imdecode(np.frombuffer(msg.data,np.uint8),cv2.IMREAD_COLOR)
  if frame is None:raise RuntimeError('Invalid JPEG')
  gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  stats.setdefault(c.topic,[]).append([float(gray.mean()),float(gray.std())])
  images[c.topic]=frame
out=run/'camera-qa';out.mkdir(exist_ok=True)
for topic,rows in stats.items():
 a=np.asarray(rows)
 print('NATIVE_PIXELS',topic,'n',len(a),'first',rows[0],'last',rows[-1],'mean_std',a.mean(axis=0).tolist(),'uniform_frac',float((a[:,1]<1).mean()))
 cv2.imwrite(str(out/('raw-'+topic.strip('/').replace('/','_')+'.jpg')),images[topic])
