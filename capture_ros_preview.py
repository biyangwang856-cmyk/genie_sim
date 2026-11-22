"""Capture actual published compressed camera frames for bottleneck inspection."""
from pathlib import Path
import time,json,os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2,numpy as np
out=Path(os.environ.get('GENIE_PREVIEW_DIR','/data/by/genie_sim/collection-runs/block-scene0-probe02/preview'));out.mkdir(exist_ok=True)
rclpy.init();node=Node('genie_by_bottleneck_preview')
stamps={};subs=[]
def callback(name,msg):
    image=cv2.imdecode(np.frombuffer(msg.data,np.uint8),cv2.IMREAD_COLOR)
    if image is not None:
        cv2.imwrite(str(out/(name+'.jpg')),image)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        stamps[name]={'stamp_s':msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,'shape':list(image.shape),
            'mean':float(gray.mean()),'spatial_std':float(gray.std())}
for name in ('head_front_Camera','Left_Camera','Right_Camera'):
    subs.append(node.create_subscription(CompressedImage,'/'+name+'_rgb_compressed',lambda m,n=name:callback(n,m),10))
deadline=time.time()+30
while time.time()<deadline and len(stamps)<3:rclpy.spin_once(node,timeout_sec=.2)
(out/'stamps.json').write_text(json.dumps(stamps,indent=2));print(stamps)
node.destroy_node();rclpy.shutdown()
