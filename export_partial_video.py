"""Decode recorded ROS camera streams without replaying or changing the scene."""
from pathlib import Path
import json,os,subprocess
import cv2,numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores,get_typestore
from rosbags.image import message_to_cvimage
root=Path('/workspace/recording_data/g2crsb90d_finemanip_insert_building_block')
out=Path(os.environ.get('GENIE_VIDEO_OUTPUT','/data/by/genie_sim/collection-runs/block-scene0-probe02/video-preview'))
out.mkdir(parents=True,exist_ok=True)
writers={};counts={};stamps={};last={}
with AnyReader([root],default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
    connections=[c for c in reader.connections if 'rgb_compressed' in c.topic]
    print('VIDEO_TOPICS',[(c.topic,c.msgcount) for c in connections],flush=True)
    for c,t,raw in reader.messages(connections=connections):
        msg=reader.deserialize(raw,c.msgtype)
        frame=cv2.imdecode(np.frombuffer(msg.data,np.uint8),cv2.IMREAD_COLOR)
        if frame is None:continue
        name=c.topic.strip('/').replace('_rgb_compressed','')
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        if name not in writers:
            h,w=frame.shape[:2]
            writers[name]=subprocess.Popen(['/home/vipuser/by/genie_sim/.venv/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','bgr24','-video_size',f'{w}x{h}','-framerate','30','-i','pipe:0','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out/(name+'.mp4'))],stdin=subprocess.PIPE)
            stamps[name]=[];counts[name]=0
            cv2.imwrite(str(out/(name+'-first.jpg')),frame)
        writers[name].stdin.write(frame.tobytes());counts[name]+=1;stamps[name].append(stamp);last[name]=frame
        if name=='head_front_Camera' and counts[name] in (60,120,180,240,300,360,420):
            cv2.imwrite(str(out/(name+f'-{counts[name]:04}.jpg')),frame)
for name,writer in writers.items():
    writer.stdin.close()
    if writer.wait()!=0:raise RuntimeError('Video encoder failed')
    cv2.imwrite(str(out/(name+'-last.jpg')),last[name])
report={'label':'Failed probe: physical grasp followed by unreachable insertion alignment; not a successful task trajectory.',
    'frames':counts,'timestamps_s':stamps,'fps':30}
for name,count in counts.items():
    result=subprocess.run(['/home/vipuser/by/genie_sim/.venv/bin/ffmpeg','-v','error','-nostats','-progress','pipe:1','-i',str(out/(name+'.mp4')),'-f','null','-'],capture_output=True,text=True,check=True)
    decoded=int([row.split('=',1)[1] for row in result.stdout.splitlines() if row.startswith('frame=')][-1])
    if decoded!=count:raise RuntimeError('Video decode mismatch')
(out/'preview-receipt.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'output':str(out),'frames':counts}),flush=True)
