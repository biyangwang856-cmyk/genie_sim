"""Extract the final real JPEG from each closed MCAP stream; no scene replay."""
import argparse,json
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores,get_typestore
p=argparse.ArgumentParser();p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);last={};counts={}
with AnyReader([a.recording],default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
    connections=[c for c in reader.connections if 'rgb_compressed' in c.topic]
    for c,t,raw in reader.messages(connections=connections):
        message=reader.deserialize(raw,c.msgtype);name=c.topic.strip('/').replace('/','_').replace('_rgb_compressed','')
        last[name]=bytes(message.data);counts[name]=counts.get(name,0)+1
for name,data in last.items():
    if not data.startswith(b'\xff\xd8'):raise ValueError('Expected original JPEG: '+name)
    (a.output/(name+'-last.jpg')).write_bytes(data)
(a.output/'raw-image-review.json').write_text(json.dumps({'recording':str(a.recording),'frames':counts,'source':'original MCAP compressed-image bytes'},indent=2))
print(json.dumps({'output':str(a.output),'frames':counts}))
