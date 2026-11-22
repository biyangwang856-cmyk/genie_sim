"""Read original ROS headers to distinguish recorder loss from extractor loss."""
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores,get_typestore

def audit(recording,output):
    recording=Path(recording);frames=json.loads((recording/'state.json').read_text())['frames'];times=[f['time_stamp'] for f in frames]
    gaps=[{'index':i,'before':a,'after':b} for i,(a,b) in enumerate(zip(times,times[1:])) if abs(b-a-1/30)>.0002]
    paths=list(recording.glob('*.mcap'));counts=defaultdict(int);headers=defaultdict(set);windows=defaultdict(list)
    with AnyReader(paths,default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections=[c for c in reader.connections if c.topic in ['/joint_states','/tf'] or c.msgtype=='sensor_msgs/msg/CompressedImage']
        for c,stamp,blob in reader.messages(connections=connections):
            message=reader.deserialize(blob,c.msgtype)
            header=message.header if hasattr(message,'header') else message.transforms[0].header
            t=header.stamp.sec+header.stamp.nanosec*1e-9;rounded=round(t,4)
            key=c.topic;counts[key]+=1;headers[key].add(rounded)
            if any(g['before']-.05<=t<=g['after']+.05 for g in gaps):windows[key].append(rounded)
    missing={key:[t for t in times if t not in stamps] for key,stamps in headers.items()}
    cameras=[key for key in headers if 'compressed' in key]
    report={'passed':not gaps and len(cameras)==3 and '/joint_states' in headers and '/tf' in headers and not any(missing.values()),
        'source':str(recording),'state_sha256':hashlib.sha256((recording/'state.json').read_bytes()).hexdigest(),
        'native_frame_count':len(times),'gaps':gaps,'topic_counts':dict(counts),'header_windows':dict(windows),
        'missing_exact_header_counts':{key:len(value) for key,value in missing.items()},
        'missing_header_examples':{key:value[:10] for key,value in missing.items() if value}}
    if any(missing.values()):report['reason']='Missing original exact-time image/joint/TF; neighboring-frame copies cannot pass this audit'
    output=Path(output);output.parent.mkdir(exist_ok=True,parents=True);output.write_text(json.dumps(report,indent=2));print(json.dumps(report));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('recording');p.add_argument('output');a=p.parse_args();audit(a.recording,a.output)
