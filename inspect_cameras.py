"""Extract the middle native frame for a task bottleneck and report camera QC."""
from pathlib import Path
import json,subprocess,sys
root=Path('/home/vipuser/by/genie_sim');run=Path(sys.argv[1])
report=json.loads((run/'collection_result.json').read_text())
raw=Path('/data/by/genie_sim/runtime-collection')/Path(report['episodes'][0]['recording_dir']).relative_to('/workspace')
if (run/'evidence/raw-archive.json').exists():raw=Path(json.loads((run/'evidence/raw-archive.json').read_text())['immutable_recording'])
state=json.loads((raw/'state.json').read_text());n=len(state['frames'])//2
if len(sys.argv)>2:n=int(sys.argv[2]);n=len(state['frames'])+n if n<0 else n
out=run/'camera-qa';out.mkdir(exist_ok=True)
for name in ['head','hand_left','hand_right']:
 subprocess.run([str(root/'.venv/bin/ffmpeg'),'-y','-v','error','-i',str(raw/'observations/videos'/(name+'_color.mp4')),
  '-vf',f'select=eq(n\\,{n})','-frames:v','1',str(out/(name+'.jpg'))],check=True)
print(json.dumps({'run':str(run),'output':str(out),'frame':n,'native_time_s':state['frames'][n]['time_stamp']}))
