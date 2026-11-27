"""Extract native camera frames at the task's measured physical bottleneck."""
import argparse,json
from pathlib import Path
import av
import numpy as np
from PIL import Image,ImageDraw


def hardpoint(run):
    source=json.loads((run/'source/request.json').read_text())
    if source['task_id']=='place_in_container':
        events=[json.loads(line) for line in (run/'evidence/skill-execution.jsonl').read_text().splitlines()]
        gates=[event for event in events if event.get('event')=='container_release_gate' and event.get('passed')]
        if not gates:raise ValueError('No verified container release')
        gate=gates[-1]
        return 'container-release',gate['sim_time_s'],{
            key:gate[key] for key in ('object','fit_margin_m','rim_clearance_m','expected_drop_m')}
    if source['task_id']=='insert_building_block':
        report=json.loads((run/'evidence/hole-crossing-audit.json').read_text())
        accepted=[attempt for attempt in report.get('passage_attempts',[]) if attempt['passed']]
        crossing=accepted[-1]['start_time_s'] if accepted else report['first_crossing_times_s'][0]
        return 'crossing',crossing,{'first_crossing_sim_time_s':report['first_crossing_times_s'][0],
            'passage_attempts':report.get('passage_attempts',[])}
    if source['task_id']=='phone_packaging':
        spec=json.loads((run/'evidence/task-spec.json').read_text())
        lid=spec['bindings']['lid']
        events=[json.loads(line) for line in (run/'evidence/skill-execution.jsonl').read_text().splitlines()]
        releases=[event for event in events if event['event']=='packing_release_gate' and event['passed'] and event['object']==lid]
        if not releases:raise ValueError('No physically validated lid release pose was reached')
        event=releases[-1]
        return 'lid-release',event['sim_time_s'],{'object':lid,
            'projected_fit_margin_m':event['projected_fit_margin_m'],'rim_clearance_m':event['rim_clearance_m']}
    if source['task_id']=='put_the_cap_on_the_pen_new':
        events=[json.loads(line) for line in (run/'evidence/stage-evidence.jsonl').read_text().splitlines()]
        seating=next((event for event in events if event['phase']=='cap_connection_gate' and event.get('passed')),None)
        if seating is None:raise ValueError('No measured cap seating event')
        return 'cap-seat',seating['snapshot']['sim_time'],{'official_snap_model':True,
            'measured_insertion_depth_m':seating['measured_insertion_depth_m']}
    raise ValueError('No hardpoint review is defined for the task')

def review(run,subdir=None):
    run=Path(run);aligned=run/'evidence/aligned-archive.json'
    raw=Path(json.loads(aligned.read_text())['aligned_recording']) if aligned.exists() else run/'raw'
    frames=json.loads((raw/'state.json').read_text())['frames']
    times=np.asarray([f['time_stamp'] for f in frames]);kind,event_time,evidence=hardpoint(run)
    if not times[0]<=event_time<=times[-1]:raise ValueError('Physical event outside the recorded camera clock')
    requested=[event_time-.35,event_time-.10,event_time+.05,event_time+.20,event_time+.40,float(times[-1])]
    indexes=[int(np.argmin(abs(times-t))) for t in requested]
    info=json.loads((raw/'recording_info.json').read_text());output=run/'review'
    if subdir:
        if Path(subdir).name!=subdir or subdir in ('.','..'):raise ValueError('Review subdirectory must be one directory name')
        output=output/subdir
    output.mkdir(parents=True,exist_ok=True)
    if (output/'review-frames.json').exists():raise FileExistsError('Existing review evidence: '+str(output))
    receipt={'hardpoint':kind,'reviewed_event_sim_time_s':event_time,**evidence,'frames':[]}
    if kind=='crossing':receipt['reviewed_crossing_sim_time_s']=event_time
    for name,camera in info['camera_info'].items():
        video=raw/'observations/videos'/camera['output']['video'];sheet=Image.new('RGB',(960,660),'#202020');draw=ImageDraw.Draw(sheet)
        selected=set(indexes);images={}
        with av.open(str(video)) as container:
            for index,frame in enumerate(container.decode(video=0)):
                if index in selected:images[index]=frame.to_image().convert('RGB')
                if len(images)==len(selected):break
        if images.keys()!=selected:raise ValueError('Missing camera frames for hardpoint review: '+name)
        for i,index in enumerate(indexes):
            frame=images[index].copy();frame.thumbnail((476,190))
            x=(i%2)*480;y=(i//2)*220;sheet.paste(frame,(x+(480-frame.width)//2,y+24))
            draw.text((x+8,y+5),f'{name} frame {index}; sim {times[index]:.4f}s',fill='white')
            receipt['frames'].append({'camera':name,'index':index,'sim_time_s':float(times[index])})
        sheet.save(output/(name+'-'+kind+'.jpg'),quality=95)
    (output/'review-frames.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'review':str(output),'hardpoint':kind,'sim_time_s':event_time}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');parser.add_argument('--subdir')
    args=parser.parse_args();review(args.run,args.subdir)
