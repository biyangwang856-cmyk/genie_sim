"""Server-side Qwen planner and embedding client; credential-free logs only."""
from pathlib import Path
import json
import time
import urllib.request
import urllib.error

ROOT=Path('/home/vipuser/by/genie_sim')

class QwenAPI:
    def __init__(self):
        self.config=json.loads((ROOT/'secrets/qwen.json').read_text())
        self.receipts=ROOT/'jobs/api-receipts.jsonl'
        self.receipts.parent.mkdir(exist_ok=True)

    def request(self, endpoint, payload):
        data=json.dumps(payload,ensure_ascii=False).encode()
        for attempt in range(3):
            request=urllib.request.Request(self.config['base_url'].rstrip('/')+'/'+endpoint,data=data,
                headers={'Authorization':'Bearer '+self.config['api_key'],'Content-Type':'application/json'})
            start=time.time()
            try:
                with urllib.request.urlopen(request,timeout=90) as r: result=json.load(r)
                receipt={'model':payload['model'],'id':result.get('id'),'usage':result.get('usage'),
                    'endpoint':endpoint,'elapsed_s':round(time.time()-start,3),'status':200}
                with self.receipts.open('a') as f: f.write(json.dumps(receipt)+'\n')
                return result,receipt
            except urllib.error.HTTPError as e:
                try: body=json.loads(e.read())
                except Exception: body={}
                error=body.get('error',{})
                code=error.get('code','unknown') if isinstance(error,dict) else 'unknown'
                if e.code in (429,500,502,503) and attempt<2:
                    time.sleep(2**attempt);continue
                raise RuntimeError(f'Qwen request rejected status={e.code} code={code}') from None
        raise RuntimeError('Qwen retry budget exhausted')

    def plan(self,instruction,catalog,robot=None):
        robot=robot or {'robot_id':'g2crsb90d'}
        schema={'robot':robot['robot_id'],'tasks':[{'task_id':'one exact registered ID','scene_quotas':[{'scene_id':'0','successes':3},{'scene_id':'1','successes':2}], 'asset_queries':['English semantic 3D object description'], 'skills':['pick','align','place','insert','release','retreat']} ]}
        prompt=('Convert the user instruction into JSON following this example structure. Select ONLY registered tasks whose actual objective matches the request. '
            'Never substitute a vaguely similar task or silently omit an unsupported task, requested object property, or unavailable robot capability. '
            'If ANY required operation is outside this existing-asset/task catalog, return '
            '{"status":"unsupported","reason":"...","missing_assets":[],"missing_skills":[],"missing_robot_capabilities":[]}. '
            'Otherwise preserve robot and per-task success quotas. Do not invent coordinates or IDs. Return all requested tasks. '
            'Each skills entry MUST be a string from this exact enum: pick, align, place, insert, release, retreat. '
            'Represent grasp as pick and close-lid as place; do not invent skill names. Robot capabilities: '+json.dumps(robot)+' Catalog: '+json.dumps(catalog)+' Schema example: '+json.dumps(schema))
        result,receipt=self.request('chat/completions',{'model':'qwen-plus','messages':[{'role':'system','content':prompt},{'role':'user','content':instruction}], 'response_format':{'type':'json_object'},'temperature':0.1,'enable_thinking':False})
        plan=json.loads(result['choices'][0]['message']['content'])
        archive=ROOT/'jobs/plan-responses';archive.mkdir(exist_ok=True)
        (archive/(result['id']+'.json')).write_text(json.dumps({'instruction':instruction,'robot':robot,'plan':plan,'raw_response':result['choices'][0]['message']['content'],'receipt':receipt},ensure_ascii=False,indent=2))
        return plan,receipt

    def embed(self,texts):
        result,receipt=self.request('embeddings',{'model':'text-embedding-v4','input':texts,'dimensions':1024,'encoding_format':'float'})
        return [item['embedding'] for item in sorted(result['data'],key=lambda x:x['index'])],receipt

if __name__=='__main__':
    api=QwenAPI()
    catalog=[{'task_id':'g2crsb90d_finemanip_'+name,'description':desc} for name,desc in [('phone_packaging','手机装盒 phone packaging'),('put_the_cap_on_the_pen_new','双臂盖笔帽 cap pen'),('insert_building_block','积木插入 matching star block insertion')]]
    instruction='用g2crsb90d完成手机装盒、盖笔帽和积木插入，每个任务采集5条完整成功轨迹，场景0三条、场景1两条。'
    report={'instruction':instruction}
    for name,fn in [('planner',lambda:api.plan(instruction,catalog)),('embedding',lambda:api.embed(['phone box packing','pen cap','star block shape sorting']))]:
        try:
            value,receipt=fn()
            report[name]={'ok':True,'receipt':receipt,'plan':value} if name=='planner' else {'ok':True,'receipt':receipt,'vectors':len(value),'dimensions':len(value[0])}
        except Exception as e: report[name]={'ok':False,'error':str(e)}
    (ROOT/'jobs/qwen-live-probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
