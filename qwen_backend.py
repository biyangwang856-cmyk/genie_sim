"""Server-side Qwen planner and embedding client; credential-free logs only."""
from pathlib import Path
import json
import os
import time
import urllib.request
import urllib.error

ROOT=Path(os.environ.get('GENIE_PROJECT_ROOT', '/home/vipuser/by/genie_sim'))

class QwenAPI:
    def __init__(self):
        config_path=Path(os.environ['GENIE_QWEN_CONFIG']) if 'GENIE_QWEN_CONFIG' in os.environ else ROOT/'secrets/qwen.json'
        self.config=json.loads(config_path.read_text())
        evidence=os.environ.get('GENIE_EPISODE_EVIDENCE')
        self.audit_directory=Path(evidence)/'qwen' if evidence else ROOT/'jobs'
        self.audit_directory.mkdir(parents=True,exist_ok=True)
        self.receipts=self.audit_directory/'api-receipts.jsonl'

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

    def plan(self,instruction,catalog,robot=None,resolve_scene=False,scene_constraints=None,
             episode_quota=None,repair=None):
        robot=robot or {'robot_id':'g2crsb90d'}
        schema={'robot':robot['robot_id'],'tasks':[{'task_id':'one exact registered ID','scene_quotas':[{'scene_id':'0','successes':3},{'scene_id':'1','successes':2}], 'asset_queries':['English semantic 3D object description'], 'skills':['pick','align','place','insert','release','retreat']} ]}
        if resolve_scene:
            schema['tasks'][0].pop('scene_quotas')
        prompt=('Convert the user instruction into JSON following this example structure. Select ONLY registered tasks whose actual objective matches the request. '
            'Never substitute a vaguely similar task or silently omit an unsupported task, requested object property, or unavailable robot capability. '
            'If ANY required operation is outside this existing-asset/task catalog, return '
            '{"status":"unsupported","reason":"...","missing_assets":[],"missing_skills":[],"missing_robot_capabilities":[]}. '
            'Preserve the robot and explicit task requirements. Do not invent coordinates or IDs. Return all requested tasks. '
            'Each skills entry MUST be a string from this exact enum: pick, align, place, insert, release, retreat. '
            'Represent grasp as pick and close-lid as place; do not invent skill names. Robot capabilities: '+json.dumps(robot)+' Catalog: '+json.dumps(catalog)+' Schema example: '+json.dumps(schema))
        if resolve_scene:
            from scene_contract import SCENE_REQUEST_FIELDS
            prompt+=(
                ' Also parse the optional scene requirements, separately from the task. When the user asks for new scenes, '
                'different tables/rooms/object placements, or composition of library assets, add a scene_request JSON object '
                'with mode="compose_from_library", optional scene_count (integer), background_query/support_query '
                '(English descriptions), exact background_id/support_id only if supplied, and variation '
                '(array drawn from background, support, layout, distractors, lighting). If they name existing task scene IDs, '
                'use mode="existing" with scene_ids as a list of strings. Preserve requested scene counts and write an explicit '
                'integer successes_per_scene only if the request specifies that count. Omit scene_request for the original '
                'two-scene task batch. Also emit trajectory_request with optional total_episodes, duplicate_policy '
                '(distinct by default, allow or replay only on an explicit repeated-trajectory request), '
                'duplicate_intent (exact source clause for allow or replay), reference_run (explicit previously accepted run label), '
                'and distinct_dimensions if the user specifies required differences (background, support, layout, distractors, lighting). '
                'If the user requires objects, dynamics or constraints that cannot be represented '
                'in these fields and in the supported task definition, return unsupported with the missing requirement. '
                'Scene counts count distinct configurations only. At "采集5条轨迹" or "record 5 episodes", set '
                'trajectory_request.total_episodes=5 and omit successes_per_scene. '
                'At "五个不同场景采五条", return scene_count=5, trajectory_request.total_episodes=5. '
                'At "一个场景采5条", return scene_count=1 and trajectory_request.total_episodes=5. '
                'Use successes_per_scene only when the user explicitly asks for that many per scene. '
                'Repeated actions within an episode do not allow duplicate episodes. '
                'The phone_packaging assets also support a NEW partial goal made from this ordered relation list: '
                'inside phone receiver, then inside insert receiver, then covers lid receiver. For an explicitly narrower goal '
                '(e.g., only place the phone inside the box base), select the asset template phone_packaging and add a goal to that task: '
                '{"effects":[{"relation":"inside","object":"phone","reference":"receiver"}],"preserve":["insert","lid"]}. '
                'Select the requested prefix and preserve any components that the user explicitly asks to leave in place. '
                'A request for a complete ordinary packaging task uses the registered task without the goal field. '
                'When the user asks to open the registered microwave door to its physical open endpoint without '
                'batch quotas, return {"status":"skill_request","robot":"g2crsb90d","skill_goal":'
                '{"skill":"open_hinged_lid","actor":"benchmark_microwave_oven_003","target_position":"open"}}. '
                'The asset determines the signed angle. Omit target_limit and angles from this semantic goal. '
                'For an explicitly different endpoint, closing direction or partial angle, report the requested effect as unsupported. '
                'For a request to lift the annotated phone and turn it exactly +15 degrees around world Z with '
                'a fixed tool position, return status=skill_request and skill_goal='
                '{"skill":"rotate_held_about_tool","actor":"benchmark_phone_000","angle_deg":15}. '
                'If the user then asks to place only the phone inside the box base while leaving the insert and lid where they are, '
                'add completion="pack_phone" to this skill_goal. Omit completion when the user ends while still holding the phone. '
                'Return unsupported for other placement requirements or a different rotation direction, angle, or axis. '
                'For unsupported actions, list the canonical missing skill names such as vacuum_pick, not an unknown task label. '
                'Numbers in asset names or identifiers never determine a count. '
                'Handoff boundary: classify the task objective separately from scene availability. Existing candidate IDs '
                'listed in the catalog include untested scenarios. Preserve any explicitly requested scene ID exactly, '
                'including IDs other than 0 and 1. The deterministic inventory and execution checks decide if it is usable. '
                'Do not reject a supported task purely because the requested scene is untested or its assets were not in this prompt.')
            prompt+=(
                ' The only allowed keys inside scene_request are '+json.dumps(sorted(SCENE_REQUEST_FIELDS))+
                '. The variation entry "distractors" means add one small library decoration in a clear tabletop margin. '
                'For this generic decoration request, use that variation entry alone without adding any field. '
                'Add "layout" when the instruction explicitly asks to rearrange or reposition task objects. '
                'If the instruction needs another quantity, a named decoration, or exact placement, return unsupported.')
            if scene_constraints is not None or episode_quota is not None:
                prompt+=(
                    ' The structured fields supplied alongside the instruction are '
                    +json.dumps({'scene_request':scene_constraints,'successes_per_scene':episode_quota},ensure_ascii=False)+
                    '. Use them for unspecified details. Report a conflict only when the natural-language instruction '
                    'explicitly contradicts a field. Keep scene counts and successful episode quotas independent.')
            if repair:
                prompt+=' The previous model response failed schema validation. Repair it without changing explicit requirements. Validation evidence: '+json.dumps(repair,ensure_ascii=False)
        result,receipt=self.request('chat/completions',{'model':'qwen-plus','messages':[{'role':'system','content':prompt},{'role':'user','content':instruction}], 'response_format':{'type':'json_object'},'temperature':0.1,'enable_thinking':False})
        plan=json.loads(result['choices'][0]['message']['content'])
        archive=self.audit_directory/'plan-responses';archive.mkdir(exist_ok=True)
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
