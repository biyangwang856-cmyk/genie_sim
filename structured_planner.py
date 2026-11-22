"""Live Qwen skill planning with bounded calls and archived, validated responses."""
import hashlib,json,time
import os
from pathlib import Path
from qwen_backend import QwenAPI
from skill_registry import SKILLS,completed_packing_objects,validate_plan
from robot_adapters import UnsupportedRequest

class PlannerBudgetExceeded(RuntimeError):
    def __init__(self,reason):super().__init__('LLM planning budget exhausted: '+reason);self.reason=reason

class StructuredPlanner:
    def __init__(self,adapter,spec,folder,instruction):
        self.adapter,self.spec,self.folder,self.instruction=adapter,spec,folder,instruction
        self.api=QwenAPI();self.calls=0;self.max_calls=3;self.max_total_tokens=22000;self.tokens=0;self.started=time.monotonic()
        self.prepared_used=False;self.planning_elapsed_s=0.

    def plan(self,state,*,failure=None):
        prepared=Path(os.environ.get('GENIE_RUN_SOURCE','/nonexistent'))/'prepared-plan.json'
        if not self.prepared_used and prepared.is_file() and failure is None:
            self.prepared_used=True;archive=json.loads(prepared.read_text())
            if archive.get('validated') is not True or archive.get('instruction')!=self.instruction:raise ValueError('Prepared plan provenance mismatch')
            plan=json.loads(archive['raw_response'])
            validate_plan(plan,self.spec,self.adapter,held=state['holding_by_role'],achieved=state['achieved'])
            digest=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
            if digest!=archive['plan_sha256']:raise ValueError('Prepared plan content digest mismatch')
            archive.update(runtime_revalidated=True,runtime_initial_state=state)
            (self.folder/'llm-plan-prepared.json').write_text(json.dumps(archive,ensure_ascii=False,indent=2))
            self.max_calls=2
            return plan,digest
        if self.calls>=self.max_calls:raise PlannerBudgetExceeded('maximum_live_calls')
        if self.tokens>=self.max_total_tokens:raise PlannerBudgetExceeded('token_budget')
        if self.planning_elapsed_s>900:raise PlannerBudgetExceeded('cumulative_api_wall_time')
        self.calls+=1
        reference={
            'insert_building_block':'Grasp the workpiece, move through geometry targets shape_approach, shape_entry_high, shape_entry, align_precision at shape_entry, release gently, retreat empty hand, wait for stability, verify_task. A precise pose alone is not insertion success.',
            'phone_packaging':'Pack phone, inner insert, then lid. For each object grasp, use its packing_i_approach then packing_i_contact targets, release, retreat, wait, verify_relation packing_stage_i. End with verify_task.',
            'put_the_cap_on_the_pen_new':'Use operator for pen and assistant for cap. Grasp both, present pen, move_held cap to cap_preinsert, MUST align_precision at cap_preinsert, then insert_axial using cap_insert_axis, verify_relation cap_seated, release cap while retaining pen, retreat assistant, sync_assembly_proxy on pen+cap, return pen to pen_return, release pen, retreat operator, wait, verify_task.'
        }[self.spec['template']]
        first=self.spec['manipulated_objects'][0];role=next(iter(self.spec['role_bindings']))
        example={'version':1,'robot_id':self.adapter.robot_id,'task_id':self.spec['task_id'],'scene_id':self.spec['scene_id'],
            'role_bindings':self.spec['role_bindings'],'steps':[{'id':'s01','skill':'grasp','parameters':{'object':first,'role':role,'grasp_profile':self.spec['grasp_profiles'][first][0]}}]}
        context={'robot_capabilities':self.adapter.describe(),'task':self.spec,'skill_registry':SKILLS,
            'current_measured_state':state,'failure':failure,'reference_strategy':reference,'output_example':example,
            'parameter_enums':{'role':list(self.spec['role_bindings']),'target':self.spec['targets'],'relation':self.spec['relations'],'mode':['native','gentle']}}
        if failure:
            facts=set(state['achieved']);holding=state['holding_by_role']
            context['recovery_constraints']={'currently_held':holding,
                'previously_released_objects':[obj for obj in self.spec['manipulated_objects'] if 'released:'+obj in facts],
                'verified_relations':sorted(f.removeprefix('relation:') for f in facts if f.startswith('relation:')),
                'instruction':'Start from currently_held and verified_relations. If a held-object motion failed, keep using that held object from its measured pose. Never start by grasping it again. Resume from the failed step or a feasible alternative. Preserve relations already verified unless the failure specifically invalidated one.'}
            if self.spec['template'] == 'phone_packaging':
                packed = completed_packing_objects(self.spec, facts)
                remaining = [{'object': obj, 'approach': f'packing_{index}_approach',
                              'contact': f'packing_{index}_contact', 'relation': f'packing_stage_{index}'}
                             for index, obj in enumerate(self.spec['manipulated_objects']) if obj not in packed]
                context['recovery_constraints'].update(completed_packing_objects=sorted(packed), remaining_packing_stages=remaining)
                context['reference_strategy'] = ('Preserve completed_packing_objects in place. For each remaining stage, '
                    'grasp if needed, move through approach and contact from the measured state, release, retreat, '
                    'wait_stable, then verify_relation for that component. Perform those final three steps for an already '
                    'released component whose relation still needs checking. Finish with verify_task.')
            if self.spec['template']=='put_the_cap_on_the_pen_new' and 'relation:cap_seated' in facts:
                cap=self.spec['bindings']['cap'];pen=self.spec['bindings']['pen']
                cap_status=(
                    'The assistant still holds the seated cap. Release it once, then withdraw the empty arm.'
                    if holding.get('assistant')==cap else
                    'The assistant already released the cap and withdrew. Omit every assistant operation.'
                    if 'cap_assistant_retreated' in facts else
                    'The assistant is empty after cap release. Withdraw assistant once. Omit regrasping and insertion.'
                )
                pen_status=(
                    f'The operator holds {pen}. If the assembly proxy is not yet synchronized, use '
                    f'sync_assembly_proxy with object={pen}, reference={cap}, role=operator. Then move the '
                    'assembled pen to pen_return unless already there, and release it once.'
                    if holding.get('operator')==pen else
                    'The operator is already empty. No new grasp, move_held, or release is needed.'
                )
                context['reference_strategy']=(
                    'The cap seating was measured and verified. Preserve it. '+cap_status+' '+pen_status+
                    ' After pen release, retreat with role=operator only if cap_operator_retreated is absent. '
                    f' When cap_pen_stable is absent wait_stable with object={pen}, '
                    'then verify_task. Omit only operations whose corresponding facts are already achieved.'
                )
            context['output_example']={**example,'steps':[]}
        prompt=('Produce an executable JSON skill plan for ONLY the task and scene in this context; other tasks in the user request have separate episodes. Use only this registry, the exact object IDs, role bindings, and named geometry targets. '
            'The executor runs your steps in their supplied order. Do not return a task-name selector or executable code. '
            'Do not invent positions, axes, asset IDs, skills, physical success labels, or new robot capabilities. '
            'Geometry and independent physical validators supply precision and success. '
            'When the task specifies release_modes, use the exact release mode for each object in both initial and recovery plans. '
            'The steps list must be nonempty. Every step has exactly id, skill, parameters; parameters exactly match the skill contract. '
            'parameters.role MUST be a logical role key (operator or assistant), NEVER left/right. Physical left/right values occur ONLY inside role_bindings. '
            'Preserve all required operations and official task arm constraints. If recovering, plan only the remaining work from measured state and achieved facts. '
            'Do not regrasp an already held object. A release requires held object. Finish with verify_task after all required objects have been released. '
            'Unsupported requests return {"status":"unsupported","reason":"...","missing_assets":[],"missing_skills":[],"missing_robot_capabilities":[]}. '
            'Return JSON only. Context: '+json.dumps(context,ensure_ascii=False))
        assignment={'original_user_request':self.instruction,
            'orchestrator_assignment':{'task_id':self.spec['task_id'],'scene_id':self.spec['scene_id'],'episodes_this_plan':1,'instruction':self.spec['official_instruction']},
            'scope':'Plan only this assigned episode. Other requested tasks, scenes and quotas are handled by separate orchestrator assignments; they are not missing requirements of this episode.'}
        api_started=time.monotonic()
        try:
            result,receipt=self.api.request('chat/completions',{'model':'qwen-plus','messages':[{'role':'system','content':prompt},{'role':'user','content':json.dumps(assignment,ensure_ascii=False)}],
                'response_format':{'type':'json_object'},'temperature':0.,'max_tokens':6500,'enable_thinking':False})
        finally:self.planning_elapsed_s+=time.monotonic()-api_started
        self.tokens+=int((receipt.get('usage') or {}).get('total_tokens',0))
        text=result['choices'][0]['message']['content']
        archive={'instruction':self.instruction,'assignment':assignment,'model':'qwen-plus','receipt':receipt,'context':context,'raw_response':text,'validated':False}
        path=self.folder/('llm-plan-'+str(self.calls)+'.json')
        try:
            plan=json.loads(text)
            if plan.get('status')=='unsupported':
                raise UnsupportedRequest(plan.get('reason','Unsupported plan'),missing_assets=plan.get('missing_assets',()),missing_skills=plan.get('missing_skills',()),missing_capabilities=plan.get('missing_robot_capabilities',()))
            validate_plan(plan,self.spec,self.adapter,held=state['holding_by_role'],achieved=state['achieved'])
            archive['plan']=plan;archive['validated']=True
            archive['plan_sha256']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        except Exception as exc:
            archive['validation_error']=str(exc);path.write_text(json.dumps(archive,ensure_ascii=False,indent=2));raise
        path.write_text(json.dumps(archive,ensure_ascii=False,indent=2))
        return plan,archive['plan_sha256']
