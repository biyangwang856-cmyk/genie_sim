"""Small executable skill contract, independent of task scripts and robot DOF count."""
from robot_adapters import UnsupportedRequest

SKILLS={
 'grasp':{'parameters':['object','role','grasp_profile'],'requires':['friction_grasp','curobo'],
   'precondition':'role is empty; object exists and has compatible grasp geometry',
   'success':'measured physical lift and native grasp completion',
   'failures':['missing_grasp','unreachable','collision','tracking_error','grasp_slip'],
   'recovery':['alternate_grasp_candidate','replan_remaining_steps']},
 'move_held':{'parameters':['object','role','target'],'requires':['held_object_motion','curobo'],
   'precondition':'object held by role; target resolved from current scene geometry',
   'success':'native motion and measured object pose',
   'failures':['unreachable','collision','tracking_error','grasp_slip'],
   'recovery':['alternate_motion_candidate','local_pose_correction','replan_remaining_steps']},
 'align_precision':{'parameters':['object','role','target'],'requires':['held_object_motion','curobo'],
   'precondition':'held object near geometry-defined entry',
   'success':'full mesh projection and measured entry pose satisfy assembly profile',
   'failures':['geometry_mismatch','tracking_error','grasp_slip'],
   'recovery':['bounded_pose_feedback','replan_remaining_steps']},
 'insert_axial':{'parameters':['object','reference','role','target'],'requires':['bimanual','held_object_motion'],
   'precondition':'two objects held by different roles; axes and mouth aligned',
   'success':'measured insertion relation and official assembly condition',
   'failures':['misalignment','contact_blocked','assembly_depth_not_reached'],
   'recovery':['withdraw_and_realign','replan_remaining_steps']},
 'release':{'parameters':['object','role','mode'],'requires':['gripper_opening_control'],
   'precondition':'role holds object; release condition checked',
   'success':'fingers reach open target; object physically separates',
   'failures':['release_disturbance','gripper_tracking_error'],
   'recovery':['slower_release_profile','restart_from_initial_state']},
 'retreat':{'parameters':['role'],'requires':['curobo'],
   'precondition':'role empty','success':'measured empty-hand withdrawal',
   'failures':['collision','tracking_error'],'recovery':['alternate_motion_candidate']},
 'wait_stable':{'parameters':['object'],'requires':[],
   'precondition':'object exists','success':'bounded physical settling; no pose setting',
   'failures':['unstable'],'recovery':['wait_within_budget','replan_remaining_steps']},
 'verify_relation':{'parameters':['object','relation'],'requires':[],
   'precondition':'geometry contract available','success':'independent geometric/physical relation checker',
   'failures':['relation_not_satisfied'],'recovery':['replan_remaining_steps']},
 'sync_assembly_proxy':{'parameters':['object','reference','role'],'requires':['curobo'],
   'precondition':'assembly verified; primary held; other role released',
   'success':'planner collision proxy includes assembled objects; no physical weld created',
   'failures':['planner_attachment_failed'],'recovery':['rebuild_collision_proxy']},
 'verify_task':{'parameters':[],'requires':[],
   'precondition':'all required objects released; complete continuous run',
   'success':'independent full-task verifier; not LLM judgment',
   'failures':['physical_task_incomplete','recording_invalid'],'recovery':['restart_from_initial_state']},
}

def completed_packing_objects(spec, achieved):
    if spec['template'] != 'phone_packaging':
        return set()
    facts = set(achieved)
    return {obj for index, obj in enumerate(spec['manipulated_objects'])
            if f'relation:packing_stage_{index}' in facts and 'released:' + obj in facts}


def validate_plan(plan,spec,adapter,*,held=None,achieved=()):
    if not isinstance(plan,dict) or set(plan)!={'version','robot_id','task_id','scene_id','role_bindings','steps'}:
        raise ValueError('Unexpected or missing plan fields')
    if plan['version']!=1 or plan['robot_id']!=adapter.robot_id or plan['task_id']!=spec['task_id'] or str(plan['scene_id'])!=spec['scene_id']:
        raise ValueError('Plan identity mismatch')
    bindings=plan['role_bindings']
    if not isinstance(bindings,dict) or bindings!=spec['role_bindings']:raise ValueError('Role binding violates official task arm constraints')
    for role in bindings:adapter.resolve_role(role,bindings)
    steps=plan['steps']
    if not isinstance(steps,list) or not 1<=len(steps)<=60:raise ValueError('Plan step budget exceeded')
    holding=dict(held or {});known=set(spec['objects']);seen=set();facts=set(achieved)
    packed=completed_packing_objects(spec,facts)
    cap_task=spec['template']=='put_the_cap_on_the_pen_new'
    packing_relations = {}
    if spec['template'] == 'phone_packaging':
        packing_relations = {f'packing_stage_{index}': obj for index, obj in enumerate(spec['manipulated_objects'])}
    for step in steps:
        if not isinstance(step,dict) or set(step)!={'id','skill','parameters'}:raise ValueError('Invalid step fields')
        if not isinstance(step['id'],str) or step['id'] in seen:raise ValueError('Duplicate step ID')
        seen.add(step['id']);skill=step['skill'];params=step['parameters']
        if skill not in SKILLS:raise UnsupportedRequest('Skill is not registered',missing_skills=[skill])
        contract=SKILLS[skill];adapter.require(contract['requires'])
        if not isinstance(params,dict) or set(params)!=set(contract['parameters']):raise ValueError('Skill parameter mismatch: '+skill)
        for field in ['object','reference']:
            if field in params and params[field] not in known:raise UnsupportedRequest('Unknown object binding',missing_assets=[params[field]])
        role=params.get('role');obj=params.get('object')
        if role is not None:
            if role not in bindings:raise ValueError('parameters.role must be a logical role key '+str(list(bindings))+'; physical arm names belong only in role_bindings')
            adapter.resolve_role(role,bindings)
        if 'target' in params and params['target'] not in spec['targets']:raise ValueError('Unknown geometry target')
        if 'target' in params and spec['target_objects'].get(params['target'])!=obj:raise ValueError('Geometry target bound to a different object')
        if 'target' in params and 'skill_targets' in spec and params['target'] not in spec['skill_targets'].get(skill,[]):
            raise UnsupportedRequest('Skill has no implementation for this target',missing_skills=[skill+':'+params['target']])
        if skill=='grasp':
            if obj in packed:
                raise ValueError('Recovery tries to regrasp a physically verified packing component: '+obj)
            if cap_task and obj==spec['bindings']['cap'] and 'relation:cap_seated' in facts:
                raise ValueError('The cap is seated. Continue the remaining operations while retaining the assembly.')
            if role in holding or obj in holding.values():raise ValueError('Grasp requires empty arm and unheld object')
            if params['grasp_profile'] not in spec['grasp_profiles'].get(obj,[]):raise UnsupportedRequest('Unsupported grasp profile',missing_skills=[params['grasp_profile']])
            holding[role]=obj
            facts.discard('released:'+obj)
            for ref,owner in spec['target_objects'].items():
                if owner==obj:facts.discard('target:'+ref)
            if spec['template']=='insert_building_block':facts.discard('shape_aligned')
        elif skill in ['move_held','align_precision','insert_axial']:
            if holding.get(role)!=obj:raise ValueError('Motion requires the specified role to hold object')
            if skill=='move_held':
                if cap_task and params['target']=='pen_return' and 'assembly_proxy_synced' not in facts:
                    raise ValueError('Returning the pen requires the native planner proxy for the assembled pen and cap')
                facts.add('target:'+params['target'])
            if skill=='align_precision':
                target=params['target'];precondition=target.replace('_contact','_approach') if target.startswith('packing_') else target
                if 'target:'+precondition not in facts:raise ValueError('Precision alignment requires a measured approach to its target')
                facts.add('target:'+target)
                if target=='cap_preinsert':facts.add('cap_aligned')
                elif target=='shape_entry':facts.add('shape_aligned')
            if skill=='insert_axial':
                if params['reference'] not in holding.values() or len(holding)<2:raise ValueError('Axial assembly requires two held objects')
                if 'cap_aligned' not in facts:raise ValueError('Axial assembly requires precision alignment')
                facts.add('assembled')
        elif skill=='release':
            if holding.get(role)!=obj:raise ValueError('Release requires held object')
            if params['mode'] not in ['native','gentle']:raise ValueError('Unknown release mode')
            required_mode = spec.get('release_modes', {}).get(obj)
            if required_mode and params['mode'] != required_mode:
                raise ValueError(f'Release of {obj} requires mode {required_mode}')
            if spec['template']=='insert_building_block' and ('shape_aligned' not in facts or params['mode']!='gentle'):
                raise ValueError('Shape release requires precision alignment and the gentle release skill')
            if spec['template']=='phone_packaging':
                index=spec['manipulated_objects'].index(obj)
                if 'target:packing_'+str(index)+'_contact' not in facts:raise ValueError('Packaging release requires its contact target')
            if cap_task and 'assembled' not in facts:raise ValueError('Release before assembly')
            if cap_task and obj==spec['bindings']['pen'] and 'target:pen_return' not in facts:
                raise ValueError('Return the assembled pen to the table before releasing it')
            holding.pop(role);facts.add('released:'+obj)
        elif skill=='retreat':
            if role in holding:raise ValueError('Retreat skill expects an empty arm')
            if cap_task and 'relation:cap_seated' in facts:
                complete='cap_'+role+'_retreated'
                if complete in facts:raise ValueError('The role has already withdrawn. Omit that retreat: '+role)
                facts.add(complete)
        elif skill=='wait_stable' and cap_task and obj==spec['bindings']['pen']:
            if 'released:'+obj not in facts or 'cap_operator_retreated' not in facts:
                raise ValueError('Check pen stability after releasing it and withdrawing the operator')
            facts.add('cap_pen_stable')
        elif skill=='verify_relation':
            relation = params['relation']
            if relation not in spec['relations']:raise ValueError('Unknown relation validator')
            if relation in packing_relations:
                if packing_relations[relation] != obj or 'released:' + obj not in facts:
                    raise ValueError('Packing relation requires its released component: '+relation)
                facts.add('relation:' + relation)
                packed.add(obj)
            if cap_task and relation=='cap_seated':
                if 'assembled' not in facts or obj!=spec['bindings']['cap']:raise ValueError('Cap seating requires the assembled cap')
                facts.add('relation:cap_seated')
        elif skill=='sync_assembly_proxy':
            if cap_task and (obj!=spec['bindings']['pen'] or params['reference']!=spec['bindings']['cap'] or role!='operator'):
                raise ValueError('Assembly proxy object must be the held pen, reference the seated cap, and role operator')
            if 'assembled' not in facts or holding.get(role)!=obj or params['reference'] in holding.values():raise ValueError('Invalid assembly proxy synchronization')
            if cap_task and 'cap_assistant_retreated' not in facts:
                raise ValueError('Verify cap seating and withdraw the empty assistant before synchronizing the assembly proxy')
            facts.add('assembly_proxy_synced')
        elif skill=='verify_task':
            if step is not steps[-1] or holding:raise ValueError('Task verification must finish the plan with released arms')
            if cap_task and 'cap_pen_stable' not in facts:
                raise ValueError('The cap task needs an explicit pen stability check after both arms withdraw')
    if steps[-1]['skill']!='verify_task':raise ValueError('Plan lacks final independent task verification')
    missing={'released:'+o for o in spec['manipulated_objects']}-facts
    if missing:raise ValueError('Plan omits required object operations: '+str(sorted(missing)))
    missing_relations = {'relation:' + relation for relation in packing_relations} - facts
    if missing_relations:
        raise ValueError('Plan omits stage verification: '+str(sorted(missing_relations)))
    return plan
