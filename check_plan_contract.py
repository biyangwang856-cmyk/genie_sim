"""Reject unsafe variants of a real model-generated plan before any motion."""
import copy,json,sys
from pathlib import Path
from robot_adapters import RobotAdapter
from skill_registry import validate_plan
archive=json.loads(Path(sys.argv[1]).read_text());plan=json.loads(archive['raw_response']);context=archive['context']
profile=context['robot_capabilities'];adapter=RobotAdapter(profile['robot_id'],tuple(profile['arms']),frozenset(profile['capabilities']),profile)
spec=context['task'];validate_plan(plan,spec,adapter)
checks={'real_model_plan':True}
def reject(name,mutate):
    candidate=copy.deepcopy(plan);mutate(candidate)
    try:validate_plan(candidate,spec,adapter)
    except ValueError:checks[name]=True
    else:raise AssertionError('Invalid plan admitted: '+name)
reject('physical_arm_cannot_replace_role',lambda p:p['steps'][0]['parameters'].update(role='left'))
reject('unknown_skill',lambda p:p['steps'][0].update(skill='teleport_object'))
reject('missing_release',lambda p:p.update(steps=[s for s in p['steps'] if s['skill']!='release']))
reject('motion_before_grasp',lambda p:p['steps'].__setitem__(slice(0,2),list(reversed(p['steps'][:2]))))
reject('unknown_asset',lambda p:p['steps'][0]['parameters'].update(object='invented_asset'))
reject('wrong_robot',lambda p:p.update(robot_id='unregistered_robot'))
reject('missing_independent_verifier',lambda p:p.update(steps=p['steps'][:-1]))
print(json.dumps({'passed':True,'checks':checks,'scope':'Plan contract only; not physical task success'}))
