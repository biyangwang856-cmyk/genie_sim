import importlib.util,os,sys
from pathlib import Path
source=Path(os.environ['GENIE_RUN_SOURCE'])
for name in ['insertion_physics','robot_adapters','skill_registry','task_geometry','qwen_backend','structured_planner','skill_executor','collection_programs']:
    spec=importlib.util.spec_from_file_location(name,source/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
execute_task=sys.modules['collection_programs'].execute_task
