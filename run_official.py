"""Start the upstream benchmark with explicit graphics-device and cache settings."""
from pathlib import Path
import os
import runpy
import faulthandler
import importlib

import isaacsim


OriginalSimulationApp = isaacsim.SimulationApp
faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)


def launch(config=None, *args, **kwargs):
    config = dict(config or {})
    device = int(os.environ.get('GENIE_PROJECT_GPU', '1'))
    config.update(active_gpu=device, physics_gpu=device, multi_gpu=False,
                  renderer='RaytracedLighting')
    config['extra_args'] = [*config.get('extra_args', []),
                            '--/rtx/hydra/supportMultiTickRate=false']
    app = OriginalSimulationApp(config, *args, **kwargs)
    from isaacsim.core.utils import extensions
    extensions.enable_extension('omni.behavior.scripting.core')
    import sys
    import omni.behavior.scripting.core as scripting
    sys.modules.setdefault('omni.kit.scripting', scripting)
    return app


isaacsim.SimulationApp = launch
import geniesim_benchmark.app.task_manager as task_manager

OriginalTaskManager = task_manager.TaskManager

class ProjectTaskManager(OriginalTaskManager):
    def __init__(self, *args, **kwargs):
        kwargs['benchmark_factory'] = lambda config, core: importlib.import_module(
            'scripted_policy').benchmark_factory(config, core)
        super().__init__(*args, **kwargs)

task_manager.TaskManager = ProjectTaskManager
repo = Path(os.environ['SIM_REPO_ROOT'])
runpy.run_path(str(repo / 'source/geniesim_benchmark/src/geniesim_benchmark/app/app.py'),
               run_name='__main__')
