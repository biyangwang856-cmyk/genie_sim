"""Public identifiers shared by the platform and the verified FineManip planner."""

from task_plugins import registered

BUILTIN_TASKS = ("insert_building_block", "phone_packaging", "put_the_cap_on_the_pen_new")
CONTRACTS = registered()
TASKS = list(BUILTIN_TASKS) + [task for task, info in CONTRACTS.items()
                             if task not in BUILTIN_TASKS and not info.get("dynamic")]
DYNAMIC_TASKS = [task for task, info in CONTRACTS.items() if info.get("dynamic")]
SCENES = ("0", "1")
DEFAULT_ROBOT = "g2crsb90d"
