"""Run a frozen cross-scene acceptance matrix through the native platform workflow."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock

from batch_collect import write
from collection_control import COLLECTION_SLOTS
from genie_extensions import Registry
from natural_backend import submit
from robot_adapters import ROOT, UnsupportedRequest
from scene_admission import pinned_code
from scene_contract import content_id, read_json, require_id


def prepare(config, state_path, state):
    registry = Registry()
    for case in config["cases"]:
        case_id = require_id(case["id"])
        if case_id in state["groups"]:
            continue
        request = registry.validate("isaac_physx", case["request"])
        try:
            result = submit(request["instruction"], robot_id=request["robot_id"],
                            task_ids=request.get("task_ids"), scenes=request["scenes"],
                            scene_options=request.get("scene_request"),
                            successes_per_scene=request.get("successes_per_scene"),
                            trajectory_options=request.get("trajectory_request"), execute=False)
        except UnsupportedRequest as exc:
            result=exc.details
        except (ValueError, RuntimeError, OSError) as exc:
            result={"status":"preparation_failed","reason":str(exc)}
        state["groups"][case_id]=result
        write(state_path,state)


def collect(cases, slot, log_root, publish):
    for label, result in cases:
        if not result.get("job_path"):
            continue
        path=Path(result["job_path"])
        env=dict(os.environ,GENIE_COLLECTION_SLOT=slot)
        with (log_root/(label+".log")).open("a",encoding="utf-8") as output:
            worker=subprocess.run([sys.executable,str(ROOT/"batch_collect.py"),"--job",str(path),
                                   "--slot",slot,"--wait-for-worker"],
                                  env=env,stdout=output,stderr=subprocess.STDOUT)
        job=read_json(path)
        publish(label, {
            "status":job["status"] if not worker.returncode else "worker_error",
            "native_status":job["status"],"exit_code":worker.returncode,
            "job_path":str(path),"scene_outcome":job.get("scene_outcome"),
            "success_counts":job.get("success_counts",{}),
            "launched_attempts":sum(bool(row.get("container")) for row in job["attempts"]),
            "finished_launched_attempts":sum(bool(row.get("container") and row.get("finished_unix")) for row in job["attempts"]),
            "admitted_motion_hashes":[item["fingerprint"] for item in job["attempts"] if item.get("accepted")],
            "dataset":job.get("dataset"),"partial_dataset":job.get("partial_dataset"),
            "raw_trajectory_path":job.get("raw_trajectory_path"),
            "metrics_path":str(path.parent/"metrics.json") if (path.parent/"metrics.json").is_file() else None,
        })


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix",type=Path)
    parser.add_argument("--run",required=True,type=require_id)
    parser.add_argument("--slots",nargs="+",choices=COLLECTION_SLOTS,default=["secondary"])
    parser.add_argument("--tasks",nargs="+",help="Validate a subset of tasks in an independent run")
    parser.add_argument("--cases",nargs="+",help="Run only the named frozen scene cases")
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    if len(set(args.slots))!=len(args.slots):
        parser.error("Distinct slots required")
    directory=ROOT/"jobs"/args.run
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/"matrix.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_matrix(args,directory)


def run_matrix(args,directory):
    config=read_json(args.matrix)
    if args.tasks:
        available={case["request"]["task_id"] for case in config["cases"]}
        if set(args.tasks)-available:
            raise ValueError("Validation tasks have no cases in the matrix")
        config["cases"]=[case for case in config["cases"] if case["request"]["task_id"] in args.tasks]
    if args.cases:
        if set(args.cases)-{case["id"] for case in config["cases"]}:
            raise ValueError("Validation case is absent from the selected matrix")
        config["cases"]=[case for case in config["cases"] if case["id"] in args.cases]
    if not config["cases"]:
        raise ValueError("Validation matrix is empty")
    state_path=directory/"validation.json"
    from scene_contract import digest
    frozen={"matrix_sha256":content_id(config),"code_sha256":pinned_code(),
            "validation_runner_sha256":digest(__file__),"slots":args.slots}
    requested={}
    for case in config["cases"]:
        request=case["request"]
        per_scene=request.get("successes_per_scene", 1)
        scenes=request.get("scene_request", {})
        requested[case["id"]]=request.get("trajectory_request", {}).get(
            "total_episodes", per_scene * scenes.get("scene_count", len(scenes.get("scene_ids", [])) or 1))
    if len(requested)!=len(config["cases"]):
        raise ValueError("Duplicate matrix case IDs")
    if state_path.is_file():
        state=read_json(state_path)
        if any(state.get(key)!=value for key,value in frozen.items()):
            raise ValueError("Code or scene matrix changed mid-validation")
    else:
        state={**frozen,"status":"preparing","groups":{},"target_episodes":sum(requested.values())}
        write(state_path,state)
    prepare(config,state_path,state)
    state["status"]="prepared" if args.prepare_only else "running"
    write(state_path,state)
    if args.prepare_only:
        return
    cases=list(state["groups"].items())
    publication=Lock()

    def publish(label, result):
        with publication:
            state["groups"][label]=result
            write(state_path,state)

    with ThreadPoolExecutor(max_workers=len(args.slots)) as threads:
        futures=[threads.submit(collect,cases[index::len(args.slots)],slot,directory,publish)
                 for index,slot in enumerate(args.slots)]
        for child in futures:
            child.result()
    accepted=sum(sum(sum(counts.values()) for counts in group.get("success_counts",{}).values())
                 for group in state["groups"].values())
    motions=[motion for row in state["groups"].values() for motion in row.get("admitted_motion_hashes",[])]
    verified=sum((row.get("dataset") or {}).get("episodes",0) for row in state["groups"].values())
    completed=(all(row["status"]=="complete" for row in state["groups"].values())
               and verified==accepted==state["target_episodes"]==len(set(motions)))
    state.update(status="complete" if completed else "partial" if accepted else "failed",
                 independently_admitted=accepted,quotas_exported_and_loader_verified=verified,
                 distinct_motion_count=len(set(motions)))
    write(state_path,state)
    print(json.dumps(state,ensure_ascii=False))


if __name__=="__main__":
    main()
