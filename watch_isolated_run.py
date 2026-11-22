"""Wait for one isolated simulator and audit its original recording."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from batch_collect import audit_recording
from collection_control import ROOT, SOCKET, STORAGE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if run.parent != STORAGE/'collection-runs':
        parser.error('Run outside the project collection root')
    launch = json.loads((run/'launch.json').read_text())
    if launch.get('collection_slot') not in ('primary','secondary') or launch.get('container') != 'genie-by-'+run.name:
        parser.error('Launch receipt does not identify a project collection')

    env = dict(os.environ, DOCKER_HOST=SOCKET)
    container = launch['container']
    deadline = time.monotonic() + 4500
    while time.monotonic() < deadline:
        running = subprocess.check_output(['docker', 'inspect', '--format', '{{.State.Running}}', container],
                                          env=env, text=True).strip()
        if running == 'false':
            break
        time.sleep(10)
    else:
        raise TimeoutError(f'Simulator is still running after the audit wait budget: {container}')

    if not (run/'collection_result.json').is_file():
        raise RuntimeError(f'Simulator exited without a collection result: {run}')
    archived, result = audit_recording(run)
    print(json.dumps({'run': str(run), 'archived': archived, 'automatic_acceptance': result['accepted'],
                      'frames': result.get('frames'), 'errors': result.get('errors', [])}), flush=True)
    if archived and result['accepted'] and not (run/'review'/'review-frames.json').is_file():
        subprocess.run([sys.executable, str(ROOT/'review_hardpoint.py'), str(run)], check=True)


if __name__ == '__main__':
    main()
