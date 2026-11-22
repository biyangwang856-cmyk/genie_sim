"""Import the same public image via verified local blobs when range resume finishes first."""
from pathlib import Path
import datetime,json,os,signal,subprocess,time,urllib.request
ROOT=Path('/data/by/genie_sim/dockerhub-sidebuild-20261119')
os.environ['DOCKER_HOST']='unix://'+str(ROOT/'docker.sock')

def main():
    while True:
        pull=json.loads((ROOT/'pull-state.json').read_text())
        if pull['status']=='pulled':print('Original Docker pull completed first');return
        state=json.loads((ROOT/'parallel-fetch-state.json').read_text()) if (ROOT/'parallel-fetch-state.json').exists() else {}
        if state.get('status')=='failed':raise RuntimeError('Parallel fetch failed; original Docker pull remains active')
        if state.get('status')=='all_blobs_verified':break
        time.sleep(5)
    # Only terminate this project's old supervisor and its exact pull child.
    for key,name in (('monitor_pid','pull_monitor.py'),('download_pid',None)):
        pid=pull.get(key)
        if not pid or not (Path('/proc')/str(pid)).exists():continue
        cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
        valid=cmd=='python3 '+str(ROOT/name) if name else cmd=='docker pull --platform linux/amd64 yonhdee/genie_sim:geniesim4'
        if not valid:raise RuntimeError('Unexpected project process identity')
        os.kill(pid,signal.SIGTERM)
    pull.update(status='importing_verified_blobs',transport='Verified Docker Hub HTTP Range blobs; loopback Docker registry')
    (ROOT/'pull-state.json').write_text(json.dumps(pull,indent=2))
    with (ROOT/'verified-registry.log').open('w') as log:
        registry=subprocess.Popen(['python3',str(ROOT/'side_verified_registry.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (ROOT/'verified-registry.pid').write_text(str(registry.pid))
    for _ in range(30):
        try:
            with urllib.request.urlopen('http://127.0.0.1:5009/v2/',timeout=2) as r:
                if r.status==200:break
        except OSError:time.sleep(1)
    else:raise RuntimeError('Verified loopback registry did not start')
    local='127.0.0.1:5009/by-geniesim/base:geniesim4'
    with (ROOT/'verified-local-import.log').open('w') as log:
        result=subprocess.run(['docker','pull',local],stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    if result.returncode:raise RuntimeError('Verified local import failed')
    details=json.loads(subprocess.check_output(['docker','image','inspect',local]))[0]
    manifest=json.loads((ROOT/'image-manifest.json').read_text())
    if details['Id']!=manifest['config']['digest']:raise RuntimeError('Imported image config digest differs from upstream')
    subprocess.run(['docker','tag',local,'yonhdee/genie_sim:geniesim4'],check=True)
    pull.update(status='pulled',image_id=details['Id'],repo_digests=details.get('RepoDigests'),size_bytes=details['Size'],architecture=details['Architecture'],os=details['Os'],
        upstream_manifest_digest=json.loads((ROOT/'verified-image-origin.json').read_text())['digest'],finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (ROOT/'base-image-inspect.json').write_text(json.dumps(details,indent=2))
    (ROOT/'pull-state.json').write_text(json.dumps(pull,indent=2));print(json.dumps(pull),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        (ROOT/'verified-import-error.json').write_text(json.dumps({'error':str(exc)},indent=2));raise
