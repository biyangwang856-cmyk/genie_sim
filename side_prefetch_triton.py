"""Fetch the exact pinned Triton wheel from PyPI with verified bounded ranges."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import hashlib,json,threading,time,urllib.request
ROOT=Path('/data/by/genie_sim/dockerhub-sidebuild-20261119')
STATE={'status':'resolving_pypi','completed_bytes':0};LOCK=threading.Lock()
def save():
    with LOCK:
        p=ROOT/'triton-prefetch.tmp';p.write_text(json.dumps(STATE,indent=2));p.replace(ROOT/'triton-prefetch.json')
def main():
    with urllib.request.urlopen('https://pypi.org/pypi/triton/3.6.0/json',timeout=30) as r:meta=json.load(r)
    wheel=next(x for x in meta['urls'] if 'cp312-cp312' in x['filename'] and 'x86_64' in x['filename'] and 'manylinux' in x['filename'])
    STATE.update(status='downloading',filename=wheel['filename'],size=wheel['size'],sha256=wheel['digests']['sha256']);save()
    parts=ROOT/'triton-parts';parts.mkdir(exist_ok=True)
    spans=[(start,min(start+8*1024*1024,wheel['size'])-1) for start in range(0,wheel['size'],8*1024*1024)]
    def fetch(span):
        start,end=span;out=parts/f'{start}-{end}.part'
        if out.exists() and out.stat().st_size==end-start+1:return out
        for attempt in range(5):
            try:
                req=urllib.request.Request(wheel['url'],headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=120) as r:
                    if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):raise RuntimeError('Range mismatch')
                    temp=out.with_suffix('.tmp')
                    with temp.open('wb') as f:
                        while chunk:=r.read(1024*1024):f.write(chunk)
                if temp.stat().st_size!=end-start+1:raise RuntimeError('Truncated part')
                temp.replace(out)
                with LOCK:STATE['completed_bytes']+=end-start+1
                save();return out
            except Exception:
                if attempt==4:raise
                time.sleep(2+attempt)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(fetch,span) for span in spans]):future.result()
    folder=ROOT/'source/data_collection/docker/wheels';folder.mkdir(exist_ok=True)
    target=folder/wheel['filename'];h=hashlib.sha256()
    with target.open('wb') as f:
        for start,end in spans:
            data=(parts/f'{start}-{end}.part').read_bytes();h.update(data);f.write(data)
    if h.hexdigest()!=wheel['digests']['sha256']:raise RuntimeError('PyPI wheel SHA-256 mismatch')
    STATE.update(status='verified',wheel=str(target),completed_bytes=wheel['size']);save()
    print(json.dumps(STATE),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:STATE.update(status='failed',error=str(exc));save();raise
