"""Resume public Docker Hub blobs from project cache with bounded parallel ranges.

All bytes are checked against the original image's SHA-256 descriptors before
the project-local registry serves them. No shared daemon is modified.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import hashlib,http.server,json,os,re,shutil,signal,subprocess,threading,time,urllib.request

ROOT=Path('/data/by/genie_sim/dockerhub-sidebuild-20261119')
CACHE=ROOT/'verified-blobs';CACHE.mkdir(exist_ok=True)
os.environ['DOCKER_HOST']='unix://'+str(ROOT/'docker.sock')
MANIFEST=json.loads((ROOT/'image-manifest.json').read_text())
STATE={'status':'collecting_cached_blobs','source':'docker.io/yonhdee/genie_sim:geniesim4','completed':0}
LOCK=threading.Lock();TOKEN={};TOKEN_LOCK=threading.Lock()

def save():
    with LOCK:
        p=ROOT/'parallel-fetch-state.tmp'
        p.write_text(json.dumps(STATE,indent=2));p.replace(ROOT/'parallel-fetch-state.json')

def token():
    with TOKEN_LOCK:
        if TOKEN.get('until',0)<time.time():
            url='https://auth.docker.io/token?service=registry.docker.io&scope=repository:yonhdee/genie_sim:pull'
            with urllib.request.urlopen(url,timeout=30) as r:d=json.load(r)
            TOKEN.update(value=d['token'],until=time.time()+min(d.get('expires_in',300),240))
        return TOKEN['value']

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return 'sha256:'+h.hexdigest()

def fetch_range(descriptor,start,end,path):
    if path.exists() and path.stat().st_size==end-start+1:return path
    temp=path.with_suffix('.incomplete')
    for attempt in range(5):
        try:
            request=urllib.request.Request('https://registry-1.docker.io/v2/yonhdee/genie_sim/blobs/'+descriptor['digest'],
                headers={'Authorization':'Bearer '+token(),'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(request,timeout=90) as r:
                if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                    raise RuntimeError('Blob endpoint did not honor the exact requested range')
                with temp.open('wb') as f:
                    while chunk:=r.read(1024*1024):f.write(chunk)
            if temp.stat().st_size!=end-start+1:raise RuntimeError('Truncated range')
            temp.replace(path)
            with LOCK:STATE['range_bytes_completed']=STATE.get('range_bytes_completed',0)+(end-start+1)
            save();return path
        except Exception:
            if attempt==4:raise
            time.sleep(2+attempt)

def fetch_blob(row,prefix=None):
    key=row['digest'].split(':',1)[1];out=CACHE/key
    if out.exists() and out.stat().st_size==row['size'] and digest(out)==row['digest']:return out
    work=CACHE/(key+'.assembling')
    prefix_size=0
    if prefix:
        prefix_size=min(prefix.stat().st_size,row['size'])
        with prefix.open('rb') as source,work.open('wb') as target:
            remaining=prefix_size
            while remaining:
                block=source.read(min(8*1024*1024,remaining))
                if not block:raise RuntimeError('Cached prefix disappeared while copying')
                target.write(block);remaining-=len(block)
    else:work.write_bytes(b'')
    STATE.update(status='fetching_ranges',current_blob=row['digest'],blob_size=row['size'],prefix_bytes=prefix_size);save()
    segments=[];chunk_size=64*1024*1024
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={}
        for start in range(prefix_size,row['size'],chunk_size):
            end=min(start+chunk_size,row['size'])-1
            part=CACHE/f'{key}.{start}.{end}.part';segments.append((start,part))
            futures[pool.submit(fetch_range,row,start,end,part)]=start
        for future in as_completed(futures):future.result()
    with work.open('ab') as f:
        for _,part in sorted(segments):
            with part.open('rb') as source:shutil.copyfileobj(source,f,8*1024*1024)
    if work.stat().st_size!=row['size'] or digest(work)!=row['digest']:
        raise RuntimeError('Full blob SHA-256 validation failed; not importing')
    work.replace(out);STATE['completed']+=1;save();return out

def main():
    rows=[MANIFEST['config'],*MANIFEST['layers']]
    unique={r['digest']:r for r in rows}
    temps=[p for p in (ROOT/'docker-data/tmp').iterdir() if p.is_file()]
    # Copy and verify complete existing Docker blobs before any cancellation.
    missing=[]
    for row in unique.values():
        target=CACHE/row['digest'].split(':',1)[1]
        if target.exists() and target.stat().st_size==row['size'] and digest(target)==row['digest']:
            STATE['completed']+=1;continue
        found=False
        for candidate in temps:
            if candidate.exists() and candidate.stat().st_size==row['size'] and digest(candidate)==row['digest']:
                shutil.copyfile(candidate,target);STATE['completed']+=1;found=True;break
        if not found:missing.append(row)
    STATE['total_blobs']=len(unique);STATE['missing_blobs']=len(missing);save()
    largest=max(MANIFEST['layers'],key=lambda x:x['size'])
    for row in sorted(missing,key=lambda x:x['size'],reverse=True):
        prefix=None
        if row['digest']==largest['digest']:
            candidates=[p for p in temps if p.exists() and 1024**3<p.stat().st_size<row['size']]
            if candidates:prefix=max(candidates,key=lambda x:x.stat().st_size)
        fetch_blob(row,prefix)
    # Keep exact manifest bytes: its digest is independently reproducible.
    auth={'Authorization':'Bearer '+token(),'Accept':'application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json'}
    request=urllib.request.Request('https://registry-1.docker.io/v2/yonhdee/genie_sim/manifests/geniesim4',headers=auth)
    with urllib.request.urlopen(request,timeout=30) as r:
        raw=r.read();declared=r.headers.get('Docker-Content-Digest')
    fetched=json.loads(raw)
    if 'manifests' in fetched:raise RuntimeError('Expected platform manifest; resolve platform explicitly before local import')
    if fetched['config']['digest']!=MANIFEST['config']['digest']:raise RuntimeError('Remote tag changed while downloading')
    sha='sha256:'+hashlib.sha256(raw).hexdigest()
    if declared and declared!=sha:raise RuntimeError('Manifest digest mismatch')
    (ROOT/'verified-image-manifest.raw.json').write_bytes(raw)
    (ROOT/'verified-image-origin.json').write_text(json.dumps({'digest':sha,'media_type':fetched.get('mediaType','application/vnd.docker.distribution.manifest.v2+json')},indent=2))
    STATE.update(status='all_blobs_verified',manifest_digest=sha);save()
    print(json.dumps(STATE),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        STATE.update(status='failed',error=str(exc));save();raise
