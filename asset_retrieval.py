"""Build resumable text-embedding-v4 asset index on the server."""
from pathlib import Path
import contextlib
import hashlib
import json
import sys
from tempfile import NamedTemporaryFile
import numpy as np
from qwen_backend import ROOT,QwenAPI

INDEX_CACHE = ROOT / 'cache/text-embedding-v4'

def load_index():
    bundle = INDEX_CACHE / 'normalized-index.npz'
    if bundle.is_file():
        with np.load(bundle, allow_pickle=False) as saved:
            records = json.loads(saved['asset_metadata'].item())
            matrix = saved['embeddings']
    else:
        records = json.loads((INDEX_CACHE / 'assets.json').read_text())
        matrix = np.load(INDEX_CACHE / 'normalized.npy', allow_pickle=False)
    if not records or matrix.ndim != 2 or matrix.shape != (len(records), 1024):
        raise ValueError('Asset index rows and text-embedding-v4 vectors do not match')
    return records, matrix

def publish_index(records, matrix):
    with NamedTemporaryFile(dir=INDEX_CACHE, prefix='index-', suffix='.tmp', delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        with staged.open('wb') as stream:
            np.savez(stream, asset_metadata=json.dumps(records, ensure_ascii=False), embeddings=matrix)
        staged.replace(INDEX_CACHE / 'normalized-index.npz')
    finally:
        staged.unlink(missing_ok=True)

def catalog():
    sys.path.insert(0,str(ROOT))
    with (ROOT/'logs/asset-catalog-import.log').open('w') as log, contextlib.redirect_stdout(log):
        import GenieSimAssets as assets
    records=[]
    for key,item in sorted(assets.ASSETS_INDEX.items()):
        path=ROOT/'GenieSimAssets'/item['url']
        if not path.is_file(): continue
        desc=item.get('description',{})
        text=json.dumps({'asset_id':key,'category':str(Path(item['url']).parent.parent),
            'description':desc,'shapes':item.get('shapes',[])},ensure_ascii=False)[:6000]
        records.append({'asset_id':key,'usd':str(path),'text':text,
            'interaction':str(ROOT/'GenieSimAssets/interaction'/key/'interaction.json')})
    return records

def main():
    api=QwenAPI(); records=catalog(); cache=INDEX_CACHE;cache.mkdir(parents=True,exist_ok=True)
    if not records:
        raise ValueError('No locally available registered USD assets')
    vectors=[]
    for start in range(0,len(records),10):
        group=records[start:start+10]
        texts=[r['text'] for r in group]
        signature=hashlib.sha256(json.dumps(texts).encode()).hexdigest()
        p=cache/(signature+'.npy')
        if p.exists(): arr=np.load(p)
        else:
            value,receipt=api.embed(texts);arr=np.asarray(value,dtype=np.float32);np.save(p,arr)
        vectors.extend(arr)
        if start%100==0: print('EMBED_PROGRESS',start+len(group),'/',len(records),flush=True)
    matrix=np.asarray(vectors,dtype=np.float32)
    if matrix.shape != (len(records), 1024) or not np.isfinite(matrix).all():
        raise ValueError('Invalid text-embedding-v4 response for the local asset catalog')
    matrix/=np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-12)
    publish_index(records, matrix)
    probe=json.loads((ROOT/'jobs/qwen-live-probe.json').read_text())
    queries=[q for task in probe['planner']['plan']['tasks'] for q in task['asset_queries']]
    query_vectors,_=api.embed(queries)
    hits=[]
    for query,v in zip(queries,query_vectors):
        v=np.asarray(v);v/=np.linalg.norm(v)
        scores=matrix@v
        hits.append({'query':query,'top5':[{'asset_id':records[i]['asset_id'],'score':float(scores[i]),'usd':records[i]['usd']} for i in np.argsort(scores)[-5:][::-1]]})
    receipt={'model':'text-embedding-v4','dimensions':matrix.shape[1],'indexed_assets':len(records),
        'source_revision':'f77c220cdf5d2e73fff77e1a2a0a9c7602384c44','queries':hits,
        'scope':'Canonical assets exported by official ASSETS_INDEX; duplicate IDs and unregistered USDs are not silently indexed as executable assets.'}
    (ROOT/'jobs/asset-retrieval.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
