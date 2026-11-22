"""Preserve each native recording under its immutable run before slot reuse."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,sys,time
MODULE=Path('/data/by/genie_sim/runtime-collection')
def archive(run):
 run=Path(run).resolve();receipt=run/'evidence/raw-archive.json'
 if receipt.exists():return json.loads(receipt.read_text())
 result=json.loads((run/'collection_result.json').read_text());episode=result['episodes'][0]
 location=episode.get('recording_dir')
 if not location:return {'archived':False,'reason':'Native result has no recording directory'}
 launch=json.loads((run/'launch.json').read_text())
 recordings=run/'isolated-runtime/recording_data' if launch.get('collection_slot')=='secondary' else MODULE/'recording_data'
 raw=Path(location)
 if raw.parent==Path('/workspace/recording_data'):raw=recordings/raw.name
 if raw.resolve().parent!=recordings.resolve() or not raw.is_dir():return {'archived':False,'reason':'Native recording absent'}
 bags=list(raw.glob('*.mcap'))
 end=datetime.fromisoformat(result['finished_at']).replace(tzinfo=timezone.utc).timestamp()
 if not bags or max(p.stat().st_mtime for p in bags)>end+30:
  return {'archived':False,'reason':'Recording slot no longer belongs to this finished run'}
 destination=run/'raw';partial=run/'raw.part'
 if destination.exists() or partial.exists():raise RuntimeError('Uncommitted archive exists: '+str(run))
 shutil.copytree(raw,partial,copy_function=shutil.copy2)
 files=[]
 for p in sorted(partial.rglob('*')):
  if p.is_file():
   digest=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
   files.append({'path':str(p.relative_to(partial)),'bytes':p.stat().st_size,'sha256':digest.hexdigest()})
 partial.rename(destination)
 output={'archived':True,'source_recording':str(raw),'immutable_recording':str(destination),
  'created_unix':time.time(),'files':files,'total_bytes':sum(p['bytes'] for p in files)}
 receipt.parent.mkdir(exist_ok=True);receipt.write_text(json.dumps(output,indent=2))
 return output
if __name__=='__main__':
 r=archive(sys.argv[1]);print(json.dumps({k:v for k,v in r.items() if k!='files'}))
