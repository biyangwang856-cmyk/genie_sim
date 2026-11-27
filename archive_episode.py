"""Preserve each native recording under its immutable run before slot reuse."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,sys,time
MODULE=Path('/data/by/genie_sim/runtime-collection')

def link_recording(source,dest):
 if not Path(dest).exists():os.link(source,dest)
 if not os.path.samefile(source,dest):raise ValueError('Isolated recording file changed: '+str(dest))

def archive(run):
 run=Path(run).resolve();receipt=run/'evidence/raw-archive.json'
 if receipt.exists():return json.loads(receipt.read_text())
 result=json.loads((run/'collection_result.json').read_text());episode=result['episodes'][0]
 location=episode.get('recording_dir')
 if not location:return {'archived':False,'reason':'Native result has no recording directory'}
 launch=json.loads((run/'launch.json').read_text())
 private=launch.get('isolated_recording',launch.get('collection_slot','primary')!='primary')
 recordings=run/'isolated-runtime/recording_data' if private else MODULE/'recording_data'
 raw=Path(location)
 if raw.parent==Path('/workspace/recording_data'):raw=recordings/raw.name
 if raw.resolve().parent!=recordings.resolve() or not raw.is_dir():return {'archived':False,'reason':'Native recording absent'}
 bags=list(raw.glob('*.mcap'))
 end=datetime.fromisoformat(result['finished_at']).replace(tzinfo=timezone.utc).timestamp()
 if not bags or max(p.stat().st_mtime for p in bags)>end+30:
  return {'archived':False,'reason':'Recording slot no longer belongs to this finished run'}
 destination=run/'raw';partial=run/'raw.part'
 isolated=private and bool(launch.get('scene_sha256'))
 if not isolated and (destination.exists() or partial.exists()):
  raise RuntimeError('Uncommitted archive exists: '+str(run))
 if isolated and destination.exists():
  archive_root=destination
 else:
  shutil.copytree(raw,partial,copy_function=link_recording if isolated else shutil.copy2,dirs_exist_ok=isolated)
  archive_root=partial
 files=[]
 for p in sorted(archive_root.rglob('*')):
  if p.is_file():
   relative=p.relative_to(archive_root)
   if isolated and not os.path.samefile(p,raw/relative):
    raise ValueError('Isolated recording file changed: '+str(p))
   digest=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
   files.append({'path':str(relative),'bytes':p.stat().st_size,'sha256':digest.hexdigest()})
 if isolated and len(files)!=sum(p.is_file() for p in raw.rglob('*')):
  raise ValueError('Isolated recording archive has incomplete file inventory')
 if archive_root==partial:partial.rename(destination)
 output={'archived':True,'source_recording':str(raw),'immutable_recording':str(destination),
  'created_unix':time.time(),'files':files,'total_bytes':sum(p['bytes'] for p in files),
  'storage_method':'hardlinks_to_stopped_isolated_recording' if isolated else 'copy'}
 receipt.parent.mkdir(exist_ok=True)
 temp=receipt.with_suffix('.json.tmp')
 temp.write_text(json.dumps(output,indent=2))
 temp.replace(receipt)
 return output
if __name__=='__main__':
 r=archive(sys.argv[1]);print(json.dumps({k:v for k,v in r.items() if k!='files'}))
