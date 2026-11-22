"""Apply the verified single-span extraction fix only to this project's SDK copy."""
import hashlib,json,time
from pathlib import Path

ROOT=Path('/home/vipuser/by/genie_sim')
source=Path('/data/by/genie_sim/runtime-collection/server/recording/extract_ros_bag.py')
before=source.read_bytes();old=hashlib.sha256(before).hexdigest()
expected='8ca61b7a2964027ffdf39a5b0dc4d4a0c1f64bae08a6ec48ff88530f77422145'
if old!=expected:raise ValueError('Runtime extractor changed; inspect instead of patching an unknown version')
text=before.decode();needle='            chunk_size = 1000\n'
if text.count(needle)!=1:raise ValueError('Unexpected extraction chunk assignment')
after=text.replace(needle,'            # Project fix: preserve native samples across former 1000-frame trims.\n            chunk_size = len(time_stamps)\n').encode()
compile(after,str(source),'exec')
backup=ROOT/'configs/postprocessor-source';backup.mkdir(parents=True,exist_ok=True)
(backup/(old+'.py')).write_bytes(before)
source.write_bytes(after)
new=hashlib.sha256(after).hexdigest();(backup/(new+'.py')).write_bytes(after)
receipt={'changed_unix':time.time(),'path':str(source),'before_sha256':old,'after_sha256':new,
    'change':'single full-span extraction; no synthetic frames; no simulator or asset changes',
    'verified_against':['block-scene0-release-pinch18','insert-building-block-s0-20261120160637'],
    'memory_limit':'First-batch episodes only; do not use this setting for unbounded long captures.'}
(ROOT/'configs/postprocessor-default-fix.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
