"""Independent official reader check; no GPU use and no environment changes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
import argparse,importlib.metadata,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset,CODEBASE_VERSION

def verify(root,receipt_path=None):
    root=Path(root);torch.set_num_threads(2)
    if CODEBASE_VERSION!='v2.1':raise ValueError('Wrong team dataset reader version')
    dataset=LeRobotDataset(repo_id='local/geniesim_g2',root=root,video_backend='pyav')
    info=json.loads((root/'meta/info.json').read_text());paths=sorted(root.glob('data/chunk-*/*.parquet'))
    source=pq.concat_tables([pq.read_table(p) for p in paths]) if hasattr(pq,'concat_tables') else None
    if source is None:
        import pyarrow as pa
        source=pa.concat_tables([pq.read_table(p) for p in paths])
    numeric=[k for k,v in info['features'].items() if v['dtype']!='video']
    for key in numeric:
        np.testing.assert_array_equal(np.asarray(dataset.hf_dataset[key]),np.asarray(source[key].to_pylist()))
    samples=[]
    boundaries=[];offset=0
    for row in (root/'meta/episodes.jsonl').read_text().splitlines():
        episode=json.loads(row);length=episode['length'];boundaries.extend([offset,offset+length//2,offset+length-1]);offset+=length
    for index in sorted(set(boundaries)):
        sample=dataset[index];shapes={}
        for key,value in sample.items():
            if isinstance(value,torch.Tensor):
                if not torch.isfinite(value).all():raise ValueError('Nonfinite loaded sample: '+key)
                shapes[key]=list(value.shape)
            if key.startswith('observation.images.'):
                expected=info['features'][key]['shape'];assert list(value.shape)==[expected[2],expected[0],expected[1]]
        samples.append({'index':index,'shapes':shapes,'task':sample.get('task')})
    distribution=importlib.metadata.distribution('lerobot')
    source=json.loads(distribution.read_text('direct_url.json') or '{}')
    receipt={'passed':True,'format':CODEBASE_VERSION,'reader_distribution':distribution.version,
        'lerobot_commit':source.get('vcs_info',{}).get('commit_id'),
        'torch_version':importlib.metadata.version('torch'),'torchvision_version':importlib.metadata.version('torchvision'),
        'datasets_version':importlib.metadata.version('datasets'),'pyav_version':importlib.metadata.version('av'),
        'frames':len(dataset),'episodes':info['total_episodes'],'numeric_readback_exact':True,'samples':samples,
        'used_gpu':False,'pi_action_adapter_verified':False}
    if receipt_path is not None:
        with Path(receipt_path).open('x') as result:json.dump(receipt,result,indent=2)
        print(json.dumps(receipt));return
    (root/'official_loader_receipt.json').write_text(json.dumps(receipt,indent=2))
    export=root/'export_receipt.json'
    if export.exists():
        result=json.loads(export.read_text());result['official_reader_verified']=True
        export.write_text(json.dumps(result,indent=2))
    print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--receipt-path',type=Path)
    args=p.parse_args();verify(args.root,args.receipt_path)
