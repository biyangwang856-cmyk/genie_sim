"""Create validated server jobs from real Qwen plans and the asset embedding index."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from asset_retrieval import load_index
from qwen_backend import ROOT,QwenAPI

SKILLS={'pick','align','place','insert','release','retreat'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('instruction');args=parser.parse_args()
    registry=json.loads((ROOT/'pipeline/task_registry.json').read_text())
    api=QwenAPI()
    plan,receipt=api.plan(args.instruction,[{'task_id':k,'description':v['scenes']['0']['instruction']} for k,v in registry.items()])
    if plan.get('robot')!='g2crsb90d':raise ValueError('Only g2crsb90d is registered')
    seen=set()
    for task in plan['tasks']:
        name=task['task_id']
        if name not in registry or name in seen:raise ValueError('Unknown or duplicate task')
        seen.add(name)
        if {p['scene_id']:p['successes'] for p in task['scene_quotas']}!={'0':3,'1':2}:raise ValueError('First batch requires fixed 3+2 quotas')
        if not set(task['skills'])<=SKILLS:raise ValueError('Unregistered atomic skill')
    if seen!=set(registry):raise ValueError('Request must include all three first-batch tasks')
    records,matrix=load_index()
    queries=[q for t in plan['tasks'] for q in t['asset_queries']]
    results=[]
    for start in range(0,len(queries),10):
        vectors,_=api.embed(queries[start:start+10])
        for q,v in zip(queries[start:start+10],vectors):
            v=np.asarray(v);v/=np.linalg.norm(v);scores=matrix@v
            results.append({'query':q,'candidates':[{'asset_id':records[i]['asset_id'],'similarity':float(scores[i])} for i in np.argsort(scores)[-5:][::-1]]})
    jobid=time.strftime('%Y%m%d-%H%M%S')+'-'+hashlib.sha256(args.instruction.encode()).hexdigest()[:8]
    output=ROOT/'jobs'/jobid;output.mkdir()
    job={'job_id':jobid,'instruction':args.instruction,'status':'prepared','plan':plan,'llm_receipt':receipt,
        'retrieved_assets':results,'task_bindings':{k:registry[k] for k in seen},
        'binding_policy':'Preserve official task objects. Semantic retrieval produces candidates; scene and task constraints determine executable binding.',
        'success_counts':{k:{'0':0,'1':0} for k in seen},'attempts':[]}
    (output/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2))
    print(json.dumps({'job_id':jobid,'path':str(output),'status':job['status']}))

if __name__=='__main__':main()
