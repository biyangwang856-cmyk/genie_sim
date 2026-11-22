"""Convert fixed benchmark task scenes into independent collection contracts."""
from pathlib import Path
import contextlib,json,shutil,sys
from pxr import Usd,UsdGeom,UsdLux,Sdf,UsdUtils
import numpy as np
ROOT=Path('/home/vipuser/by/genie_sim')
REPO=ROOT/'genie_sim-main'
BENCH=REPO/'source/geniesim_benchmark/src/geniesim_benchmark'
ASSETS=ROOT/'GenieSimAssets'
NAMES={'phone_packaging':'phone_packaging_g2_crsb90d','put_the_cap_on_the_pen_new':'table_task_g2_crsb_90d_room2_pen_cap_main','insert_building_block':'table_task_g2_crsb_90d_room2_building_block_main'}

def main():
    sys.path.insert(0,str(ROOT))
    with (ROOT/'logs/task-asset-registration.log').open('w') as log,contextlib.redirect_stdout(log):
        import GenieSimAssets as lib
    sample=REPO/'source/geniesim_scene_agent/examples/tasks/assemble_gear_or_bearing_g2_crsb_90d/task_info.json'
    base_info=json.loads(sample.read_text())
    registry={}
    for task,evalname in NAMES.items():
        name='g2crsb90d_finemanip_'+task
        dest=ROOT/'tasks'/name;dest.mkdir(parents=True,exist_ok=True)
        conf=json.loads((BENCH/'benchmark/config/eval_tasks'/(evalname+'.json')).read_text())
        pose=next(iter(conf['robot']['robot_init_pose'].values()))
        metadata=json.loads(json.dumps(base_info))
        metadata['task_name']=name
        metadata['robot']['init_arm_pose_noise']={}
        metadata['robot']['robot_init_pose']={k:pose[k] for k in ('position','quaternion')}
        metadata['recording_setting']={'camera_list':['/G2/head_link3/head_front_Camera','/G2/gripper_l_camera_link/Left_Camera','/G2/gripper_r_camera_link/Right_Camera'],'fps':30,'noised_probability':0.0}
        metadata['task_metric']={}
        metadata['backend']='gpu'
        metadata['physics_frequency']=120
        metadata['enable_deformable']=False
        entry={'robot_profile':'g2crsb90d','template':task,'scenes':{},'status':'adapted_unvalidated'}
        for scene in ('0','1'):
            src=BENCH/'benchmark/config/llm_task'/task/scene
            out=dest/scene;out.mkdir(exist_ok=True)
            layer=Sdf.Layer.CreateAnonymous('.usda');layer.ImportFromString((src/'scene.usda').read_text())
            missing=[]
            def resolve(p):
                if '/geniesim_assets/' in p:
                    target=ASSETS/p.split('/geniesim_assets/',1)[1]
                    if not target.exists():missing.append(str(target))
                    return str(target)
                return p
            UsdUtils.ModifyAssetPaths(layer,resolve)
            stage=Usd.Stage.Open(layer)
            if not stage:raise RuntimeError('USD composition failed')
            world=stage.GetDefaultPrim()
            if task=='phone_packaging':
                # Preserve the official benchmark background that supplies its table.
                world.GetReferences().AddReference(str(ASSETS/conf['scene']['scene_usd']))
            else:
                dome=UsdLux.DomeLight.Define(stage,str(world.GetPath())+'/CollectionDome')
                dome.CreateIntensityAttr(800.)
            old=json.loads((src/'scene_info.json').read_text())
            layout={}
            bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'],useExtentsHint=False)
            objects=stage.GetPrimAtPath(str(world.GetPath())+'/Objects')
            if not objects:raise RuntimeError(f'Missing Objects scope: {task}/{scene}')
            for prim in objects.GetChildren():
                label=prim.GetName()
                info=old.get('layout',{}).get(label,{})
                asset=info.get('usd')
                if not asset:
                    payload=prim.GetMetadata('payload')
                    if payload:
                        paths=payload.GetAddedOrExplicitItems()
                        if paths:asset=Path(paths[0].assetPath).parent.name
                if not asset:
                    raise RuntimeError(f'Cannot resolve asset id for {label}')
                mat=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                q=mat.ExtractRotationQuat()
                layout[label]={**info,'usd':asset,'xyz':list(mat.ExtractTranslation()),
                    'xyzw':[*q.GetImaginary(),q.GetReal()], 'registered':asset in lib.ASSETS_INDEX}
                entity=stage.GetPrimAtPath(str(prim.GetPath())+'/entity')
                if entity:
                    bound=bbox.ComputeUntransformedBound(entity).ComputeAlignedRange()
                    layout[label]['entity_local_bbox']={'min':list(bound.GetMin()),'max':list(bound.GetMax())}
            if missing:raise RuntimeError(f'Missing task references: {missing}')
            instructions=json.loads((src/'instructions.json').read_text())
            scene_info={**old,'scene_id':scene,'layout':layout,'official_instruction':instructions['instructions'][0],
                'official_subtask_steps':instructions.get('subtask_steps',[]),'task_template':task,'source_scene':str(src)}
            metadata_scene=json.loads(json.dumps(metadata))
            instruction=instructions['instructions'][0]['instruction']
            metadata_scene['task_description']={'task_name':instruction,'english_task_name':instruction,'init_scene_text':''}
            (out/'task_info.json').write_text(json.dumps(metadata_scene,indent=2))
            (out/'scene_info.json').write_text(json.dumps(scene_info,indent=2))
            stage.GetRootLayer().Export(str(out/'scene.usda'))
            for filename in ('instructions.json','problems.json'):shutil.copy2(src/filename,out/filename)
            entry['scenes'][scene]={'success_quota':3 if scene=='0' else 2,'instruction':instruction,
                'objects':{k:v['usd'] for k,v in layout.items()},'unregistered':[k for k,v in layout.items() if not v['registered']]}
        (dest/'task_info.json').write_text(json.dumps(metadata,indent=2))
        (dest/'data_gen.py').write_text('from collection_programs import execute_task\n')
        registry[name]=entry
    (ROOT/'pipeline/task_registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2))
    print(json.dumps(registry,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
