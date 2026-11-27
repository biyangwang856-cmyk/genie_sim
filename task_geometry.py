"""Existing-asset task contracts and measured geometry; no robot motion here."""
import json,os
from itertools import product
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from robot_adapters import UnsupportedRequest

# Conservative inner face bounds from the official collision meshes in both
# packaging scenes. Coordinates are X/Z in the entity's own frame, in metres.
PACKING_CASE_OPENING = np.array([[-.03423,-.07385],[.03478,.07462]])
PACKING_LID_OPENING = np.array([[-.0373,-.0772],[.0373,.0771]])
PACKING_RELEASE_CLEARANCE_M = .005


def bbox_corners(lower, upper):
    return np.asarray(list(product(*zip(lower, upper))))


def transformed_points(matrix, points):
    return points @ matrix[:3,:3].T + matrix[:3,3]


def opening_margin(points, opening):
    projected = points[:,[0,2]]
    return float(min((projected.min(axis=0)-opening[0]).min(), (opening[1]-projected.max(axis=0)).min()))

def task_spec(scene,robot_id):
    name=scene['task_template'];instruction=scene['official_instruction'];layout=scene['layout']
    spec={'template':name,'task_id':name,'source_task_id':'g2crsb90d_finemanip_'+name,'scene_id':str(scene['scene_id']),
        'robot_id':robot_id,'official_instruction':instruction['instruction'],
        'objects':{k:{'asset_id':v['usd'],'bbox':v.get('entity_local_bbox'),'registered':v.get('registered',False)} for k,v in layout.items()},
        'targets':[],'relations':[],'role_bindings':{},'grasp_profiles':{},'manipulated_objects':[],'target_objects':{}}
    if name=='insert_building_block':
        obj=instruction['target']['id1']
        # Use the actual certificate binding, not a guessed asset alias.
        from insertion_physics import certificate
        box=certificate()['box']
        spec.update(role_bindings={'operator':instruction['gripper']['1']['side']},manipulated_objects=[obj],
            grasp_profiles={obj:['release_compatible']},targets=['shape_approach','shape_entry_high','shape_entry'],
            relations=['correct_hole_and_stable'],target_objects={k:obj for k in ['shape_approach','shape_entry_high','shape_entry']},
            bindings={'workpiece':obj,'receiver':box},required_capabilities=['friction_grasp','angular_parallel_gripper','curobo'])
    elif name=='phone_packaging':
        objects=['benchmark_phone_000','benchmark_packaging_box_000','benchmark_packaging_box_002']
        targets={f'packing_{i}_{phase}':obj for i,obj in enumerate(objects) for phase in ['approach','contact']}
        spec.update(role_bindings={'operator':'right'},manipulated_objects=objects,
            environment_collision_paths=['/World/background/benchmark_table_019'],
            grasp_profiles={o:['topdown_box'] for o in objects},targets=list(targets),target_objects=targets,
            relations=['packing_stage_0','packing_stage_1','packing_stage_2'],
            bindings={'phone':objects[0],'insert':objects[1],'lid':objects[2],'receiver':'benchmark_packaging_box_001'},
            release_modes={objects[0]:'native',objects[1]:'native',objects[2]:'gentle'},
            required_capabilities=['friction_grasp','angular_parallel_gripper','curobo'])
        if 'project_packing_goal' in scene:
            from goal_composition import bind_packing_goal
            bind_packing_goal(spec, scene['project_packing_goal'])
    elif name=='put_the_cap_on_the_pen_new':
        pen=instruction['target']['id1'];cap=instruction['target']['id2']
        spec.update(role_bindings={'operator':instruction['gripper']['1']['side'],'assistant':instruction['gripper']['2']['side']},
            manipulated_objects=[pen,cap],grasp_profiles={pen:['pen_barrel'],cap:['cap_body']},
            targets=['pen_present','cap_preinsert','cap_insert_axis','pen_return'],
            target_objects={'pen_present':pen,'pen_return':pen,'cap_preinsert':cap,'cap_insert_axis':cap},relations=['cap_seated'],
            bindings={'pen':pen,'cap':cap},required_capabilities=['bimanual','friction_grasp','angular_parallel_gripper','curobo'],
            assembly={'pen_tip_x_m':-.07802,'cap_mouth_x_m':.02127,'preinsert_gap_m':.03,'insertion_depth_m':.03152,
                'bore_half_width_m':.00919,'tip_half_width_m':.00451,'transition_half_width_m':.00811,'barrel_half_width_m':.00877,
                'transition_start_depth_m':.02336,'barrel_start_depth_m':.03152,
                'model':'official_asset_physics_with_pose_triggered_snap_joint'})
    else:
        from task_plugins import load
        plugin=load(name)
        spec.update(plugin.bind(scene,robot_id))
        spec['task_plugin']=name
    if transfer := scene.get('scene_transfer'):
        support=transfer['support'];spec['support']=support
        spec['profile_scene_id']=transfer['profile_scene_id']
        spec['support_height_offset_m']=transfer['support_height_offset_m']
        spec['environment_collision_paths']=transfer.get('collision_roots',[support['prim']])
        if name=='phone_packaging' and transfer.get('mode')=='compose_from_library':
            spec['release_modes'][spec['bindings']['insert']]='gentle'
    if 'task_plugin' not in spec:
        precision=['shape_entry'] if name=='insert_building_block' else ['cap_preinsert'] if name=='put_the_cap_on_the_pen_new' else [t for t in spec['targets'] if t.endswith('_contact')]
        spec['skill_targets']={'move_held':[t for t in spec['targets'] if t!='cap_insert_axis'],
            'align_precision':precision,'insert_axial':['cap_insert_axis'] if name=='put_the_cap_on_the_pen_new' else []}
    for obj in sorted(set(spec['manipulated_objects']) | set(spec['bindings'].values())):
        info=spec['objects'].get(obj)
        if not info or not info['registered']:raise UnsupportedRequest('Task object is not a registered asset',missing_assets=[str(obj)])
        if not info['bbox']:raise UnsupportedRequest('Asset lacks size/frame metadata',missing_assets=[obj+':bbox'])
    return spec

class Geometry:
    def cap_preinsert_pose(self,pen_pose):
        assembly=self.spec['assembly']
        gap=assembly['pen_tip_x_m']-assembly['cap_mouth_x_m']-assembly['preinsert_gap_m']
        result=pen_pose.copy();result[:3,3]+=gap*pen_pose[:3,0]
        return result

    def __init__(self,robot,spec,ops,adapter):
        self.robot,self.spec,self.ops,self.adapter=robot,spec,ops,adapter;self.cache={}
        self.initial={o:ops.pose(robot,o).copy()
                      for o in [*spec['manipulated_objects'], *spec.get('preserve_objects', [])]}
        seed=os.environ.get('GENIE_EPISODE_SEED')
        self.packing_approach_extra_m=0.
        self.pen_return_offset_m=np.zeros(2)
        self.shape_approach_extra_m=0.
        self.shape_transport_lift_m=0.
        if seed is not None:
            rng=np.random.default_rng(int(seed))
            self.packing_approach_extra_m=float(rng.uniform(.004,.014))
            angle=float(rng.uniform(-np.pi,np.pi))
            self.pen_return_offset_m=float(rng.uniform(.012,.022))*np.array([np.cos(angle),np.sin(angle)])
            if spec['template']=='insert_building_block':
                lift = float(rng.uniform(.008, .016))
                if spec.get('profile_scene_id', spec['scene_id']) == '0':
                    self.shape_approach_extra_m = lift
                elif 'support' in spec:
                    self.shape_transport_lift_m = lift

    def gripper_collision_points(self, arm):
        hand = self.robot.client.command_any({'command':'ProjectGripperCollisionBounds','arm':arm})
        if not isinstance(hand,dict) or hand.get('success') is not True:raise ValueError('Native collision geometry of the grasping hand is unavailable')
        corners = np.concatenate([np.asarray(shape['corners_world_m']) for shape in hand['shapes']])
        if not np.isfinite(corners).all():raise ValueError('Nonfinite gripper collision bounds')
        return corners, hand

    def packing_release_geometry(self, ref, obj, arm, *, yaw=None):
        """Plan to the mouth using the live held payload and native gripper bounds."""
        goal, excluded = self.target(ref, obj)
        receiver = self.spec['bindings']['receiver']
        base = self.ops.pose(self.robot, receiver)
        actual = self.ops.pose(self.robot, obj)
        world_to_base = np.linalg.inv(base)
        if base[2,1] < .995:raise ValueError('Packaging receiver axis is not upright')
        own_corners = bbox_corners(*self.ops.local_box(self.robot, obj))
        base_bbox = self.ops.local_box(self.robot, receiver)
        # Only the contact skill uses the mouth height. The old target remains
        # the orientation/centering reference, never a commanded deep descent.
        nominal = world_to_base @ goal
        if yaw is None:
            yaw = self.cache.get('release_yaw:'+ref, 0.)
        if yaw:
            center = own_corners.mean(axis=0)
            target_center = nominal[:3, :3] @ center + nominal[:3, 3]
            nominal[:3, :3] = Rotation.from_rotvec([0., yaw, 0.]).as_matrix() @ nominal[:3, :3]
            nominal[:3, 3] = target_center - nominal[:3, :3] @ center
        nominal[1,3] = 0.
        hand_world, hand = self.gripper_collision_points(arm)
        hand_object = transformed_points(np.linalg.inv(actual), hand_world)
        held_geometry = np.concatenate([own_corners, hand_object])
        clearance = self.cache.get('release_clearance:'+ref,PACKING_RELEASE_CLEARANCE_M)
        required_y = float(base_bbox[1][1]) + clearance
        min_nominal_y = float(transformed_points(nominal, held_geometry)[:,1].min())
        nominal[1,3] = required_y-min_nominal_y
        goal = base @ nominal
        local = world_to_base @ actual
        if obj == self.spec['bindings']['lid']:
            # The lid surrounds the outside of the base. Test the outer base
            # corners in the lid frame, so the small skirt clearance is kept.
            margin = opening_margin(transformed_points(np.linalg.inv(local),bbox_corners(*base_bbox)),PACKING_LID_OPENING)
            goal_margin = opening_margin(transformed_points(np.linalg.inv(nominal),bbox_corners(*base_bbox)),PACKING_LID_OPENING)
        else:
            margin = opening_margin(transformed_points(local,own_corners),PACKING_CASE_OPENING)
            goal_margin = opening_margin(transformed_points(nominal,own_corners),PACKING_CASE_OPENING)
        min_y = float(transformed_points(local, held_geometry)[:,1].min())
        metrics = {'sim_time_s':hand['sim_time'],'rim_clearance_m':min_y-float(base_bbox[1][1]),
            'projected_fit_margin_m':margin,'projected_goal_fit_margin_m':goal_margin,
            'release_yaw_rad':float(yaw),'gripper_collision_shapes':len(hand['shapes']),
            'gripper_collision_paths':[s['path'] for s in hand['shapes']],
            'target_clearance_m':clearance,'receiver_collisions_active':True}
        return goal, excluded, metrics

    def shape_release_geometry(self, obj, arm, target_gap):
        from insertion_physics import certificate,measure
        goal, excluded = self.target('shape_entry', obj)
        receiver = self.spec['bindings']['receiver']
        base = self.ops.pose(self.robot, receiver)
        actual = self.ops.pose(self.robot, obj)
        if base[2,1] < .995:raise ValueError('Shape panel axis is not upright')
        native = certificate()
        plane = max(native['hole_plane_y_box_frame'],float(self.ops.local_box(self.robot,receiver)[1][1]))
        world_to_base = np.linalg.inv(base)
        hand_world, hand = self.gripper_collision_points(arm)
        hand_object = transformed_points(np.linalg.inv(actual),hand_world)
        mesh = np.asarray(native['star_triangles_entity_frame']).reshape(-1,3)
        local_goal = world_to_base @ goal
        correction = self.adapter.config.get('shape_release_lateral_correction',{}).get(arm,{})
        offset = np.zeros(2)
        if correction.get('scene_id')==self.spec.get('profile_scene_id',self.spec['scene_id']):
            offset=np.asarray(correction['box_xz_m'],dtype=float)
            if offset.shape!=(2,) or not np.isfinite(offset).all() or np.linalg.norm(offset)>.0011:
                raise ValueError('Invalid submillimetre shape release correction')
            local_goal[[0,2],3]+=offset
            if measure(base@local_goal,base,contract=native)['outside_hole_area_fraction']>.001:
                raise ValueError('Corrected target projection would not fit the original aperture')
        p = transformed_points(local_goal,np.concatenate([mesh,hand_object]))
        old_gap = float(p[:,1].min()-plane)
        adjustment = target_gap-old_gap
        if abs(adjustment)>.06:raise ValueError('Measured gripper clearance is inconsistent with the shape entry profile')
        local_goal[1,3] += adjustment
        local_actual = world_to_base @ actual
        hand_y = transformed_points(world_to_base, hand_world)[:, 1].reshape(len(hand['shapes']), 8)
        lowest_collider = int(np.argmin(hand_y.min(axis=1)))
        metrics = {'sim_time_s':hand['sim_time'],'target_minimum_clearance_m':target_gap,
            'target_workpiece_clearance_m':float(p[:len(mesh),1].min()-plane+adjustment),
            'clearance_at_original_target_m':old_gap,'height_adjustment_m':adjustment,
            'box_xz_correction_m':offset.tolist(),'correction_reference_run':correction.get('reference_run'),
            'actual_hand_clearance_m':float(hand_y.min()-plane),
            'actual_workpiece_clearance_m':float(transformed_points(local_actual,mesh)[:,1].min()-plane),
            'limiting_gripper_collision_path':hand['shapes'][lowest_collider]['path'],
            'gripper_collision_paths':[s['path'] for s in hand['shapes']]}
        return base @ local_goal, excluded, metrics

    def target(self,ref,obj):
        if self.spec['target_objects'].get(ref)!=obj:raise ValueError('Target/object binding mismatch')
        if self.spec.get('task_plugin'):
            from task_plugins import load
            return load(self.spec['task_plugin']).resolve_target(self,ref,obj)
        ops=self.ops;r=self.robot;b=self.spec['bindings']
        if ref.startswith('shape_'):
            from insertion_physics import measure
            if 'shape_profile' not in self.cache:
                source=Path(os.environ['GENIE_RUN_SOURCE'])/'insertion-target.json'
                self.cache['shape_profile']=json.loads(source.read_text())
            profile=self.cache['shape_profile'];box=b['receiver'];base=ops.pose(r,box)@getattr(r,'_project_shape_goal_local',np.asarray(profile['object_goal_in_box_frame']))
            if ref=='shape_approach':
                center=np.asarray(profile['star_center_entity']);world_center=(base@np.r_[center,1])[:3];candidates=[]
                for angle in np.arange(5)*2*np.pi/5:
                    t=base.copy();t[:3,:3]=Rotation.from_rotvec([0,0,angle]).as_matrix()@base[:3,:3];t[:3,3]=world_center-t[:3,:3]@center
                    if measure(t,ops.pose(r,box))['outside_hole_area_fraction']<=.001:candidates.append(t)
                if not candidates:raise UnsupportedRequest('Native workpiece mesh does not fit selected aperture',missing_assets=['compatible_workpiece_aperture_pair'])
                self.cache['shape_variants']=candidates
                # This offset applies to the above-aperture travel. The cached
                # goals provide the alignment and release geometry.
                result=np.asarray(candidates);result[:,2,3]+=.07+self.shape_approach_extra_m
                return result,[]
            if 'shape_goal_local' not in self.cache:
                actual=ops.pose(r,obj)
                variants=self.cache.get('shape_variants')
                if not variants:raise ValueError('Resolve approach before aperture entry')
                selected=min(variants,key=lambda v:Rotation.from_matrix(v[:3,:3].T@actual[:3,:3]).magnitude())
                self.cache['shape_goal_local']=np.linalg.inv(ops.pose(r,box))@selected
            result=ops.pose(r,box)@self.cache['shape_goal_local'];result[2,3]+=.06 if ref=='shape_entry_high' else .05
            return result,[r.get_object_prim_path(box)+'/body']
        if ref.startswith('packing_'):
            index=int(ref.split('_')[1]);result=ops.phone_target(r,obj,[.008,.020,.026][index])
            if ref.endswith('approach'):result[2,3]+=.10+self.packing_approach_extra_m
            return result,[] if ref.endswith('approach') else [r.get_object_prim_path(b['receiver'])]
        if ref=='pen_return':
            result=self.initial[obj].copy();result[2,3]+=.004;return result,[]
        if ref=='pen_present':
            direction=np.array([0.,-1. if self.spec['role_bindings']['operator']=='right' else 1.,0.])
            workspace=self.adapter.config['cooperation_workspace']
            t=np.eye(4);t[:3,:3]=np.column_stack([direction,[0,0,1],np.cross(direction,[0,0,1])]);t[:3,3]=workspace['center_world_m']
            t[1,3]+=np.sign(direction[1])*workspace['operator_side_offset_m']
            t[2,3]+=self.spec.get('support_height_offset_m',0.)
            return t,[]
        if ref in ['cap_preinsert','cap_insert_axis']:
            return self.cap_preinsert_pose(ops.pose(r,b['pen'])),[]
        raise ValueError('Unknown geometry resolver')
