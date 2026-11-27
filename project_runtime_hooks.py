"""Project-local read-only diagnostics and arm-scoped planner attachment cleanup."""
import json,math,os
from itertools import product
from pathlib import Path

_controller=None
MAX_LOCAL_ARM_JOINT_STEP_RAD = .18
G2_COUPLED_WRIST_LIMIT_RAD = 1.92
G2_COUPLED_WRIST_PAIRS = (
    ('idx26_arm_l_joint6', 'idx27_arm_l_joint7'),
    ('idx66_arm_r_joint6', 'idx67_arm_r_joint7'),
)


def arm_joint_indices(names):
    return [index for index, name in enumerate(names)
            if '_arm_l_joint' in name or '_arm_r_joint' in name]

def g2_coupled_wrist_mask(names, positions):
    """Evaluate both native sum-and-difference limits on every waypoint."""
    pairs = []
    for first, second in G2_COUPLED_WRIST_PAIRS:
        if first not in names and second not in names:
            continue
        if first not in names or second not in names:
            raise ValueError('Incomplete G2 wrist pair in native plan')
        pairs.append((names.index(first), names.index(second)))
    if not pairs:
        raise ValueError('G2 wrist joints missing from native plan')
    valid = abs(positions[..., pairs[0][0]]) + abs(positions[..., pairs[0][1]]) <= G2_COUPLED_WRIST_LIMIT_RAD
    for first, second in pairs[1:]:
        valid = valid & (abs(positions[..., first]) + abs(positions[..., second]) <= G2_COUPLED_WRIST_LIMIT_RAD)
    return valid


def g2_trajectory_candidate_mask(names, positions, successes):
    within_limit = g2_coupled_wrist_mask(names, positions).all(-1)
    try:
        return successes & within_limit.reshape(successes.shape)
    except (ValueError, RuntimeError) as exc:
        raise ValueError('G2 joint-path candidate dimensions do not match the planner result') from exc


def g2_endpoint_ik_seed(names, positions, successes):
    """Select the first native-ranked endpoint seed that meets wrist limits."""
    import numpy as np

    positions, successes = np.asarray(positions), np.asarray(successes, dtype=bool)
    if positions.ndim != 3 or successes.shape != positions.shape[:2] or not successes.shape[1]:
        raise ValueError('G2 endpoint IK candidate dimensions do not match the planner result')
    eligible = successes & g2_coupled_wrist_mask(names, positions)
    return eligible.argmax(axis=1), eligible.any(axis=1)


def probe_support_ray(query, origin, support):
    """Find the table below a workpiece without changing either collider."""
    hits = []

    def collect(hit):
        collision, body = str(hit.collision), str(hit.rigid_body)
        hits.append({'distance_m': float(hit.distance), 'position': [float(v) for v in hit.position],
                     'normal': [float(v) for v in hit.normal], 'collision': collision, 'body': body,
                     'requested_support': any(path == support or path.startswith(support + '/')
                                              for path in (collision, body))})
        return True

    query.raycast_all(origin, (0., 0., -1.), .34, collect)
    hits.sort(key=lambda hit: hit['distance_m'])
    chosen = next((hit for hit in hits if hit['requested_support']), None)
    if chosen is None:
        return hits[0] if hits else None
    if hits[0] is not chosen:
        chosen['nearest_occluder'] = hits[0]
    return chosen


def swept_payload_clearance(tool_poses, payload_in_tool, payload_bounds, fixture_bounds):
    """Conservative oriented-payload clearance in the registered fixture frame."""
    import numpy as np

    transforms = np.asarray(tool_poses) @ np.asarray(payload_in_tool)
    corners = np.asarray(list(product(*zip(*payload_bounds))))
    points = corners @ transforms[:, :3, :3].transpose(0, 2, 1) + transforms[:, None, :3, 3]
    bounds = np.asarray(fixture_bounds, dtype=float)
    separation = np.maximum(bounds[0] - points.max(axis=1), points.min(axis=1) - bounds[1])
    if not np.isfinite(separation).all():
        raise ValueError('Nonfinite transport clearance witness')
    return float(separation.max(axis=1).min())


def tool_pose_matrices(positions, quaternions_wxyz):
    """Convert native FK components, including interleaved quaternion views."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    xyz = np.asarray(positions).reshape(-1, 3)
    wxyz = np.asarray(quaternions_wxyz).reshape(-1, 4)
    if len(xyz) != len(wxyz) or not len(xyz):
        raise ValueError('Mismatched native FK pose components')
    matrices = np.repeat(np.eye(4)[None], len(xyz), axis=0)
    matrices[:, :3, :3] = Rotation.from_quat(wxyz[:, [1, 2, 3, 0]]).as_matrix()
    matrices[:, :3, 3] = xyz
    return matrices


def continuous_ik_candidates(names, positions, current_positions):
    """Keep the IK solutions reachable from the previous waypoint's branch."""
    import numpy as np

    arm = arm_joint_indices(names)
    if not arm:
        raise ValueError('The G2 IK solve has no arm joints')
    solutions = np.asarray(positions)[..., arm]
    current = np.asarray(current_positions)[:, arm]
    if solutions.ndim == 3:
        current = current[:, None, :]
    steps = np.abs(solutions - current)
    return np.all(np.isfinite(steps) & (steps <= MAX_LOCAL_ARM_JOINT_STEP_RAD), axis=-1)


def solve_from_local_joint_seed(solver, solve, args, options):
    """Run the native collision-aware optimizer from the last continuous branch."""
    config = solver.config
    current = options['current_state']
    seeds = current.position.unsqueeze(1).expand(-1, config.num_seeds, -1).contiguous()
    old_lm, old_early = config.use_lm_seed, config.exit_early
    try:
        # The LM seed generator otherwise replaces even caller-provided starts.
        config.use_lm_seed = False
        config.exit_early = False
        return solve(*args, **{**options, 'seed_config': seeds})
    finally:
        config.use_lm_seed, config.exit_early = old_lm, old_early


def largest_arm_joint_step(planned_names, planned_positions, measured_names, measured_positions):
    """Measure the largest adjacent arm command, including the live start."""
    import numpy as np

    names = list(planned_names)
    indices = arm_joint_indices(names)
    if not indices:
        raise ValueError('The G2 local joint plan has no arm joints')
    start_by_name = dict(zip(measured_names, measured_positions, strict=True))
    start = [start_by_name[names[i]] for i in indices]
    positions = np.asarray(planned_positions)[:, indices]
    differences = np.abs(np.diff(np.vstack([start, positions]), axis=0))
    if not np.isfinite(differences).all():
        raise ValueError('The G2 local joint plan contains nonfinite positions')
    waypoint, joint = np.unravel_index(np.argmax(differences), differences.shape)
    return {'waypoint': int(waypoint), 'joint': names[indices[joint]],
            'step_rad': float(differences[waypoint, joint])}

def record_render_sample(world):
    """Observe live and rendered wrists after rendering; never set a body pose."""
    if _controller is None or not world.is_playing():return
    now=float(world.current_time)
    if now-getattr(_controller,'_project_render_sample_time',-1)<.1-1e-5:return
    import numpy as np
    from pxr import UsdGeom
    import omni.usd
    from sim_backend import physics,to_numpy
    from scipy.spatial.transform import Rotation
    cache=UsdGeom.XformCache();stage=omni.usd.get_context().get_stage();links={}
    for side in ['l','r']:
        path='/G2/arm_'+side+'_end_link'
        p,q=_controller._get_object_pose(path)
        actual=np.eye(4);actual[:3,3]=to_numpy(p)
        actual[:3,:3]=Rotation.from_quat(to_numpy(q)[[1,2,3,0]]).as_matrix()
        usd=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T
        links[side]={'physics':actual.tolist(),'usd':usd.tolist()}
    row={'sim_time':now,'links':links,'joint_positions':to_numpy(physics.get_joint_positions(_controller.ui_builder.articulation,as_torch=False)).tolist()}
    if not hasattr(_controller,'_project_render_samples'):
        _controller._project_render_samples=(Path(os.environ['GENIE_EPISODE_EVIDENCE'])/'render-motion-samples.jsonl').open('a',buffering=1)
    _controller._project_render_samples.write(json.dumps(row)+'\n')
    _controller._project_render_sample_time=now

def camera_stamp_for_render(requested, rendered, physics_dt):
    delta = rendered - requested
    if not all(math.isfinite(value) for value in (delta, physics_dt)) or not -1e-6 <= delta <= physics_dt + 1e-6:
        raise ValueError('Native render did not follow the matching physical capture')
    return requested


def pending_direct_cameras():
    if _controller is None:
        return ()
    return tuple(node for node in _controller.ros_publishers_persistent
                 if getattr(node, '_pending_capture_time', None) is not None)


def matching_camera_packet(data, previous_reference, requested, physics_dt):
    import numpy as np

    try:
        numerator, denominator = data['reference_time']
        reference = (int(numerator), int(denominator))
        previous_numerator, previous_denominator = previous_reference or (-1, 1)
        simulation_time = float(data['IsaacReadSimulationTime']['simulationTime'])
        rgb = data['rgb']['data']
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    fresh_reference = reference[0] * previous_denominator > previous_numerator * reference[1]
    # The two native clocks accumulate float32 and float64 time differently.
    matching_step = math.isclose(simulation_time, requested, rel_tol=0., abs_tol=physics_dt / 100)
    valid_rgb = (isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8
                 and rgb.ndim == 3 and rgb.size > 0 and rgb.shape[2] in (3, 4))
    if reference[1] <= 0 or not (fresh_reference and matching_step and valid_rgb):
        return None
    return {'reference_time': reference, 'simulation_time': simulation_time, 'rgb': rgb[:, :, :3]}


def capture_exact_camera_packets(world, render_only, writers):
    """Acquire fresh RGB and render-clock metadata from each Writer's same-frame payload."""
    requested = world.current_time
    step = world.current_time_step_index
    physics_dt = world.get_physics_dt()
    before = [tuple((writer.get_data() or {}).get('reference_time') or ()) for writer in writers]
    for attempt in range(1, 5):
        for writer in writers:
            writer.schedule_write()
        render_only()
        if (world.current_time, world.current_time_step_index) != (requested, step):
            raise RuntimeError('Render-only camera capture advanced physics')
        payloads = [writer.get_data() for writer in writers]
        packets = [matching_camera_packet(data, ref, requested, physics_dt) for data, ref in zip(payloads, before)]
        if all(packet is not None for packet in packets):
            for packet in packets:
                packet['rgb'] = packet['rgb'].copy()
            return packets, attempt
    times = [(data or {}).get('IsaacReadSimulationTime') for data in payloads]
    raise RuntimeError(f'No matching RGB packets for step {step} at {requested:.9f} s, observed {times}')


def prepare_exact_camera_samples(world, render_only, update_camera_poses):
    if _controller is None or not _controller.publish_ros or not world.is_playing():
        return None
    cameras = tuple(node for node in _controller.ros_publishers_persistent
                    if getattr(node, 'project_direct_camera', False) and node.capture_due)
    if not cameras:
        return None
    if len(cameras) != 3 or any(not hasattr(node, '_project_camera_writer') for node in cameras):
        raise RuntimeError('Exact render capture requires all three RGB cameras on the same recording tick')
    update_camera_poses(_controller, force=True)
    packets, attempts = capture_exact_camera_packets(world, render_only, [node._project_camera_writer for node in cameras])
    return {'cameras': cameras, 'packets': packets, 'sim_time': world.current_time,
            'physics_step': world.current_time_step_index, 'attempts': attempts}


def publish_exact_camera_samples(world, prepared):
    pending = pending_direct_cameras()
    if prepared is None and not pending:
        return
    if (prepared is None or pending != prepared['cameras']
            or any(node._pending_capture_time != prepared['sim_time'] for node in pending)):
        raise RuntimeError('Verified camera packets and native state capture do not share the requested stamp')
    if len(pending) != 3 or len(prepared['packets']) != len(pending):
        raise RuntimeError('Publication requires a complete three-camera capture')
    stamp = prepared['sim_time']
    physics_dt = world.get_physics_dt()
    camera_stamp_for_render(stamp, world.current_time, physics_dt)
    # Finish encoding every camera before publishing any JPEG for this state.
    messages = [node.encode_exact_message(stamp, packet['rgb']) for node, packet in zip(pending, prepared['packets'])]
    rows = []
    for node, packet, message in zip(pending, prepared['packets'], messages):
        node._publisher.publish(message)
        node._pending_capture_time = None
        rows.append({'topic': node._topic, 'header_sim_time_s': stamp, 'world_sim_time_after_render_s': stamp,
                     'physics_dt_s': physics_dt, 'capture_physics_step': prepared['physics_step'],
                     'render_attempts': prepared['attempts'], 'render_reference_time': packet['reference_time'],
                     'render_sim_time_s': packet['simulation_time']})
    write_camera_render_boundary(rows)


def write_camera_render_boundary(rows):
    with (Path(os.environ['GENIE_EPISODE_EVIDENCE']) / 'camera-render-boundary.jsonl').open('a') as journal:
        for frame in rows:
            journal.write(json.dumps(frame) + '\n')


def publish_pending_camera_samples(world):
    cameras = pending_direct_cameras()
    if not cameras:
        return
    render_time = float(world.current_time)
    physics_dt = float(world.get_physics_dt())
    rows = [camera.publish_rendered_sample(render_time, physics_dt) for camera in cameras]
    write_camera_render_boundary(rows)


def install_direct_camera_recording(record_handler, controller, *, exact_render=False):
    """Publish the renderer's exact RGB sample as JPEG without a relay queue."""
    global _controller
    import inspect
    import cv2
    import numpy as np
    from sensor_msgs.msg import CompressedImage
    from common.base_utils.ros_nodes.sim_ros_node import SimPubRosNode,QOS_PROFILE_VOLATILE
    from server.ros_publisher import camera

    if 'GENIE_DIRECT_COMPRESSED_CAMERA' not in inspect.getsource(record_handler):
        raise RuntimeError('The native recorder needs the project direct-camera relay guard')
    _controller = controller

    class DirectCameraPublisher(SimPubRosNode):
        project_direct_camera = True

        def __init__(self, topic, get_msg_callback, frame_id, robot_name='G1_120s',
                     node_name='image_pub_node', step_size=1):
            super().__init__(CompressedImage, topic+'_compressed', get_msg_callback,
                             frame_id, robot_name, node_name, step_size)
            self._pending_capture_time = None

        def reset_publish_phase(self):
            super().reset_publish_phase()
            self._pending_capture_time = None

        @property
        def capture_due(self):
            return self._cur_step + 1 >= self._step_size

        def tick(self, current_time):
            # RGB sampling waits for the render which follows joint/TF capture.
            self._cur_step += 1
            if self._cur_step >= self._step_size:
                if self._pending_capture_time is not None:
                    raise RuntimeError('Cannot label a new capture before the previous scene rendered')
                self._cur_step = 0
                self._pending_capture_time = float(current_time)

        def publish_rendered_sample(self, render_time, physics_dt):
            requested = camera_stamp_for_render(self._pending_capture_time, render_time, physics_dt)
            self._cur_step = self._step_size - 1
            try:
                super().tick(requested)
            finally:
                self._pending_capture_time = None
            return {'topic': self._topic, 'header_sim_time_s': requested,
                    'world_sim_time_after_render_s': render_time, 'physics_dt_s': physics_dt}

        def create_publisher(self, message_type, topic, qos_profile, **kwargs):
            return super().create_publisher(message_type, topic, QOS_PROFILE_VOLATILE, **kwargs)

        def encode_exact_message(self, stamp, rgb):
            sec = int(stamp)
            self._header.stamp.sec = sec
            self._header.stamp.nanosec = int((stamp - sec) * 1e9)
            return self.handle_msg(rgb)

        def destroy_node(self):
            writer = getattr(self, '_project_camera_writer', None)
            if writer is not None:
                writer.detach()
            return super().destroy_node()

        def handle_msg(self, rgb):
            bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            ok, image = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError('Could not encode the live rendered frame')
            msg = CompressedImage()
            msg.header = self._header
            msg.format = 'rgb8; jpeg compressed bgr8'
            msg.data = image.reshape(-1)
            return msg

    camera.ImagePubRosNode = DirectCameraPublisher
    os.environ['GENIE_DIRECT_COMPRESSED_CAMERA'] = '1'
    recording = {'mode':'single_hop_jpeg','quality':95,'qos_history':'keep_last',
                 'qos_reliability':'reliable','qos_depth':QOS_PROFILE_VOLATILE.depth,
                 'sampling':'after_native_render_of_capture_step',
                 'controller_source':inspect.getsourcefile(record_handler)}
    if exact_render:
        import omni.replicator.core as rep

        class CameraPacketWriter(rep.Writer):
            def __init__(self):
                self.annotators = [rep.AnnotatorRegistry.get_annotator('rgb', device='cpu', do_array_copy=True),
                                   rep.AnnotatorRegistry.get_annotator('IsaacReadSimulationTime',
                                                                     init_params={'resetOnStop': True})]

            def write(self, data):
                # Replicator keeps the complete packet accessible through Writer.get_data().
                pass

        def publish_rgb_with_render_time(sensor, step_size, topic=''):
            frame_id = sensor.prim_path.rsplit('/', 1)[-1]
            node = DirectCameraPublisher(topic or f'/{frame_id}_rgb', None, frame_id,
                                         node_name=f'{frame_id}_rgb_node', step_size=step_size)
            writer = CameraPacketWriter()
            writer.attach(sensor.get_render_product_path(), trigger=None)
            node._project_camera_writer = writer
            return node

        camera.publish_rgb = publish_rgb_with_render_time
        recording['render_provenance'] = 'replicator_writer_simulation_time'
    return recording


def wait_for_recording_subscribers(nodes, timeout_s=15.):
    import time

    sensors = {node._topic:node._publisher for node in nodes
               if getattr(node,'project_direct_camera',False) or getattr(node,'_topic',None) in ('/joint_states','/tf')}
    if len(sensors) != 5:
        raise RuntimeError('Expected three cameras, joint states and TF for recorded collection')
    started = time.monotonic()
    while True:
        peers = {topic:publisher.get_subscription_count() for topic,publisher in sensors.items()}
        elapsed = time.monotonic()-started
        if all(peers.values()) or elapsed >= timeout_s:
            return {'passed':all(peers.values()),'wait_s':elapsed,'matched_subscribers':peers}
        time.sleep(.05)


def install():
    from server.motion_generator.motion_gen_reacher import CuroboMotion
    from server.command_controller import CommandController
    from pxr import Usd,UsdGeom,UsdPhysics
    import omni.usd
    from isaacsim.core.api.controllers.articulation_controller import ArticulationController
    from isaacsim.core.api import World
    from sim_backend import physics, to_numpy
    import numpy as np
    from scipy.spatial.transform import Rotation
    original_extract=CuroboMotion._extract_cached_obstacles
    def extract_physical_world(self,*args,**kwargs):
        result=original_extract(self,*args,**kwargs)
        # The phone scene contains a 100 m photographic HDR enclosure. It has
        # no PhysX CollisionAPI, but the USD mesh parser registers it as a solid
        # enclosing every robot pose. Exclude only this proven render backdrop;
        # keep the real table, task geometry and all enabled physical colliders.
        stage=omni.usd.get_context().get_stage();excluded=[]
        for path in list(CuroboMotion.cached_obstacle_info):
            if not path.startswith('/World/hdr_backdrop/'):continue
            prim=stage.GetPrimAtPath(path)
            if not prim:raise RuntimeError('Missing HDR prim during collision audit')
            enabled=prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
            if enabled:continue
            excluded.append({'path':path,'reason':'render_only_hdr_backdrop_without_enabled_physical_collision'})
            CuroboMotion.cached_obstacle_info.pop(path)
        evidence=Path(os.environ['GENIE_EPISODE_EVIDENCE']);evidence.mkdir(parents=True,exist_ok=True)
        policy=evidence/'planner-geometry-policy.json'
        previous=json.loads(policy.read_text()).get('excluded_render_only',[]) if policy.exists() else []
        excluded=list({r['path']:r for r in previous+excluded}.values())
        policy.write_text(json.dumps({'excluded_render_only':excluded,
            'retained_obstacle_paths':sorted(CuroboMotion.cached_obstacle_info),'physical_geometry_modified':False},indent=2))
        return result
    CuroboMotion._extract_cached_obstacles=extract_physical_world
    original_motion_init=CuroboMotion.__init__
    def motion_init(self,*args,**kwargs):
        original_motion_init(self,*args,**kwargs)
        self.max_attempts=int(os.environ.get('GENIE_PLANNER_MAX_ATTEMPTS','8'))
        self.plan_config.max_attempts=self.max_attempts
        self._reach_settle_max_delay_s=2.0
    CuroboMotion.__init__=motion_init
    original_slowdown=CommandController._resolve_slowdown_factor
    def bounded_speed(self,user_factor):
        # Native 1.34 s paths requested multi-radian wrist changes that the
        # physical drives could not follow. Resample commands without setting q.
        return original_slowdown(self,user_factor)*4.0
    CommandController._resolve_slowdown_factor=bounded_speed
    original_apply=ArticulationController.apply_action
    journal_path=Path(os.environ['GENIE_EPISODE_EVIDENCE'])/'native-control-commands.jsonl'
    journal_path.parent.mkdir(parents=True,exist_ok=True)
    journal=journal_path.open('a',buffering=1)
    strict_wrist = os.environ.get('GENIE_ENFORCE_G2_COUPLED_WRIST') == '1'
    guard_local = os.environ.get('GENIE_GUARD_LOCAL_JOINT_BRANCH') == '1'
    transport = None
    if strict_wrist or guard_local:
        original_paths = CuroboMotion._extract_candidate_paths

        def filter_g2_wrist(self, solution, flags, *, in_batch=False):
            before = int(flags.sum().item())
            flags.copy_(g2_trajectory_candidate_mask(solution.joint_names, solution.position, flags))
            remaining = int(flags.sum().item())
            witness = {'sim_time': float(World.instance().current_time),
                       'native_valid_candidates': before, 'wrist_compliant_candidates': remaining,
                       'coupled_limit_rad': G2_COUPLED_WRIST_LIMIT_RAD,
                       'stage': 'optimizer_attempt' if in_batch else 'final_selection'}
            with (journal_path.parent / 'coupled-wrist-plan-candidates.jsonl').open('a') as stream:
                stream.write(json.dumps(witness) + '\n')
            if not remaining and not in_batch:
                self.cmd_plan = None
                self.cmd_idx = 0
                raise ValueError('No collision-checked candidate also satisfies G2 coupled wrist limits')

        def filter_transport(self, solution, flags, request):
            from curobo.types import JointState

            tool = f"gripper_{request['arm'][0]}_center_link"
            current, _ = self._get_current_joint_state()
            names = list(current.joint_names)

            def tool_poses(joints):
                fk = self.planner.compute_kinematics(joints)
                pose = fk.tool_poses.get_link_pose(tool)
                return tool_pose_matrices(to_numpy(pose.position), to_numpy(pose.quaternion))

            predicted = request['world_from_robot'] @ tool_poses(current[0:1])[0]
            measured = request['live_tool_world']
            position_error = float(np.linalg.norm(predicted[:3, 3] - measured[:3, 3]))
            rotation_error = float(Rotation.from_matrix(predicted[:3, :3].T @ measured[:3, :3]).magnitude())
            if position_error > .003 or rotation_error > .03:
                raise ValueError(f'Transport clearance frame mismatch {position_error:.6f} m, {rotation_error:.6f} rad')
            fixture_from_robot = request['fixture_from_robot']
            paths = solution.position.reshape(-1, *solution.position.shape[-2:])
            mask = to_numpy(flags).reshape(-1).copy()
            if len(paths) != len(mask):
                raise ValueError('Native transport candidate count mismatch')
            clearances = []
            for i in np.flatnonzero(mask):
                joints = JointState.from_position(paths[i], joint_names=solution.joint_names).reorder(names)
                poses = fixture_from_robot @ tool_poses(joints)
                clearance = swept_payload_clearance(poses, request['payload_in_tool'],
                                                     request['payload_bounds'], request['fixture_bounds'])
                clearances.append(clearance)
                mask[i] = clearance >= request['margin_m']
            flags.logical_and_(flags.new_tensor(mask.reshape(flags.shape)))
            with (journal_path.parent / 'held-transport-clearance.jsonl').open('a') as stream:
                stream.write(json.dumps({
                    'sim_time': float(World.instance().current_time), 'arm': request['arm'],
                    'position_frame_error_m': position_error, 'rotation_frame_error_rad': rotation_error,
                    'margin_m': request['margin_m'], 'clearances_m': clearances,
                    'accepted_candidates': int(mask.sum())}) + '\n')
            if not mask.any():
                # The native failure path restores its temporary collision filters.
                self.cmd_plan = None
                self.cmd_idx = 0
                raise ValueError('No native path keeps the held object clear of the registered fixture')

        def extract_guarded_paths(self, result):
            if result is None or result.js_solution is None:
                return original_paths(self, result)
            if strict_wrist and 'G2' in self.robot_prim_path:
                # Filter before the native selector can fall back to unfiltered seeds.
                filter_g2_wrist(self, result.js_solution, result.success)
            if transport is not None:
                filter_transport(self, result.js_solution, result.success, transport)
            return original_paths(self, result)

        CuroboMotion._extract_candidate_paths = extract_guarded_paths
    if strict_wrist:
        original_endpoints = CuroboMotion.solve_batch_ik

        def solve_wrist_compatible_endpoints(self, positions, rotations, active_ee_name, output_link_pose=False):
            if 'G2' not in self.robot_prim_path:
                return original_endpoints(self, positions, rotations, active_ee_name, output_link_pose)
            import torch
            from curobo.types import JointState

            solver = self.planner.ik_solver
            solve = solver.solve_pose

            def solve_eligible(goal, **options):
                options['return_seeds'] = min(8, solver.config.num_seeds)
                result = solve(goal, **options)
                joints = result.js_solution
                native = to_numpy(result.success)
                indices, accepted = g2_endpoint_ik_seed(joints.joint_names, to_numpy(joints.position), native)
                device = joints.position.device
                row = torch.arange(len(indices), device=device)
                column = torch.as_tensor(indices, device=device, dtype=torch.long)
                chosen = joints.position[row, column][:, None, :].contiguous()
                result.js_solution = JointState.from_position(chosen, joint_names=joints.joint_names)
                result.success = result.success.new_tensor(accepted[:, None])
                with (journal_path.parent / 'coupled-wrist-endpoint-ik.jsonl').open('a') as stream:
                    stream.write(json.dumps({
                        'sim_time': float(World.instance().current_time), 'tool': active_ee_name,
                        'batch_size': len(indices), 'seeds_returned': int(native.shape[-1]),
                        'native_reachable_count': int(native.any(axis=1).sum()),
                        'wrist_reachable_count': int(accepted.sum()),
                        'wrist_seed_substitutions': int((indices > 0).sum()),
                    }) + '\n')
                return result

            solver.solve_pose = solve_eligible
            try:
                return original_endpoints(self, positions, rotations, active_ee_name, output_link_pose)
            finally:
                solver.solve_pose = solve

        CuroboMotion.solve_batch_ik = solve_wrist_compatible_endpoints
        original_calculate = CuroboMotion.calculate_ik_goal

        def calculate_with_wrist_filter(self, *args, **kwargs):
            if 'G2' not in self.robot_prim_path:
                return original_calculate(self, *args, **kwargs)
            solver = self.planner.trajopt_solver
            solve = solver.solve_pose

            def solve_admissible(*call_args, **call_kwargs):
                result = solve(*call_args, **call_kwargs)
                if result is not None and result.js_solution is not None and result.success.any():
                    # The batch planner retains the first successful path per target.
                    # Exclude wrist violations before it locks that batch entry.
                    filter_g2_wrist(self, result.js_solution, result.success, in_batch=True)
                return result

            solver.solve_pose = solve_admissible
            try:
                return original_calculate(self, *args, **kwargs)
            finally:
                solver.solve_pose = solve

        CuroboMotion.calculate_ik_goal = calculate_with_wrist_filter
    if guard_local:
        original_follow = CuroboMotion.follow_trajectory

        def guarded_follow(self, *args, **kwargs):
            if 'G2' not in self.robot_prim_path:
                return original_follow(self, *args, **kwargs)
            solver = self.planner.ik_solver
            solve = solver.solve_pose
            rejected_candidates = 0
            seed_attempts = 0
            waypoint_solves = 0
            first_unsolved = None

            def filter_local(result, current):
                nonlocal rejected_candidates
                allowed = continuous_ik_candidates(current.joint_names, to_numpy(result.solution),
                                                   to_numpy(current.position))
                mask = result.success.new_tensor(allowed)
                if strict_wrist:
                    mask &= g2_coupled_wrist_mask(current.joint_names, result.solution)
                native_count = int(result.success.sum().item())
                result.success.logical_and_(mask)
                count = int(result.success.sum().item())
                rejected_candidates += native_count - count
                return native_count, count

            def solve_continuous(*call_args, **call_kwargs):
                nonlocal seed_attempts, waypoint_solves, first_unsolved
                result = solve(*call_args, **call_kwargs)
                current = call_kwargs.get('current_state')
                if current is None:
                    return result
                waypoint = waypoint_solves
                waypoint_solves += 1
                raw, retained = filter_local(result, current)
                if raw and not retained and seed_attempts < 4:
                    seed_attempts += 1
                    retry = solve_from_local_joint_seed(solver, solve, call_args, call_kwargs)
                    seeded_raw, seeded_retained = filter_local(retry, current)
                    with (journal_path.parent / 'locally-seeded-ik.jsonl').open('a') as stream:
                        stream.write(json.dumps({'sim_time': float(World.instance().current_time),
                            'initial_native_successes': raw, 'seeded_native_successes': seeded_raw,
                            'seeded_continuous_wrist_successes': seeded_retained,
                            'attempt_in_local_plan': seed_attempts}) + '\n')
                    if seeded_retained:
                        return retry
                if not retained and first_unsolved is None:
                    first_unsolved = {'waypoint': waypoint, 'native_ik_solutions': raw,
                                      'joint_filtered_solutions': retained}
                return result

            solver.solve_pose = solve_continuous
            try:
                result = original_follow(self, *args, **kwargs)
            finally:
                solver.solve_pose = solve
            plan = self.cmd_plan
            witness = {}
            if self.success and plan is not None:
                measured = to_numpy(physics.get_joint_positions(self.robot, as_torch=False)).reshape(-1)
                plan_positions = to_numpy(plan.position)
                witness = largest_arm_joint_step(plan.joint_names, plan_positions, physics.get_dof_names(self.robot), measured)
                if strict_wrist:
                    witness['coupled_wrist_passed'] = bool(g2_coupled_wrist_mask(plan.joint_names, plan_positions).all())
            passed = bool(witness) and witness['step_rad'] <= MAX_LOCAL_ARM_JOINT_STEP_RAD
            passed = passed and witness.get('coupled_wrist_passed', True)
            record = {'passed': passed, **witness, 'limit_rad': MAX_LOCAL_ARM_JOINT_STEP_RAD,
                      'ik_candidates_rejected_by_continuity': rejected_candidates,
                      'local_seed_attempts': seed_attempts, 'first_unsolved_waypoint': first_unsolved,
                      'sim_time': float(World.instance().current_time)}
            with (journal_path.parent / 'local-joint-plan-continuity.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            if not passed:
                # The native follower has only constructed a plan at this point.
                # Reject it before the first drive command is issued.
                self.cmd_plan = None
                self.cmd_idx = 0
                self.reached = True
                self.success = False
            return result

        CuroboMotion.follow_trajectory = guarded_follow
    def apply(self,*args,**kwargs):
        action=args[0] if args else kwargs.get('control_actions',kwargs.get('action'))
        view=getattr(self,'_articulation_view',None)
        measured=to_numpy(view.get_joint_positions())[0].tolist()
        result=original_apply(self,*args,**kwargs)
        row={'sim_time':float(World.instance().current_time)}
        row['measured_joint_positions_before']=measured
        row['joint_names']=list(getattr(view,'dof_names',[]))
        for field in ('joint_indices','joint_positions','joint_velocities','joint_efforts'):
            value=getattr(action,field,None)
            row[field]=None if value is None else to_numpy(value).tolist()
        applied=self.get_applied_action()
        for field in ('joint_positions','joint_velocities','joint_efforts'):
            value=getattr(applied,field,None)
            row['applied_'+field]=None if value is None else to_numpy(value).tolist()
        kp,kd=view.get_gains()
        row['drive_stiffness']=to_numpy(kp).reshape(-1).tolist()
        row['drive_damping']=to_numpy(kd).reshape(-1).tolist()
        row['drive_max_efforts']=to_numpy(view.get_max_efforts()).reshape(-1).tolist()
        journal.write(json.dumps(row)+'\n')
        return result
    ArticulationController.apply_action=apply
    original_callback=CommandController.callback_physics
    def track_objects(self,*args,**kwargs):
        paths=getattr(self,'_project_track_paths',{})
        tcp_arm=getattr(self,'_project_track_tcp_arm',None)
        if paths or tcp_arm:
            poses={}
            for key,path in paths.items():
                p,q=self._get_object_pose(path);t=np.eye(4)
                t[:3,3]=to_numpy(p);t[:3,:3]=Rotation.from_quat(to_numpy(q)[[1,2,3,0]]).as_matrix();poses[key]=t.tolist()
            if tcp_arm:
                position,quaternion=self._get_ee_pose(tcp_arm=='right')
                tcp=np.eye(4)
                tcp[:3,3]=to_numpy(position)
                tcp[:3,:3]=Rotation.from_quat(to_numpy(quaternion)[[1,2,3,0]]).as_matrix()
                poses['tcp']=tcp.tolist()
            self._project_track_file.write(json.dumps({'sim_time':float(self.ui_builder.my_world.current_time),'poses':poses})+'\n')
        return original_callback(self,*args,**kwargs)
    CommandController.callback_physics=track_objects
    original_attach=CuroboMotion.attach_obj
    original_add=CuroboMotion.add_obstacle_from_prim_path
    def add_obstacle(self,prim_path,usd_path):
        if not prim_path.startswith(('/World/background/','/World/Background/')):
            return original_add(self,prim_path,usd_path)
        saved=self.ignore_substring_list
        self.ignore_substring_list=[x for x in saved if x not in ['/World/background','/World/Background']]
        try:return original_add(self,prim_path,usd_path)
        finally:self.ignore_substring_list=saved
    CuroboMotion.add_obstacle_from_prim_path=add_obstacle
    original_obstacles=CuroboMotion.set_obstacles
    original_pose_update=CuroboMotion._update_obstacle_poses_fast
    def shared_mapping(self):return getattr(self.planner,'_project_attachments',{})
    def update_obstacle_poses(self):
        scene,updated=original_pose_update(self)
        return scene,updated or getattr(self,'_project_force_world_refresh',False)
    CuroboMotion._update_obstacle_poses_fast=update_obstacle_poses
    def obstacles(self, *, force=False):
        mapping=shared_mapping(self)
        self._project_attachments=mapping
        self.attached_objects=list(dict.fromkeys(p for paths in mapping.values() for p in paths))
        if not force:return original_obstacles(self)
        self._project_force_world_refresh=True
        try:return original_obstacles(self)
        finally:self._project_force_world_refresh=False
    CuroboMotion.set_obstacles=obstacles
    def detach(self):
        selected=getattr(self,'_project_detach_link',None)
        mapping=shared_mapping(self).copy()
        links=[selected] if selected else ['attached_object','left_attached_object']
        released={link:mapping[link] for link in links if link in mapping}
        if not released:return []
        restored=sorted({path for paths in released.values() for path in paths})
        remaining={link:paths for link,paths in mapping.items() if link not in released}
        # cuRobo's manager enables the exact world obstacles during detach.
        # Held geometry was omitted from the last world rebuild, so restore its
        # real current pose and collider names before the native manager looks them up.
        self.planner._project_attachments=remaining
        try:
            for path in restored:
                if path not in CuroboMotion.cached_obstacle_info:self.add_obstacle_from_prim_path(path,None)
            self.set_obstacles(force=True)
            world=set(self.planner.scene_collision_checker.get_obstacle_names(env_idx=0) or [])
            if not set(restored)<=world:raise RuntimeError('Missing native obstacles for planner detach: '+str(sorted(set(restored)-world)))
            for link,paths in released.items():
                for manager in self._attachment_managers():
                    manager.detach(link_name=link,enable_obstacle_names=paths)
        except Exception:
            self.planner._project_attachments=mapping
            self.set_obstacles(force=True)
            raise
        with (journal_path.parent/'planner-detachment-events.jsonl').open('a') as f:
            f.write(json.dumps({'removed_links':sorted(released),'restored_world_obstacles':restored,
                'remaining_attachments':remaining,'success':True})+'\n')
        return restored
    def attach(self,prim_paths,link_name='attached_object',ee_position=[0,0,0],ee_rotation=[1,0,0,0]):
        # Contact planning may exclude this mesh. Reintroduce its geometry for
        # the attachment manager to fit; the manager then disables its world
        # slot and carries it with the arm. No physical attachment is created.
        for path in prim_paths:
            if path not in CuroboMotion.cached_obstacle_info:self.add_obstacle_from_prim_path(path,None)
        self.set_obstacles()
        self._project_detach_link=link_name
        try:success=original_attach(self,prim_paths,link_name,ee_position,ee_rotation)
        finally:self._project_detach_link=None
        # The native attach only appends names whose meshes are in its world.
        # Unregistered render meshes must stay outside the active link mapping.
        attached_paths=[path for path in prim_paths if success and path in self.attached_objects]
        success=bool(success and attached_paths)
        if success:
            mapping=shared_mapping(self)
            mapping[link_name]=attached_paths;self._project_attachments=mapping
            self.planner._project_attachments=mapping
            self.attached_objects=list(dict.fromkeys(p for paths in mapping.values() for p in paths))
        with (journal_path.parent/'planner-attachment-events.jsonl').open('a') as f:
            f.write(json.dumps({'link':link_name,'paths':attached_paths,'requested_paths':list(prim_paths),'success':success})+'\n')
        return success
    CuroboMotion.detach_obj=detach
    CuroboMotion.attach_obj=attach
    old_command=CommandController.handle_command_any
    def command(self):
        nonlocal transport
        global _controller
        _controller=self
        cfg=self.data['command_config'];name=cfg.get('command')
        if name=='ProjectTransportClearance':
            if not guard_local:
                raise RuntimeError('Native transport clearance guard is unavailable')
            if cfg.get('enabled') is False:
                transport=None
                self.data_to_send={'success': True, 'enabled': False}
                return
            request={key: np.asarray(cfg[key], dtype=float) for key in (
                'live_tool_world','fixture_world','payload_in_tool','payload_bounds','fixture_bounds')}
            for key, value in request.items():
                expected_shape=(2,3) if key.endswith('_bounds') else (4,4)
                if value.shape!=expected_shape or not np.isfinite(value).all():
                    raise ValueError('Invalid transport clearance geometry: '+key)
            if any((request[key][1] <= request[key][0]).any() for key in ('payload_bounds','fixture_bounds')):
                raise ValueError('Transport clearance bounding boxes must have positive volume')
            margin=float(cfg['margin_m'])
            if cfg['arm'] not in ('left','right') or not 0. <= margin <= .01:
                raise ValueError('Invalid transport clearance arm or margin')
            world_from_robot=np.eye(4)
            world_from_robot[:3,3]=to_numpy(self.robot_init_position)
            wxyz=to_numpy(self.robot_init_rotation)
            world_from_robot[:3,:3]=Rotation.from_quat(wxyz[[1,2,3,0]]).as_matrix()
            transport={**request, 'arm': cfg['arm'], 'margin_m': margin, 'world_from_robot': world_from_robot,
                       'fixture_from_robot': np.linalg.inv(request['fixture_world']) @ world_from_robot}
            self.data_to_send={'success': True, 'enabled': True}
            return
        if name=='ProjectProbeSupport':
            from omni.physx import get_physx_scene_query_interface
            support=cfg['path']
            lo,hi=np.asarray(cfg['bounds_xy_m'],dtype=float)
            expected=float(cfg['height_m'])
            if (not support.startswith('/World/') or not np.isfinite([lo,hi]).all() or
                    not np.isfinite(expected) or np.any(hi<=lo)):
                raise ValueError('Invalid support probe geometry')
            query=get_physx_scene_query_interface()
            hits=[]
            for u,v in product((.17,.5,.83),repeat=2):
                x,y=lo+(hi-lo)*[u,v]
                hit=probe_support_ray(query,(float(x),float(y),expected+.20),support)
                if hit is not None:
                    hits.append(hit)
            self.data_to_send={'support':support,'expected_height_m':expected,'hits':hits};return
        if name=='ProjectGripperConfiguration':
            self._init_grippers();contracts={}
            for side,gripper in [('left',self.gripper_L),('right',self.gripper_R)]:
                contracts[side]={'joint_names':list(gripper._joint_prim_names),
                    'opened_positions':np.asarray(gripper.joint_opened_positions).tolist(),
                    'closed_positions':np.asarray(gripper.joint_closed_positions).tolist(),
                    'actuator':gripper._joint_control_prim.rsplit('/',1)[-1],
                    'type':gripper.gripper_type}
            self.data_to_send={'success':True,'grippers':contracts};return
        if name == 'ProjectGripUnload':
            from isaacsim.core.utils.types import ArticulationAction
            from sim_backend import physics

            arm = cfg['arm']
            fraction = float(cfg['effort_fraction'])
            if arm not in ('left', 'right') or not 0. < fraction <= 1.:
                raise ValueError('Invalid grip unloading command')
            robot = self._init_grippers()
            view = robot._articulation_view
            gripper = self.gripper_R if arm == 'right' else self.gripper_L
            if fraction == 1.:
                actuator = gripper._joint_control_prim.rsplit('/', 1)[-1]
                names = physics.get_dof_names(robot)
                index = names.index(actuator)
                finger = gripper._joint_prim_names.index(actuator)
                positions = to_numpy(physics.get_joint_positions(robot, as_torch=False)).reshape(-1)
                gains = [float(to_numpy(g).reshape(-1)[index]) for g in view.get_gains()]
                effort = float(to_numpy(view.get_max_efforts()).reshape(-1)[index])
                velocity = float(gripper._joint_closed_velocities[finger])
                gap = float(gripper.joint_opened_positions[finger] - positions[index])
                if gains[0] != 0. or gains[1] <= 0. or not np.isfinite(effort) or effort <= 0. or not velocity * gap < 0.:
                    raise ValueError('Unload requires an active native closing velocity drive')
                self._project_unload = {'arm': arm, 'index': index, 'actuator': actuator,
                                        'effort': effort, 'velocity': velocity, 'last_fraction': 1.}
            unload = self._project_unload
            if unload['arm'] != arm or fraction > unload['last_fraction']:
                raise ValueError('Unload must continue on the same actuator and decrease effort')
            index = unload['index']
            indices = np.asarray([index])
            target_effort = unload['effort'] * fraction
            if fraction < 1.:
                physics.set_max_efforts(robot, np.asarray([target_effort]), joint_indices=indices)
                position = to_numpy(physics.get_joint_positions(robot, as_torch=False)).reshape(-1)[index]
                physics.apply_action(robot, ArticulationAction(
                    joint_positions=np.asarray([position]),
                    joint_velocities=np.asarray([unload['velocity']]), joint_indices=indices))
            unload['last_fraction'] = fraction
            actual_effort = float(to_numpy(view.get_max_efforts()).reshape(-1)[index])
            self.data_to_send = {'success': bool(np.isclose(actual_effort, target_effort, atol=.0001)),
                                 'sim_time_s': float(World.instance().current_time),
                                 'actuator': unload['actuator'], 'closing_velocity_rad_s': unload['velocity'],
                                 'max_effort_Nm': actual_effort}
            return
        if name=='ProjectPrepareFingerRelease':
            from sim_backend import physics
            self.robot=self._init_grippers();side=cfg['arm']
            if side not in ['left','right']:raise ValueError('Invalid arm')
            gripper=self.gripper_R if side=='right' else self.gripper_L
            ids=np.asarray(gripper._joint_dof_indicies[:2],dtype=int)
            current=to_numpy(physics.get_joint_positions(self.robot,as_torch=False)).reshape(-1)[ids]
            lo,hi=physics.get_dof_limits(self.robot,as_torch=False)
            target=np.clip(np.asarray(gripper.joint_opened_positions[:2],dtype=float),to_numpy(lo)[ids],to_numpy(hi)[ids])
            kp,kd=self.robot._articulation_view.get_gains()
            original={'stiffness':to_numpy(kp).reshape(-1)[ids].tolist(),'damping':to_numpy(kd).reshape(-1)[ids].tolist()}
            hold_command=None
            if side=='right':
                from isaacsim.core.utils.types import ArticulationAction
                velocity=np.zeros_like(current)
                # Clear the old close targets while the existing gains are in
                # effect. The next physics step uses both new position drives.
                physics.apply_action(self.robot,ArticulationAction(
                    joint_positions=current.copy(),joint_velocities=velocity,joint_indices=ids))
                hold_command={'position_rad':current.tolist(),'velocity_rad_s':velocity.tolist(),
                              'joint_indices':ids.tolist()}
            physics.set_gains(self.robot,kps=np.full(2,100.),kds=np.full(2,5.),joint_indices=ids)
            physics.set_max_efforts(self.robot,np.full(2,5.),joint_indices=ids)
            self._project_finger_release={'arm':side,'indices':ids,'initial':current.copy(),'target':target,'fraction':0.}
            self.data_to_send={'success':True,'arm':side,'initial_rad':current.tolist(),'target_rad':target.tolist(),
                'original_gains':original,'hold_command_before_gain_switch':hold_command,
                'release_gains':{'stiffness':100.,'damping':5.,'max_effort_Nm':5.}}
            return
        if name=='ProjectFingerReleaseTarget':
            from sim_backend import physics
            from isaacsim.core.utils.types import ArticulationAction
            session=self._project_finger_release;fraction=float(cfg['fraction'])
            if not session['fraction']<=fraction<=1.:raise ValueError('Opening must advance monotonically')
            target=session['initial']+fraction*(session['target']-session['initial'])
            physics.apply_action(self.robot,ArticulationAction(joint_positions=target,joint_velocities=np.zeros(2),joint_indices=session['indices']))
            session['fraction']=fraction
            current=to_numpy(physics.get_joint_positions(self.robot,as_torch=False)).reshape(-1)[session['indices']]
            self.data_to_send={'success':True,'fraction':fraction,'actual_rad':current.tolist(),'target_rad':target.tolist()}
            return
        if name=='ProjectTrackObjects':
            paths=cfg.get('paths',{})
            tcp_arm=cfg.get('tcp_arm')
            if not isinstance(paths,dict) or any(not isinstance(path,str) or not path.startswith('/World/Objects/')
                                                  for path in paths.values()):
                raise ValueError('Track outside task objects')
            if tcp_arm not in (None,'left','right') or tcp_arm and 'tcp' in paths:
                raise ValueError('Invalid TCP tracking arm or key')
            self._project_track_paths=paths
            self._project_track_tcp_arm=tcp_arm
            if not hasattr(self,'_project_track_file'):
                self._project_track_file=(journal_path.parent/'object-physics-120hz.jsonl').open('a',buffering=1)
            self.data_to_send={'success':True,'tracked':list(paths)+(['tcp'] if tcp_arm else [])};return
        if name=='ProjectGripperCollisionBounds':
            side={'left':'l','right':'r'}[cfg['arm']]
            stage=omni.usd.get_context().get_stage()
            xforms=UsdGeom.XformCache()
            bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy])
            shapes=[]
            for link in stage.GetPrimAtPath('/G2').GetChildren():
                if not link.GetName().startswith('gripper_'+side+'_'):continue
                colliders=[prim for prim in Usd.PrimRange(link,Usd.TraverseInstanceProxies())
                           if prim.IsA(UsdGeom.Boundable) and prim.HasAPI(UsdPhysics.CollisionAPI)
                           and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]
                if not colliders:continue
                authored_link=np.asarray(xforms.GetLocalToWorldTransform(link)).T
                p,q=self._get_object_pose(str(link.GetPath()))
                live_link=np.eye(4);live_link[:3,3]=to_numpy(p)
                live_link[:3,:3]=Rotation.from_quat(to_numpy(q)[[1,2,3,0]]).as_matrix()
                for prim in colliders:
                    extents=bounds.ComputeUntransformedBound(prim).ComputeAlignedRange()
                    if extents.IsEmpty():raise RuntimeError('Enabled gripper collider has no bounds: '+str(prim.GetPath()))
                    mount=np.linalg.inv(authored_link)@np.asarray(xforms.GetLocalToWorldTransform(prim)).T
                    world=live_link@mount
                    points=np.asarray(list(product(*zip(extents.GetMin(),extents.GetMax()))))
                    corners=points@world[:3,:3].T+world[:3,3]
                    shapes.append({'path':str(prim.GetPath()),'corners_world_m':corners.tolist()})
            self.data_to_send={'success':bool(shapes),'shapes':shapes,'sim_time':float(self.ui_builder.my_world.current_time)};return
        if name=='ProjectKinematicSnapshot':
            from sim_backend import physics
            names=physics.get_dof_names(self.ui_builder.articulation)
            joints=to_numpy(physics.get_joint_positions(self.ui_builder.articulation,as_torch=False)).reshape(-1)
            lo,hi=physics.get_dof_limits(self.ui_builder.articulation,as_torch=False)
            limits=np.stack([to_numpy(lo),to_numpy(hi)],axis=-1)
            links={};usd_links={}
            cache=UsdGeom.XformCache()
            stage=omni.usd.get_context().get_stage()
            for side in ['l','r']:
                for suffix in ['arm_'+side+'_end_link','gripper_'+side+'_base_link',
                    *['gripper_'+side+'_'+part+'_link'+str(n) for part in ['inner','outer','support'] for n in [1,2]]]:
                    p,q=self._get_object_pose('/G2/'+suffix)
                    links[suffix]={'position':to_numpy(p).tolist(),'quaternion_wxyz':to_numpy(q).tolist()}
                    usd_links[suffix]=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/G2/'+suffix))).T.tolist()
            self.data_to_send={'success':True,'joint_names':names,'joint_positions':joints.tolist(),
                'native_limits':limits.tolist(),'native_limits_layout':'joint_min_max','links':links,'usd_links':usd_links,'robot_base_position':to_numpy(self.robot_init_position).tolist(),
                'robot_base_quaternion_wxyz':to_numpy(self.robot_init_rotation).tolist()}
            kp,kd=self.ui_builder.articulation._articulation_view.get_gains()
            self.data_to_send['drive_stiffness']=to_numpy(kp).tolist()
            self.data_to_send['drive_damping']=to_numpy(kd).tolist()
            return
        if name=='ProjectSnapshot':
            stage=omni.usd.get_context().get_stage();attributes={}
            for path in cfg.get('joint_paths',[]):
                if not path.startswith('/World/Objects/'):raise ValueError('Diagnostic path outside task objects')
                prim=stage.GetPrimAtPath(path)
                attr=prim.GetAttribute('physics:jointEnabled') if prim else None
                attributes[path]=bool(attr.Get()) if attr else False
            mappings={}
            obstacles=set()
            for motion in self.ui_builder.curoboMotion.values():
                mappings.update(shared_mapping(motion));motion.set_obstacles()
                obstacles.update(motion.planner.scene_collision_checker.get_obstacle_names(env_idx=0) or [])
            self.data_to_send={'success':True,'attach_states':self.attach_states.copy(),'planner_attachments':mappings,
                'planner_obstacles':sorted(obstacles),'joint_enabled':attributes,'sim_time':self.get_current_simulation_time()}
            return
        if name=='ProjectDetachPlannerArm':
            right=cfg['arm']=='right';link='attached_object' if right else 'left_attached_object'
            seen=set();restored=set();remaining={}
            for motion in self.ui_builder.curoboMotion.values():
                if id(motion) in seen:continue
                seen.add(id(motion));motion._project_detach_link=link
                try:restored.update(motion.detach_obj())
                finally:motion._project_detach_link=None
                remaining.update(shared_mapping(motion))
            self.attach_states={p:r for p,r in self.attach_states.items() if r!=right}
            self.data_to_send={'success':True,'attach_states':self.attach_states.copy(),
                'restored_world_obstacles':sorted(restored),'planner_attachments':remaining};return
        return old_command(self)
    CommandController.handle_command_any=command
    original_restore=CommandController.restore_state
    def restore(self,state):
        if self.playback_waited_frame_num==0 and state and not state.get('attach_states') and self.attach_states:
            self.ui_builder.detach_objs();self.attach_states={}
        return original_restore(self,state)
    CommandController.restore_state=restore
