"""Compare original head-camera JPEGs with the star's recorded physics mesh.

Example for V38:
    python audit_star_camera_reprojection.py RUN 124.34 124.65 --baseline-end 124.45

Requires numpy, Pillow, rosbags and scipy. Reads one indexed MCAP window and
prints a JSON receipt. The lag fit is a diagnostic estimate from the upper edge
of the purple region. Occlusion can remove pixels from the projected mesh.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path


CAMERA = 'head_front_Camera'
CAMERA_TOPIC = f'/{CAMERA}_rgb_compressed'


def head_messages(recording, start, stop):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    with AnyReader([recording], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
        tf, = (item for item in reader.connections if item.topic == '/tf')
        rgb, = (item for item in reader.connections if item.topic == CAMERA_TOPIC)
        low, high = reader.start_time, reader.end_time
        # Seek by the MCAP receipt clock. The ROS simulation clock is in the payload.
        while high - low > 250_000_000:
            mid = (low + high) // 2
            row = next(reader.messages(connections=[tf], start=mid), None)
            if row and stamp_ns(reader.deserialize(row[2], tf.msgtype).transforms[0].header) < start * 1e9:
                low = mid
            else:
                high = mid
        for connection, _, blob in reader.messages(connections=[tf, rgb], start=max(reader.start_time, low - 1_000_000_000)):
            message = reader.deserialize(blob, connection.msgtype)
            header = message.header if connection == rgb else message.transforms[0].header
            ns = stamp_ns(header)
            if ns * 1e-9 > stop + .2:
                break
            if start <= ns * 1e-9 <= stop:
                yield connection.topic, ns, message


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def purple_pixels(jpeg, roi):
    import numpy as np
    from PIL import Image

    x0, y0, x1, y1 = roi
    with Image.open(io.BytesIO(jpeg)) as image:
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise ValueError('ROI must lie inside the camera image')
        pixels = np.asarray(image.convert('RGB').crop(roi), dtype=np.int16)
    red, green, blue = np.moveaxis(pixels, -1, 0)
    y, x = np.where((blue > 30) & (blue > red + 15) & (blue > green + 15) & (red > green + 3))
    if not y.size:
        return {'area_px': 0}
    return {'area_px': int(y.size), 'y_top_px': int(y.min() + y0),
            'y_centroid_px': round(float(y.mean() + y0), 3),
            'bbox_xyxy_px': [int(x.min() + x0), int(y.min() + y0), int(x.max() + x0), int(y.max() + y0)]}


def projected_top(vertices, body, camera, lens):
    xyz = (vertices @ body[:3, :3].T + body[:3, 3] - camera[0]) @ camera[1]
    if (xyz[:, 2] <= 0).any():
        raise ValueError('Star vertices fall behind the camera')
    x = xyz[:, 0] / xyz[:, 2]
    y = xyz[:, 1] / xyz[:, 2]
    r2 = x * x + y * y
    radial = 1 + r2 * (lens['k1'] + r2 * (lens['k2'] + r2 * lens['k3']))
    distorted_y = y * radial + lens['p1'] * (r2 + 2 * y * y) + 2 * lens['p2'] * x * y
    return float(lens['Cy'] + lens['Fy'] * distorted_y.min())


def audit(run, start, stop, baseline_end, roi, max_lag_steps):
    import numpy as np
    from scipy.spatial.transform import Rotation

    if not start < baseline_end < stop or max_lag_steps < 1:
        raise ValueError('Expected start < baseline end < stop and a positive lag search window')
    evidence = run / 'evidence'
    raw = run / 'raw'
    certificate = json.loads((run / 'source/insertion-certificate.json').read_text())
    vertices = np.asarray(certificate['star_triangles_entity_frame'], dtype=float).reshape(-1, 3)
    lens = json.loads((raw / 'parameters/sensor/intrinsic_head_front_rgb.json').read_text())
    with (evidence / 'camera-render-boundary.jsonl').open() as stream:
        boundary = [row for row in map(json.loads, stream)
                    if row['topic'] == CAMERA_TOPIC and start <= row['header_sim_time_s'] <= stop]
    if not boundary:
        raise ValueError('No head-camera render boundary records in this window')
    physics_dt = boundary[0]['physics_dt_s']

    def physics_step(seconds):
        return round(seconds / physics_dt)

    with (evidence / 'object-physics-120hz.jsonl').open() as stream:
        physics = {physics_step(row['sim_time']): np.asarray(row['poses'][certificate['star']], dtype=float)
                   for row in map(json.loads, stream)
                   if start - (max_lag_steps + 1) * physics_dt <= row['sim_time'] <= stop + 2 * physics_dt}

    images, cameras = {}, {}
    for topic, ns, message in head_messages(raw, start, stop):
        if topic == CAMERA_TOPIC:
            if ns in images:
                raise ValueError(f'Duplicate original head JPEG header {ns}')
            jpeg = bytes(message.data)
            images[ns] = {**purple_pixels(jpeg, roi), 'jpeg_sha256': hashlib.sha256(jpeg).hexdigest()}
        else:
            if ns in cameras:
                raise ValueError(f'Duplicate original camera TF header {ns}')
            camera, = (row.transform for row in message.transforms
                       if row.header.frame_id == 'world' and row.child_frame_id == CAMERA and stamp_ns(row.header) == ns)
            p, q = camera.translation, camera.rotation
            cameras[ns] = (np.array([p.x, p.y, p.z]), Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix())
    render_rows = {physics_step(row['header_sim_time_s']): row for row in boundary}
    if (len(render_rows) != len(boundary) or images.keys() != cameras.keys()
            or {physics_step(ns * 1e-9) for ns in images} != render_rows.keys()):
        raise ValueError('Expected original JPEG, camera TF and render boundary for every window frame')

    measurements, projections = [], []
    for ns in sorted(images):
        seconds = ns * 1e-9
        step = physics_step(seconds)
        render_row = render_rows[step]
        if abs(render_row['header_sim_time_s'] - seconds) > 2e-9:
            raise ValueError(f'Render boundary does not match the original camera header at {seconds:.9f} s')
        render_time = render_row['world_sim_time_after_render_s']
        poses = [physics[step - lag] for lag in range(max_lag_steps + 1)]
        tops = [projected_top(vertices, pose, cameras[ns], lens) for pose in poses]
        projections.append(tops)
        measurements.append({**images[ns], 'ros_header_s': seconds, 'world_after_render_s': render_time,
                             'star_origin_z_m': float(poses[0][2, 3]), 'projected_top_at_header_px': tops[0],
                             'projected_top_after_render_px': projected_top(
                                 vertices, physics[physics_step(render_time)], cameras[ns], lens)})

    baseline = [row for row in measurements if row['ros_header_s'] <= baseline_end and row['area_px'] > 0]
    if len(baseline) < 2:
        raise ValueError('Choose a baseline with at least two visibly stationary frames')
    minimum_area = np.median([row['area_px'] for row in baseline]) / 2
    visible = [i for i, row in enumerate(measurements) if row['area_px'] >= minimum_area]
    if sum(measurements[i]['ros_header_s'] > baseline_end for i in visible) < 2:
        raise ValueError('Choose a window with at least two sufficiently visible frames after the baseline')
    bias = float(np.median([row['y_top_px'] - row['projected_top_at_header_px'] for row in baseline]))
    delta = np.array([measurements[i]['y_top_px'] - bias for i in visible])[:, None] - np.array(projections)[visible]
    errors = np.sqrt(np.mean(np.square(delta), axis=0))
    best_lag = int(np.argmin(errors))
    for row, tops in zip(measurements, projections):
        row['projected_top_at_best_lag_px'] = tops[best_lag]
        if row['area_px']:
            row['pixels_above_unoccluded_projection_px'] = row['projected_top_at_header_px'] - row['y_top_px']
    return {'run': str(run), 'camera_topic': CAMERA_TOPIC, 'physics_step_s': physics_dt,
            'mesh_vertex_count': len(vertices), 'baseline_bias_px': bias, 'lag_fit_frames': len(visible),
            'best_lag_physics_steps': best_lag, 'best_lag_from_header_ms': best_lag * physics_dt * 1e3,
            'top_edge_rmse_px_by_lag_step': errors.tolist(), 'measurements': measurements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('start', type=float, help='Window start in ROS simulation seconds')
    parser.add_argument('stop', type=float, help='Window stop in ROS simulation seconds')
    parser.add_argument('--baseline-end', type=float, required=True, help='Last time of the stationary baseline')
    parser.add_argument('--roi', type=int, nargs=4, default=(286, 146, 337, 195), metavar=('X0', 'Y0', 'X1', 'Y1'))
    parser.add_argument('--max-lag-steps', type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.start, args.stop, args.baseline_end, args.roi, args.max_lag_steps), indent=2))


if __name__ == '__main__':
    main()
