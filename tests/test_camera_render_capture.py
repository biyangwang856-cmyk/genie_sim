import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import project_runtime_hooks as hooks


POSE_TIME = 124.54167316202074
RENDER_TIME = 124.54166666666667
PHYSICS_DT = 1 / 120


def writer_payload(simulation_time, reference_time, pixel_value):
    return {
        'reference_time': reference_time,
        'rgb': {'data': np.full((2, 3, 4), pixel_value, dtype=np.uint8)},
        'IsaacReadSimulationTime': {'simulationTime': simulation_time},
    }


class Writer:
    def __init__(self):
        self.payload = writer_payload(RENDER_TIME - 7 * PHYSICS_DT, (10, 60), 7)

    def get_data(self):
        return self.payload

    def schedule_write(self):
        pass


class CameraRenderCaptureTests(unittest.TestCase):
    def setUp(self):
        self.world = SimpleNamespace(current_time=POSE_TIME, current_time_step_index=14945,
                                     get_physics_dt=lambda: PHYSICS_DT)
        self.writers = [Writer(), Writer(), Writer()]
        self.render_count = 0

    def test_waits_for_all_new_packets_at_the_same_physics_step(self):
        def render_only():
            self.render_count += 1
            if self.render_count == 1:
                self.writers[0].payload = writer_payload(RENDER_TIME, (11, 60), 32)
                self.writers[1].payload = writer_payload(RENDER_TIME, (11, 60), 64)
                self.writers[2].payload = writer_payload(RENDER_TIME + PHYSICS_DT, (11, 60), 7)
            else:
                self.writers[2].payload = writer_payload(RENDER_TIME, (12, 60), 96)

        packets, attempts = hooks.capture_exact_camera_packets(self.world, render_only, self.writers)
        self.assertEqual(attempts, 2)
        self.assertEqual([packet['reference_time'] for packet in packets], [(11, 60), (11, 60), (12, 60)])
        for index, packet in enumerate(packets):
            self.writers[index].payload['rgb']['data'][:] = 255
            np.testing.assert_array_equal(packet['rgb'], np.full((2, 3, 3), (32, 64, 96)[index]))
            self.assertEqual(packet['simulation_time'], RENDER_TIME)
        self.assertEqual((self.world.current_time, self.world.current_time_step_index), (POSE_TIME, 14945))

    def test_rejects_a_cached_reference_even_with_the_requested_time(self):
        self.writers[2].payload = writer_payload(RENDER_TIME, (10, 60), 7)

        def render_only():
            self.render_count += 1
            self.writers[0].payload = writer_payload(RENDER_TIME, (11, 60), 32)
            self.writers[1].payload = writer_payload(RENDER_TIME, (11, 60), 64)
            self.writers[2].payload = writer_payload(RENDER_TIME, (1, 6), 7)

        with self.assertRaisesRegex(RuntimeError, 'matching RGB packets'):
            hooks.capture_exact_camera_packets(self.world, render_only, self.writers)
        self.assertEqual(self.render_count, 4)

    def test_physics_progress_aborts_even_if_all_packets_match(self):
        for field, changed in (('current_time', POSE_TIME + 1e-9), ('current_time_step_index', 14946)):
            with self.subTest(field=field):
                self.setUp()

                def render_only():
                    for writer in self.writers:
                        writer.payload = writer_payload(RENDER_TIME, (11, 60), 32)
                    setattr(self.world, field, changed)

                with self.assertRaisesRegex(RuntimeError, 'advanced physics'):
                    hooks.capture_exact_camera_packets(self.world, render_only, self.writers)

    def test_publication_rejects_an_incomplete_group_before_any_camera_is_emitted(self):
        published = []
        cameras = tuple(SimpleNamespace(_topic=f'/camera_{index}', _pending_capture_time=POSE_TIME,
                                        encode_exact_message=lambda stamp, rgb: (stamp, rgb),
                                        _publisher=SimpleNamespace(publish=published.append))
                        for index in range(3))
        packets = [{'reference_time': (11, 60), 'simulation_time': RENDER_TIME,
                    'rgb': np.zeros((2, 3, 3), dtype=np.uint8)} for _ in range(2)]
        prepared = {'cameras': cameras, 'packets': packets, 'sim_time': POSE_TIME, 'attempts': 1,
                    'physics_step': 14945}
        with (patch.object(hooks, '_controller', SimpleNamespace(ros_publishers_persistent=cameras)),
              patch.object(hooks, 'write_camera_render_boundary')):
            with self.assertRaisesRegex(RuntimeError, 'complete three-camera capture'):
                hooks.publish_exact_camera_samples(self.world, prepared)
        self.assertEqual(published, [])


if __name__ == '__main__':
    unittest.main()
