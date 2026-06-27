import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from pi0_cr5a_bridge import Pi0ActionAdapter, extract_action_chunk
from toolframe_governor_teleop import CartesianSafetyEnvelope


class Pi0Cr5aBridgeTest(unittest.TestCase):
    def test_tool_frame_delta_uses_current_tool_orientation(self):
        adapter = Pi0ActionAdapter(
            action_format="cartesian_delta_mm_deg",
            position_transform=np.eye(3),
            rotation_transform=np.eye(3),
            delta_frame="tool",
            rotation_delta_representation="euler",
            normalized_linear_scale_mm=1.0,
            normalized_angular_scale_deg=1.0,
            gripper_index=6,
            gripper_scale=1.0,
            gripper_offset=0.0,
        )
        target, gripper = adapter.adapt(
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
            [0.0, 0.0, 300.0, 0.0, 0.0, 90.0],
        )

        self.assertTrue(np.allclose(target[:3], [0.0, 10.0, 300.0], atol=1e-9))
        self.assertAlmostEqual(gripper, 0.7)

    def test_safety_envelope_caps_workspace_and_orientation_from_origin(self):
        envelope = CartesianSafetyEnvelope(
            max_total_translation_mm=50.0,
            max_total_rotation_deg=10.0,
            workspace_min_x_mm=-20.0,
            workspace_max_x_mm=20.0,
            workspace_min_y_mm=-100.0,
            workspace_max_y_mm=100.0,
            workspace_min_z_mm=50.0,
            workspace_max_z_mm=500.0,
        )
        origin = [0.0, 0.0, 300.0, 0.0, 0.0, 0.0]
        envelope.reset(origin)
        target = envelope.apply([200.0, 0.0, 300.0, 45.0, 0.0, 0.0])

        self.assertAlmostEqual(target[0], 20.0)
        relative = R.from_euler("XYZ", target[3:], degrees=True).magnitude()
        self.assertLessEqual(np.degrees(relative), 10.0 + 1e-9)

    def test_chunk_accepts_single_cr5a_action_and_honors_open_loop_limit(self):
        chunk = extract_action_chunk({"actions": [1, 2, 3, 4, 5, 6, 0]}, "actions", 3)
        self.assertEqual(len(chunk), 1)
        self.assertTrue(np.array_equal(chunk.popleft(), np.array([1, 2, 3, 4, 5, 6, 0])))


if __name__ == "__main__":
    unittest.main()
