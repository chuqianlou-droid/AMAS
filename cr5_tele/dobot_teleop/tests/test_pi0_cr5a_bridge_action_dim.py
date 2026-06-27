import unittest

import numpy as np

from pi0_cr5a_bridge import Pi0ActionAdapter, classify_cr5a_action_chunk, extract_action_array


class Pi0Cr5aBridgeActionDimTest(unittest.TestCase):
    def test_execute_refuses_aloha_14d_actions(self):
        actions = np.zeros((50, 14), dtype=float)
        with self.assertRaisesRegex(ValueError, "Received ALOHA-style 14D action"):
            classify_cr5a_action_chunk(actions, action_format="delta", execute=True)

    def test_dry_run_marks_aloha_14d_without_adapting_it(self):
        actions = np.zeros((50, 14), dtype=float)
        self.assertEqual(
            classify_cr5a_action_chunk(actions, action_format="delta", execute=False), "aloha_14d"
        )

    def test_seven_dimensional_action_reaches_adapter(self):
        actions = np.zeros((1, 7), dtype=float)
        self.assertEqual(
            classify_cr5a_action_chunk(actions, action_format="delta", execute=False), "cr5a_cartesian_7d"
        )
        adapter = Pi0ActionAdapter(
            action_format="delta",
            position_transform=np.eye(3),
            rotation_transform=np.eye(3),
            delta_frame="base",
            rotation_delta_representation="euler",
            normalized_linear_scale_mm=1.0,
            normalized_angular_scale_deg=1.0,
            gripper_index=6,
            gripper_scale=1.0,
            gripper_offset=0.0,
        )
        target, gripper = adapter.adapt(actions[0], [0, 0, 300, 0, 0, 0])
        self.assertEqual(target, [0.0, 0.0, 300.0, 0.0, 0.0, 0.0])
        self.assertEqual(gripper, 0.0)

    def test_nan_or_inf_actions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            extract_action_array({"actions": [[0, 0, 0, 0, 0, 0, np.nan]]}, "actions")
