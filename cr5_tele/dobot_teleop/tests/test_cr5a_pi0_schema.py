import unittest

import numpy as np

from cr5a_pi0_schema import CR5A_ACTION_DIM, CR5A_IMAGE_KEYS, build_cr5a_observation, validate_cr5a_action


class Cr5aPi0SchemaTest(unittest.TestCase):
    def test_build_observation_normalizes_chw_images_to_uint8_hwc(self):
        d435 = np.ones((3, 4, 5), dtype=np.float32)
        d415 = np.zeros((4, 5, 3), dtype=np.uint8)
        observation = build_cr5a_observation(
            d435, d415, [1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5], 0.25, "pick"
        )
        self.assertEqual(set(observation["images"]), set(CR5A_IMAGE_KEYS.values()))
        self.assertEqual(observation["images"]["image_primary"].shape, (4, 5, 3))
        self.assertEqual(observation["images"]["image_primary"].dtype, np.uint8)
        self.assertEqual(observation["state"]["tcp_pose"].shape, (6,))
        self.assertEqual(observation["state"]["joints"].shape, (6,))
        self.assertEqual(observation["prompt"], "pick")

    def test_action_validator_requires_exactly_seven_finite_values(self):
        self.assertEqual(validate_cr5a_action(np.zeros(CR5A_ACTION_DIM)).shape, (CR5A_ACTION_DIM,))
        with self.assertRaises(ValueError):
            validate_cr5a_action(np.zeros(6))
        with self.assertRaises(ValueError):
            validate_cr5a_action([0, 0, 0, 0, 0, 0, np.nan])
