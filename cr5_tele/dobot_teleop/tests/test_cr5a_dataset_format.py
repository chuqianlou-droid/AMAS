import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
sys.path.insert(0, str(ROOT / "dobot_teleop"))

from record_cr5a_pi0_dataset import _feedback_delta_actions, teleop_action_sample, write_raw_episode
from teleop_action_stream import TeleopAction


class Cr5aDatasetFormatTest(unittest.TestCase):
    def test_mock_episode_has_required_arrays_and_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            episode_dir = Path(temp_dir) / "episode_000000"
            image = np.zeros((8, 10, 3), dtype=np.uint8)
            write_raw_episode(
                episode_dir,
                images_d435=[image, image],
                images_d415=[image, image],
                tcp_pose=np.zeros((2, 6), dtype=np.float32),
                joints=np.zeros((2, 6), dtype=np.float32),
                gripper=np.zeros(2, dtype=np.float32),
                actions=np.zeros((2, 7), dtype=np.float32),
                timestamps=np.array([1.0, 2.0]),
                prompt="format test",
            )
            with np.load(episode_dir / "steps.npz") as steps:
                self.assertEqual(steps["actions"].shape, (2, 7))
                self.assertEqual(
                    set(steps.files),
                    {
                        "tcp_pose",
                        "joints",
                        "gripper",
                        "actions",
                        "timestamps",
                        "action_timestamps",
                        "action_age_ms",
                        "deadman",
                        "servo_sent",
                        "gripper_action",
                    },
                )
            self.assertTrue((episode_dir / "images_d435" / "000000.png").is_file())
            self.assertTrue((episode_dir / "images_d415" / "000001.png").is_file())

    def test_teleop_action_sample_rejects_stale_action(self):
        stream_action = TeleopAction(
            timestamp=10.0,
            seq=1,
            source="test",
            action=(1, 0, 0, 0, 0, 0, 0),
            current_pose=(0, 0, 0, 0, 0, 0),
            target_pose=(1, 0, 0, 0, 0, 0),
            current_joints=None,
            deadman=True,
            servo_sent=True,
            gripper_command=0.0,
            gripper_state=0.25,
        )
        sample, reason = teleop_action_sample(stream_action, now_s=10.1, max_action_age_ms=200)
        self.assertIsNotNone(sample)
        self.assertIsNone(reason)
        self.assertEqual(sample.gripper_state, 0.25)
        stale, reason = teleop_action_sample(stream_action, now_s=10.3, max_action_age_ms=200)
        self.assertIsNone(stale)
        self.assertEqual(reason, "stale_action")

    def test_feedback_delta_actions_use_recorded_pose_differences(self):
        poses = [
            np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            np.array([1, 2, 3, 0, 0, 10], dtype=np.float32),
            np.array([3, 5, 9, 0, 0, 15], dtype=np.float32),
        ]
        actions = _feedback_delta_actions(poses, [0.0, 0.5, 1.0])

        self.assertEqual(actions.shape, (3, 7))
        np.testing.assert_allclose(actions[0, :3], [1, 2, 3], atol=1e-6)
        np.testing.assert_allclose(actions[0, 3:6], [0, 0, 10], atol=1e-6)
        np.testing.assert_allclose(actions[1, :3], [2, 3, 6], atol=1e-6)
        np.testing.assert_allclose(actions[1, 3:6], [0, 0, 5], atol=1e-6)
        np.testing.assert_allclose(actions[2, :6], np.zeros(6), atol=1e-6)
        np.testing.assert_allclose(actions[:, 6], [0.5, 1.0, 1.0], atol=1e-6)
