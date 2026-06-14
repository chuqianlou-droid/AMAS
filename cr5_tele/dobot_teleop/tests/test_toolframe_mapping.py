import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from dobot_teleop.quest_udp import QuestPose
from dobot_teleop.toolframe_mapping import QuestTeleopConfig, QuestTeleopMapper
from servoj_toolframe_teleop import should_send_target


def pose(x=0.0, y=0.0, z=0.0, rot=None):
    if rot is None:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    else:
        qx, qy, qz, qw = rot.as_quat()
    return QuestPose(
        x=x,
        y=y,
        z=z,
        qx=qx,
        qy=qy,
        qz=qz,
        qw=qw,
        buttons={},
        timestamp=0.0,
        count=0,
        address=("test", 0),
        raw={},
    )


def matrix_from_target(target):
    return R.from_euler("XYZ", target[3:], degrees=True).as_matrix()


class ToolFrameMappingTest(unittest.TestCase):
    def test_pure_rotation_reports_rotation_step(self):
        cfg = QuestTeleopConfig(
            position_scale=0.0,
            rotation_scale=1.0,
            rotation_mode="origin_delta",
            rot_transform=np.eye(3).tolist(),
            target_deadband_mm=0.0,
            target_deadband_deg=0.0,
            max_step_deg=90.0,
        )
        mapper = QuestTeleopMapper(cfg)
        mapper.reset(pose(), [0.0, 0.0, 300.0, 0.0, 0.0, 0.0])

        target, info = mapper.target_from_quest(
            pose(rot=R.from_euler("Z", 10.0, degrees=True)),
            rg_pressed=True,
        )

        self.assertEqual(target[:3], [0.0, 0.0, 300.0])
        self.assertAlmostEqual(info["sent_step_mm"], 0.0)
        self.assertGreater(info["sent_step_deg"], 0.0)
        self.assertTrue(should_send_target(info))

    def test_origin_delta_uses_tool_frame_right_multiply(self):
        cfg = QuestTeleopConfig(
            position_scale=0.0,
            rotation_scale=1.0,
            rotation_mode="origin_delta",
            rot_transform=np.eye(3).tolist(),
            target_deadband_mm=0.0,
            target_deadband_deg=0.0,
            max_step_deg=180.0,
        )
        mapper = QuestTeleopMapper(cfg)
        robot_origin = [0.0, 0.0, 300.0, 20.0, 30.0, 40.0]
        mapper.reset(pose(), robot_origin)

        target, _ = mapper.target_from_quest(
            pose(rot=R.from_euler("X", 15.0, degrees=True)),
            rg_pressed=True,
        )

        origin_R = R.from_euler("XYZ", robot_origin[3:], degrees=True).as_matrix()
        delta_R = R.from_euler("X", 15.0, degrees=True).as_matrix()
        expected = origin_R @ delta_R
        wrong = delta_R @ origin_R
        actual = matrix_from_target(target)

        self.assertTrue(np.allclose(actual, expected, atol=1e-9))
        self.assertFalse(np.allclose(actual, wrong, atol=1e-3))

    def test_rg_false_freezes_last_target(self):
        cfg = QuestTeleopConfig(
            position_scale=1.0,
            rotation_scale=1.0,
            target_deadband_mm=0.0,
            target_deadband_deg=0.0,
        )
        mapper = QuestTeleopMapper(cfg)
        mapper.reset(pose(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        target, info = mapper.target_from_quest(
            pose(x=1.0, rot=R.from_euler("Z", 45.0, degrees=True)),
            rg_pressed=False,
        )

        self.assertEqual(target, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertAlmostEqual(info["sent_step_mm"], 0.0)
        self.assertAlmostEqual(info["sent_step_deg"], 0.0)

    def test_total_translation_limit_caps_accumulator(self):
        cfg = QuestTeleopConfig(
            pos_transform=np.eye(3).tolist(),
            position_scale=1.0,
            rotation_scale=0.0,
            target_deadband_mm=0.0,
            target_deadband_deg=0.0,
            max_step_mm=1000.0,
            max_total_translation_mm=5.0,
            workspace_min_x_mm=-1000.0,
            workspace_max_x_mm=1000.0,
            workspace_min_y_mm=-1000.0,
            workspace_max_y_mm=1000.0,
            workspace_min_z_mm=-1000.0,
            workspace_max_z_mm=1000.0,
        )
        mapper = QuestTeleopMapper(cfg)
        mapper.reset(pose(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        mapper.target_from_quest(pose(x=1.0), rg_pressed=True)

        self.assertLessEqual(np.linalg.norm(mapper._accum_delta_pos), 5.0 + 1e-9)

    def test_workspace_clamp_syncs_accumulator_for_reverse_motion(self):
        cfg = QuestTeleopConfig(
            pos_transform=np.eye(3).tolist(),
            position_scale=1.0,
            rotation_scale=0.0,
            target_deadband_mm=0.0,
            target_deadband_deg=0.0,
            max_step_mm=1000.0,
            max_total_translation_mm=1000.0,
            workspace_min_x_mm=-1000.0,
            workspace_max_x_mm=10.0,
        )
        mapper = QuestTeleopMapper(cfg)
        mapper.reset(pose(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        target, _ = mapper.target_from_quest(pose(x=1.0), rg_pressed=True)
        self.assertAlmostEqual(target[0], 10.0)
        self.assertAlmostEqual(mapper._accum_delta_pos[0], 10.0)

        target, _ = mapper.target_from_quest(pose(x=0.9), rg_pressed=True)
        self.assertLess(target[0], 10.0)


if __name__ == "__main__":
    unittest.main()
