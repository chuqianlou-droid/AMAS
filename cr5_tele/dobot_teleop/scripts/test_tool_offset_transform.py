#!/usr/bin/env python3
"""Self-check script for the TCP → gripper-center pose transforms.

Run from the project root:

    python3 dobot_teleop/scripts/test_tool_offset_transform.py

Or with pytest:

    pytest dobot_teleop/scripts/test_tool_offset_transform.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

# Ensure the dobot_teleop package is importable.
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
from scipy.spatial.transform import Rotation as R

from dobot_teleop.transforms import (
    DOBOT_EULER_SEQ,
    ToolOffsetConfig,
    apply_tool_offset,
    format_tool_offset_config,
    gripper_center_pose_to_tcp_pose,
    matrix_to_pose6d,
    pose6d_to_matrix,
    remove_tool_offset,
    tcp_pose_to_gripper_center_pose,
)


class TestPose6dMatrixRoundTrip(unittest.TestCase):
    """pose6d ↔ matrix conversions are invertible."""

    def test_round_trip_identity(self):
        pose = [100.0, 200.0, 300.0, 0.0, 0.0, 0.0]
        T = pose6d_to_matrix(pose)
        back = matrix_to_pose6d(T)
        for a, b in zip(pose, back):
            self.assertAlmostEqual(a, b, places=10)

    def test_round_trip_rotated(self):
        pose = [-450.0, 123.4, 567.8, 30.0, -45.0, 90.0]
        T = pose6d_to_matrix(pose)
        back = matrix_to_pose6d(T)
        for a, b in zip(pose, back):
            self.assertAlmostEqual(a, b, places=10)

    def test_matrix_inverse(self):
        pose = [100.0, -200.0, 350.0, -15.0, 25.0, -60.0]
        T = pose6d_to_matrix(pose)
        T_inv = np.linalg.inv(T)
        inv_pose = matrix_to_pose6d(T_inv)
        # Applying T @ T_inv should give near-identity transform
        T_reconstructed = pose6d_to_matrix(inv_pose)
        should_be_identity = T @ T_reconstructed
        np.testing.assert_allclose(
            should_be_identity, np.eye(4), atol=1e-12
        )


class TestTcpToGripperCenter(unittest.TestCase):
    """Core forward / inverse offset transforms."""

    # --- Case 1: zero rotation → offset is purely along world Z ----------

    def test_no_rotation_offset_along_z(self):
        """TCP with identity orientation: gripper center is Z + 160 mm."""
        tcp = [100.0, 200.0, 300.0, 0.0, 0.0, 0.0]
        offset = (0.0, 0.0, 160.0)
        gripper = tcp_pose_to_gripper_center_pose(tcp, offset=offset)

        # Position: Z should increase by exactly 160 mm.
        self.assertAlmostEqual(gripper[0], 100.0, places=10)
        self.assertAlmostEqual(gripper[1], 200.0, places=10)
        self.assertAlmostEqual(gripper[2], 460.0, places=10)

        # Orientation unchanged.
        for i in range(3, 6):
            self.assertAlmostEqual(gripper[i], tcp[i], places=10)

    # --- Case 2: TCP has rotation → offset is NOT world-Z-aligned -------

    def test_rotated_tcp_offset_is_local_z(self):
        """Verify p_gripper - p_tcp == R_tcp @ [0, 0, 160]."""
        tcp = [50.0, -100.0, 200.0, 30.0, -45.0, 90.0]
        offset = (0.0, 0.0, 160.0)

        gripper = tcp_pose_to_gripper_center_pose(tcp, offset=offset)

        R_tcp = R.from_euler(DOBOT_EULER_SEQ, tcp[3:], degrees=True).as_matrix()
        expected_delta_world = R_tcp @ np.array([0.0, 0.0, 160.0])

        actual_delta = np.array(gripper[:3]) - np.array(tcp[:3])
        np.testing.assert_allclose(actual_delta, expected_delta_world, atol=1e-10)

    def test_rotated_tcp_not_simple_z_add(self):
        """A 90° rotation around X means the TCP Z is no longer world-Z.

        The test verifies that we are NOT doing a naive Z += 160.  With 90°
        X rotation (scipy "XYZ" intrinsic), the local Z axis points to world −Y.
        """
        tcp = [0.0, 0.0, 100.0, 90.0, 0.0, 0.0]  # 90° about X
        offset = (0.0, 0.0, 160.0)

        gripper = tcp_pose_to_gripper_center_pose(tcp, offset=offset)

        # Verify this is NOT a naive Z += 160:
        self.assertNotAlmostEqual(gripper[2], 260.0, places=10)
        # With 90° X rotation the local Z is world −Y, so
        # p_gripper = [0, 0, 100] + R_x(90°) @ [0, 0, 160] = [0, -160, 100].
        self.assertAlmostEqual(gripper[0], 0.0, places=10)
        self.assertAlmostEqual(gripper[1], -160.0, places=10)
        self.assertAlmostEqual(gripper[2], 100.0, places=10)

    # --- Case 3: round-trip consistency ---------------------------------

    def test_forward_inverse_round_trip(self):
        """Applying forward then inverse should recover the original.

        Rotation is compared on SO(3) (via the rotation matrix) to handle
        Euler angle wrap-around (e.g. −180° ↔ 180°).
        """
        poses = [
            [100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
            [-450.0, 123.4, 567.8, 30.0, -45.0, 90.0],
            [0.0, 0.0, 50.0, -180.0, 0.0, 90.0],
            [320.0, -240.0, 180.0, -15.0, 85.0, -175.0],
        ]
        offset = (0.0, 0.0, 160.0)
        for tcp in poses:
            gripper = tcp_pose_to_gripper_center_pose(tcp, offset=offset)
            tcp_back = gripper_center_pose_to_tcp_pose(gripper, offset=offset)

            # Position: exact float compare
            for i in range(3):
                self.assertAlmostEqual(
                    tcp[i], tcp_back[i], places=10,
                    msg=f"Position mismatch at index {i} for pose {tcp}"
                )
            # Rotation: compare on SO(3) to tolerate Euler wrap-around
            R_orig = R.from_euler(DOBOT_EULER_SEQ, tcp[3:], degrees=True).as_matrix()
            R_back = R.from_euler(
                DOBOT_EULER_SEQ, tcp_back[3:], degrees=True
            ).as_matrix()
            np.testing.assert_allclose(
                R_orig, R_back, atol=1e-10,
                err_msg=f"Rotation matrix mismatch for pose {tcp}"
            )

    def test_inverse_forward_round_trip(self):
        """Starting from gripper center, inverse then forward recovers it."""
        gripper_orig = [-400.0, 100.0, 520.0, 45.0, -20.0, 135.0]
        offset = (0.0, 0.0, 160.0)
        tcp = gripper_center_pose_to_tcp_pose(gripper_orig, offset=offset)
        gripper_back = tcp_pose_to_gripper_center_pose(tcp, offset=offset)
        # Position: exact float compare
        for i in range(3):
            self.assertAlmostEqual(gripper_orig[i], gripper_back[i], places=10)
        # Rotation: compare on SO(3)
        R_orig = R.from_euler(DOBOT_EULER_SEQ, gripper_orig[3:], degrees=True).as_matrix()
        R_back = R.from_euler(DOBOT_EULER_SEQ, gripper_back[3:], degrees=True).as_matrix()
        np.testing.assert_allclose(R_orig, R_back, atol=1e-10)

    # --- Case 4: config object API --------------------------------------

    def test_config_enabled(self):
        cfg = ToolOffsetConfig(
            enabled=True,
            xyz_mm=(0.0, 0.0, 160.0),
            rpy_deg=(0.0, 0.0, 0.0),
            controller_tool_offset_already_set=False,
        )
        self.assertTrue(cfg.effective())

        tcp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        gripper, raw = apply_tool_offset(tcp, cfg)
        self.assertAlmostEqual(gripper[2], 160.0)
        self.assertAlmostEqual(raw[2], 0.0)

        tcp_back = remove_tool_offset(gripper, cfg)
        self.assertAlmostEqual(tcp_back[2], 0.0)

    def test_config_disabled(self):
        cfg = ToolOffsetConfig(
            enabled=False,
            xyz_mm=(0.0, 0.0, 160.0),
            controller_tool_offset_already_set=False,
        )
        self.assertFalse(cfg.effective())

        tcp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        gripper, raw = apply_tool_offset(tcp, cfg)
        # Disabled → passthrough
        self.assertAlmostEqual(gripper[2], 0.0)

    def test_config_controller_already_set(self):
        cfg = ToolOffsetConfig(
            enabled=True,
            xyz_mm=(0.0, 0.0, 160.0),
            controller_tool_offset_already_set=True,
        )
        self.assertFalse(cfg.effective())

        tcp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        gripper, raw = apply_tool_offset(tcp, cfg)
        # Controller already set → passthrough
        self.assertAlmostEqual(gripper[2], 0.0)


class TestToolOffsetConfigFormatting(unittest.TestCase):
    """Ensure the startup log helper does not raise."""

    def test_format_active(self):
        cfg = ToolOffsetConfig(enabled=True, controller_tool_offset_already_set=False)
        text = format_tool_offset_config(cfg)
        self.assertIn("software offset active: True", text)

    def test_format_passthrough(self):
        cfg = ToolOffsetConfig(enabled=True, controller_tool_offset_already_set=True)
        text = format_tool_offset_config(cfg)
        self.assertIn("software offset active: False", text)


class TestNonDefaultOffset(unittest.TestCase):
    """Exercise the configurable offset parameters."""

    def test_custom_translation(self):
        tcp = [100.0, 0.0, 0.0, 0.0, 0.0, 90.0]  # 90° about Z
        # With 90° Z rotation, TCP X points to world +Y, TCP Z still points to world +Z
        # (Z rotation doesn't affect Z axis direction)
        offset = (50.0, 0.0, 0.0)  # 50mm along TCP X → world +Y
        gripper = tcp_pose_to_gripper_center_pose(tcp, offset=offset)
        # TCP X with 90° Z → world +Y = [0, 1, 0]
        # So 50 * [0, 1, 0] = [0, 50, 0] in world
        self.assertAlmostEqual(gripper[0], 100.0, places=10)
        self.assertAlmostEqual(gripper[1], 50.0, places=10)
        self.assertAlmostEqual(gripper[2], 0.0, places=10)

    def test_custom_rotation_offset(self):
        tcp = [0.0, 0.0, 300.0, 0.0, 0.0, 0.0]
        offset = (0.0, 0.0, 0.0)
        offset_rpy = (0.0, 0.0, 90.0)  # 90° Z rotation in TCP frame
        gripper = tcp_pose_to_gripper_center_pose(
            tcp, offset=offset, offset_rpy_deg=offset_rpy
        )
        # Position unchanged (no translation offset)
        self.assertAlmostEqual(gripper[0], 0.0, places=10)
        self.assertAlmostEqual(gripper[1], 0.0, places=10)
        self.assertAlmostEqual(gripper[2], 300.0, places=10)
        # Rotation: TCP Rz=0 + offset Rz=90 = Rz=90
        self.assertAlmostEqual(gripper[5], 90.0, places=10)

    def test_meter_unit_mode(self):
        """Verify that xyz_unit='m' works correctly for pi0 / LeRobot data."""
        tcp = [0.5, -0.3, 0.8, 0.0, 0.0, 0.0]  # meters
        offset = (0.0, 0.0, 0.16)  # meters
        gripper = tcp_pose_to_gripper_center_pose(
            tcp, offset=offset, xyz_unit="m"
        )
        self.assertAlmostEqual(gripper[2], 0.96, places=10)  # 0.8 + 0.16

        tcp_back = gripper_center_pose_to_tcp_pose(
            gripper, offset=offset, xyz_unit="m"
        )
        self.assertAlmostEqual(tcp_back[2], 0.8, places=10)


if __name__ == "__main__":
    unittest.main()
