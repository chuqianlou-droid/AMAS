#!/usr/bin/env python3

import time
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from tf2_ros import Buffer, TransformListener, TransformException

from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import RobotState

from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration as DurationMsg


class TestCartesianExecute(Node):
    def __init__(self):
        super().__init__('test_cartesian_execute')

        self.group_name = 'cr5_group'
        self.base_frame = 'base_link'
        self.ee_link = 'Link6'

        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'
        ]

        # =========================
        # 目标末端位置
        # =========================
        self.target_x = -0.140
        self.target_y = -0.484
        self.target_z = 0.399

        # =========================
        # 关节限幅：相对当前关节角的最大允许偏移
        # 单位：degree
        #
        # joint1 是 base 关节，重点限制它，防止底座绕大圈。
        # 如果你发现正常运动也被拒绝，可以稍微放宽 joint1 到 20~30 deg。
        # =========================
        self.max_joint_delta_deg = {
            'joint1': 15.0,   # base，重点限制
            'joint2': 45.0,
            'joint3': 45.0,
            'joint4': 60.0,
            'joint5': 60.0,
            'joint6': 90.0,
        }

        # 相邻轨迹点之间的最大关节跳变限制
        # 用来检测 IK 突然切换分支
        self.max_step_delta_deg = {
            'joint1': 5.0,
            'joint2': 8.0,
            'joint3': 8.0,
            'joint4': 12.0,
            'joint5': 12.0,
            'joint6': 20.0,
        }

        # 笛卡尔插值步长，单位 m
        self.cartesian_max_step = 0.005

        # Jump threshold 不要设为 0
        # 0 表示禁用关节跳变检测
        # 1.5~2.0 通常比较适合先测试
        self.jump_threshold = 2.0

        # 轨迹时间缩放，数值越大，执行越慢
        self.time_scale = 5.0

        # 如果 MoveIt 返回的轨迹没有时间戳，就用这个基础时间间隔
        self.default_dt = 0.08

        self.latest_joint_state = None

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cartesian_client = self.create_client(
            GetCartesianPath,
            '/compute_cartesian_path'
        )

        self.traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/cr5_group_controller/follow_joint_trajectory'
        )

        self.get_logger().info('Waiting for /compute_cartesian_path service...')
        while not self.cartesian_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/compute_cartesian_path not available, waiting...')
        self.get_logger().info('/compute_cartesian_path available.')

        self.get_logger().info('Waiting for /cr5_group_controller/follow_joint_trajectory action...')
        self.traj_client.wait_for_server()
        self.get_logger().info('follow_joint_trajectory action available.')

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def wait_for_joint_state(self, timeout_sec=3.0):
        self.get_logger().info('Waiting for /joint_states...')

        start = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.latest_joint_state is not None:
                if all(j in self.latest_joint_state.name for j in self.joint_names):
                    self.get_logger().info('/joint_states received.')
                    return True

            if time.time() - start > timeout_sec:
                self.get_logger().error('Timeout waiting for joint states.')
                return False

    def get_current_joint_positions(self):
        js = self.latest_joint_state
        name_to_pos = dict(zip(js.name, js.position))
        return [float(name_to_pos[j]) for j in self.joint_names]

    def make_start_state(self, positions):
        rs = RobotState()
        rs.joint_state.name = list(self.joint_names)
        rs.joint_state.position = list(positions)
        return rs

    def get_current_pose(self, timeout_sec=3.0):
        self.get_logger().info(f'Waiting for TF {self.base_frame} -> {self.ee_link}...')

        start = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.ee_link,
                    rclpy.time.Time()
                )

                pose = Pose()
                pose.position.x = tf_msg.transform.translation.x
                pose.position.y = tf_msg.transform.translation.y
                pose.position.z = tf_msg.transform.translation.z

                pose.orientation.x = tf_msg.transform.rotation.x
                pose.orientation.y = tf_msg.transform.rotation.y
                pose.orientation.z = tf_msg.transform.rotation.z
                pose.orientation.w = tf_msg.transform.rotation.w

                self.get_logger().info('Current TF pose received.')
                return pose

            except TransformException:
                pass

            if time.time() - start > timeout_sec:
                self.get_logger().error('Timeout waiting for TF.')
                return None

    @staticmethod
    def shortest_angle_delta(angle, reference):
        """
        返回 angle 相对 reference 的最短角度差，范围 [-pi, pi]。
        """
        return math.atan2(math.sin(angle - reference), math.cos(angle - reference))

    def unwrap_trajectory_positions(self, trajectory, current_positions):
        """
        将 trajectory 中的角度连续化。

        例如当前 joint6 = 6.25 rad，
        IK 结果可能给出 -0.02 rad。
        这两个角度物理上接近，但如果直接发给控制器，
        机械臂可能会反向转一大圈。

        这里把 -0.02 unwrap 到接近 6.25 的连续角度。
        """
        prev = list(current_positions)

        for point in trajectory.joint_trajectory.points:
            if len(point.positions) != len(self.joint_names):
                self.get_logger().error('Trajectory point position size mismatch.')
                return False

            new_positions = []

            for i, raw_angle in enumerate(point.positions):
                delta = self.shortest_angle_delta(raw_angle, prev[i])
                continuous_angle = prev[i] + delta
                new_positions.append(continuous_angle)

            point.positions = new_positions
            prev = new_positions

        return True

    def validate_trajectory_joint_limits(self, trajectory, current_positions):
        """
        检查整条轨迹是否满足：
        1. 每个关节不偏离当前构型太远
        2. 相邻轨迹点之间没有突跳
        """
        points = trajectory.joint_trajectory.points

        if len(points) == 0:
            self.get_logger().error('Trajectory has no points.')
            return False

        prev_positions = list(current_positions)

        for point_idx, point in enumerate(points):
            positions = list(point.positions)

            if len(positions) != len(self.joint_names):
                self.get_logger().error(
                    f'Point {point_idx}: position size mismatch.'
                )
                return False

            for j_idx, joint_name in enumerate(self.joint_names):
                pos = positions[j_idx]
                cur = current_positions[j_idx]
                prev = prev_positions[j_idx]

                total_delta_deg = math.degrees(pos - cur)
                step_delta_deg = math.degrees(pos - prev)

                max_total = self.max_joint_delta_deg[joint_name]
                max_step = self.max_step_delta_deg[joint_name]

                if abs(total_delta_deg) > max_total:
                    self.get_logger().error(
                        f'Trajectory rejected: {joint_name} exceeds total limit at point {point_idx}. '
                        f'current={math.degrees(cur):.2f} deg, '
                        f'planned={math.degrees(pos):.2f} deg, '
                        f'delta={total_delta_deg:.2f} deg, '
                        f'limit=±{max_total:.2f} deg'
                    )
                    return False

                if abs(step_delta_deg) > max_step:
                    self.get_logger().error(
                        f'Trajectory rejected: {joint_name} has sudden jump at point {point_idx}. '
                        f'prev={math.degrees(prev):.2f} deg, '
                        f'planned={math.degrees(pos):.2f} deg, '
                        f'step_delta={step_delta_deg:.2f} deg, '
                        f'limit=±{max_step:.2f} deg'
                    )
                    return False

            prev_positions = positions

        self.get_logger().info('Trajectory joint-limit validation PASSED.')
        return True

    @staticmethod
    def duration_to_sec(duration_msg):
        return duration_msg.sec + duration_msg.nanosec * 1e-9

    @staticmethod
    def sec_to_duration(sec_float):
        if sec_float < 0.0:
            sec_float = 0.0

        msg = DurationMsg()
        msg.sec = int(sec_float)
        msg.nanosec = int((sec_float - msg.sec) * 1e9)
        return msg

    def retime_trajectory(self, trajectory):
        """
        让轨迹执行更慢、更安全。

        如果 MoveIt 返回的 time_from_start 全是 0，
        则手动生成时间戳。
        如果已有时间戳，则整体乘以 time_scale。
        """
        points = trajectory.joint_trajectory.points

        if len(points) == 0:
            return trajectory

        final_time = self.duration_to_sec(points[-1].time_from_start)

        if final_time <= 1e-6:
            self.get_logger().warn(
                'Trajectory has no valid timing. Assigning manual timing.'
            )

            for i, point in enumerate(points):
                t = (i + 1) * self.default_dt * self.time_scale
                point.time_from_start = self.sec_to_duration(t)

                # 只使用位置 + 时间，让控制器自己插值
                point.velocities = []
                point.accelerations = []
                point.effort = []

        else:
            for point in points:
                old_t = self.duration_to_sec(point.time_from_start)
                new_t = old_t * self.time_scale
                point.time_from_start = self.sec_to_duration(new_t)

                if len(point.velocities) == len(self.joint_names):
                    point.velocities = [v / self.time_scale for v in point.velocities]
                else:
                    point.velocities = []

                if len(point.accelerations) == len(self.joint_names):
                    point.accelerations = [
                        a / (self.time_scale * self.time_scale)
                        for a in point.accelerations
                    ]
                else:
                    point.accelerations = []

                point.effort = []

        total_time = self.duration_to_sec(points[-1].time_from_start)
        self.get_logger().info(
            f'Trajectory retimed. points={len(points)}, duration={total_time:.3f} s'
        )

        return trajectory

    def print_current_joints(self, current_positions):
        self.get_logger().info('Current joint positions:')
        for name, pos in zip(self.joint_names, current_positions):
            self.get_logger().info(
                f'  {name}: {pos:.6f} rad = {math.degrees(pos):.2f} deg'
            )

    def print_goal_summary(self, current_pose, target_pose):
        self.get_logger().info(
            'Current EE pose: '
            f'x={current_pose.position.x:.4f}, '
            f'y={current_pose.position.y:.4f}, '
            f'z={current_pose.position.z:.4f}'
        )
        self.get_logger().info(
            'Target EE pose: '
            f'x={target_pose.position.x:.4f}, '
            f'y={target_pose.position.y:.4f}, '
            f'z={target_pose.position.z:.4f}'
        )
        self.get_logger().info(
            'Joint total limits: '
            + ', '.join([
                f'{j}=±{self.max_joint_delta_deg[j]:.0f}deg'
                for j in self.joint_names
            ])
        )
        self.get_logger().info(
            f'Cartesian max_step={self.cartesian_max_step:.4f} m, '
            f'jump_threshold={self.jump_threshold:.2f}, '
            f'time_scale={self.time_scale:.1f}'
        )

    def execute_cartesian_path(self):
        if not self.wait_for_joint_state():
            return

        current_positions = self.get_current_joint_positions()
        self.print_current_joints(current_positions)

        current_pose = self.get_current_pose()
        if current_pose is None:
            return

        target_pose = Pose()
        target_pose.position.x = self.target_x
        target_pose.position.y = self.target_y
        target_pose.position.z = self.target_z

        # 保持当前姿态
        target_pose.orientation.x = current_pose.orientation.x
        target_pose.orientation.y = current_pose.orientation.y
        target_pose.orientation.z = current_pose.orientation.z
        target_pose.orientation.w = current_pose.orientation.w

        self.print_goal_summary(current_pose, target_pose)

        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.header.stamp = self.get_clock().now().to_msg()

        req.start_state = self.make_start_state(current_positions)
        req.group_name = self.group_name
        req.link_name = self.ee_link

        req.waypoints = [target_pose]
        req.max_step = self.cartesian_max_step

        # 关键：不要再用 0.0
        # 0.0 会关闭 jump 检测
        req.jump_threshold = self.jump_threshold

        req.avoid_collisions = True

        self.get_logger().info('Calling /compute_cartesian_path...')

        future = self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('Failed to call /compute_cartesian_path.')
            return

        res = future.result()

        self.get_logger().info(f'Cartesian path fraction: {res.fraction:.4f}')
        self.get_logger().info(f'Error code: {res.error_code.val}')

        if res.error_code.val != 1:
            self.get_logger().error(
                f'Cartesian path failed with error code {res.error_code.val}. Abort execution.'
            )
            return

        if res.fraction < 0.99:
            self.get_logger().error(
                'Cartesian path is not complete. Abort execution.'
            )
            return

        trajectory = res.solution

        # 关键 1：角度连续化，避免 2pi 跳变
        if not self.unwrap_trajectory_positions(trajectory, current_positions):
            self.get_logger().error('Failed to unwrap trajectory. Abort execution.')
            return

        # 关键 2：检查关节限幅，尤其 joint1/base
        if not self.validate_trajectory_joint_limits(trajectory, current_positions):
            self.get_logger().error('Unsafe trajectory. Abort execution.')
            return

        # 关键 3：放慢轨迹
        trajectory = self.retime_trajectory(trajectory)

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory.joint_trajectory

        # 设置一个稍微未来的 stamp，避免控制器认为轨迹时间已经过期
        start_time = self.get_clock().now() + RclpyDuration(seconds=0.5)
        goal_msg.trajectory.header.stamp = start_time.to_msg()

        # 执行容差时间
        goal_msg.goal_time_tolerance = self.sec_to_duration(2.0)

        self.get_logger().info('Sending trajectory to controller...')
        send_future = self.traj_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()

        if goal_handle is None:
            self.get_logger().error('No goal handle returned.')
            return

        if not goal_handle.accepted:
            self.get_logger().error('Trajectory goal rejected by controller.')
            return

        self.get_logger().info('Trajectory goal accepted. Executing...')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result

        self.get_logger().info(
            f'Trajectory execution finished. '
            f'error_code={result.error_code}, '
            f'error_string="{result.error_string}"'
        )


def main(args=None):
    rclpy.init(args=args)
    node = TestCartesianExecute()

    try:
        node.execute_cartesian_path()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()