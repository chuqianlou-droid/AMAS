#!/usr/bin/env python3

import math
import threading
from typing import Iterable, Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from rclpy.node import Node

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetCartesianPath
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener


class MoveItPoseController:
    """Small wrapper around MoveIt Cartesian planning and trajectory execution."""

    def __init__(
        self,
        node: Node,
        group_name: str = 'cr5_group',
        base_frame: str = 'base_link',
        ee_link: str = 'Link6',
        joint_names: Optional[Iterable[str]] = None,
        cartesian_service: str = '/compute_cartesian_path',
        trajectory_action: str = '/cr5_group_controller/follow_joint_trajectory',
        cartesian_max_step: float = 0.005,
        jump_threshold: float = 2.0,
        time_scale: float = 5.0,
        default_dt: float = 0.08,
        dry_run: bool = True,
        plan_fraction_threshold: float = 0.99,
        max_joint_delta_deg: Optional[dict] = None,
        max_step_delta_deg: Optional[dict] = None,
    ):
        self.node = node
        self.group_name = group_name
        self.base_frame = base_frame
        self.ee_link = ee_link
        self.joint_names = list(joint_names or [
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'
        ])
        self.cartesian_max_step = cartesian_max_step
        self.jump_threshold = jump_threshold
        self.time_scale = time_scale
        self.default_dt = default_dt
        self.dry_run = dry_run
        self.plan_fraction_threshold = plan_fraction_threshold

        self.max_joint_delta_deg = max_joint_delta_deg or {
            'joint1': 15.0,
            'joint2': 45.0,
            'joint3': 45.0,
            'joint4': 60.0,
            'joint5': 60.0,
            'joint6': 90.0,
        }
        self.max_step_delta_deg = max_step_delta_deg or {
            'joint1': 5.0,
            'joint2': 8.0,
            'joint3': 8.0,
            'joint4': 12.0,
            'joint5': 12.0,
            'joint6': 20.0,
        }

        self.latest_joint_state: Optional[JointState] = None
        self.joint_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        self.cartesian_client = self.node.create_client(
            GetCartesianPath,
            cartesian_service,
        )
        self.traj_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            trajectory_action,
        )

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def wait_for_servers(self, timeout_sec: float = 1.0):
        while rclpy.ok() and not self.cartesian_client.wait_for_service(timeout_sec=timeout_sec):
            self.node.get_logger().info('/compute_cartesian_path not available, waiting...')

        if self.dry_run:
            self.node.get_logger().info('dry_run=true: trajectory action server is not required.')
            return

        self.node.get_logger().info('Waiting for follow_joint_trajectory action...')
        self.traj_client.wait_for_server()
        self.node.get_logger().info('follow_joint_trajectory action available.')

    def has_joint_state(self) -> bool:
        js = self.latest_joint_state
        return js is not None and all(j in js.name for j in self.joint_names)

    def get_current_joint_positions(self) -> Optional[list]:
        if not self.has_joint_state():
            self.node.get_logger().warn('No complete /joint_states received yet.')
            return None

        name_to_pos = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
        return [float(name_to_pos[j]) for j in self.joint_names]

    def get_current_pose(self, timeout_sec: float = 0.5) -> Optional[Pose]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_link,
                rclpy.time.Time(),
                timeout=RclpyDuration(seconds=timeout_sec),
            )
        except TransformException as exc:
            self.node.get_logger().warn(
                f'Failed to lookup TF {self.base_frame}->{self.ee_link}: {exc}'
            )
            return None

        pose = Pose()
        pose.position.x = tf_msg.transform.translation.x
        pose.position.y = tf_msg.transform.translation.y
        pose.position.z = tf_msg.transform.translation.z
        pose.orientation = tf_msg.transform.rotation
        return pose

    def make_start_state(self, positions: list) -> RobotState:
        robot_state = RobotState()
        robot_state.joint_state.name = list(self.joint_names)
        robot_state.joint_state.position = list(positions)
        return robot_state

    @staticmethod
    def shortest_angle_delta(angle: float, reference: float) -> float:
        return math.atan2(math.sin(angle - reference), math.cos(angle - reference))

    def unwrap_trajectory_positions(self, trajectory, current_positions: list) -> bool:
        prev = list(current_positions)

        for point in trajectory.joint_trajectory.points:
            if len(point.positions) != len(self.joint_names):
                self.node.get_logger().error('Trajectory point position size mismatch.')
                return False

            new_positions = []
            for i, raw_angle in enumerate(point.positions):
                delta = self.shortest_angle_delta(raw_angle, prev[i])
                new_positions.append(prev[i] + delta)

            point.positions = new_positions
            prev = new_positions

        return True

    def validate_trajectory_joint_limits(self, trajectory, current_positions: list) -> bool:
        points = trajectory.joint_trajectory.points
        if not points:
            self.node.get_logger().error('Trajectory has no points.')
            return False

        prev_positions = list(current_positions)

        for point_idx, point in enumerate(points):
            positions = list(point.positions)
            if len(positions) != len(self.joint_names):
                self.node.get_logger().error(
                    f'Point {point_idx}: position size mismatch.'
                )
                return False

            for j_idx, joint_name in enumerate(self.joint_names):
                pos = positions[j_idx]
                cur = current_positions[j_idx]
                prev = prev_positions[j_idx]

                total_delta_deg = math.degrees(pos - cur)
                step_delta_deg = math.degrees(pos - prev)

                max_total = self.max_joint_delta_deg.get(joint_name, 180.0)
                max_step = self.max_step_delta_deg.get(joint_name, 180.0)

                if abs(total_delta_deg) > max_total:
                    self.node.get_logger().error(
                        f'Trajectory rejected: {joint_name} total delta '
                        f'{total_delta_deg:.2f} deg exceeds {max_total:.2f} deg.'
                    )
                    return False

                if abs(step_delta_deg) > max_step:
                    self.node.get_logger().error(
                        f'Trajectory rejected: {joint_name} step delta '
                        f'{step_delta_deg:.2f} deg exceeds {max_step:.2f} deg.'
                    )
                    return False

            prev_positions = positions

        return True

    @staticmethod
    def duration_to_sec(duration_msg) -> float:
        return duration_msg.sec + duration_msg.nanosec * 1e-9

    @staticmethod
    def sec_to_duration(sec_float: float) -> DurationMsg:
        sec_float = max(0.0, sec_float)
        msg = DurationMsg()
        msg.sec = int(sec_float)
        msg.nanosec = int((sec_float - msg.sec) * 1e9)
        return msg

    def retime_trajectory(self, trajectory):
        points = trajectory.joint_trajectory.points
        if not points:
            return trajectory

        final_time = self.duration_to_sec(points[-1].time_from_start)

        if final_time <= 1e-6:
            for i, point in enumerate(points):
                point.time_from_start = self.sec_to_duration(
                    (i + 1) * self.default_dt * self.time_scale
                )
                point.velocities = []
                point.accelerations = []
                point.effort = []
        else:
            for point in points:
                old_t = self.duration_to_sec(point.time_from_start)
                point.time_from_start = self.sec_to_duration(old_t * self.time_scale)

                if len(point.velocities) == len(self.joint_names):
                    point.velocities = [v / self.time_scale for v in point.velocities]
                else:
                    point.velocities = []

                if len(point.accelerations) == len(self.joint_names):
                    scale = self.time_scale * self.time_scale
                    point.accelerations = [a / scale for a in point.accelerations]
                else:
                    point.accelerations = []

                point.effort = []

        return trajectory

    def wait_for_future(self, future, timeout_sec: float):
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())

        if not event.wait(timeout_sec):
            return None

        return future.result()

    def plan_cartesian_path(self, target_pose: Pose, timeout_sec: float = 5.0):
        current_positions = self.get_current_joint_positions()
        if current_positions is None:
            return None

        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.header.stamp = self.node.get_clock().now().to_msg()
        req.start_state = self.make_start_state(current_positions)
        req.group_name = self.group_name
        req.link_name = self.ee_link
        req.waypoints = [target_pose]
        req.max_step = self.cartesian_max_step
        req.jump_threshold = self.jump_threshold
        req.avoid_collisions = True

        future = self.cartesian_client.call_async(req)
        res = self.wait_for_future(future, timeout_sec)
        if res is None:
            self.node.get_logger().warn('Timed out waiting for /compute_cartesian_path.')
            return None

        self.node.get_logger().info(
            f'Cartesian path fraction={res.fraction:.4f}, error_code={res.error_code.val}'
        )

        if res.error_code.val != 1 or res.fraction < self.plan_fraction_threshold:
            self.node.get_logger().warn('Cartesian plan rejected by MoveIt.')
            return None

        trajectory = res.solution
        if not self.unwrap_trajectory_positions(trajectory, current_positions):
            return None

        if not self.validate_trajectory_joint_limits(trajectory, current_positions):
            return None

        return self.retime_trajectory(trajectory)

    def execute_pose(self, target_pose: Pose, timeout_sec: float = 20.0) -> bool:
        trajectory = self.plan_cartesian_path(target_pose)
        if trajectory is None:
            return False

        if self.dry_run:
            self.node.get_logger().info('dry_run=true: planned target accepted, not executing.')
            return True

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory.joint_trajectory
        goal_msg.trajectory.header.stamp = (
            self.node.get_clock().now() + RclpyDuration(seconds=0.5)
        ).to_msg()
        goal_msg.goal_time_tolerance = self.sec_to_duration(2.0)

        self.node.get_logger().info(
            f'Sending trajectory goal to controller. points={len(goal_msg.trajectory.points)}'
        )
        send_future = self.traj_client.send_goal_async(goal_msg)
        goal_handle = self.wait_for_future(send_future, timeout_sec=5.0)
        if goal_handle is None:
            self.node.get_logger().warn('Timed out sending trajectory goal.')
            return False

        if not goal_handle.accepted:
            self.node.get_logger().warn('Trajectory goal rejected by controller.')
            return False

        result_future = goal_handle.get_result_async()
        result_wrapper = self.wait_for_future(result_future, timeout_sec=timeout_sec)
        if result_wrapper is None:
            self.node.get_logger().warn('Timed out waiting for trajectory execution result.')
            return False

        result = result_wrapper.result
        ok = result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        if ok:
            self.node.get_logger().info('Trajectory execution succeeded.')
        else:
            self.node.get_logger().warn(
                f'Trajectory execution failed: error_code={result.error_code}, '
                f'error_string="{result.error_string}"'
            )

        return ok
