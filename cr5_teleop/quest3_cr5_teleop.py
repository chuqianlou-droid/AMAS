#!/usr/bin/env python3

import math
import threading
from copy import deepcopy
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Vector3
from std_srvs.srv import SetBool

from cr5_teleop.moveit_pose_controller import MoveItPoseController


class Quest3CR5Teleop(Node):
    def __init__(self):
        super().__init__('quest3_cr5_teleop')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('vr_pose_topic', '/quest/right_controller_pose'),
                ('base_frame', 'base_link'),
                ('ee_link', 'Link6'),
                ('move_group_name', 'cr5_group'),
                ('trajectory_action', '/cr5_group_controller/follow_joint_trajectory'),
                ('position_scale', 0.25),
                ('rotation_scale', 1.0),
                ('orientation_mode', 'fixed'),
                ('command_rate', 3.0),
                ('min_target_delta', 0.001),
                ('workspace_min_x', -0.70),
                ('workspace_max_x', 0.70),
                ('workspace_min_y', -0.70),
                ('workspace_max_y', 0.35),
                ('workspace_min_z', 0.05),
                ('workspace_max_z', 0.80),
                ('max_step_translation', 0.02),
                ('max_total_translation', 0.25),
                ('max_consecutive_failures', 5),
                ('dry_run', True),
                ('axis_map_robot_x', 'vr_z'),
                ('axis_map_robot_y', 'vr_x'),
                ('axis_map_robot_z', 'vr_y'),
                ('axis_sign_robot_x', 1.0),
                ('axis_sign_robot_y', -1.0),
                ('axis_sign_robot_z', 1.0),
                ('cartesian_max_step', 0.005),
                ('jump_threshold', 2.0),
                ('time_scale', 5.0),
                ('plan_fraction_threshold', 0.99),
                ('max_joint1_delta_deg', 30.0),
                ('max_joint2_delta_deg', 45.0),
                ('max_joint3_delta_deg', 45.0),
                ('max_joint4_delta_deg', 60.0),
                ('max_joint5_delta_deg', 60.0),
                ('max_joint6_delta_deg', 90.0),
                ('log_targets', True),
            ],
        )

        self.vr_pose_topic = self.get_parameter('vr_pose_topic').value
        self.base_frame = self.get_parameter('base_frame').value
        self.ee_link = self.get_parameter('ee_link').value
        self.move_group_name = self.get_parameter('move_group_name').value
        self.trajectory_action = self.get_parameter('trajectory_action').value

        self.position_scale = float(self.get_parameter('position_scale').value)
        self.rotation_scale = float(self.get_parameter('rotation_scale').value)
        self.orientation_mode = self.get_parameter('orientation_mode').value
        self.command_rate = max(0.1, float(self.get_parameter('command_rate').value))
        self.min_target_delta = float(self.get_parameter('min_target_delta').value)

        self.workspace_min_x = float(self.get_parameter('workspace_min_x').value)
        self.workspace_max_x = float(self.get_parameter('workspace_max_x').value)
        self.workspace_min_y = float(self.get_parameter('workspace_min_y').value)
        self.workspace_max_y = float(self.get_parameter('workspace_max_y').value)
        self.workspace_min_z = float(self.get_parameter('workspace_min_z').value)
        self.workspace_max_z = float(self.get_parameter('workspace_max_z').value)

        self.max_step_translation = float(self.get_parameter('max_step_translation').value)
        self.max_total_translation = float(self.get_parameter('max_total_translation').value)
        self.max_consecutive_failures = int(self.get_parameter('max_consecutive_failures').value)
        self.dry_run = bool(self.get_parameter('dry_run').value)
        self.log_targets = bool(self.get_parameter('log_targets').value)

        self.axis_map = {
            'x': self.get_parameter('axis_map_robot_x').value,
            'y': self.get_parameter('axis_map_robot_y').value,
            'z': self.get_parameter('axis_map_robot_z').value,
        }
        self.axis_sign = {
            'x': float(self.get_parameter('axis_sign_robot_x').value),
            'y': float(self.get_parameter('axis_sign_robot_y').value),
            'z': float(self.get_parameter('axis_sign_robot_z').value),
        }

        self.controller = MoveItPoseController(
            self,
            group_name=self.move_group_name,
            base_frame=self.base_frame,
            ee_link=self.ee_link,
            trajectory_action=self.trajectory_action,
            cartesian_max_step=float(self.get_parameter('cartesian_max_step').value),
            jump_threshold=float(self.get_parameter('jump_threshold').value),
            time_scale=float(self.get_parameter('time_scale').value),
            dry_run=self.dry_run,
            plan_fraction_threshold=float(self.get_parameter('plan_fraction_threshold').value),
            max_joint_delta_deg={
                'joint1': float(self.get_parameter('max_joint1_delta_deg').value),
                'joint2': float(self.get_parameter('max_joint2_delta_deg').value),
                'joint3': float(self.get_parameter('max_joint3_delta_deg').value),
                'joint4': float(self.get_parameter('max_joint4_delta_deg').value),
                'joint5': float(self.get_parameter('max_joint5_delta_deg').value),
                'joint6': float(self.get_parameter('max_joint6_delta_deg').value),
            },
        )

        self.latest_vr_pose: Optional[PoseStamped] = None
        self.vr_pose_count = 0
        self.no_vr_warned = False
        self.last_status_pose_count = 0
        self.vr_start_pose: Optional[PoseStamped] = None
        self.robot_start_pose: Optional[Pose] = None
        self.last_commanded_pose: Optional[Pose] = None
        self.consecutive_failures = 0
        self.enabled = False
        self.worker_busy = False
        self.teleop_session = 0
        self.state_lock = threading.Lock()
        self.warned_vr_delta_orientation = False

        self.vr_sub = self.create_subscription(
            PoseStamped,
            self.vr_pose_topic,
            self.vr_pose_callback,
            10,
        )
        self.enable_srv = self.create_service(
            SetBool,
            '/cr5_teleop/set_enabled',
            self.set_enabled_callback,
        )
        self.timer = self.create_timer(1.0 / self.command_rate, self.control_loop)
        self.status_timer = self.create_timer(5.0, self.status_loop)

        self.get_logger().info('Quest3 CR5 teleop node started.')
        self.get_logger().info(f'Subscribing VR pose: {self.vr_pose_topic}')
        self.get_logger().info('Enable service: /cr5_teleop/set_enabled')
        self.get_logger().info(f'dry_run={self.dry_run}, command_rate={self.command_rate:.2f} Hz')
        self.get_logger().info(
            'Workspace: '
            f'x=[{self.workspace_min_x:.3f},{self.workspace_max_x:.3f}], '
            f'y=[{self.workspace_min_y:.3f},{self.workspace_max_y:.3f}], '
            f'z=[{self.workspace_min_z:.3f},{self.workspace_max_z:.3f}]'
        )

        self.controller.wait_for_servers()

    def vr_pose_callback(self, msg: PoseStamped):
        with self.state_lock:
            self.latest_vr_pose = msg
            self.vr_pose_count += 1

            if self.vr_pose_count == 1:
                p = msg.pose.position
                self.get_logger().info(
                    'First Quest3 pose received: '
                    f'x={p.x:.3f}, y={p.y:.3f}, z={p.z:.3f}, '
                    f'frame_id="{msg.header.frame_id}"'
                )
                self.no_vr_warned = False

    def status_loop(self):
        with self.state_lock:
            pose_count = self.vr_pose_count
            already_warned = self.no_vr_warned
            last_status_pose_count = self.last_status_pose_count

        if pose_count == 0:
            if not already_warned:
                self.get_logger().warn(
                    f'No Quest3 pose received yet. Is quest_udp_receiver running '
                    f'and publishing {self.vr_pose_topic}?'
                )
                with self.state_lock:
                    self.no_vr_warned = True
            return

        if pose_count == last_status_pose_count:
            self.get_logger().warn(
                f'No new Quest3 pose since last status check. '
                f'total_received={pose_count}'
            )
        else:
            self.get_logger().info(
                f'Quest3 pose received. total_received={pose_count}'
            )

        with self.state_lock:
            self.last_status_pose_count = pose_count

    def set_enabled_callback(self, request: SetBool.Request, response: SetBool.Response):
        if request.data:
            ok, message = self.enable_teleop()
        else:
            ok, message = self.disable_teleop('Disabled by service request.')

        response.success = ok
        response.message = message
        return response

    def enable_teleop(self):
        with self.state_lock:
            latest_vr_pose = deepcopy(self.latest_vr_pose)

        if latest_vr_pose is None:
            return False, 'Cannot enable: no Quest3 pose received yet.'

        robot_pose = self.controller.get_current_pose()
        if robot_pose is None:
            return False, 'Cannot enable: current robot end-effector TF is unavailable.'

        with self.state_lock:
            self.vr_start_pose = latest_vr_pose
            self.robot_start_pose = robot_pose
            self.last_commanded_pose = deepcopy(robot_pose)
            self.consecutive_failures = 0
            self.enabled = True
            self.teleop_session += 1

        p = robot_pose.position
        v = latest_vr_pose.pose.position
        self.get_logger().info(
            f'Teleop enabled. Robot start pose: x={p.x:.3f}, y={p.y:.3f}, z={p.z:.3f}'
        )
        self.get_logger().info(
            f'VR start pose: x={v.x:.3f}, y={v.y:.3f}, z={v.z:.3f}'
        )
        return True, 'Teleop enabled.'

    def disable_teleop(self, message: str):
        with self.state_lock:
            self.enabled = False
            self.vr_start_pose = None
            self.robot_start_pose = None
            self.last_commanded_pose = None
            self.consecutive_failures = 0
            self.teleop_session += 1

        self.get_logger().info(message)
        return True, message

    @staticmethod
    def clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def position_distance(a, b) -> float:
        return math.sqrt(
            (a.x - b.x) * (a.x - b.x)
            + (a.y - b.y) * (a.y - b.y)
            + (a.z - b.z) * (a.z - b.z)
        )

    @staticmethod
    def normalize_vector(vec):
        norm = math.sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z)
        if norm <= 1e-9:
            return vec, 0.0
        return Vector3(x=vec.x / norm, y=vec.y / norm, z=vec.z / norm), norm

    def map_vr_delta_to_robot_delta(self, vr_delta: Vector3) -> Vector3:
        values = {
            'vr_x': vr_delta.x,
            'vr_y': vr_delta.y,
            'vr_z': vr_delta.z,
        }

        robot_delta = Vector3()
        robot_delta.x = self.axis_sign['x'] * self.position_scale * values.get(self.axis_map['x'], 0.0)
        robot_delta.y = self.axis_sign['y'] * self.position_scale * values.get(self.axis_map['y'], 0.0)
        robot_delta.z = self.axis_sign['z'] * self.position_scale * values.get(self.axis_map['z'], 0.0)
        return robot_delta

    def limit_total_translation(self, target_pose: Pose, robot_start_pose: Pose) -> Pose:
        start = robot_start_pose.position
        delta = Vector3(
            x=target_pose.position.x - start.x,
            y=target_pose.position.y - start.y,
            z=target_pose.position.z - start.z,
        )
        direction, distance = self.normalize_vector(delta)
        if distance <= self.max_total_translation:
            return target_pose

        target_pose.position.x = start.x + direction.x * self.max_total_translation
        target_pose.position.y = start.y + direction.y * self.max_total_translation
        target_pose.position.z = start.z + direction.z * self.max_total_translation
        self.get_logger().warn('Target limited by max_total_translation.')
        return target_pose

    def clamp_workspace(self, target_pose: Pose) -> Pose:
        old = deepcopy(target_pose.position)
        target_pose.position.x = self.clamp(
            target_pose.position.x, self.workspace_min_x, self.workspace_max_x
        )
        target_pose.position.y = self.clamp(
            target_pose.position.y, self.workspace_min_y, self.workspace_max_y
        )
        target_pose.position.z = self.clamp(
            target_pose.position.z, self.workspace_min_z, self.workspace_max_z
        )

        if self.position_distance(old, target_pose.position) > 1e-9:
            self.get_logger().warn(
                'Target clamped to workspace: '
                f'x={target_pose.position.x:.3f}, '
                f'y={target_pose.position.y:.3f}, '
                f'z={target_pose.position.z:.3f}'
            )

        return target_pose

    def limit_step_translation(self, target_pose: Pose, last_commanded_pose: Optional[Pose]) -> Pose:
        if last_commanded_pose is None:
            return target_pose

        last = last_commanded_pose.position
        delta = Vector3(
            x=target_pose.position.x - last.x,
            y=target_pose.position.y - last.y,
            z=target_pose.position.z - last.z,
        )
        direction, distance = self.normalize_vector(delta)
        if distance <= self.max_step_translation:
            return target_pose

        target_pose.position.x = last.x + direction.x * self.max_step_translation
        target_pose.position.y = last.y + direction.y * self.max_step_translation
        target_pose.position.z = last.z + direction.z * self.max_step_translation
        self.get_logger().warn('Target limited by max_step_translation.')
        return target_pose

    @staticmethod
    def normalize_quaternion(q: Quaternion) -> Quaternion:
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if norm <= 1e-9:
            return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        return Quaternion(x=q.x / norm, y=q.y / norm, z=q.z / norm, w=q.w / norm)

    @staticmethod
    def inverse_quaternion(q: Quaternion) -> Quaternion:
        qn = Quest3CR5Teleop.normalize_quaternion(q)
        return Quaternion(x=-qn.x, y=-qn.y, z=-qn.z, w=qn.w)

    @staticmethod
    def multiply_quaternion(a: Quaternion, b: Quaternion) -> Quaternion:
        return Quest3CR5Teleop.normalize_quaternion(Quaternion(
            x=a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            y=a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            z=a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            w=a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        ))

    def choose_orientation(
        self,
        current_vr_pose: PoseStamped,
        vr_start_pose: PoseStamped,
        robot_start_pose: Pose,
    ) -> Quaternion:
        if self.orientation_mode == 'fixed':
            return robot_start_pose.orientation

        if self.orientation_mode == 'vr_delta':
            if not self.warned_vr_delta_orientation:
                self.get_logger().warn(
                    'orientation_mode=vr_delta is experimental and does not remap VR axes yet.'
                )
                self.warned_vr_delta_orientation = True

            q_delta = self.multiply_quaternion(
                current_vr_pose.pose.orientation,
                self.inverse_quaternion(vr_start_pose.pose.orientation),
            )
            return self.multiply_quaternion(q_delta, robot_start_pose.orientation)

        self.get_logger().warn(
            f'Unknown orientation_mode "{self.orientation_mode}", falling back to fixed.'
        )
        return robot_start_pose.orientation

    def build_target_pose(self) -> Optional[Pose]:
        with self.state_lock:
            if not self.enabled:
                return None
            current_vr = deepcopy(self.latest_vr_pose)
            vr_start = deepcopy(self.vr_start_pose)
            robot_start = deepcopy(self.robot_start_pose)
            last_commanded = deepcopy(self.last_commanded_pose)

        if current_vr is None or vr_start is None or robot_start is None:
            return None

        vr_delta = Vector3(
            x=current_vr.pose.position.x - vr_start.pose.position.x,
            y=current_vr.pose.position.y - vr_start.pose.position.y,
            z=current_vr.pose.position.z - vr_start.pose.position.z,
        )
        robot_delta = self.map_vr_delta_to_robot_delta(vr_delta)

        target = Pose()
        target.position.x = robot_start.position.x + robot_delta.x
        target.position.y = robot_start.position.y + robot_delta.y
        target.position.z = robot_start.position.z + robot_delta.z
        target.orientation = self.choose_orientation(current_vr, vr_start, robot_start)

        target = self.limit_total_translation(target, robot_start)
        target = self.clamp_workspace(target)
        target = self.limit_step_translation(target, last_commanded)

        if self.log_targets:
            self.get_logger().info(
                'Target built: '
                f'vr_delta=({vr_delta.x:+.3f},{vr_delta.y:+.3f},{vr_delta.z:+.3f}) '
                f'robot_delta=({robot_delta.x:+.3f},{robot_delta.y:+.3f},{robot_delta.z:+.3f}) '
                f'target=({target.position.x:+.3f},{target.position.y:+.3f},{target.position.z:+.3f})'
            )

        return target

    def should_send_target(self, target_pose: Pose) -> bool:
        with self.state_lock:
            if self.last_commanded_pose is None:
                return True
            distance = self.position_distance(
                target_pose.position,
                self.last_commanded_pose.position,
            )

        epsilon = 1e-6
        if distance + epsilon < self.min_target_delta:
            if self.log_targets:
                self.get_logger().info(
                    f'Target skipped: delta={distance:.4f} m < min_target_delta={self.min_target_delta:.4f} m'
                )
            return False

        return True

    def control_loop(self):
        with self.state_lock:
            enabled = self.enabled
            busy = self.worker_busy

        if not enabled or busy:
            return

        target_pose = self.build_target_pose()
        if target_pose is None or not self.should_send_target(target_pose):
            return

        with self.state_lock:
            self.worker_busy = True
            session = self.teleop_session

        worker = threading.Thread(
            target=self.execute_target_worker,
            args=(deepcopy(target_pose), session),
            daemon=True,
        )
        worker.start()

    def execute_target_worker(self, target_pose: Pose, session: int):
        try:
            if self.log_targets:
                p = target_pose.position
                self.get_logger().info(
                    f'Sending target to MoveIt: x={p.x:.3f}, y={p.y:.3f}, z={p.z:.3f}'
                )

            ok = self.controller.execute_pose(target_pose)

            with self.state_lock:
                if session != self.teleop_session:
                    return

                if ok:
                    self.last_commanded_pose = deepcopy(target_pose)
                    self.consecutive_failures = 0
                    self.get_logger().info('MoveIt command accepted/executed.')
                else:
                    self.consecutive_failures += 1
                    failures = self.consecutive_failures

                    if failures >= self.max_consecutive_failures:
                        self.enabled = False
                        self.vr_start_pose = None
                        self.robot_start_pose = None
                        self.last_commanded_pose = None
                        self.consecutive_failures = 0
                        self.get_logger().error(
                            'Too many consecutive MoveIt failures; teleop disabled.'
                        )
                    else:
                        self.get_logger().warn(
                            f'MoveIt command failed ({failures}/{self.max_consecutive_failures}).'
                        )
        finally:
            with self.state_lock:
                self.worker_busy = False


def main(args=None):
    rclpy.init(args=args)
    node = Quest3CR5Teleop()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
