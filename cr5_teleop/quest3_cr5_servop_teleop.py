#!/usr/bin/env python3

import math
import re
import threading
from copy import deepcopy
from typing import Optional

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from dobot_msgs_v4.srv import GetPose, ServoP
from geometry_msgs.msg import PoseStamped, Vector3
from std_srvs.srv import SetBool


class Quest3CR5ServoPTeleop(Node):
    def __init__(self):
        super().__init__('quest3_cr5_servop_teleop')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('vr_pose_topic', '/quest/right_controller_pose'),
                ('get_pose_service', '/dobot_bringup_ros2/srv/GetPose'),
                ('servop_service', '/dobot_bringup_ros2/srv/ServoP'),
                ('position_scale', 0.20),
                ('command_rate', 10.0),
                ('min_target_delta_mm', 0.0),
                ('raw_target_filter_ratio', 0.80),
                ('target_deadband_mm', 2.0),
                ('max_speed_mm_s', 50.0),
                ('max_accel_mm_s2', 250.0),
                ('max_total_translation_mm', 120.0),
                ('workspace_min_x_mm', -700.0),
                ('workspace_max_x_mm', 700.0),
                ('workspace_min_y_mm', -700.0),
                ('workspace_max_y_mm', 350.0),
                ('workspace_min_z_mm', 50.0),
                ('workspace_max_z_mm', 800.0),
                ('servo_t', 0.10),
                ('servo_aheadtime', 50.0),
                ('servo_gain', 200.0),
                ('axis_map_robot_x', 'vr_x'),
                ('axis_map_robot_y', 'vr_z'),
                ('axis_map_robot_z', 'vr_y'),
                ('axis_sign_robot_x', -1.0),
                ('axis_sign_robot_y', -1.0),
                ('axis_sign_robot_z', 1.0),
                ('log_targets', False),
            ],
        )

        self.vr_pose_topic = self.get_parameter('vr_pose_topic').value
        self.get_pose_service = self.get_parameter('get_pose_service').value
        self.servop_service = self.get_parameter('servop_service').value

        self.position_scale = float(self.get_parameter('position_scale').value)
        self.command_rate = max(1.0, float(self.get_parameter('command_rate').value))
        self.min_target_delta_mm = float(self.get_parameter('min_target_delta_mm').value)
        self.raw_target_filter_ratio = self.clamp(
            float(self.get_parameter('raw_target_filter_ratio').value),
            0.0,
            0.98,
        )
        self.target_deadband_mm = float(self.get_parameter('target_deadband_mm').value)
        self.max_speed_mm_s = float(self.get_parameter('max_speed_mm_s').value)
        self.max_accel_mm_s2 = float(self.get_parameter('max_accel_mm_s2').value)
        self.max_total_translation_mm = float(self.get_parameter('max_total_translation_mm').value)

        self.workspace_min_x_mm = float(self.get_parameter('workspace_min_x_mm').value)
        self.workspace_max_x_mm = float(self.get_parameter('workspace_max_x_mm').value)
        self.workspace_min_y_mm = float(self.get_parameter('workspace_min_y_mm').value)
        self.workspace_max_y_mm = float(self.get_parameter('workspace_max_y_mm').value)
        self.workspace_min_z_mm = float(self.get_parameter('workspace_min_z_mm').value)
        self.workspace_max_z_mm = float(self.get_parameter('workspace_max_z_mm').value)

        self.servo_t = float(self.get_parameter('servo_t').value)
        expected_servo_t = 1.0 / self.command_rate
        if abs(self.servo_t - expected_servo_t) > 0.02:
            self.get_logger().warn(
                f'servo_t={self.servo_t:.3f}s does not match command_rate={self.command_rate:.1f}Hz '
                f'(period={expected_servo_t:.3f}s). Fixed-interval ServoP works best when they match.'
            )
        self.servo_aheadtime = float(self.get_parameter('servo_aheadtime').value)
        self.servo_gain = float(self.get_parameter('servo_gain').value)
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

        self.latest_vr_pose: Optional[PoseStamped] = None
        self.vr_pose_count = 0
        self.no_vr_warned = False
        self.last_status_pose_count = 0
        self.enable_requested = False
        self.requested_vr_start_pose: Optional[PoseStamped] = None
        self.pending_get_pose_future = None
        self.vr_start_pose: Optional[PoseStamped] = None
        self.robot_start_pose_mm_deg: Optional[list] = None
        self.filtered_raw_target_mm_deg: Optional[list] = None
        self.planned_target_mm_deg: Optional[list] = None
        self.planned_velocity_mm_s = [0.0, 0.0, 0.0]
        self.last_commanded_mm_deg: Optional[list] = None
        self.pending_servo_future = None
        self.next_servo_send_time = self.get_clock().now()
        self.enabled = False
        self.state_lock = threading.Lock()

        self.vr_sub = self.create_subscription(
            PoseStamped,
            self.vr_pose_topic,
            self.vr_pose_callback,
            1,
        )
        self.enable_srv = self.create_service(
            SetBool,
            '/cr5_teleop/servop/set_enabled',
            self.set_enabled_callback,
        )

        self.get_pose_client = self.create_client(GetPose, self.get_pose_service)
        self.servop_client = self.create_client(ServoP, self.servop_service)

        self.timer = self.create_timer(1.0 / self.command_rate, self.control_loop)
        self.status_timer = self.create_timer(5.0, self.status_loop)

        self.get_logger().info('Quest3 CR5 ServoP teleop node started.')
        self.get_logger().info(f'Subscribing VR pose: {self.vr_pose_topic}')
        self.get_logger().info('Enable service: /cr5_teleop/servop/set_enabled')
        self.get_logger().info(f'GetPose service: {self.get_pose_service}')
        self.get_logger().info(f'ServoP service: {self.servop_service}')
        self.get_logger().info(
            f'command_rate={self.command_rate:.1f} Hz, '
            f'max_speed={self.max_speed_mm_s:.1f} mm/s, '
            f'max_accel={self.max_accel_mm_s2:.1f} mm/s^2, '
            f'servo_t={self.servo_t:.3f}s, '
            f'target_deadband={self.target_deadband_mm:.1f} mm'
        )
        self.get_logger().info(
            'Axis mapping: '
            f'robot_x={self.axis_sign["x"]:+.0f}*{self.axis_map["x"]}, '
            f'robot_y={self.axis_sign["y"]:+.0f}*{self.axis_map["y"]}, '
            f'robot_z={self.axis_sign["z"]:+.0f}*{self.axis_map["z"]}'
        )

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
            ok, message = self.disable_teleop('ServoP teleop disabled.')

        response.success = ok
        response.message = message
        return response

    def enable_teleop(self):
        with self.state_lock:
            latest_vr_pose = deepcopy(self.latest_vr_pose)

        if latest_vr_pose is None:
            return False, 'Cannot enable: no Quest3 pose received yet.'

        with self.state_lock:
            self.enable_requested = True
            self.requested_vr_start_pose = latest_vr_pose
            self.enabled = False
            self.pending_get_pose_future = None

        self.get_logger().info('Enable requested. Waiting for current robot pose from GetPose...')
        return True, 'ServoP teleop enable requested.'

    def finish_enable_teleop(self, robot_pose: list):
        with self.state_lock:
            if self.requested_vr_start_pose is None:
                return

            self.vr_start_pose = deepcopy(self.requested_vr_start_pose)
            self.robot_start_pose_mm_deg = robot_pose
            self.filtered_raw_target_mm_deg = list(robot_pose)
            self.planned_target_mm_deg = list(robot_pose)
            self.planned_velocity_mm_s = [0.0, 0.0, 0.0]
            self.last_commanded_mm_deg = None
            self.pending_servo_future = None
            self.next_servo_send_time = self.get_clock().now()
            self.enabled = True
            self.enable_requested = False
            self.pending_get_pose_future = None
            vr_start_pose = deepcopy(self.vr_start_pose)

        v = vr_start_pose.pose.position
        self.get_logger().info(
            'ServoP teleop enabled. '
            f'robot_start=({robot_pose[0]:.1f},{robot_pose[1]:.1f},{robot_pose[2]:.1f},'
            f'{robot_pose[3]:.1f},{robot_pose[4]:.1f},{robot_pose[5]:.1f}), '
            f'vr_start=({v.x:.3f},{v.y:.3f},{v.z:.3f})'
        )

    def disable_teleop(self, message: str):
        with self.state_lock:
            self.enabled = False
            self.enable_requested = False
            self.requested_vr_start_pose = None
            self.pending_get_pose_future = None
            self.vr_start_pose = None
            self.robot_start_pose_mm_deg = None
            self.filtered_raw_target_mm_deg = None
            self.planned_target_mm_deg = None
            self.planned_velocity_mm_s = [0.0, 0.0, 0.0]
            self.last_commanded_mm_deg = None
            self.pending_servo_future = None
            self.next_servo_send_time = self.get_clock().now()

        self.get_logger().info(message)
        return True, message

    def start_get_pose_request(self):
        if not self.get_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f'GetPose service not available: {self.get_pose_service}')
            with self.state_lock:
                self.enable_requested = False
            return

        req = GetPose.Request()
        req.user = 0
        req.tool = 0

        with self.state_lock:
            self.pending_get_pose_future = self.get_pose_client.call_async(req)

    def check_get_pose_response(self):
        with self.state_lock:
            pending_get_pose_future = self.pending_get_pose_future

        if pending_get_pose_future is None:
            return

        if not pending_get_pose_future.done():
            return

        future = pending_get_pose_future
        with self.state_lock:
            self.pending_get_pose_future = None

        if future.result() is None:
            self.get_logger().warn('GetPose returned no response. Enable cancelled.')
            with self.state_lock:
                self.enable_requested = False
            return None

        res = future.result()
        if res.res != 0:
            self.get_logger().warn(f'GetPose service res={res.res}, raw="{res.robot_return}". Enable cancelled.')
            with self.state_lock:
                self.enable_requested = False
            return

        values = self.parse_dobot_return_values(res.robot_return)
        if len(values) < 6:
            self.get_logger().warn(f'Failed to parse GetPose return: "{res.robot_return}". Enable cancelled.')
            with self.state_lock:
                self.enable_requested = False
            return

        self.finish_enable_teleop(values[:6])

    @staticmethod
    def parse_dobot_return_values(text: str) -> list:
        match = re.search(r'\{([^}]*)\}', text)
        if not match:
            return []
        values = []
        for item in match.group(1).split(','):
            item = item.strip()
            if not item:
                continue
            try:
                values.append(float(item))
            except ValueError:
                return []
        return values

    @staticmethod
    def clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def distance_mm(a: list, b: list) -> float:
        return math.sqrt(
            (a[0] - b[0]) * (a[0] - b[0])
            + (a[1] - b[1]) * (a[1] - b[1])
            + (a[2] - b[2]) * (a[2] - b[2])
        )

    @staticmethod
    def norm3(vec: list) -> float:
        return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])

    def map_vr_delta_to_robot_delta_mm(self, vr_delta: Vector3) -> list:
        values = {
            'vr_x': vr_delta.x,
            'vr_y': vr_delta.y,
            'vr_z': vr_delta.z,
        }
        return [
            self.axis_sign['x'] * self.position_scale * values.get(self.axis_map['x'], 0.0) * 1000.0,
            self.axis_sign['y'] * self.position_scale * values.get(self.axis_map['y'], 0.0) * 1000.0,
            self.axis_sign['z'] * self.position_scale * values.get(self.axis_map['z'], 0.0) * 1000.0,
        ]

    def build_raw_target(self) -> Optional[list]:
        with self.state_lock:
            latest_vr_pose = deepcopy(self.latest_vr_pose)
            vr_start_pose = deepcopy(self.vr_start_pose)
            robot_start_pose = (
                list(self.robot_start_pose_mm_deg)
                if self.robot_start_pose_mm_deg is not None
                else None
            )

        if latest_vr_pose is None or vr_start_pose is None or robot_start_pose is None:
            return None

        current = latest_vr_pose.pose.position
        start = vr_start_pose.pose.position
        vr_delta = Vector3(
            x=current.x - start.x,
            y=current.y - start.y,
            z=current.z - start.z,
        )
        robot_delta = self.map_vr_delta_to_robot_delta_mm(vr_delta)

        target = list(robot_start_pose)
        target[0] += robot_delta[0]
        target[1] += robot_delta[1]
        target[2] += robot_delta[2]

        target = self.limit_total_translation(target)
        target = self.clamp_workspace(target)

        if self.log_targets:
            self.get_logger().info(
                'ServoP raw target: '
                f'vr_delta=({vr_delta.x:+.3f},{vr_delta.y:+.3f},{vr_delta.z:+.3f}) '
                f'robot_delta_mm=({robot_delta[0]:+.1f},{robot_delta[1]:+.1f},{robot_delta[2]:+.1f}) '
                f'target=({target[0]:+.1f},{target[1]:+.1f},{target[2]:+.1f})'
            )

        return target

    def limit_total_translation(self, target: list) -> list:
        with self.state_lock:
            start = (
                list(self.robot_start_pose_mm_deg)
                if self.robot_start_pose_mm_deg is not None
                else None
            )

        if start is None:
            return target

        dx = target[0] - start[0]
        dy = target[1] - start[1]
        dz = target[2] - start[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist <= self.max_total_translation_mm or dist <= 1e-9:
            return target

        scale = self.max_total_translation_mm / dist
        target[0] = start[0] + dx * scale
        target[1] = start[1] + dy * scale
        target[2] = start[2] + dz * scale
        return target

    def clamp_workspace(self, target: list) -> list:
        target[0] = self.clamp(target[0], self.workspace_min_x_mm, self.workspace_max_x_mm)
        target[1] = self.clamp(target[1], self.workspace_min_y_mm, self.workspace_max_y_mm)
        target[2] = self.clamp(target[2], self.workspace_min_z_mm, self.workspace_max_z_mm)
        return target

    def filter_raw_target(self, raw_target: list) -> list:
        with self.state_lock:
            if self.filtered_raw_target_mm_deg is None:
                self.filtered_raw_target_mm_deg = list(raw_target)

            filtered = list(self.filtered_raw_target_mm_deg)
            robot_start_pose = (
                list(self.robot_start_pose_mm_deg)
                if self.robot_start_pose_mm_deg is not None
                else None
            )

        new_weight = 1.0 - self.raw_target_filter_ratio
        for i in range(3):
            filtered[i] = (
                filtered[i] * self.raw_target_filter_ratio
                + raw_target[i] * new_weight
            )

        if robot_start_pose is not None:
            for i in range(3, 6):
                filtered[i] = robot_start_pose[i]

        filtered = self.clamp_workspace(filtered)

        with self.state_lock:
            self.filtered_raw_target_mm_deg = list(filtered)

        if self.log_targets:
            raw_delta = self.distance_mm(raw_target, filtered)
            self.get_logger().info(
                f'Filtered raw target: lag={raw_delta:.2f} mm, '
                f'target=({filtered[0]:.1f},{filtered[1]:.1f},{filtered[2]:.1f})'
            )

        return filtered

    def plan_velocity_limited_target(self, raw_target: list) -> list:
        with self.state_lock:
            if self.planned_target_mm_deg is None:
                self.planned_target_mm_deg = list(raw_target)

            planned = list(self.planned_target_mm_deg)
            velocity = list(self.planned_velocity_mm_s)
            robot_start_pose = (
                list(self.robot_start_pose_mm_deg)
                if self.robot_start_pose_mm_deg is not None
                else None
            )

        dt = max(1e-3, self.servo_t)
        error = [
            raw_target[0] - planned[0],
            raw_target[1] - planned[1],
            raw_target[2] - planned[2],
        ]
        distance = self.norm3(error)

        if distance <= self.target_deadband_mm:
            desired_velocity = [0.0, 0.0, 0.0]
        else:
            direction = [error[0] / distance, error[1] / distance, error[2] / distance]
            braking_speed = math.sqrt(max(0.0, 2.0 * self.max_accel_mm_s2 * distance))
            desired_speed = min(self.max_speed_mm_s, braking_speed)
            desired_velocity = [
                direction[0] * desired_speed,
                direction[1] * desired_speed,
                direction[2] * desired_speed,
            ]

        velocity_delta = [
            desired_velocity[0] - velocity[0],
            desired_velocity[1] - velocity[1],
            desired_velocity[2] - velocity[2],
        ]
        velocity_delta_norm = self.norm3(velocity_delta)
        max_velocity_delta = self.max_accel_mm_s2 * dt
        if velocity_delta_norm > max_velocity_delta and velocity_delta_norm > 1e-9:
            scale = max_velocity_delta / velocity_delta_norm
            velocity_delta = [
                velocity_delta[0] * scale,
                velocity_delta[1] * scale,
                velocity_delta[2] * scale,
            ]

        velocity = [
            velocity[0] + velocity_delta[0],
            velocity[1] + velocity_delta[1],
            velocity[2] + velocity_delta[2],
        ]
        speed = self.norm3(velocity)
        if speed > self.max_speed_mm_s and speed > 1e-9:
            scale = self.max_speed_mm_s / speed
            velocity = [velocity[0] * scale, velocity[1] * scale, velocity[2] * scale]

        step = [velocity[0] * dt, velocity[1] * dt, velocity[2] * dt]
        step_norm = self.norm3(step)
        if step_norm > distance and distance > 1e-9:
            step = error
            velocity = [0.0, 0.0, 0.0]

        planned[0] += step[0]
        planned[1] += step[1]
        planned[2] += step[2]

        if robot_start_pose is not None:
            for i in range(3, 6):
                planned[i] = robot_start_pose[i]

        planned = self.clamp_workspace(planned)
        with self.state_lock:
            self.planned_target_mm_deg = list(planned)
            self.planned_velocity_mm_s = list(velocity)

        if self.log_targets:
            self.get_logger().info(
                'Velocity plan: '
                f'error={distance:.2f} mm, deadband={self.target_deadband_mm:.1f} mm, '
                f'speed={self.norm3(velocity):.1f} mm/s, '
                f'planned=({planned[0]:.1f},{planned[1]:.1f},{planned[2]:.1f})'
            )

        return planned

    def control_loop(self):
        with self.state_lock:
            enable_requested = self.enable_requested
            pending_get_pose_future = self.pending_get_pose_future
            enabled = self.enabled
            pending_servo_future = self.pending_servo_future

        if enable_requested and pending_get_pose_future is None:
            self.start_get_pose_request()

        if pending_get_pose_future is not None:
            self.check_get_pose_response()

        if not enabled:
            return

        if pending_servo_future is not None and not pending_servo_future.done():
            if self.log_targets:
                self.get_logger().info('ServoP skip: previous ServoP request is still pending.')
            return

        raw_target = self.build_raw_target()
        if raw_target is None:
            return

        filtered_raw_target = self.filter_raw_target(raw_target)
        target = self.plan_velocity_limited_target(filtered_raw_target)
        with self.state_lock:
            last_commanded = (
                list(self.last_commanded_mm_deg)
                if self.last_commanded_mm_deg is not None
                else None
            )
            robot_start_pose = (
                list(self.robot_start_pose_mm_deg)
                if self.robot_start_pose_mm_deg is not None
                else None
            )

        if last_commanded is not None:
            target_delta = self.distance_mm(target, last_commanded)
            if target_delta < self.min_target_delta_mm:
                if self.log_targets:
                    self.get_logger().info(
                        f'ServoP skip: target delta {target_delta:.2f} mm '
                        f'< min_target_delta_mm {self.min_target_delta_mm:.2f} mm.'
                    )
                return

        command_reference = last_commanded or robot_start_pose
        if command_reference is None:
            return

        command_distance = self.distance_mm(target, command_reference)

        if self.log_targets:
            self.get_logger().info(
                f'ServoP command: distance={command_distance:.2f} mm, '
                f't={self.servo_t:.3f}s, '
                f'target=({target[0]:.1f},{target[1]:.1f},{target[2]:.1f})'
            )

        if not self.servop_client.service_is_ready():
            if not self.servop_client.wait_for_service(timeout_sec=0.0):
                self.get_logger().warn(f'ServoP service not available: {self.servop_service}')
                return

        req = ServoP.Request()
        req.a = target[0]
        req.b = target[1]
        req.c = target[2]
        req.d = target[3]
        req.e = target[4]
        req.f = target[5]
        req.param_value = [
            f't={self.servo_t:.3f}',
            f'aheadtime={self.servo_aheadtime:.1f}',
            f'gain={self.servo_gain:.1f}',
        ]

        pending_servo_future = self.servop_client.call_async(req)
        pending_servo_future.add_done_callback(self.handle_servo_response)

        with self.state_lock:
            self.pending_servo_future = pending_servo_future
            self.next_servo_send_time = self.get_clock().now() + Duration(seconds=self.servo_t)
            self.last_commanded_mm_deg = list(target)

    def handle_servo_response(self, future):
        try:
            res = future.result()
        except Exception as exc:
            self.get_logger().warn(f'ServoP call failed: {exc}')
            return

        if res is None:
            self.get_logger().warn('ServoP returned no response.')
            return

        if res.res != 0:
            self.get_logger().warn(f'ServoP response res={res.res}')


def main(args=None):
    rclpy.init(args=args)
    node = Quest3CR5ServoPTeleop()

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
