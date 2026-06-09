from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState

import tf2_ros


class CartesianIKTeleop(Node):
    def __init__(self):
        super().__init__('cartesian_ik_teleop')

        # =========================
        # Robot / MoveIt 参数
        # =========================
        self.group_name = 'cr5_group'
        self.base_frame = 'base_link'
        self.ee_link = 'Link6'

        self.quest_topic = '/quest/right_controller_pose'
        self.traj_topic = '/cr5_group_controller/joint_trajectory'

        self.joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
        ]

        # =========================
        # 控制参数
        # =========================
        # MoveIt IK 不能太高频，先用 5Hz 保守测试
        self.control_rate = 5.0

        # 手柄移动 1m，末端移动 0.4m
        self.position_scale = 0.4

        # 轨迹执行时间
        self.traj_time = 0.20

        # 手柄微小抖动过滤
        self.deadzone = 0.005

        # 末端相对初始位姿的最大位移
        self.max_dx = 0.25
        self.max_dy = 0.25
        self.max_dz = 0.25

        # IK 请求超时时间
        self.ik_timeout_sec = 0.60

        # =========================
        # 坐标轴映射
        # =========================
        # Quest / Unity:
        #   x: 右
        #   y: 上
        #   z: 前
        #
        # Robot base_link:
        #   x: 前
        #   y: 左/右
        #   z: 上
        #
        # 默认：
        #   Quest +z 前  -> Robot +x
        #   Quest +x 右  -> Robot -y
        #   Quest +y 上  -> Robot +z
        self.sign_forward = 1.0
        self.sign_right = -1.0
        self.sign_up = 1.0

        # =========================
        # 状态变量
        # =========================
        self.latest_quest_pose: Optional[PoseStamped] = None
        self.latest_joint_state: Optional[JointState] = None

        self.initial_quest_pos = None
        self.initial_ee_pose = None

        # pending IK
        self.pending_ik_future = None
        self.pending_ik_start_time = None

        # =========================
        # TF
        # =========================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # =========================
        # ROS 通信
        # =========================
        self.quest_sub = self.create_subscription(
            PoseStamped,
            self.quest_topic,
            self.quest_callback,
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.traj_pub = self.create_publisher(
            JointTrajectory,
            self.traj_topic,
            10
        )

        self.ik_client = self.create_client(
            GetPositionIK,
            '/compute_ik'
        )

        self.timer = self.create_timer(
            1.0 / self.control_rate,
            self.control_loop
        )

        self.get_logger().info('Cartesian IK teleop started.')
        self.get_logger().info(f'group_name = {self.group_name}')
        self.get_logger().info(f'base_frame = {self.base_frame}')
        self.get_logger().info(f'ee_link    = {self.ee_link}')
        self.get_logger().info(f'quest_topic = {self.quest_topic}')
        self.get_logger().info(f'traj_topic  = {self.traj_topic}')
        self.get_logger().info('Mapping: Quest +z -> Robot +x, Quest +x -> Robot -y, Quest +y -> Robot +z')

    def quest_callback(self, msg: PoseStamped):
        self.latest_quest_pose = msg

    def joint_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def apply_deadzone(self, v: float) -> float:
        if abs(v) < self.deadzone:
            return 0.0
        return v

    def clamp(self, v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def lookup_current_ee_pose(self) -> Optional[PoseStamped]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_link,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.3)
            )
        except Exception as e:
            self.get_logger().warn(
                f'Failed to lookup TF {self.base_frame} -> {self.ee_link}: {e}'
            )
            return None

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = tf.transform.translation.x
        pose.pose.position.y = tf.transform.translation.y
        pose.pose.position.z = tf.transform.translation.z
        pose.pose.orientation = tf.transform.rotation

        return pose

    def build_target_pose(self) -> Optional[PoseStamped]:
        if self.latest_quest_pose is None:
            return None

        q = self.latest_quest_pose.pose.position

        if self.initial_quest_pos is None:
            self.initial_quest_pos = [q.x, q.y, q.z]
            self.get_logger().info(
                f'Initial Quest pose: x={q.x:.3f}, y={q.y:.3f}, z={q.z:.3f}'
            )

        if self.initial_ee_pose is None:
            self.initial_ee_pose = self.lookup_current_ee_pose()
            if self.initial_ee_pose is None:
                return None

            p0 = self.initial_ee_pose.pose.position
            o0 = self.initial_ee_pose.pose.orientation
            self.get_logger().info(
                f'Initial EE pose in {self.base_frame}: '
                f'x={p0.x:.3f}, y={p0.y:.3f}, z={p0.z:.3f}, '
                f'qx={o0.x:.3f}, qy={o0.y:.3f}, qz={o0.z:.3f}, qw={o0.w:.3f}'
            )

        # Quest 相对位移
        dx_q = q.x - self.initial_quest_pos[0]  # 右
        dy_q = q.y - self.initial_quest_pos[1]  # 上
        dz_q = q.z - self.initial_quest_pos[2]  # 前

        dx_q = self.apply_deadzone(dx_q)
        dy_q = self.apply_deadzone(dy_q)
        dz_q = self.apply_deadzone(dz_q)

        # 末端到末端映射
        robot_dx = self.sign_forward * self.position_scale * dz_q
        robot_dy = self.sign_right * self.position_scale * dx_q
        robot_dz = self.sign_up * self.position_scale * dy_q

        robot_dx = self.clamp(robot_dx, -self.max_dx, self.max_dx)
        robot_dy = self.clamp(robot_dy, -self.max_dy, self.max_dy)
        robot_dz = self.clamp(robot_dz, -self.max_dz, self.max_dz)

        p0 = self.initial_ee_pose.pose.position

        target = PoseStamped()
        target.header.frame_id = self.base_frame
        target.header.stamp = self.get_clock().now().to_msg()

        target.pose.position.x = p0.x + robot_dx
        target.pose.position.y = p0.y + robot_dy
        target.pose.position.z = p0.z + robot_dz

        # 固定初始末端姿态，只控制位置
        target.pose.orientation = self.initial_ee_pose.pose.orientation

        return target

    def send_ik_request(self, target_pose: PoseStamped):
        if self.latest_joint_state is None:
            self.get_logger().warn('No /joint_states received yet.')
            return

        if not self.ik_client.service_is_ready():
            self.get_logger().warn('/compute_ik service not ready.')
            return

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.pose_stamped = target_pose
        req.ik_request.avoid_collisions = False

        seed_state = RobotState()
        seed_state.joint_state = self.latest_joint_state
        req.ik_request.robot_state = seed_state

        req.ik_request.timeout.sec = 0
        req.ik_request.timeout.nanosec = int(self.ik_timeout_sec * 1e9)

        self.pending_ik_future = self.ik_client.call_async(req)
        self.pending_ik_start_time = self.get_clock().now()

    def check_pending_ik(self):
        if self.pending_ik_future is None:
            return

        now = self.get_clock().now()
        elapsed = (now - self.pending_ik_start_time).nanoseconds / 1e9

        if self.pending_ik_future.done():
            future = self.pending_ik_future
            self.pending_ik_future = None
            self.pending_ik_start_time = None

            res = future.result()

            if res is None:
                self.get_logger().warn('IK result is None.')
                return

            if res.error_code.val != res.error_code.SUCCESS:
                self.get_logger().warn(f'IK failed, error_code={res.error_code.val}')
                return

            positions = self.extract_positions(res.solution.joint_state)
            if positions is None:
                return

            self.publish_trajectory(positions)
            return

        if elapsed > self.ik_timeout_sec + 0.2:
            self.get_logger().warn('IK timeout, dropping pending request.')
            self.pending_ik_future = None
            self.pending_ik_start_time = None

    def extract_positions(self, ik_joint_state: JointState) -> Optional[List[float]]:
        name_to_pos: Dict[str, float] = {}

        for name, pos in zip(ik_joint_state.name, ik_joint_state.position):
            name_to_pos[name] = pos

        positions = []

        for name in self.joint_names:
            if name not in name_to_pos:
                self.get_logger().warn(
                    f'IK solution missing joint {name}. '
                    f'Available: {list(name_to_pos.keys())}'
                )
                return None

            positions.append(name_to_pos[name])

        return positions

    def publish_trajectory(self, positions: List[float]):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(self.traj_time * 1e9)

        msg.points.append(point)
        self.traj_pub.publish(msg)

    def control_loop(self):
        # 先处理上一次 IK 的返回
        self.check_pending_ik()

        # 如果还有未完成 IK，不再发新的，避免请求堆积
        if self.pending_ik_future is not None:
            return

        target_pose = self.build_target_pose()
        if target_pose is None:
            return

        self.send_ik_request(target_pose)


def main(args=None):
    rclpy.init(args=args)

    node = CartesianIKTeleop()

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
