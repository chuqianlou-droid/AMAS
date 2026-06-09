import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class PoseToJointTeleop(Node):
    def __init__(self):
        super().__init__('pose_to_joint_teleop')

        self.sub = self.create_subscription(
            PoseStamped,
            '/quest/right_controller_pose',
            self.pose_callback,
            10
        )

        self.pub = self.create_publisher(
            JointTrajectory,
            '/cr5_group_controller/joint_trajectory',
            10
        )

        self.joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6'
        ]

        self.initial_pose = None

        # CR5 初始关节角
        self.base_q = [0.0, 0.2, -0.2, 0.0, 0.2, 0.0]

        # 当前输出关节角
        self.q = self.base_q.copy()

        # 手柄位移到关节角的比例
        self.scale_x_to_j1 = 3.0
        self.scale_y_to_j2 = 3.0
        self.scale_z_to_j3 = 3.0

        # 关节限幅，避免运动过大
        self.joint_limits = [
            (-1.0, 1.0),
            (-0.8, 0.8),
            (-0.8, 0.8),
            (-1.0, 1.0),
            (-0.8, 0.8),
            (-1.0, 1.0),
        ]

        self.get_logger().info('Pose to joint teleop started.')
        self.get_logger().info('Subscribing: /quest/right_controller_pose')
        self.get_logger().info('Publishing: /cr5_group_controller/joint_trajectory')

    def clamp(self, value, lower, upper):
        return max(lower, min(upper, value))

    def pose_callback(self, msg: PoseStamped):
        p = msg.pose.position

        if self.initial_pose is None:
            self.initial_pose = [p.x, p.y, p.z]
            self.get_logger().info(
                f'Initial controller pose set: '
                f'x={p.x:.3f}, y={p.y:.3f}, z={p.z:.3f}'
            )
            return

        dx = p.x - self.initial_pose[0]
        dy = p.y - self.initial_pose[1]
        dz = p.z - self.initial_pose[2]

        # 简单映射：
        # 手柄 x 位移 → joint1
        # 手柄 y 位移 → joint2
        # 手柄 z 位移 → joint3
        self.q[0] = self.base_q[0] + self.scale_x_to_j1 * dx
        self.q[1] = self.base_q[1] + self.scale_y_to_j2 * dy
        self.q[2] = self.base_q[2] + self.scale_z_to_j3 * dz

        # 限幅
        for i in range(6):
            lower, upper = self.joint_limits[i]
            self.q[i] = self.clamp(self.q[i], lower, upper)

        self.publish_trajectory()

    def publish_trajectory(self):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = self.q.copy()
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0

        msg.points.append(point)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PoseToJointTeleop()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()