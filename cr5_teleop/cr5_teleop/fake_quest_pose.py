import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class FakeQuestPose(Node):
    def __init__(self):
        super().__init__('fake_quest_pose')

        self.pub = self.create_publisher(
            PoseStamped,
            '/quest/right_controller_pose',
            10
        )

        self.t = 0.0
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info('Fake Quest right controller pose publisher started.')

    def timer_callback(self):
        self.t += 0.05

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'quest_world'

        # 模拟手柄左右、前后、上下移动
        msg.pose.position.x = 0.10 * math.sin(0.5 * self.t)
        msg.pose.position.y = 0.05 * math.sin(0.8 * self.t)
        msg.pose.position.z = 1.0 + 0.05 * math.sin(0.6 * self.t)

        # 暂时不控制姿态，给单位四元数
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeQuestPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()