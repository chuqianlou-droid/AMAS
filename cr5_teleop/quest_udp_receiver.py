import json
import socket

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class QuestUdpReceiver(Node):
    def __init__(self):
        super().__init__('quest_udp_receiver')

        self.declare_parameter('log_received_pose', False)
        self.declare_parameter('log_period_sec', 1.0)

        self.log_received_pose = bool(
            self.get_parameter('log_received_pose').value
        )
        self.log_period_sec = float(
            self.get_parameter('log_period_sec').value
        )
        self.received_count = 0
        self.last_log_time = None

        # 发布给后面的 pose_to_joint_teleop.py 使用
        self.pub = self.create_publisher(
            PoseStamped,
            '/quest/right_controller_pose',
            10
        )

        # 监听所有网卡上的 UDP 5005 端口
        self.udp_ip = '0.0.0.0'
        self.udp_port = 5005

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))
        self.sock.setblocking(False)

        # 100 Hz 检查 UDP 数据
        self.timer = self.create_timer(0.01, self.timer_callback)

        self.get_logger().info(
            f'Quest UDP receiver started on {self.udp_ip}:{self.udp_port}'
        )
        self.get_logger().info(
            'Publishing PoseStamped to /quest/right_controller_pose'
        )
        self.get_logger().info(
            f'log_received_pose={self.log_received_pose}'
        )

    def timer_callback(self):
        try:
            data, addr = self.sock.recvfrom(2048)
        except BlockingIOError:
            return

        try:
            # 期望收到 JSON 字符串，例如：
            # {"x":0.1,"y":0.0,"z":1.0,"qx":0.0,"qy":0.0,"qz":0.0,"qw":1.0}
            msg_json = json.loads(data.decode('utf-8'))

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'quest_world'

            pose_msg.pose.position.x = float(msg_json.get('x', 0.0))
            pose_msg.pose.position.y = float(msg_json.get('y', 0.0))
            pose_msg.pose.position.z = float(msg_json.get('z', 1.0))

            pose_msg.pose.orientation.x = float(msg_json.get('qx', 0.0))
            pose_msg.pose.orientation.y = float(msg_json.get('qy', 0.0))
            pose_msg.pose.orientation.z = float(msg_json.get('qz', 0.0))
            pose_msg.pose.orientation.w = float(msg_json.get('qw', 1.0))

            self.pub.publish(pose_msg)
            self.received_count += 1

            if self.should_log_pose():
                self.get_logger().info(
                    f"Received UDP from {addr}: "
                    f"x={pose_msg.pose.position.x:.3f}, "
                    f"y={pose_msg.pose.position.y:.3f}, "
                    f"z={pose_msg.pose.position.z:.3f}"
                )

        except Exception as e:
            self.get_logger().warn(f'Failed to parse UDP packet: {e}')

    def should_log_pose(self):
        if self.received_count == 1:
            self.last_log_time = self.get_clock().now()
            return True

        if not self.log_received_pose:
            return False

        now = self.get_clock().now()
        if self.last_log_time is None:
            self.last_log_time = now
            return True

        elapsed = (now - self.last_log_time).nanoseconds / 1e9
        if elapsed >= self.log_period_sec:
            self.last_log_time = now
            return True

        return False


def main(args=None):
    rclpy.init(args=args)
    node = QuestUdpReceiver()
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
