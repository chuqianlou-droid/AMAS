#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped


class QuestPoseMonitor(Node):
    def __init__(self):
        super().__init__('quest_pose_monitor')

        self.declare_parameter('pose_topic', '/quest/right_controller_pose')
        self.declare_parameter('print_rate', 10.0)
        self.declare_parameter('one_line', True)

        self.pose_topic = self.get_parameter('pose_topic').value
        self.print_rate = max(0.1, float(self.get_parameter('print_rate').value))
        self.one_line = bool(self.get_parameter('one_line').value)

        self.latest_pose = None
        self.pose_count = 0
        self.waiting_printed = False

        self.sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            10,
        )
        self.timer = self.create_timer(1.0 / self.print_rate, self.print_pose)

        self.get_logger().info(
            f'Quest pose monitor started. topic={self.pose_topic}, '
            f'print_rate={self.print_rate:.1f} Hz'
        )

    def pose_callback(self, msg: PoseStamped):
        self.latest_pose = msg
        self.pose_count += 1
        self.waiting_printed = False

    def print_pose(self):
        if self.latest_pose is None:
            if not self.waiting_printed:
                print(
                    f'waiting for PoseStamped on {self.pose_topic} ...',
                    flush=True,
                )
                self.waiting_printed = True
            return

        p = self.latest_pose.pose.position
        q = self.latest_pose.pose.orientation
        text = (
            f'count={self.pose_count:06d} '
            f'x={p.x:+.3f} y={p.y:+.3f} z={p.z:+.3f} '
            f'qx={q.x:+.3f} qy={q.y:+.3f} qz={q.z:+.3f} qw={q.w:+.3f}'
        )

        if self.one_line:
            print('\r' + text, end='', flush=True)
        else:
            print(text, flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = QuestPoseMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.one_line:
            print()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
