import sys
import termios
import tty
import select

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class KeyboardJointTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_joint_teleop')

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

        # 初始关节角，单位 rad
        self.q = [0.0, 0.2, -0.2, 0.0, 0.2, 0.0]

        # 每次按键改变的角度，单位 rad
        self.step = 0.05

        self.get_logger().info('Keyboard teleop started.')
        self.get_logger().info('Controls:')
        self.get_logger().info('  a/d: joint1 -/+')
        self.get_logger().info('  w/s: joint2 +/-')
        self.get_logger().info('  q/e: joint3 +/-')
        self.get_logger().info('  r: reset')
        self.get_logger().info('  x: exit')

    def publish_trajectory(self):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = self.q.copy()
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0

        msg.points.append(point)
        self.pub.publish(msg)

        self.get_logger().info(
            f'Published q = {[round(v, 3) for v in self.q]}'
        )

    def handle_key(self, key):
        if key == 'a':
            self.q[0] -= self.step
        elif key == 'd':
            self.q[0] += self.step
        elif key == 'w':
            self.q[1] += self.step
        elif key == 's':
            self.q[1] -= self.step
        elif key == 'q':
            self.q[2] += self.step
        elif key == 'e':
            self.q[2] -= self.step
        elif key == 'r':
            self.q = [0.0, 0.2, -0.2, 0.0, 0.2, 0.0]
        elif key == 'x':
            return False
        else:
            return True

        self.publish_trajectory()
        return True


def get_key(timeout=0.1):
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardJointTeleop()

    node.publish_trajectory()

    running = True
    while rclpy.ok() and running:
        key = get_key()
        if key:
            running = node.handle_key(key)
        rclpy.spin_once(node, timeout_sec=0.01)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()