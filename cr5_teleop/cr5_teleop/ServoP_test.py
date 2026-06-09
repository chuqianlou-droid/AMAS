#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from dobot_msgs_v4.srv import ServoP
import time

class ServoPTester(Node):
    def __init__(self):
        super().__init__('servo_p_tester')
        self.cli = self.create_client(ServoP, '/dobot_bringup_ros2/srv/ServoP')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for ServoP service...')
        self.get_logger().info('ServoP service available.')

        # 当前机械臂位姿（ToolVectorActual）
        self.current_pose = {
            'x': -117.6109,
            'y': -482.6342,
            'z': 558.1356,
            'rx': 46.6632,
            'ry': -70.8836,
            'rz': 46.0216
        }

        # 平动增量
        self.delta_x = 100 # mm, 5 cm
        self.delta_y = 0
        self.delta_z = 0
        # ServoP 可选参数
        self.speed_mm_s = 50.0  # 期望末端速度，mm/s
        self.min_t = 0.4      # ServoP t 参数下限
        self.max_t = 0.8        # 单次测试目标时间上限
        self.aheadtime = 50    # ms
        self.gain = 200        # P增益

        # 循环调用次数
        self.cycles = 1
        self.interval = 0.05   # 50ms 循环间隔

    def send_servoP(self, x, y, z, rx, ry, rz, t, aheadtime, gain):
        req = ServoP.Request()
        req.a = x
        req.b = y
        req.c = z
        req.d = rx
        req.e = ry
        req.f = rz
        req.param_value = [f't={t}', f'aheadtime={aheadtime}', f'gain={gain}']

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            if future.result().res == 0:
                self.get_logger().info(f'ServoP command accepted: X={x:.3f}')
            else:
                self.get_logger().warn(f'ServoP command failed: {future.result().res}')
        else:
            self.get_logger().error('Service call failed.')

    def calc_servo_time(self, target_x, target_y, target_z):
        dx = target_x - self.current_pose['x']
        dy = target_y - self.current_pose['y']
        dz = target_z - self.current_pose['z']
        distance = (dx * dx + dy * dy + dz * dz) ** 0.5
        if distance <= 1e-9:
            return self.min_t

        t = distance / max(self.speed_mm_s, 1e-6)
        return max(self.min_t, min(self.max_t, t))

    def run_test(self):
        # 构建目标末端位置：X方向 + delta
        target_x = self.current_pose['x'] + self.delta_x
        target_y = self.current_pose['y'] + self.delta_y
        target_z = self.current_pose['z'] + self.delta_z
        rx = self.current_pose['rx']
        ry = self.current_pose['ry']
        rz = self.current_pose['rz']
        t = self.calc_servo_time(target_x, target_y, target_z)

        self.get_logger().info(
            f'Target delta=({self.delta_x:.1f},{self.delta_y:.1f},{self.delta_z:.1f}) mm, '
            f'speed={self.speed_mm_s:.1f} mm/s, t={t:.3f}s'
        )

        for i in range(self.cycles):
            self.get_logger().info(f'Sending ServoP command cycle {i+1}')
            self.send_servoP(target_x, target_y, target_z, rx, ry, rz,
                             t, self.aheadtime, self.gain)
            time.sleep(self.interval)


def main(args=None):
    rclpy.init(args=args)
    node = ServoPTester()
    try:
        node.run_test()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
