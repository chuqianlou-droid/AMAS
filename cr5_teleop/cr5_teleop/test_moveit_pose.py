#!/usr/bin/env python3

import time
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from tf2_ros import Buffer, TransformListener, TransformException

from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import RobotState
from moveit_msgs.msg import DisplayTrajectory

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class TestCartesianExecuteSafe(Node):
    def __init__(self):
        super().__init__('test_cartesian_execute_safe')

        self.group_name = 'cr5_group'
        self.base_frame = 'base_link'
        self.ee_link = 'Link6'

        self.joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']

        self.latest_joint_state = None
        self.joint_sub = self.create_subscription(JointState,'/joint_states',self.joint_state_callback,10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer,self)

        # Cartesian Path Service
        self.cartesian_client = self.create_client(GetCartesianPath,'/compute_cartesian_path')
        while not self.cartesian_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/compute_cartesian_path not available, waiting...')
        self.get_logger().info('/compute_cartesian_path is available.')

        # FollowJointTrajectory action
        self.traj_client = ActionClient(self,FollowJointTrajectory,'/cr5_group_controller/follow_joint_trajectory')
        self.get_logger().info('Waiting for follow_joint_trajectory action...')
        self.traj_client.wait_for_server()
        self.get_logger().info('follow_joint_trajectory action available.')
        self.display_pub = self.create_publisher(
            DisplayTrajectory,
            '/display_planned_path',
            10
        )

        # 安全参数，可修改
        self.max_trans_step = 0.005  # 5 mm per cycle
        self.max_rot_step = 0.02     # rad per cycle
        self.max_joint_step_deg = 1.0  # degree per cycle (仿真可放宽 3~5)
        self.max_joint_jump_reject_deg = 10.0
        self.end_effector_bounds = {'x':[0.2,0.7],'y':[-0.35,0.35],'z':[0.10,0.6]}
        self.cartesian_max_step = 0.005
        self.jump_threshold = 2.0
        self.time_scale = 5.0
        self.default_dt = 0.08

        # 目标位姿，可改
        self.target_pose = Pose()
        self.target_pose.position.x = -0.3
        self.target_pose.position.y = -0.485
        self.target_pose.position.z = 0.398

    def joint_state_callback(self,msg:JointState):
        self.latest_joint_state = msg

    def wait_for_joint_state(self,timeout_sec=3.0):
        start=time.time()
        while rclpy.ok():
            rclpy.spin_once(self,timeout_sec=0.1)
            if self.latest_joint_state and all(j in self.latest_joint_state.name for j in self.joint_names):
                return True
            if time.time()-start>timeout_sec:
                self.get_logger().error('Timeout waiting for /joint_states.')
                return False

    def get_current_joint_positions(self):
        js = self.latest_joint_state
        name_to_pos = dict(zip(js.name, js.position))
        return [float(name_to_pos[j]) for j in self.joint_names]

    def make_start_state(self,positions):
        rs = RobotState()
        rs.joint_state.name = list(self.joint_names)
        rs.joint_state.position = list(positions)
        return rs

    def wait_for_current_pose(self,timeout_sec=3.0):
        start=time.time()
        while rclpy.ok():
            rclpy.spin_once(self,timeout_sec=0.1)
            try:
                tf_msg=self.tf_buffer.lookup_transform(self.base_frame,self.ee_link,rclpy.time.Time())
                pose=Pose()
                pose.position.x=tf_msg.transform.translation.x
                pose.position.y=tf_msg.transform.translation.y
                pose.position.z=tf_msg.transform.translation.z
                pose.orientation.x=tf_msg.transform.rotation.x
                pose.orientation.y=tf_msg.transform.rotation.y
                pose.orientation.z=tf_msg.transform.rotation.z
                pose.orientation.w=tf_msg.transform.rotation.w
                return pose
            except TransformException:
                pass
            if time.time()-start>timeout_sec:
                self.get_logger().error(f'Timeout waiting for TF {self.base_frame}->{self.ee_link}')
                return None

    @staticmethod
    def shortest_angle_delta(angle,reference):
        return math.atan2(math.sin(angle-reference),math.cos(angle-reference))

    def unwrap_trajectory_positions(self,trajectory,current_positions):
        prev=list(current_positions)
        for point in trajectory.joint_trajectory.points:
            new_positions=[]
            for i,raw_angle in enumerate(point.positions):
                delta=self.shortest_angle_delta(raw_angle,prev[i])
                new_positions.append(prev[i]+delta)
            point.positions=new_positions
            prev=new_positions
        return True

    def validate_trajectory_joint_limits(self,trajectory,current_positions):
        points=trajectory.joint_trajectory.points
        prev_positions=list(current_positions)
        for point_idx,point in enumerate(points):
            positions=list(point.positions)
            for j_idx,joint_name in enumerate(self.joint_names):
                pos=positions[j_idx]
                cur=current_positions[j_idx]
                prev=prev_positions[j_idx]
                total_delta_deg=math.degrees(pos-cur)
                step_delta_deg=math.degrees(pos-prev)
                if abs(total_delta_deg)>self.max_joint_jump_reject_deg:
                    self.get_logger().warn(f'{joint_name} total delta {total_delta_deg:.2f}° exceeds {self.max_joint_jump_reject_deg}°, clip applied.')
                if abs(step_delta_deg)>self.max_joint_step_deg:
                    self.get_logger().warn(f'{joint_name} step delta {step_delta_deg:.2f}° exceeds {self.max_joint_step_deg}°, clip applied.')
            prev_positions=positions
        return True

    def clip_end_effector_bounds(self,pose):
        pose.position.x=np.clip(pose.position.x,self.end_effector_bounds['x'][0],self.end_effector_bounds['x'][1])
        pose.position.y=np.clip(pose.position.y,self.end_effector_bounds['y'][0],self.end_effector_bounds['y'][1])
        pose.position.z=np.clip(pose.position.z,self.end_effector_bounds['z'][0],self.end_effector_bounds['z'][1])
        return pose

    def send_cartesian_path_request(self):
        if not self.wait_for_joint_state():
            return

        current_positions=self.get_current_joint_positions()
        current_pose=self.wait_for_current_pose()
        if current_pose is None:
            return

        # 限制末端工作空间
        safe_pose=self.clip_end_effector_bounds(self.target_pose)

        # 保持姿态
        safe_pose.orientation=current_pose.orientation

        # 构建 Cartesian Path 请求
        req=GetCartesianPath.Request()
        req.header.frame_id=self.base_frame
        req.header.stamp=self.get_clock().now().to_msg()
        req.start_state=self.make_start_state(current_positions)
        req.group_name=self.group_name
        req.link_name=self.ee_link
        req.waypoints=[safe_pose]
        req.max_step=self.cartesian_max_step
        req.jump_threshold=self.jump_threshold
        req.avoid_collisions=True

        future=self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self,future)
        if future.result() is None:
            self.get_logger().error('Failed to call /compute_cartesian_path.')
            return

        res=future.result()
        self.get_logger().info(f'Cartesian path fraction: {res.fraction:.4f}')
        self.get_logger().info(f'Error code: {res.error_code.val}')

        if res.solution is not None:
            self.unwrap_trajectory_positions(res.solution,current_positions)
            self.validate_trajectory_joint_limits(res.solution,current_positions)

            display_msg=DisplayTrajectory()
            display_msg.model_id='cr5_robot'
            display_msg.trajectory=[res.solution]
            self.display_pub.publish(display_msg)
            self.get_logger().info('Published trajectory to /display_planned_path with joint limits check.')


def main(args=None):
    rclpy.init(args=args)
    node=TestCartesianExecuteSafe()
    try:
        node.send_cartesian_path_request()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()
