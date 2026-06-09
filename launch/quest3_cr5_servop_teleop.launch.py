#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    start_quest_app = LaunchConfiguration('start_quest_app')
    quest_app_package = LaunchConfiguration('quest_app_package')

    quest_udp_receiver = Node(
        package='cr5_teleop',
        executable='quest_udp_receiver',
        name='quest_udp_receiver',
        output='screen',
        parameters=[{'log_received_pose': False}],
    )

    servop_teleop = Node(
        package='cr5_teleop',
        executable='quest3_cr5_servop_teleop',
        name='quest3_cr5_servop_teleop',
        output='screen',
        parameters=[{
            'get_pose_service': LaunchConfiguration('get_pose_service'),
            'servop_service': LaunchConfiguration('servop_service'),
            'position_scale': ParameterValue(LaunchConfiguration('position_scale'), value_type=float),
            'command_rate': ParameterValue(LaunchConfiguration('command_rate'), value_type=float),
            'min_target_delta_mm': ParameterValue(LaunchConfiguration('min_target_delta_mm'), value_type=float),
            'raw_target_filter_ratio': ParameterValue(LaunchConfiguration('raw_target_filter_ratio'), value_type=float),
            'target_deadband_mm': ParameterValue(LaunchConfiguration('target_deadband_mm'), value_type=float),
            'max_speed_mm_s': ParameterValue(LaunchConfiguration('max_speed_mm_s'), value_type=float),
            'max_accel_mm_s2': ParameterValue(LaunchConfiguration('max_accel_mm_s2'), value_type=float),
            'max_total_translation_mm': ParameterValue(LaunchConfiguration('max_total_translation_mm'), value_type=float),
            'servo_t': ParameterValue(LaunchConfiguration('servo_t'), value_type=float),
            'servo_aheadtime': ParameterValue(LaunchConfiguration('servo_aheadtime'), value_type=float),
            'servo_gain': ParameterValue(LaunchConfiguration('servo_gain'), value_type=float),
            'axis_map_robot_x': LaunchConfiguration('axis_map_robot_x'),
            'axis_map_robot_y': LaunchConfiguration('axis_map_robot_y'),
            'axis_map_robot_z': LaunchConfiguration('axis_map_robot_z'),
            'axis_sign_robot_x': ParameterValue(LaunchConfiguration('axis_sign_robot_x'), value_type=float),
            'axis_sign_robot_y': ParameterValue(LaunchConfiguration('axis_sign_robot_y'), value_type=float),
            'axis_sign_robot_z': ParameterValue(LaunchConfiguration('axis_sign_robot_z'), value_type=float),
            'log_targets': ParameterValue(LaunchConfiguration('log_targets'), value_type=bool),
        }],
    )

    launch_quest_app = TimerAction(
        period=1.0,
        actions=[
            ExecuteProcess(
                cmd=['adb', 'shell', 'monkey', '-p', quest_app_package, '1'],
                output='screen',
                condition=IfCondition(start_quest_app),
            )
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('start_quest_app', default_value='true'),
        DeclareLaunchArgument('quest_app_package', default_value='com.sjtu.questcr5teleop'),
        DeclareLaunchArgument('get_pose_service', default_value='/dobot_bringup_ros2/srv/GetPose'),
        DeclareLaunchArgument('servop_service', default_value='/dobot_bringup_ros2/srv/ServoP'),
        DeclareLaunchArgument('position_scale', default_value='0.20'),
        DeclareLaunchArgument('command_rate', default_value='10.0'),
        DeclareLaunchArgument('min_target_delta_mm', default_value='0.0'),
        DeclareLaunchArgument('raw_target_filter_ratio', default_value='0.80'),
        DeclareLaunchArgument('target_deadband_mm', default_value='2.0'),
        DeclareLaunchArgument('max_speed_mm_s', default_value='50.0'),
        DeclareLaunchArgument('max_accel_mm_s2', default_value='250.0'),
        DeclareLaunchArgument('max_total_translation_mm', default_value='120.0'),
        DeclareLaunchArgument('servo_t', default_value='0.10'),
        DeclareLaunchArgument('servo_aheadtime', default_value='50.0'),
        DeclareLaunchArgument('servo_gain', default_value='200.0'),
        DeclareLaunchArgument('axis_map_robot_x', default_value='vr_x'),
        DeclareLaunchArgument('axis_map_robot_y', default_value='vr_z'),
        DeclareLaunchArgument('axis_map_robot_z', default_value='vr_y'),
        DeclareLaunchArgument('axis_sign_robot_x', default_value='-1.0'),
        DeclareLaunchArgument('axis_sign_robot_y', default_value='-1.0'),
        DeclareLaunchArgument('axis_sign_robot_z', default_value='1.0'),
        DeclareLaunchArgument('log_targets', default_value='false'),
        quest_udp_receiver,
        servop_teleop,
        launch_quest_app,
    ])
