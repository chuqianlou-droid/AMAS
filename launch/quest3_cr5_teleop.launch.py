#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('cr5_teleop')
    params_file = os.path.join(package_share, 'config', 'quest3_cr5_teleop.yaml')
    start_quest_app = LaunchConfiguration('start_quest_app')
    quest_app_package = LaunchConfiguration('quest_app_package')
    dry_run = LaunchConfiguration('dry_run')
    position_scale = LaunchConfiguration('position_scale')
    command_rate = LaunchConfiguration('command_rate')
    min_target_delta = LaunchConfiguration('min_target_delta')
    max_step_translation = LaunchConfiguration('max_step_translation')
    max_total_translation = LaunchConfiguration('max_total_translation')
    trajectory_action = LaunchConfiguration('trajectory_action')
    workspace_min_x = LaunchConfiguration('workspace_min_x')
    workspace_max_x = LaunchConfiguration('workspace_max_x')
    workspace_min_y = LaunchConfiguration('workspace_min_y')
    workspace_max_y = LaunchConfiguration('workspace_max_y')
    workspace_min_z = LaunchConfiguration('workspace_min_z')
    workspace_max_z = LaunchConfiguration('workspace_max_z')
    max_joint1_delta_deg = LaunchConfiguration('max_joint1_delta_deg')
    udp_log_received_pose = LaunchConfiguration('udp_log_received_pose')

    quest_udp_receiver = Node(
        package='cr5_teleop',
        executable='quest_udp_receiver',
        name='quest_udp_receiver',
        output='screen',
        parameters=[
            {
                'log_received_pose': ParameterValue(udp_log_received_pose, value_type=bool),
            },
        ],
    )

    quest3_cr5_teleop = Node(
        package='cr5_teleop',
        executable='quest3_cr5_teleop',
        name='quest3_cr5_teleop',
        output='screen',
        parameters=[
            params_file,
            {
                'dry_run': ParameterValue(dry_run, value_type=bool),
                'position_scale': ParameterValue(position_scale, value_type=float),
                'command_rate': ParameterValue(command_rate, value_type=float),
                'min_target_delta': ParameterValue(min_target_delta, value_type=float),
                'max_step_translation': ParameterValue(max_step_translation, value_type=float),
                'max_total_translation': ParameterValue(max_total_translation, value_type=float),
                'trajectory_action': trajectory_action,
                'workspace_min_x': ParameterValue(workspace_min_x, value_type=float),
                'workspace_max_x': ParameterValue(workspace_max_x, value_type=float),
                'workspace_min_y': ParameterValue(workspace_min_y, value_type=float),
                'workspace_max_y': ParameterValue(workspace_max_y, value_type=float),
                'workspace_min_z': ParameterValue(workspace_min_z, value_type=float),
                'workspace_max_z': ParameterValue(workspace_max_z, value_type=float),
                'max_joint1_delta_deg': ParameterValue(max_joint1_delta_deg, value_type=float),
            },
        ],
    )

    launch_quest_app = TimerAction(
        period=1.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'adb',
                    'shell',
                    'monkey',
                    '-p',
                    quest_app_package,
                    '1',
                ],
                output='screen',
                condition=IfCondition(start_quest_app),
            )
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_quest_app',
            default_value='true',
            description='Start the Quest Unity app through adb shell monkey.',
        ),
        DeclareLaunchArgument(
            'quest_app_package',
            default_value='com.sjtu.questcr5teleop',
            description='Android package name of the Quest Unity app.',
        ),
        DeclareLaunchArgument(
            'dry_run',
            default_value='true',
            description='Plan only when true; execute trajectories when false.',
        ),
        DeclareLaunchArgument(
            'position_scale',
            default_value='0.25',
            description='Scale from Quest translation delta to robot translation delta.',
        ),
        DeclareLaunchArgument(
            'command_rate',
            default_value='3.0',
            description='Maximum MoveIt command rate in Hz.',
        ),
        DeclareLaunchArgument(
            'min_target_delta',
            default_value='0.001',
            description='Minimum target change before sending a new MoveIt command, in meters.',
        ),
        DeclareLaunchArgument(
            'max_step_translation',
            default_value='0.02',
            description='Maximum target translation step per command, in meters.',
        ),
        DeclareLaunchArgument(
            'max_total_translation',
            default_value='0.25',
            description='Maximum translation away from teleop start pose, in meters.',
        ),
        DeclareLaunchArgument(
            'trajectory_action',
            default_value='/cr5_group_controller/follow_joint_trajectory',
            description='FollowJointTrajectory action used to execute planned paths.',
        ),
        DeclareLaunchArgument('workspace_min_x', default_value='-0.70'),
        DeclareLaunchArgument('workspace_max_x', default_value='0.70'),
        DeclareLaunchArgument('workspace_min_y', default_value='-0.70'),
        DeclareLaunchArgument('workspace_max_y', default_value='0.35'),
        DeclareLaunchArgument('workspace_min_z', default_value='0.05'),
        DeclareLaunchArgument('workspace_max_z', default_value='0.80'),
        DeclareLaunchArgument('max_joint1_delta_deg', default_value='30.0'),
        DeclareLaunchArgument(
            'udp_log_received_pose',
            default_value='false',
            description='Print received Quest UDP pose periodically when true.',
        ),
        quest_udp_receiver,
        quest3_cr5_teleop,
        launch_quest_app,
    ])
