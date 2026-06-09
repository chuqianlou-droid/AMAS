import os
import glob
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import yaml
import xacro

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    with open(absolute_file_path, 'r') as file:
        return yaml.safe_load(file)

def generate_launch_description():
    dobot_type = os.environ.get('DOBOT_TYPE', 'cr5')

    cr5_moveit_path = get_package_share_directory('cr5_moveit')
    cra_description_path = get_package_share_directory('cra_description')

    xacro_file = os.path.join(
        cra_description_path,
        'urdf',
        f'{dobot_type}_robot.xacro'
    )
    robot_description_config = xacro.process_file(xacro_file)
    robot_description = {'robot_description': robot_description_config.toxml()}

    # SRDF
    srdf_candidates = glob.glob(os.path.join(cr5_moveit_path, 'config', '*.srdf'))
    if not srdf_candidates:
        raise RuntimeError('No SRDF file found in cr5_moveit/config')
    with open(srdf_candidates[0], 'r') as f:
        robot_description_semantic = {'robot_description_semantic': f.read()}

    # Kinematics / joint limits
    robot_description_kinematics = {'robot_description_kinematics': load_yaml('cr5_moveit', 'config/kinematics.yaml')}
    joint_limits_yaml = load_yaml('cr5_moveit', 'config/joint_limits.yaml')
    joint_limits = {'robot_description_planning': joint_limits_yaml} if joint_limits_yaml else {}

    servo_params = load_yaml('cr5_moveit', 'config/cr5_servo.yaml')

    servo_node = Node(
        package='moveit_servo',
        executable='servo_node_main',
        name='servo_node',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            joint_limits,
            servo_params,
        ],
    )

    return LaunchDescription([servo_node])
