import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'cr5_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jiaotan',
    maintainer_email='jiaotan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'keyboard_joint_teleop = cr5_teleop.keyboard_joint_teleop:main',
        'fake_quest_pose = cr5_teleop.fake_quest_pose:main',
        'pose_to_joint_teleop = cr5_teleop.pose_to_joint_teleop:main',
        'quest_udp_receiver = cr5_teleop.quest_udp_receiver:main',
        'quest_pose_monitor = cr5_teleop.quest_pose_monitor:main',
        'cartesian_ik_teleop = cr5_teleop.cartesian_ik_teleop:main',
        'quest3_cr5_teleop = cr5_teleop.quest3_cr5_teleop:main',
        'quest3_cr5_servop_teleop = cr5_teleop.quest3_cr5_servop_teleop:main',
        'test_moveit_pose = cr5_teleop.test_moveit_pose:main',
        'test_cartesian_execute = cr5_teleop.test_cartesian_execute:main',
        'ServoP_test = cr5_teleop.ServoP_test: main',
    ],
},
)
