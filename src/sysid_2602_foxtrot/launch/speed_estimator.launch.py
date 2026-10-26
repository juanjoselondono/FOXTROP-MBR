#!/usr/bin/env python3
"""
speed_estimator.launch.py
=========================
Launch the online speed estimator node with parameters loaded from YAML.

Usage:
    ros2 launch sysid_2602_foxtrot speed_estimator.launch.py
    ros2 launch sysid_2602_foxtrot speed_estimator.launch.py K:=1.23 tau:=0.28 u_dead:=0.06
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('sysid_2602_foxtrot')
    params_file = os.path.join(pkg_share, 'config', 'speed_estimator_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('K',      default_value='1.0',  description='Steady-state gain [m/s per unit]'),
        DeclareLaunchArgument('tau',    default_value='0.3',  description='Time constant [s]'),
        DeclareLaunchArgument('u_dead', default_value='0.05', description='Dead zone threshold (normalised)'),
        DeclareLaunchArgument('dt',     default_value='0.05', description='Estimator update rate [s]'),

        Node(
            package='sysid_2602_foxtrot',
            executable='speed_estimator_node',
            name='speed_estimator_node',
            output='screen',
            parameters=[
                params_file,
                {
                    'K':      LaunchConfiguration('K'),
                    'tau':    LaunchConfiguration('tau'),
                    'u_dead': LaunchConfiguration('u_dead'),
                    'dt':     LaunchConfiguration('dt'),
                }
            ],
            remappings=[
                # If your ESC node publishes on a different topic, change here:
                # ('/esc/command', '/my_esc/command'),
            ],
        ),
    ])
