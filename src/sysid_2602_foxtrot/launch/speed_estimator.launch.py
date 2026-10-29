#!/usr/bin/env python3
"""
speed_estimator.launch.py
=========================
Launch the online speed/yaw-rate estimator with the identified parameters
from config/speed_estimator_params.yaml.

Usage:
    ros2 launch sysid_2602_foxtrot speed_estimator.launch.py
    ros2 launch sysid_2602_foxtrot speed_estimator.launch.py max_speed_cap:=2.5 yaw_rate_source:=gyro
    ros2 launch sysid_2602_foxtrot speed_estimator.launch.py use_imu:=false use_mag:=false   # command model only
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

OVERRIDABLE = ('K_fwd', 'tau_fwd', 'u_dead_fwd', 'K_rev', 'decel_neutral', 'cornering_drag',
               'impact_decel_threshold', 'max_speed_cap', 'yaw_rate_source', 'use_mag', 'tau_mag',
               'steer_curve', 'K_steer_pos', 'K_steer_neg', 'use_imu', 'w_imu_max', 'dt')


def generate_launch_description():
    pkg_share = get_package_share_directory('sysid_2602_foxtrot')
    params_file = os.path.join(pkg_share, 'config', 'speed_estimator_params.yaml')
    with open(params_file) as f:
        defaults = yaml.safe_load(f)['speed_estimator_node']['ros__parameters']

    # Argument defaults come FROM the YAML, so launching without arguments
    # never overrides the identified values with stale hardcoded numbers.
    args = [DeclareLaunchArgument(name, default_value=str(defaults[name])) for name in OVERRIDABLE]

    return LaunchDescription(args + [
        Node(
            package='sysid_2602_foxtrot',
            executable='speed_estimator_node',
            name='speed_estimator_node',
            output='screen',
            parameters=[params_file, {name: LaunchConfiguration(name) for name in OVERRIDABLE}],
        ),
    ])
