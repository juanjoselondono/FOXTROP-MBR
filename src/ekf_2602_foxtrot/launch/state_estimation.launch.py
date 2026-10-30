#!/usr/bin/env python3
"""
state_estimation.launch.py
==========================
Laser odometry (rf2o) + IMU fused by robot_localization's EKF:

  /scan --> rf2o_laser_odometry --> /odom_rf2o --+
                                                 +--> covariance_relay --> /odom_laser --+
  /imu/data_raw ---------------------------------+                     --> /imu/data_cov +--> ekf_node --> /odometry/filtered
                                                                                                         + TF odom -> base_link
  static TF: base_link -> <laser_frame>, base_link -> imu_link

The EKF must be the ONLY odom -> base_link publisher: do not run
speed_estimator_node or ekf_node_2602_foxtrot at the same time.
If robot_state_publisher (rsp.launch.py) is running, it already publishes the
sensor frames from the URDF: launch with publish_static_tf:=false.

Usage:
    ros2 launch ekf_2602_foxtrot state_estimation.launch.py
    ros2 launch ekf_2602_foxtrot state_estimation.launch.py laser_frame:=laser laser_x:=0.12 laser_yaw:=3.1416
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def static_tf(name, child, prefix):
    lc = LaunchConfiguration
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=name,
        arguments=['--x', lc(f'{prefix}_x'), '--y', lc(f'{prefix}_y'), '--z', lc(f'{prefix}_z'),
                   '--yaw', lc(f'{prefix}_yaw'), '--pitch', '0', '--roll', '0',
                   '--frame-id', 'base_link', '--child-frame-id', child],
        condition=IfCondition(lc('publish_static_tf')),
    )


def generate_launch_description():
    share = get_package_share_directory('ekf_2602_foxtrot')
    rf2o_params = os.path.join(share, 'config', 'rf2o_laser_odometry.yaml')
    ekf_params = os.path.join(share, 'config', 'ekf_laser_imu.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')

    args = [
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('laser_frame', default_value='laser_frame',
                              description='MUST equal /scan header.frame_id '
                                          '(ros2 topic echo /scan --field header.frame_id --once)'),
        DeclareLaunchArgument('publish_static_tf', default_value='true',
                              description='false if robot_state_publisher already publishes the sensor frames'),
        DeclareLaunchArgument('laser_odom_topic', default_value='/odom_laser',
                              description='where the laser odometry (with covariances) is published'),
        DeclareLaunchArgument('start_ekf', default_value='true',
                              description='robot_localization EKF; false when another EKF owns odom -> base_link'),
    ]
    for prefix in ('laser', 'imu'):
        for axis in ('x', 'y', 'z', 'yaw'):
            args.append(DeclareLaunchArgument(f'{prefix}_{axis}', default_value='0.0',
                                              description=f'base_link -> {prefix} {axis} [m or rad]'))

    rf2o = Node(
        package='rf2o_laser_odometry',
        executable='rf2o_laser_odometry_node',
        name='rf2o_laser_odometry',
        output='screen',
        parameters=[rf2o_params, {'laser_scan_topic': LaunchConfiguration('scan_topic'),
                                  'use_sim_time': use_sim_time}],
    )

    relay = Node(
        package='ekf_2602_foxtrot',
        executable='covariance_relay',
        name='covariance_relay',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'odom_out': LaunchConfiguration('laser_odom_topic')}],
    )

    ekf = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_params, {'use_sim_time': use_sim_time, 'odom0': LaunchConfiguration('laser_odom_topic')}],
        condition=IfCondition(LaunchConfiguration('start_ekf')),
    )

    return LaunchDescription(args + [
        static_tf('static_tf_base_to_laser', LaunchConfiguration('laser_frame'), 'laser'),
        static_tf('static_tf_base_to_imu', 'imu_link', 'imu'),
        rf2o,
        relay,
        ekf,
    ])
