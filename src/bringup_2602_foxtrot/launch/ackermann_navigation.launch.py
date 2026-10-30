#!/usr/bin/env python3
"""
ackermann_navigation.launch.py
==============================
Full autonomy stack for the PHYSICAL Ackermann car. Run on the Pi, after the
hardware drivers (LiDAR -> /scan, twist_cmd_node -> /imu/data_raw and
/cmd_vel_stamped -> motors).

TF tree (REP-105):  map --(SLAM or AMCL)--> odom --(EKF)--> base_link --(static)--> laser_frame, imu_link

  /scan --> rf2o --> covariance_relay --> /estimated_odo --+
  /imu/data_raw -------------------------------------------+--> ekf_node_2602_foxtrot --> /ekf/odometry, TF odom->base_link
  /scan --> slam_toolbox (mode:=slam)  |  map_server + amcl (mode:=nav) --> TF map->odom, /map
  /scan --> line_extraction_node --> /line_segments (MarkerArray)
  Nav2 (SmacPlannerHybrid + RegulatedPurePursuit, no in-place rotation) --> /cmd_vel (m/s, rad/s)
        --> ackermann_cmd_converter --> /cmd_vel_stamped (hardware units, DIRECT CONTROL)

Usage:
  # 1) build a map
  ros2 launch bringup_2602_foxtrot ackermann_navigation.launch.py mode:=slam laser_frame:=<scan frame_id>
  ros2 run nav2_map_server map_saver_cli -f ~/maps/lab        # when the map looks complete
  # 2) navigate on it
  ros2 launch bringup_2602_foxtrot ackermann_navigation.launch.py mode:=nav map:=$HOME/maps/lab.yaml laser_frame:=<...>

Only ONE odom -> base_link publisher may run: this file starts the custom EKF,
so do NOT also run state_estimation.launch.py (robot_localization) or
speed_estimator_node.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    bringup = get_package_share_directory('bringup_2602_foxtrot')
    ekf_pkg = get_package_share_directory('ekf_2602_foxtrot')
    nav2 = get_package_share_directory('nav2_bringup')
    slam = get_package_share_directory('slam_toolbox')
    lc = LaunchConfiguration
    use_sim_time = lc('use_sim_time')

    args = [
        DeclareLaunchArgument('mode', default_value='slam', choices=['slam', 'nav'],
                              description='slam: build a map (slam_toolbox) | nav: localize on a saved map (AMCL)'),
        DeclareLaunchArgument('map', default_value='', description='map .yaml (required for mode:=nav)'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('laser_frame', default_value='laser_frame',
                              description='MUST equal /scan header.frame_id'),
        DeclareLaunchArgument('publish_static_tf', default_value='true'),
        DeclareLaunchArgument('start_laser_odom', default_value='true',
                              description='rf2o + covariance_relay -> /estimated_odo (false if already running)'),
        DeclareLaunchArgument('start_nav2', default_value='true'),
        DeclareLaunchArgument('start_line_extraction', default_value='true'),
        DeclareLaunchArgument('cmd_out_topic', default_value='/cmd_vel_stamped',
                              description='hardware-unit commands; bypassing AEB to directly feed twist_cmd_node'),
        DeclareLaunchArgument('nav2_params', default_value=os.path.join(bringup, 'config', 'nav2_ackermann_params.yaml')),
        DeclareLaunchArgument('slam_params', default_value=os.path.join(bringup, 'config', 'mapper_params_online_async.yaml')),
        DeclareLaunchArgument('ekf_params', default_value=os.path.join(ekf_pkg, 'config', 'ekf_hw.yaml')),
    ]
    for prefix in ('laser', 'imu'):
        for axis in ('x', 'y', 'z', 'yaw'):
            args.append(DeclareLaunchArgument(f'{prefix}_{axis}', default_value='0.0'))

    # ── 0. Static sensor frames (independent of whether laser odometry is started here) ──
    def static_tf(name, child, prefix):
        return Node(
            package='tf2_ros', executable='static_transform_publisher', name=name,
            arguments=['--x', lc(f'{prefix}_x'), '--y', lc(f'{prefix}_y'), '--z', lc(f'{prefix}_z'),
                       '--yaw', lc(f'{prefix}_yaw'), '--pitch', '0', '--roll', '0',
                       '--frame-id', 'base_link', '--child-frame-id', child],
            condition=IfCondition(lc('publish_static_tf')),
        )
    static_tfs = [static_tf('static_tf_base_to_laser', lc('laser_frame'), 'laser'),
                  static_tf('static_tf_base_to_imu', 'imu_link', 'imu')]

    # ── 1. Laser odometry -> /estimated_odo, WITHOUT robot_localization (the custom EKF replaces it) ──
    laser_odom = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ekf_pkg, 'launch', 'state_estimation.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time, 'scan_topic': lc('scan_topic'), 'laser_frame': lc('laser_frame'),
            'publish_static_tf': 'false', 'laser_odom_topic': '/estimated_odo', 'start_ekf': 'false',
        }.items(),
        condition=IfCondition(lc('start_laser_odom')),
    )

    # ── 2. Custom EKF: /estimated_odo + /imu/data_raw -> /ekf/odometry, TF odom -> base_link ──
    ekf = Node(
        package='ekf_2602_foxtrot',
        executable='ekf_node_2602_foxtrot',
        name='ekf_node_2602_foxtrot',
        output='screen',
        parameters=[lc('ekf_params'), {'use_sim_time': use_sim_time}],
    )

    # ── 3. Perception: line segments from the raw scan ──
    lines = Node(
        package='perception_2602_foxtrot',
        executable='line_extraction_node',
        name='line_extraction_node',
        output='screen',
        parameters=[{'scan_topic': lc('scan_topic'), 'use_sim_time': use_sim_time}],
        condition=IfCondition(lc('start_line_extraction')),
    )

    # ── 4a. mode:=slam -> slam_toolbox provides /map and TF map -> odom ──
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(slam, 'launch', 'online_async_launch.py')),
        launch_arguments={'slam_params_file': lc('slam_params'), 'use_sim_time': use_sim_time,
                          'autostart': lc('autostart')}.items(),
        condition=LaunchConfigurationEquals('mode', 'slam'),
    )

    # Nav2 params with the Ackermann behavior trees resolved to their installed paths
    nav2_params = RewrittenYaml(
        source_file=lc('nav2_params'),
        param_rewrites={
            'default_nav_to_pose_bt_xml': os.path.join(bringup, 'behavior_trees', 'navigate_to_pose_ackermann.xml'),
            'default_nav_through_poses_bt_xml': os.path.join(bringup, 'behavior_trees',
                                                             'navigate_through_poses_ackermann.xml'),
        },
        convert_types=True,
    )

    # ── 4b. mode:=nav -> map_server + AMCL provide /map and TF map -> odom ──
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2, 'launch', 'localization_launch.py')),
        launch_arguments={'map': lc('map'), 'params_file': nav2_params, 'use_sim_time': use_sim_time,
                          'autostart': lc('autostart'), 'use_composition': 'False'}.items(),
        condition=LaunchConfigurationEquals('mode', 'nav'),
    )

    # ── 5. Nav2 planning + tracking, then Nav2 /cmd_vel -> hardware units ──
    # Delayed so EKF/SLAM/AMCL have TF up before the costmaps start asking for it.
    navigation = TimerAction(period=5.0, actions=[GroupAction(
        condition=IfCondition(lc('start_nav2')),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(nav2, 'launch', 'navigation_launch.py')),
                launch_arguments={'params_file': nav2_params, 'use_sim_time': use_sim_time,
                                  'autostart': lc('autostart'), 'use_composition': 'False'}.items(),
            ),
            Node(
                package='control_2602_foxtrot',
                executable='ackermann_cmd_converter.py',
                name='ackermann_cmd_converter',
                output='screen',
                parameters=[{'in_topic': '/cmd_vel', 'out_topic': lc('cmd_out_topic'),
                             'use_sim_time': use_sim_time}],
            ),
        ],
    )])

    return LaunchDescription(args + static_tfs + [laser_odom, ekf, lines, slam_toolbox, localization, navigation])