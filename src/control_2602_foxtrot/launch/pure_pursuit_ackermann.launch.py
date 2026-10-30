#!/usr/bin/env python3
"""
pure_pursuit_ackermann.launch.py
================================
A* + Pure Pursuit on a saved map, for the REAL Ackermann car. Runs on the LAPTOP.

Prerequisites (already running):
  [RASPBERRY PI] ros2 launch ybeb_2602_zulu odom_op.launch.py
  [RASPBERRY PI] ros2 run ybeb_2602_zulu aeb_ttc
  [LAPTOP]       ros2 launch ekf_2602_foxtrot state_estimation.launch.py laser_frame:=laser_frame
                 (EKF -> TF odom -> base_link)
Do NOT run slam_toolbox at the same time: AMCL here publishes map -> odom.

  map_server (map:=...yaml) --> /map --> astar_planner --> /plan
  AMCL (/scan, TF odom->base_link) --> TF map -> odom
  /plan + TF map->base_link --> pure_pursuit_ackermann --> /cmd_vel_pp (m/s, rad/s)
        --> ackermann_cmd_converter --> /cmd_vel_raw (hardware units) --> AEB (Pi) --> car

Usage:
  ros2 launch control_2602_foxtrot pure_pursuit_ackermann.launch.py map:=$HOME/.../slam/maps/lab.yaml
  then in RViz: "2D Goal Pose" (-> /goal_pose)   or   ros2 topic pub --once /laps_cmd std_msgs/msg/Int32 "{data: 1}"
The initial pose defaults to (0, 0, 0): the spot where the map was started. Otherwise
use RViz "2D Pose Estimate" or the init_x / init_y / init_yaw arguments.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup = get_package_share_directory('bringup_2602_foxtrot')
    lc = LaunchConfiguration
    cmd_topic = '/cmd_vel_pp'

    args = [
        DeclareLaunchArgument('map', description='map .yaml saved with map_saver_cli'),
        DeclareLaunchArgument('localization', default_value='true',
                              description='start map_server + AMCL (false if map -> odom comes from elsewhere)'),
        DeclareLaunchArgument('nav2_params', default_value=os.path.join(bringup, 'config', 'nav2_ackermann_params.yaml'),
                              description='only its amcl section is used'),
        DeclareLaunchArgument('init_x', default_value='0.0'),
        DeclareLaunchArgument('init_y', default_value='0.0'),
        DeclareLaunchArgument('init_yaw', default_value='0.0'),
        DeclareLaunchArgument('v_nominal', default_value='0.5'),
        DeclareLaunchArgument('v_min', default_value='0.2'),
        DeclareLaunchArgument('cmd_out_topic', default_value='/cmd_vel_raw',
                              description='hardware-unit commands; /cmd_vel_raw goes through the AEB'),
    ]

    # ── Localization on the saved map ──
    loc = IfCondition(lc('localization'))
    map_server = Node(
        package='nav2_map_server', executable='map_server', name='map_server', output='screen',
        parameters=[{'yaml_filename': lc('map'), 'use_sim_time': False}], condition=loc)
    amcl = Node(
        package='nav2_amcl', executable='amcl', name='amcl', output='screen',
        parameters=[lc('nav2_params'), {
            'use_sim_time': False, 'set_initial_pose': True,
            'initial_pose.x': lc('init_x'), 'initial_pose.y': lc('init_y'), 'initial_pose.yaw': lc('init_yaw')}],
        condition=loc)
    lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'use_sim_time': False, 'autostart': True, 'node_names': ['map_server', 'amcl']}],
        condition=loc)

    # ── Planning: A* on /map -> /plan ──
    planner = Node(
        package='control_2602_foxtrot', executable='astar_planner.py', name='astar_planner', output='screen')

    # ── Tracking: Pure Pursuit -> physical Twist ──
    tracker = Node(
        package='control_2602_foxtrot', executable='pure_pursuit_ackermann.py', name='pure_pursuit_ackermann',
        output='screen',
        parameters=[{'path_topic': '/plan', 'cmd_topic': cmd_topic,
                     'v_nominal': lc('v_nominal'), 'v_min': lc('v_min')}])

    # ── Physical Twist -> hardware units (PWM dithering + Ackermann steering inverse) ──
    converter = Node(
        package='control_2602_foxtrot', executable='ackermann_cmd_converter.py', name='ackermann_cmd_converter',
        output='screen',
        parameters=[{'in_topic': cmd_topic, 'out_topic': lc('cmd_out_topic')}])

    return LaunchDescription(args + [map_server, amcl, lifecycle, planner, tracker, converter])
