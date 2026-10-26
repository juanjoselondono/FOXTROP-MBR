#!/usr/bin/env python3
"""
manual_cruise.launch.py
=========================
Launch the button-driven manual cruise control (see manual_cruise_node.py
for the full control scheme: A=start+record, Up/Down=speed, B=brake+stop
recording).

Also starts joy_node (NOT teleop_node -- that would fight this node over
/cmd_vel_stamped).

Usage:
    ros2 launch sysid_2602_foxtrot manual_cruise.launch.py
    ros2 launch sysid_2602_foxtrot manual_cruise.launch.py \\
        max_speed:=0.25 start_speed:=0.08 speed_step:=0.02
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    joy_params = os.path.join(
        get_package_share_directory('bringup_2602_foxtrot'), 'config', 'joystick.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('max_speed', default_value='0.25',
                               description='Upper clamp for commanded speed [m/s]'),
        DeclareLaunchArgument('start_speed', default_value='0.08',
                               description='Speed applied the moment A is pressed [m/s]'),
        DeclareLaunchArgument('speed_step', default_value='0.02',
                               description='Speed delta per Up/Down press [m/s]'),
        DeclareLaunchArgument('start_button_index', default_value='0',
                               description='/joy buttons[] index for A (start + record)'),
        DeclareLaunchArgument('brake_button_index', default_value='1',
                               description='/joy buttons[] index for B (brake + stop record)'),
        DeclareLaunchArgument('increase_button_index', default_value='-1',
                               description='/joy buttons[] index for Up, -1 = use dpad axis instead'),
        DeclareLaunchArgument('decrease_button_index', default_value='-1',
                               description='/joy buttons[] index for Down, -1 = use dpad axis instead'),
        DeclareLaunchArgument('dpad_axis_index', default_value='7',
                               description='/joy axes[] index for the D-pad vertical hat'),
        DeclareLaunchArgument('bag_output_dir', default_value='./bag_files/manual',
                               description='Directory for auto-started bag recordings'),

        Node(
            package='joy',
            executable='joy_node',
            parameters=[joy_params],
        ),

        Node(
            package='sysid_2602_foxtrot',
            executable='manual_cruise_node',
            name='manual_cruise_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'max_speed': LaunchConfiguration('max_speed'),
                'start_speed': LaunchConfiguration('start_speed'),
                'speed_step': LaunchConfiguration('speed_step'),
                'start_button_index': LaunchConfiguration('start_button_index'),
                'brake_button_index': LaunchConfiguration('brake_button_index'),
                'increase_button_index': LaunchConfiguration('increase_button_index'),
                'decrease_button_index': LaunchConfiguration('decrease_button_index'),
                'dpad_axis_index': LaunchConfiguration('dpad_axis_index'),
                'bag_output_dir': LaunchConfiguration('bag_output_dir'),
            }],
        ),
    ])
