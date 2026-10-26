#!/usr/bin/env python3
"""
throttle_profile.launch.py
===========================
Launch the automated throttle-step sequencer for sysid Session A.

Also starts joy_node so the controller's B button can be used as a live
brake / E-stop (see throttle_profile_node.py docstring) -- it does NOT
start teleop_node, which would otherwise fight this node over
/cmd_vel_stamped.

Usage:
    ros2 launch sysid_2602_foxtrot throttle_profile.launch.py
    ros2 launch sysid_2602_foxtrot throttle_profile.launch.py \\
        throttle_levels_pct:="[20.0, 50.0, 70.0]" hold_duration:=3.0 \\
        neutral_duration:=3.0 repeats:=1 interactive_pause:=true
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
        DeclareLaunchArgument('throttle_levels_pct', default_value='[20.0, 50.0, 70.0]',
                               description='Throttle levels to step through, as %% of max_linear_x'),
        DeclareLaunchArgument('hold_duration', default_value='2.0',
                               description='Seconds to hold each throttle level [s]'),
        DeclareLaunchArgument('neutral_duration', default_value='3.0',
                               description='Seconds to hold neutral between reps [s]'),
        DeclareLaunchArgument('repeats', default_value='1',
                               description='Repeats per throttle level'),
        DeclareLaunchArgument('max_linear_x', default_value='0.11',
                               description='linear.x value for 100%% throttle [m/s] -- matches '
                                           'velocidad_maxima in ftg_advanced.py'),
        DeclareLaunchArgument('start_delay', default_value='3.0',
                               description='Delay before sequence starts, to arm ros2 bag record [s]'),
        DeclareLaunchArgument('interactive_pause', default_value='true',
                               description='Pause between levels to reposition the car'),
        DeclareLaunchArgument('pause_duration', default_value='8.0',
                               description='Countdown length for the between-levels pause [s]'),
        DeclareLaunchArgument('brake_button_index', default_value='1',
                               description='/joy buttons[] index that brakes + aborts (B on most pads)'),

        Node(
            package='joy',
            executable='joy_node',
            parameters=[joy_params],
        ),

        Node(
            package='sysid_2602_foxtrot',
            executable='throttle_profile_node',
            name='throttle_profile_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'throttle_levels_pct': LaunchConfiguration('throttle_levels_pct'),
                'hold_duration': LaunchConfiguration('hold_duration'),
                'neutral_duration': LaunchConfiguration('neutral_duration'),
                'repeats': LaunchConfiguration('repeats'),
                'max_linear_x': LaunchConfiguration('max_linear_x'),
                'start_delay': LaunchConfiguration('start_delay'),
                'interactive_pause': LaunchConfiguration('interactive_pause'),
                'pause_duration': LaunchConfiguration('pause_duration'),
                'brake_button_index': LaunchConfiguration('brake_button_index'),
            }],
        ),
    ])
