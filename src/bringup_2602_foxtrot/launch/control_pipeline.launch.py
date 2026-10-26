import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    bringup_pkg_name = "bringup_2602_foxtrot"
    
    # FIXED: Updated to match your laptop's control package
    control_pkg_name = "control_2602_foxtrot" 
    
    joy_params = os.path.join(get_package_share_directory(bringup_pkg_name), 'config', 'joystick.yaml')

    return LaunchDescription([
        Node(
            package='joy', 
            executable='joy_node',
            parameters=[joy_params],
        ),
        Node(
            package='teleop_twist_joy', 
            executable='teleop_node',
            name="teleop_node",
            parameters=[joy_params],
            remappings=[('/cmd_vel', '/cmd_vel_raw')]
        ),
        Node(
            package=control_pkg_name,
            executable='aeb_ttc.py',
            name='kinematic_aeb_filter',
            output='screen'
        ),
        Node(
            package=control_pkg_name,
            executable='supervisor.py',
            name='mode_supervisor',
            output='screen'
        )
    ])