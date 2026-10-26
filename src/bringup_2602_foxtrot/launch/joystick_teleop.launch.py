import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    bringup_pkg_name = "bringup_2602_foxtrot"
    joy_params = os.path.join(get_package_share_directory(bringup_pkg_name),'config','joystick.yaml')

    joy_node = Node(package='joy', 
                    executable='joy_node',
                    parameters=[joy_params],
    )

    teleop_node = Node(package='teleop_twist_joy', 
                    executable='teleop_node',
                    name="teleop_node",
                    parameters=[joy_params],
                    # REMAPPED to match the RPi5 node exactly
                    remappings=[('/cmd_vel','/cmd_vel_stamped')]
    )

    return LaunchDescription([
        joy_node,
        teleop_node
    ])
