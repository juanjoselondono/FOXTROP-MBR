import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    gazebo_pkg_name = "gazebo_2602_foxtrot"
    description_pkg_name = "description_2602_foxtrot"

    use_sim_time = LaunchConfiguration("use_sim_time")

    # --- Robot description (xacro -> URDF XML string) ---
    xacro_file = os.path.join(get_package_share_directory(description_pkg_name), "ackermann_urdf", "robot.urdf.xacro")
    robot_description = xacro.process_file(xacro_file).toxml()

    # --- State Publisher ---
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description,
                     "use_sim_time": use_sim_time}],
    )
    # --- Joint State Publisher GUI (Manual Telemetry) ---
    jsp_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        output="screen"
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false", # Best set to false when testing standalone in RViz
            description="Use simulation (Gazebo) clock if true",
        ),
        DeclareLaunchArgument(
            "world",
            default_value=os.path.join(get_package_share_directory(gazebo_pkg_name), "worlds", "empty_world.sdf"),
            description="Full path to world SDF file",
        ),
        rsp,
        jsp_gui,
    ])