"""
ekf.launch.py

    Launch robot_localization EKF for a differential-drive robot (wheel odom + IMU).

    Assumptions:
    - Wheel odometry topic: /diffdrive_controller/odom  (nav_msgs/Odometry)
    - IMU topic:            /imu                          (sensor_msgs/Imu)
    - Config file:          <your_pkg>/config/ekf.yaml
    - Frames:               odom -> base_link (published by EKF)

    Put this file in: <your_pkg>/launch/ekf.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition

import xacro

def generate_launch_description():

    ekf_pkg_name = "ekf_2602_foxtrot"

    use_sim_time = LaunchConfiguration("use_sim_time")

    ekf_params = os.path.join(get_package_share_directory(ekf_pkg_name), 'config', 'ekf.yaml')

    # imu_stamper_node = Node(
    #     package="ekf_2602_zulu",
    #     executable="imu_stamper",
    #     name="imu_stamper_node",
    #     output="screen",
    #     parameters=[{"use_sim_time": True,
    #            "in_topic": "/imu",
    #            "out_topic": "/imu/data",
    #            "frame_id": "imu_link"}],
    # )

    ekf_node = Node(
        package=ekf_pkg_name,
        executable="ekf_node_2602_foxtrot",
        name="ekf_node_2602_foxtrot",
        output="screen",
        parameters=[
            ekf_params,
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            # Uncomment if your topics differ:
            # ("/wheel/odom", "/your_wheel_odom"),
            # ("/imu/data", "/your_imu_topic"),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock if true",
        ),
        #imu_stamper_node,
        ekf_node,
    ])