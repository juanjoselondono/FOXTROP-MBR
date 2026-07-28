import os
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PythonExpression, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    gazebo_pkg_name = "gazebo_2602_foxtrot"
    bringup_pkg_name = "bringup_2602_foxtrot"
    description_pkg_name = "description_2602_foxtrot"

    use_sim_time = LaunchConfiguration("use_sim_time")
    world = LaunchConfiguration("world")
    robot_type = LaunchConfiguration("robot_type")

    # --- Robot description (Using Command Substitution for runtime evaluation) ---
    xacro_path = PathJoinSubstitution([
            FindPackageShare(description_pkg_name),
            PythonExpression(["'", robot_type, "_urdf'"]),
            'robot.urdf.xacro'
        ])
        
    # Force the parser to evaluate the XML as a raw string
    robot_description = ParameterValue(Command(['xacro ', xacro_path]), value_type=str)

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description,
                     "use_sim_time": use_sim_time}],
    )

    # --- Launch Gazebo (via ros_gz_sim launch file) ---
    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py",
            )
        ),
        launch_arguments={"gz_args": ['-r -v4 --render-engine ogre ', world], 'on_exit_shutdown': 'true'}.items(),    )

    # --- Spawn entity into Gazebo from robot_description topic ---
    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name", "foxtrot_bot",
            "-topic", "robot_description",
            "-x", "0.0", "-y", "0.0", "-z", "0.5",
        ],
    )

    # --- Time clock bridge ---
    bridge_params = os.path.join(get_package_share_directory(gazebo_pkg_name),'config','topic_bridge.yaml')
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_params}',
        ],
    )

    # --- Base Control Node (Always Active) ---
    joint_broad_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_broadcaster_controller"],
    )

    # --- Dynamic Controller Spawners ---
    diff_drive_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diffdrive_controller"],
        condition=IfCondition(PythonExpression(["'", robot_type, "' == 'diffdrive'"]))
    )

    ackermann_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["ackermann_controller"],
        condition=IfCondition(PythonExpression(["'", robot_type, "' == 'ackermann'"]))
    )

    # --- Dynamic Multiplexer Nodes ---
    twist_mux_params = os.path.join(get_package_share_directory(bringup_pkg_name), 'config', 'twist_mux.yaml')
    
    twist_mux_node_diff = Node(
        package='twist_mux', 
        executable='twist_mux',
        parameters=[twist_mux_params, {'use_sim_time': True}],
        remappings=[('/cmd_vel_out', '/diffdrive_controller/cmd_vel')],
        condition=IfCondition(PythonExpression(["'", robot_type, "' == 'diffdrive'"]))
    )

    twist_mux_node_ack = Node(
        package='twist_mux', 
        executable='twist_mux',
        parameters=[twist_mux_params, {'use_sim_time': True}],
        remappings=[('/cmd_vel_out', '/ackermann_controller/reference')],
        condition=IfCondition(PythonExpression(["'", robot_type, "' == 'ackermann'"]))
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_type",
            default_value="diffdrive",
            description="Choose robot architecture: 'diffdrive' or 'ackermann'",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock if true",
        ),
        DeclareLaunchArgument(
            "world",
            default_value=os.path.join(get_package_share_directory(gazebo_pkg_name), "worlds", "empty_world.sdf"),
            description="Full path to world SDF file",
        ),
        gz_launch,
        rsp,
        spawn,
        bridge,
        # joint_broad_spawner,
        # diff_drive_spawner,
        # ackermann_spawner,
        # twist_mux_node_diff,
        # twist_mux_node_ack
    ])