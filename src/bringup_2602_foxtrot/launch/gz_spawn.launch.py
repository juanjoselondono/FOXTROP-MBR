import os
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, AppendEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PythonExpression, PathJoinSubstitution
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    gazebo_pkg_name = "gazebo_2602_foxtrot"
    bringup_pkg_name = "bringup_2602_foxtrot"
    description_pkg_name = "description_2602_foxtrot"

    # --- Gazebo Resource Path Resolution ---
    gazebo_pkg_share = get_package_share_directory(gazebo_pkg_name)
    workspace_share = os.path.dirname(gazebo_pkg_share)
    
    gz_resource_path_update = AppendEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=workspace_share
    )

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
        launch_arguments={"gz_args": ['-r -v4 ', world], 'on_exit_shutdown': 'true'}.items(),
    )   
    
    #Spawn the robot in Gazebo using ros_gz_sim's create node
    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name", "foxtrot_bot",
            "-topic", "robot_description",
            "-x", "-2.0", 
            "-y", "0.0", 
            "-z", "3.5", # Drop altitude above Bridge 1 deck
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

    # --- Event Handlers for Sequential Execution ---
    # These prevent the spawners from executing until the 'spawn' node exits successfully
    delay_joint_broad_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn,
            on_exit=[joint_broad_spawner],
        )
    )

    delay_diff_drive_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn,
            on_exit=[diff_drive_spawner],
        )
    )

    delay_ackermann_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn,
            on_exit=[ackermann_spawner],
        )
    )

    # --- Dynamic Multiplexer Nodes ---
    joy_params = os.path.join(get_package_share_directory(bringup_pkg_name),'config','joystick.yaml')

    joy_node = Node(package='joy', 
                    executable='joy_node',
                    parameters=[joy_params],
    )

    teleop_node = Node(package='teleop_twist_joy', 
                    executable='teleop_node',
                    name="teleop_node",
                    parameters=[joy_params],
                    remappings=[('/cmd_vel','/cmd_vel_joy')]
    )

    twist_mux_params = os.path.join(get_package_share_directory(bringup_pkg_name),'config','twist_mux.yaml')
    
    twist_mux_node = Node(package='twist_mux', 
                    executable='twist_mux',
                    parameters=[twist_mux_params,{'use_sim_time': True}],
                    # REROUTED: Send multiplexed commands to the AEB filter instead of the wheels
                    remappings=[('/cmd_vel_out','/cmd_vel_raw')] 
    )

    twist_mux_node_ack = Node(
        package='twist_mux', 
        executable='twist_mux',
        parameters=[twist_mux_params, {'use_sim_time': True}],
        remappings=[('/cmd_vel_out', '/ackermann_controller/reference')],
        condition=IfCondition(PythonExpression(["'", robot_type, "' == 'ackermann'"]))
    )
    
    # Autonomous Emergency Braking (AEB) Node
    aeb_filter_node = Node(
        package='control_2602_foxtrot',
        executable='aeb_ttc.py',
        name='kinematic_aeb_filter',
        output='screen',
        parameters=[{'use_sim_time': True}],
        remappings=[
            ('/cmd_vel', '/diffdrive_controller/cmd_vel') # Final filtered output to the wheels
        ]
    )
    
    supervisor_node = Node(
        package='control_2602_foxtrot',
        executable='supervisor.py',
        name='mode_supervisor',
        output='screen'
    )
    
    # Wall Following Node
    wall_follower_node = Node(
        package='control_2602_foxtrot',
        executable='wall_follower.py',
        name='wall_follower_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}] # Forces sync with Gazebo
    )
    
    # lane keeping node
    lane_keeper_node = Node(
        package='control_2602_foxtrot',
        executable='lane_keeper.py',
        name='lane_keeper_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}] # Forces sync with Gazebo
    )
    
    # Follow the gap
    follow_gap_node = Node(
        package='control_2602_foxtrot',
        executable='gap.py',
        name='follow_gap_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}] # Forces sync with Gazebo
    )
    
    return LaunchDescription([
        gz_resource_path_update, # Injected environment variable
        
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
            default_value=os.path.join(get_package_share_directory(gazebo_pkg_name), "worlds", "exam1_world.sdf"),
            description="Full path to world SDF file",
        ),
        gz_launch,
        rsp,
        spawn,
        bridge,
        joy_node,
        teleop_node,
        twist_mux_node,
        aeb_filter_node,
        delay_joint_broad_spawner,
        delay_diff_drive_spawner,
        supervisor_node,     
        wall_follower_node,
        follow_gap_node,
        #lane_keeper_node   
        # delay_ackermann_spawner,
        # twist_mux_node_ack
        # twist_mux_node_ack,
        # We replace the direct node calls with the delayed event handlers
    ])