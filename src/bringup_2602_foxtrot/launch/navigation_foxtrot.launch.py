import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Directorios de los paquetes
    gazebo_pkg_dir = get_package_share_directory('gazebo_2602_foxtrot')
    default_map_path = os.path.join(gazebo_pkg_dir, 'maps', 'map2.yaml')

    # Variables de configuración
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml_file = LaunchConfiguration('map')

    # Argumentos de lanzamiento
    declare_use_sim_time_cmd = DeclareLaunchArgument('use_sim_time', default_value='true')
    declare_map_yaml_cmd = DeclareLaunchArgument('map', default_value=default_map_path)

    # 1. Custom Global Path Planning: A* Search
    astar_planner_node = Node(
        package='control_2602_foxtrot',
        executable='astar_planner.py',
        name='astar_planner',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 4. Custom Path Tracking: MPPI
    mppi_tracker_node = Node(
        package='control_2602_foxtrot',
        executable='mppi_tracker.py',
        name='mppi_tracker',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 5. Safety Override: Autonomous Emergency Braking (AEB)
    aeb_ttc_node = Node(
        package='control_2602_foxtrot',
        executable='aeb_ttc.py',
        name='aeb_ttc',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 6. Integración del Filtro Extendido de Kalman (EKF)
    ekf_launch_path = os.path.join(get_package_share_directory('ekf_2602_foxtrot'), 'launch', 'ekf.launch.py')
    ekf_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ekf_launch_path),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # Ensamblaje del Launch Description
    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_map_yaml_cmd)
    
    ld.add_action(astar_planner_node)
    ld.add_action(mppi_tracker_node)
    ld.add_action(aeb_ttc_node)
    ld.add_action(ekf_include)

    return ld