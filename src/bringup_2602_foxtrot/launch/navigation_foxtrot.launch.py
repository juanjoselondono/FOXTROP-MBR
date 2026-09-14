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

    # 1. Servidor de Mapas
    # map_server_node = Node(
    #     package='nav2_map_server',
    #     executable='map_server',
    #     name='map_server',
    #     output='screen',
    #     parameters=[{'use_sim_time': use_sim_time, 'yaml_filename': map_yaml_file}]
    # )

    # # 2. Administrador del Ciclo de Vida (Bond Timeout extendido para evitar caídas en simulación)
    # lifecycle_manager_node = Node(
    #     package='nav2_lifecycle_manager',
    #     executable='lifecycle_manager',
    #     name='lifecycle_manager_map',
    #     output='screen',
    #     parameters=[{
    #         'use_sim_time': use_sim_time, 
    #         'autostart': True,  # Obliga a transicionar automáticamente a 'active'
    #         'node_names': ['map_server'],
    #         'bond_timeout': 15.0
    #     }]
    # )
    # Integración de Nav2 Oficial (MPPI + NavFn)
    bringup_pkg_dir = get_package_share_directory('bringup_2602_foxtrot')
    nav2_params_path = os.path.join(bringup_pkg_dir, 'config', 'nav2_params.yaml')

    nav2_launch_path = os.path.join(get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')
    
    nav2_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch_path),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': nav2_params_path
        }.items()
    )
    
    # Nodo conversor de Twist a TwistStamped (para sortear problemas de tipos de mensaje en Gazebo/ROS2)
    nav2_converter_node = Node(
        package='control_2602_foxtrot',
        executable='nav2_converter.py',
        name='nav2_converter',
        output='screen'
    )

    # 6. Integración del Filtro Extendido de Kalman (EKF)
    # Se encarga de publicar dinámicamente la transformación odom -> base_link
    ekf_launch_path = os.path.join(get_package_share_directory('ekf_2602_foxtrot'), 'launch', 'ekf.launch.py')
    ekf_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ekf_launch_path),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # Ensamblaje del Launch Description
    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_map_yaml_cmd)
    
    ld.add_action(nav2_include)
    ld.add_action(nav2_converter_node)
    ld.add_action(ekf_include)

    return ld