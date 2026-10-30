cmake_minimum_required(VERSION 3.8)
project(control_2602_foxtrot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Install Python nodes as executables
install(PROGRAMS
  AEB/aeb_ttc.py
  AEB/gap.py
  AEB/lane_keeper.py
  AEB/supervisor.py
  AEB/wall_follower.py
  planning/rrt_pp_2602_foxtrot.py
  planning/nav2_converter.py
  planning/ackermann_cmd_converter.py
  planning/astar_planner.py
  tracking/pid_pt_2602_foxtrot.py
  tracking/track2.py
  tracking/rpp_dwa_tracker.py
  tracking/mppi_tracker.py
  tracking/pure_pursuit_ackermann.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files to the package share directory
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()