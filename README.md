## Mobile Robotics: Autonomous Driving Workspace

This is a GitHub template. You can make your own copy by clicking the green "Use this template" button.

## Requirements knwoledge

- ROS 2 Jazzy
- ROS2_Control
- Gazebo Harmonic
- Python3 & C++
## Command setup

## Requirements package
- colcon build --symlink-install
- source install/setup.bash
- ros2 launch bringup_2602_foxtrot gz_spawn.launch.py robot_type:=diffdrive
- ros2 launch bringup_2602_foxtrot gz_spawn.launch.py robot_type:=ackermann


- `rclpy`
- `geometry_msgs`, `nav_msgs`, `sensor_msgs`, `tf2_ros`
- `numpy`

## Reference

This project was develop with resources available form:

Josh Newans: https://github.com/joshnewans

And the work done by:

M. O’Kelly, H. Zheng, D. Karthik, and R. Mangharam, “*F1tenth: An open-source evaluation environment for continuous control and reinforcement learning*”, in NeurIPS 2019 Competition and Demonstration Track. PMLR, 2020,pp. 77–89

RoboRacer - Course Documentation: https://roboracer.ai/learn

## License

Apache 2.0

## Author

*Professor*: David Rozo-Osorio, I.M. M.Sc. email: david.rozo31@eia.edu.co

**EIA University**, Mechatronical Eng. - Industrial Robotics

Version: 2026-02