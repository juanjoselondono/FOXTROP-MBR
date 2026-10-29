#!/usr/bin/env python3
"""
covariance_relay.py
===================
Gives robot_localization usable covariances for the two sources that ship
without them:

  /odom_rf2o    (rf2o_laser_odometry: pose and twist covariance are all zeros)
      -> /odom_laser
  /imu/data_raw (twist_cmd_node / Rosmaster: all covariances zero, no orientation)
      -> /imu/data_cov

A zero covariance tells the EKF a measurement is perfect: it would follow rf2o
blindly and the IMU would contribute nothing. Covariances are only filled in
when the incoming matrix is all zeros, so a source that starts reporting its
own keeps them. The IMU orientation is marked unavailable (covariance[0] = -1,
per the sensor_msgs/Imu convention): this IMU has no orientation estimate.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def diag6(d):
    m = [0.0] * 36
    for i, v in enumerate(d):
        m[i * 7] = float(v)
    return m


def diag3(d):
    m = [0.0] * 9
    for i, v in enumerate(d):
        m[i * 4] = float(v)
    return m


class CovarianceRelay(Node):

    def __init__(self):
        super().__init__('covariance_relay')
        self.declare_parameter('odom_in', '/odom_rf2o')
        self.declare_parameter('odom_out', '/odom_laser')
        self.declare_parameter('imu_in', '/imu/data_raw')
        self.declare_parameter('imu_out', '/imu/data_cov')
        self.declare_parameter('imu_frame_id', 'imu_link')
        # variances [x, y, z, roll, pitch, yaw]; huge values = "not measured" (2D)
        self.declare_parameter('odom_pose_variance', [0.01, 0.01, 1e6, 1e6, 1e6, 0.01])
        self.declare_parameter('odom_twist_variance', [0.02, 0.02, 1e6, 1e6, 1e6, 0.02])
        self.declare_parameter('imu_angular_velocity_variance', [0.0025, 0.0025, 0.0025])
        self.declare_parameter('imu_linear_acceleration_variance', [0.25, 0.25, 0.25])

        p = lambda n: self.get_parameter(n).value
        self._pose_cov = diag6(p('odom_pose_variance'))
        self._twist_cov = diag6(p('odom_twist_variance'))
        self._gyro_cov = diag3(p('imu_angular_velocity_variance'))
        self._acc_cov = diag3(p('imu_linear_acceleration_variance'))
        self._imu_frame = p('imu_frame_id')

        self._odom_pub = self.create_publisher(Odometry, p('odom_out'), 10)
        self._imu_pub = self.create_publisher(Imu, p('imu_out'), 10)
        self.create_subscription(Odometry, p('odom_in'), self._odom_cb, 10)
        self.create_subscription(Imu, p('imu_in'), self._imu_cb, qos_profile_sensor_data)

    def _odom_cb(self, msg: Odometry):
        if not any(msg.pose.covariance):
            msg.pose.covariance = self._pose_cov
        if not any(msg.twist.covariance):
            msg.twist.covariance = self._twist_cov
        self._odom_pub.publish(msg)

    def _imu_cb(self, msg: Imu):
        if self._imu_frame:
            msg.header.frame_id = self._imu_frame
        if not any(msg.angular_velocity_covariance):
            msg.angular_velocity_covariance = self._gyro_cov
        if not any(msg.linear_acceleration_covariance):
            msg.linear_acceleration_covariance = self._acc_cov
        if not any(msg.orientation_covariance):
            msg.orientation_covariance[0] = -1.0
        self._imu_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CovarianceRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
