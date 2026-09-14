#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

def yaw_from_quaternion(q):
    """Extracts the yaw (Z) angle from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def wrap_pi(a):
    """Wraps an angle to the [-pi, pi] range."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi

class SimpleEKF(Node):
    """
    Minimal 2D EKF:
      x = [x, y, yaw, v, omega]^T

    Fuses:
      - wheel odom twist (v, omega)
      - IMU yaw-rate (omega_z)

    Publishes:
      - nav_msgs/Odometry (odom -> base_link)
      - TF (odom -> base_link) optionally
    """

    def __init__(self):
        # Nomenclatura unificada para la arquitectura Foxtrot
        super().__init__('ekf_node_2602_foxtrot')

        # ---- Parameters ----
        self.declare_parameter('frequency', 30.0)
        self.declare_parameter('sensor_timeout', 0.2)
        self.declare_parameter('frames.map_frame', 'map')
        self.declare_parameter('frames.odom_frame', 'odom')
        self.declare_parameter('frames.base_frame', 'base_link')
        self.declare_parameter('frames.publish_tf', True)

        self.declare_parameter('topics.wheel_odom', '/diffdrive_controller/odom')
        self.declare_parameter('topics.imu', '/imu')

        self.declare_parameter('wheel_odom_use.v', True)
        self.declare_parameter('wheel_odom_use.yaw_rate', True)
        self.declare_parameter('imu_use.yaw_rate', True)

        self.declare_parameter('P0_diag', [0.5, 0.8, 0.5, 1.0, 1.0])
        self.declare_parameter('Q_diag',  [0.05, 0.05, 0.02, 0.50, 0.50])
        self.declare_parameter('R_wheel_diag', [0.20, 0.20])
        self.declare_parameter('R_imu_yaw_rate', 0.05)

        self.freq = float(self.get_parameter('frequency').value)
        self.sensor_timeout = float(self.get_parameter('sensor_timeout').value)

        self.map_frame = self.get_parameter('frames.map_frame').value
        self.odom_frame = self.get_parameter('frames.odom_frame').value
        self.base_frame = self.get_parameter('frames.base_frame').value
        self.publish_tf = bool(self.get_parameter('frames.publish_tf').value)

        self.topic_wheel = self.get_parameter('topics.wheel_odom').value
        self.topic_imu = self.get_parameter('topics.imu').value

        self.use_wheel_v = bool(self.get_parameter('wheel_odom_use.v').value)
        self.use_wheel_w = bool(self.get_parameter('wheel_odom_use.yaw_rate').value)
        self.use_imu_w = bool(self.get_parameter('imu_use.yaw_rate').value)

        P0_diag = np.array(self.get_parameter('P0_diag').value, dtype=float)
        Q_diag  = np.array(self.get_parameter('Q_diag').value, dtype=float)
        Rwheel_diag = np.array(self.get_parameter('R_wheel_diag').value, dtype=float)
        Rimu = float(self.get_parameter('R_imu_yaw_rate').value)

        # ---- EKF state ----
        self.x = np.zeros((5, 1), dtype=float)  # [x, y, yaw, v, omega]
        self.P = np.diag(P0_diag)

        self.Q_base = np.diag(Q_diag)  # will be scaled by dt in predict

        # Measurement noises
        self.R_wheel = np.diag(Rwheel_diag)           # for [v, omega]
        self.R_imu = np.array([[Rimu]], dtype=float)  # for [omega]

        # Timing
        self.last_predict_stamp = None
        self.last_meas_stamp = None

        # Subscribers
        self.sub_wheel = self.create_subscription(Odometry, self.topic_wheel, self.on_wheel_odom, 50)
        self.sub_imu = self.create_subscription(Imu, self.topic_imu, self.on_imu, 50)

        # Publisher
        self.pub_odom = self.create_publisher(Odometry, 'ekf/odometry', 10)

        # TF
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer at nominal EKF rate (predict + publish)
        self.timer = self.create_timer(1.0 / self.freq, self.on_timer)

        self.get_logger().info("SimpleEKF initialized. Publishing: /ekf/odometry (and TF if enabled).")

    def now_sec(self):
        """Works with sim time if /clock is present and use_sim_time is true."""
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------- EKF core ----------
    def predict(self, dt):
        if dt <= 0.0:
            return

        x, y, yaw, v, w = self.x.flatten()

        # Nonlinear motion model
        x_new = x + v * math.cos(yaw) * dt
        y_new = y + v * math.sin(yaw) * dt
        yaw_new = wrap_pi(yaw + w * dt)
        v_new = v
        w_new = w

        self.x = np.array([[x_new], [y_new], [yaw_new], [v_new], [w_new]], dtype=float)

        # Jacobian F = df/dx
        F = np.eye(5)
        F[0, 2] = -v * math.sin(yaw) * dt
        F[0, 3] =  math.cos(yaw) * dt
        F[1, 2] =  v * math.cos(yaw) * dt
        F[1, 3] =  math.sin(yaw) * dt
        F[2, 4] =  dt

        # Simple dt scaling for process noise
        Q = self.Q_base * max(dt, 1e-3)

        self.P = F @ self.P @ F.T + Q

    def update(self, z, H, R):
        """
        Standard EKF measurement update for linear measurement model:
          z = H x + noise
        """
        y = z - (H @ self.x)                       # innovation
        S = H @ self.P @ H.T + R                   # innovation cov
        K = self.P @ H.T @ np.linalg.inv(S)        # Kalman gain

        self.x = self.x + K @ y
        self.x[2, 0] = wrap_pi(self.x[2, 0])       # keep yaw bounded
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P

    # ---------- Callbacks ----------
    def on_wheel_odom(self, msg: Odometry):
        # Use message stamp if present; fallback to node time
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = self.now_sec()
        self.last_meas_stamp = stamp

        v_meas = msg.twist.twist.linear.x
        w_meas = msg.twist.twist.angular.z

        if self.use_wheel_v and self.use_wheel_w:
            z = np.array([[v_meas], [w_meas]], dtype=float)
            H = np.zeros((2, 5))
            H[0, 3] = 1.0  # measure v
            H[1, 4] = 1.0  # measure omega
            self.update(z, H, self.R_wheel)

        elif self.use_wheel_v and (not self.use_wheel_w):
            z = np.array([[v_meas]], dtype=float)
            H = np.zeros((1, 5))
            H[0, 3] = 1.0
            R = np.array([[self.R_wheel[0, 0]]], dtype=float)
            self.update(z, H, R)

    def on_imu(self, msg: Imu):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = self.now_sec()
        self.last_meas_stamp = stamp

        if not self.use_imu_w:
            return

        w_meas = msg.angular_velocity.z
        z = np.array([[w_meas]], dtype=float)
        H = np.zeros((1, 5))
        H[0, 4] = 1.0  # measure omega
        self.update(z, H, self.R_imu)

    # ---------- Periodic predict/publish ----------
    def on_timer(self):
        now = self.now_sec()

        if self.last_predict_stamp is None:
            self.last_predict_stamp = now
            return

        dt = now - self.last_predict_stamp
        self.last_predict_stamp = now

        # Handle sensor timeout logic if desired
        if (self.last_meas_stamp is not None) and ((now - self.last_meas_stamp) > self.sensor_timeout):
            pass # Currently configured to predict even if sensors timeout

        self.predict(dt)
        self.publish_outputs(now)

    def publish_outputs(self, now_sec):
        x, y, yaw, v, w = self.x.flatten()

        # Build odom message
        odom = Odometry()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        # Stamp: use node clock
        t = self.get_clock().now().to_msg()
        odom.header.stamp = t

        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)

        # yaw -> quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        odom.pose.pose.orientation.w = cy
        odom.pose.pose.orientation.z = sy

        odom.twist.twist.linear.x = float(v)
        odom.twist.twist.angular.z = float(w)

        # Fill covariance from P
        pose_cov = np.zeros((6, 6))
        pose_cov[0, 0] = self.P[0, 0]
        pose_cov[1, 1] = self.P[1, 1]
        pose_cov[5, 5] = self.P[2, 2]
        odom.pose.covariance = pose_cov.flatten().tolist()

        twist_cov = np.zeros((6, 6))
        twist_cov[0, 0] = self.P[3, 3]
        twist_cov[5, 5] = self.P[4, 4]
        odom.twist.covariance = twist_cov.flatten().tolist()

        self.pub_odom.publish(odom)

        if self.publish_tf:
            tfm = TransformStamped()
            tfm.header.stamp = t
            tfm.header.frame_id = self.odom_frame
            tfm.child_frame_id = self.base_frame
            tfm.transform.translation.x = float(x)
            tfm.transform.translation.y = float(y)
            tfm.transform.rotation.w = cy
            tfm.transform.rotation.z = sy
            self.tf_broadcaster.sendTransform(tfm)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleEKF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Validación de contexto para prevenir colapso de rclpy
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()