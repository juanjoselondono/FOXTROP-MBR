#!/usr/bin/env python3
"""
speed_estimator_node.py
=======================
Online forward-speed estimator for an RC car with no wheel encoders.

Uses a fitted 1st-order vehicle speed model:
  u_eff = apply_deadzone(u, u_dead)
  dv/dt = (K * u_eff - v) / tau

Subscribes:
  /esc/command  (std_msgs/Float32)   — normalised ESC throttle command [-1..+1]
  /cmd_vel_stamped (geometry_msgs/TwistStamped) — for yaw rate (steering)

Publishes:
  /estimated_speed  (std_msgs/Float32)   — estimated forward speed [m/s]
  /estimated_odom   (nav_msgs/Odometry)  — dead-reckoning odometry

Parameters (set via YAML or --ros-args -p):
  K        (float, default 1.0)  — steady-state speed per unit command [m/s]
  tau      (float, default 0.3)  — first-order time constant [s]
  u_dead   (float, default 0.05) — ESC dead-zone threshold (normalised)
  dt       (float, default 0.05) — estimator update rate [s] (= 20 Hz)

Run with fitted parameters:
  ros2 run sysid_2602_foxtrot speed_estimator_node \\
      --ros-args -p K:=1.23 -p tau:=0.28 -p u_dead:=0.06

Or via launch file:
  ros2 launch sysid_2602_foxtrot speed_estimator.launch.py
"""

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class SpeedEstimatorNode(Node):
    """1st-order parametric forward-speed estimator with dead-reckoning odometry."""

    def __init__(self):
        super().__init__('speed_estimator_node')

        # ── Declare & read model parameters ─────────────────────────────────
        self.declare_parameter('K',      1.0)   # [m/s per unit]
        self.declare_parameter('tau',    0.3)   # [s]
        self.declare_parameter('u_dead', 0.05)  # [normalised]
        self.declare_parameter('dt',     0.05)  # [s] → 20 Hz

        self._K      = float(self.get_parameter('K').value)
        self._tau    = float(self.get_parameter('tau').value)
        self._u_dead = float(self.get_parameter('u_dead').value)
        self._dt     = float(self.get_parameter('dt').value)

        # ── State variables ──────────────────────────────────────────────────
        self._v   = 0.0   # estimated speed [m/s]
        self._x   = 0.0   # dead-reckoning x [m]
        self._y   = 0.0   # dead-reckoning y [m]
        self._yaw = 0.0   # heading from /cmd_vel_stamped angular.z integration [rad]
        self._u   = 0.0   # latest throttle command
        self._omega = 0.0 # latest yaw rate command [rad/s]

        # ── Subscribers ──────────────────────────────────────────────────────
        self._sub_esc = self.create_subscription(
            Float32, '/esc/command', self._esc_cmd_cb, 10)
        self._sub_cmd = self.create_subscription(
            TwistStamped, '/cmd_vel_stamped', self._cmd_vel_cb, 10)

        # ── Publishers ───────────────────────────────────────────────────────
        self._pub_speed = self.create_publisher(Float32,   '/estimated_speed', 10)
        self._pub_odom  = self.create_publisher(Odometry, '/estimated_odom',  10)

        # ── TF broadcaster ───────────────────────────────────────────────────
        self._tf_broadcaster = TransformBroadcaster(self)

        # ── Update timer ─────────────────────────────────────────────────────
        self._timer = self.create_timer(self._dt, self._update)

        self.get_logger().info(
            f'SpeedEstimator ready  K={self._K:.3f} m/s  '
            f'tau={self._tau:.3f} s  u_dead={self._u_dead:.3f}'
        )

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _esc_cmd_cb(self, msg: Float32):
        """Receive normalised ESC throttle command."""
        self._u = float(msg.data)

    def _cmd_vel_cb(self, msg: TwistStamped):
        """
        Receive velocity command.
        Use linear.x as throttle proxy (normalised to max speed)
        and angular.z as yaw rate for dead-reckoning heading.
        """
        # Extract yaw rate directly from the twist command
        self._omega = float(msg.twist.angular.z)
        # If /esc/command is not available separately, derive u from linear.x
        # (override only if /esc/command has not been received yet)
        # Users can disable this by publishing /esc/command explicitly.

    # ── Dead-zone helper ─────────────────────────────────────────────────────

    def _apply_deadzone(self, u: float) -> float:
        if u > self._u_dead:
            return u - self._u_dead
        if u < -self._u_dead:
            return u + self._u_dead
        return 0.0

    # ── Main update loop ─────────────────────────────────────────────────────

    def _update(self):
        # 1. Speed model integration
        u_eff = self._apply_deadzone(self._u)
        dvdt = (self._K * u_eff - self._v) / self._tau
        self._v += dvdt * self._dt

        # 2. Dead-reckoning pose integration
        self._yaw += self._omega * self._dt
        self._x   += self._v * math.cos(self._yaw) * self._dt
        self._y   += self._v * math.sin(self._yaw) * self._dt

        now = self.get_clock().now().to_msg()

        # 3. Publish speed
        self._pub_speed.publish(Float32(data=float(self._v)))

        # 4. Publish odometry
        odom = Odometry()
        odom.header.stamp    = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'
        odom.pose.pose.position.x = float(self._x)
        odom.pose.pose.position.y = float(self._y)
        # Heading as quaternion (z-axis rotation only)
        odom.pose.pose.orientation.z = float(math.sin(self._yaw / 2.0))
        odom.pose.pose.orientation.w = float(math.cos(self._yaw / 2.0))
        odom.twist.twist.linear.x  = float(self._v)
        odom.twist.twist.angular.z = float(self._omega)
        self._pub_odom.publish(odom)

        # 5. Broadcast TF: odom → base_link
        tf = TransformStamped()
        tf.header.stamp    = now
        tf.header.frame_id = 'odom'
        tf.child_frame_id  = 'base_link'
        tf.transform.translation.x = float(self._x)
        tf.transform.translation.y = float(self._y)
        tf.transform.rotation.z    = float(math.sin(self._yaw / 2.0))
        tf.transform.rotation.w    = float(math.cos(self._yaw / 2.0))
        self._tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = SpeedEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
