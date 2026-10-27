#!/usr/bin/env python3
"""
speed_estimator_node.py
=======================
Online forward-speed and Ackermann yaw-rate estimator for an RC car.
Implements a Hybrid Digital Twin with Asymmetric Dynamics and Manual Scaling.
"""

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, TransformStamped
from tf2_ros import TransformBroadcaster


class SpeedEstimatorNode(Node):
    """Hybrid parametric speed estimator with Asymmetric dead-reckoning and scaling."""

    def __init__(self):
        super().__init__('speed_estimator_node')

        # ── Parámetros Asimétricos Longitudinales ────────────────────────────
        self.declare_parameter('K_fwd',      5.0)     # Ganancia base hacia adelante
        self.declare_parameter('K_rev',      5.0)     # Ganancia base en reversa
        self.declare_parameter('tau',        0.167)   # Constante de tiempo
        self.declare_parameter('u_dead_fwd', 0.08)    # Zona muerta hacia adelante
        self.declare_parameter('u_dead_rev', 0.036)   # Zona muerta en reversa

        # ── ESCALARES DE AJUSTE MANUAL (Prueba y Error) ──────────────────────
        self.declare_parameter('scale_fwd',  5.0)     # Multiplicador de distancia hacia adelante
        self.declare_parameter('scale_rev',  1.0)     # Multiplicador de distancia en reversa

        # ── Parámetros Laterales (Ackermann) ─────────────────────────────────
        self.declare_parameter('L_eff',   0.245)  
        self.declare_parameter('K_steer', 0.436)  
        self.declare_parameter('steering_axis', 2)

        # Execution
        self.declare_parameter('dt', 0.05)        # 20 Hz update rate

        # ── Carga de Parámetros ──────────────────────────────────────────────
        self._K_fwd = float(self.get_parameter('K_fwd').value)
        self._K_rev = float(self.get_parameter('K_rev').value)
        self._tau = float(self.get_parameter('tau').value)
        self._u_dead_fwd = float(self.get_parameter('u_dead_fwd').value)
        self._u_dead_rev = float(self.get_parameter('u_dead_rev').value)
        
        self._scale_fwd = float(self.get_parameter('scale_fwd').value)
        self._scale_rev = float(self.get_parameter('scale_rev').value)

        self._L_eff = float(self.get_parameter('L_eff').value)
        self._K_steer = float(self.get_parameter('K_steer').value)
        self._steering_axis = int(self.get_parameter('steering_axis').value)
        self._dt = float(self.get_parameter('dt').value)

        # ── Discretización Exponencial con Escalares Aplicados ───────────────
        self._alpha = math.exp(-self._dt / self._tau)
        self._beta_fwd = (self._K_fwd * self._scale_fwd) * (1.0 - self._alpha)
        self._beta_rev = (self._K_rev * self._scale_rev) * (1.0 - self._alpha)

        # ── Variables de Estado ──────────────────────────────────────────────
        self._v   = 0.0   
        self._x   = 0.0   
        self._y   = 0.0   
        self._yaw = 0.0   

        self._u_throttle = 0.0
        self._u_steer = 0.0

        # ── Suscriptores y Publicadores ──────────────────────────────────────
        self._sub_cmd = self.create_subscription(
            TwistStamped, '/cmd_vel_stamped', self._cmd_vel_cb, 10)
        self._sub_joy = self.create_subscription(
            Joy, '/joy', self._joy_cb, 10)

        self._pub_speed = self.create_publisher(Float32, '/estimated_speed', 10)
        self._pub_odom  = self.create_publisher(Odometry, '/estimated_odom', 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._timer = self.create_timer(self._dt, self._update)

    def _cmd_vel_cb(self, msg: TwistStamped):
        self._u_throttle = float(msg.twist.linear.x)

    def _joy_cb(self, msg: Joy):
        if len(msg.axes) > self._steering_axis:
            self._u_steer = float(msg.axes[self._steering_axis])

    def _update(self):
        # 1. Lógica Asimétrica de Zona Muerta y Ganancia
        if self._u_throttle > self._u_dead_fwd:
            u_eff = self._u_throttle - self._u_dead_fwd
            beta_current = self._beta_fwd
        elif self._u_throttle < -self._u_dead_rev:
            u_eff = self._u_throttle + self._u_dead_rev
            beta_current = self._beta_rev
        else:
            u_eff = 0.0
            beta_current = self._beta_fwd

        # 2. Integración de Velocidad
        self._v = (self._alpha * self._v) + (beta_current * u_eff)

        # 3. Cinemática de Ackermann
        delta = self._K_steer * self._u_steer
        omega = (self._v / self._L_eff) * math.tan(delta)

        # 4. Integración Odometría
        self._yaw += omega * self._dt
        self._x   += self._v * math.cos(self._yaw) * self._dt
        self._y   += self._v * math.sin(self._yaw) * self._dt

        now = self.get_clock().now().to_msg()

        self._pub_speed.publish(Float32(data=float(self._v)))

        # 5. Publicación de Mensajes
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = float(self._x)
        odom.pose.pose.position.y = float(self._y)
        odom.pose.pose.orientation.z = float(math.sin(self._yaw / 2.0))
        odom.pose.pose.orientation.w = float(math.cos(self._yaw / 2.0))
        odom.twist.twist.linear.x = float(self._v)
        odom.twist.twist.angular.z = float(omega)
        self._pub_odom.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x = float(self._x)
        tf.transform.translation.y = float(self._y)
        tf.transform.rotation.z = float(math.sin(self._yaw / 2.0))
        tf.transform.rotation.w = float(math.cos(self._yaw / 2.0))
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