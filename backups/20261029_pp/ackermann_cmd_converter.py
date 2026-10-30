#!/usr/bin/env python3
"""
ackermann_cmd_converter.py
==========================
Nav2 /cmd_vel (geometry_msgs/Twist, physical units: v [m/s], omega [rad/s])
  -> /cmd_vel_raw (TwistStamped, HARDWARE units for twist_cmd_node, through the AEB)

twist_cmd_node does not take m/s and rad/s: linear.x is an ESC throttle
fraction and angular.z a steering-servo command, both mapped to INTEGER PWM.
This node inverts the identified models (VICON, bag_files/sysid2):

Speed -> throttle: the integer PWM leaves the ESC only a few sustainable
speeds (measured, straight runs held >= 2 s):
    PWM 90: 0.00   95: 0.09   96: 0.18   97: 1.30   98: 1.84  m/s
There is nothing between 0.18 and 1.30 m/s, so a single PWM level cannot hold
e.g. 0.5 m/s. The node DITHERS between the two neighbouring levels
(first-order sigma-delta at the publish rate); the car's slow longitudinal
dynamics (tau ~ 1-1.8 s) average the pulses into the requested mean speed.

Yaw rate -> steering: Ackermann bicycle model, delta = atan(L * omega / v),
normalized steering s = delta / K_steer (asymmetric: +0.2495 / -0.3244 rad at
full lock), angular.z = s * steer_full_scale.

Reverse is NOT identified: negative v is sent as neutral unless allow_reverse.
"""

import bisect
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped


class AckermannCmdConverter(Node):

    def __init__(self):
        super().__init__('ackermann_cmd_converter')
        self.declare_parameter('in_topic', '/cmd_vel')
        self.declare_parameter('out_topic', '/cmd_vel_raw')
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('timeout_s', 0.5)
        # twist_cmd_node mapping: throttle PWM = int(linear.x * 85 + 90)
        self.declare_parameter('pwm_neutral', 90.0)
        self.declare_parameter('pwm_per_unit', 85.0)
        # measured steady speed per PWM level (forward), VICON sysid2
        self.declare_parameter('pwm_levels', [90, 95, 96, 97, 98])
        self.declare_parameter('pwm_speeds', [0.0, 0.094, 0.184, 1.304, 1.842])
        self.declare_parameter('v_min', 0.03)            # below: neutral
        self.declare_parameter('allow_reverse', False)
        self.declare_parameter('reverse_linear_x', -0.12)  # fixed reverse command if allowed (not identified)
        # steering
        self.declare_parameter('L_eff', 0.245)
        self.declare_parameter('K_steer_pos', 0.2495)
        self.declare_parameter('K_steer_neg', 0.3244)
        self.declare_parameter('steer_full_scale', 0.5)

        p = lambda n: self.get_parameter(n).value
        self._levels = [int(x) for x in p('pwm_levels')]
        self._speeds = [float(x) for x in p('pwm_speeds')]
        if len(self._levels) != len(self._speeds) or self._speeds != sorted(self._speeds):
            raise ValueError('pwm_levels/pwm_speeds must have equal length and increasing speeds')
        self._pwm0, self._ppu = float(p('pwm_neutral')), float(p('pwm_per_unit'))
        self._v_min = float(p('v_min'))
        self._allow_rev = bool(p('allow_reverse'))
        self._rev_lx = float(p('reverse_linear_x'))
        self._L = float(p('L_eff'))
        self._k_pos, self._k_neg = float(p('K_steer_pos')), float(p('K_steer_neg'))
        self._s_full = float(p('steer_full_scale'))
        self._timeout = float(p('timeout_s'))

        self._v = self._w = 0.0
        self._last_rx = None
        self._sigma = 0.0            # sigma-delta error accumulator [m/s * steps]

        self._pub = self.create_publisher(TwistStamped, p('out_topic'), 10)
        self.create_subscription(Twist, p('in_topic'), self._cb, 10)
        self.create_timer(1.0 / float(p('rate_hz')), self._tick)
        self.get_logger().info(f'Nav2 {p("in_topic")} -> {p("out_topic")} (hardware units), '
                               f'PWM speed table {dict(zip(self._levels, self._speeds))}')

    def _cb(self, msg: Twist):
        self._v, self._w = float(msg.linear.x), float(msg.angular.z)
        self._last_rx = self.get_clock().now()

    def _pwm_to_linear_x(self, pwm: int) -> float:
        # centre of the integer bin, so int(lx * 85 + 90) lands exactly on `pwm`
        return (pwm - self._pwm0 + 0.5) / self._ppu if pwm != int(self._pwm0) else 0.0

    def _throttle(self, v: float) -> float:
        if v < -self._v_min:
            return self._rev_lx if self._allow_rev else 0.0
        if v < self._v_min:
            self._sigma = 0.0
            return 0.0
        v = min(v, self._speeds[-1])
        i = bisect.bisect_left(self._speeds, v)
        if self._speeds[i] == v:
            self._sigma = 0.0
            return self._pwm_to_linear_x(self._levels[i])
        lo, hi = i - 1, i
        # sigma-delta between neighbouring levels: the mean output speed tracks v
        self._sigma += v - self._speeds[lo]
        if self._sigma >= self._speeds[hi] - self._speeds[lo]:
            self._sigma -= self._speeds[hi] - self._speeds[lo]
            return self._pwm_to_linear_x(self._levels[hi])
        return self._pwm_to_linear_x(self._levels[lo])

    def _steering(self, v: float, w: float) -> float:
        if abs(v) < self._v_min:
            return 0.0
        delta = math.atan(self._L * w / v)
        s = delta / (self._k_pos if delta >= 0.0 else self._k_neg)
        return max(-1.0, min(1.0, s)) * self._s_full

    def _tick(self):
        stale = (self._last_rx is None or
                 (self.get_clock().now() - self._last_rx).nanoseconds * 1e-9 > self._timeout)
        v, w = (0.0, 0.0) if stale else (self._v, self._w)
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.twist.linear.x = self._throttle(v)
        out.twist.angular.z = self._steering(v, w)
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = AckermannCmdConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
