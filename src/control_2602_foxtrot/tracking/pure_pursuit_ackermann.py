#!/usr/bin/env python3
"""
pure_pursuit_ackermann.py
=========================
Simple Pure Pursuit for the REAL Ackermann car.

  /plan (nav_msgs/Path, frame map, from astar_planner.py)
  TF map -> base_link (AMCL or slam_toolbox + robot_localization EKF)
      --> Twist (v [m/s], omega [rad/s]) on cmd_topic (/cmd_vel_pp)
      --> ackermann_cmd_converter.py --> /cmd_vel_raw (hardware units) --> AEB --> car

Pure pursuit: the lookahead point (x_l, y_l) in base_link gives the arc curvature
    kappa = 2 * y_l / Ld^2           omega = v * kappa
and the converter turns it into steering with the bicycle model,
delta = atan(L * omega / v) = atan(L * kappa), i.e. the classic PP steering law.

Real-car limits (sysid2, same values as the converter):
- kappa is clipped to tan(K_steer_pos) / L_eff (~1.04 1/m, R_min ~0.96 m):
    the tighter (right-hand) steering lock is the binding one.
- forward only (reverse is not identified).
- v in [v_min, v_nominal]; v_min ~0.2 m/s is the lowest speed the ESC holds.

Safety:
- Publishes ONLY while tracking. When it stops publishing, the converter sends one
    neutral command and goes silent, so the joystick teleop (also on /cmd_vel_raw) works.
- Pressing the abort button (default 5 = RB) drops the path. Button 4 (LB) is reserved
    for the hardware dead-man switch to maintain teleop hardware override.
- No pose (TF missing or older than pose_timeout_s): stop publishing -> converter neutral.
"""

import math

import rclpy
import rclpy.time
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64
from tf2_ros import Buffer, TransformListener


class PurePursuitAckermann(Node):

    def __init__(self):
        super().__init__('pure_pursuit_ackermann')
        self.declare_parameter('path_topic', '/plan')
        self.declare_parameter('cmd_topic', '/cmd_vel_pp')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('rate_hz', 20.0)
        # speed
        self.declare_parameter('v_nominal', 0.5)
        self.declare_parameter('v_min', 0.2)
        self.declare_parameter('slow_down_dist', 0.8)     # ramp to v_min when this close to the goal
        # lookahead Ld = ld_min + ld_gain * v, clipped to [ld_min, ld_max]
        self.declare_parameter('ld_min', 0.6)
        self.declare_parameter('ld_gain', 0.6)
        self.declare_parameter('ld_max', 1.5)
        # vehicle (identified, sysid2)
        self.declare_parameter('L_eff', 0.245)
        self.declare_parameter('K_steer_pos', 0.2495)
        self.declare_parameter('K_steer_neg', 0.3244)
        # termination / safety
        self.declare_parameter('goal_tolerance', 0.3)
        self.declare_parameter('search_window_m', 2.0)   # nearest-point search ahead of the last index
        self.declare_parameter('pose_timeout_s', 0.5)
        self.declare_parameter('abort_button', 5)        # 5 = RB (4 is the hardware dead-man switch)

        p = lambda n: self.get_parameter(n).value
        self._map, self._base = p('map_frame'), p('base_frame')
        self._v_nom, self._v_min = float(p('v_nominal')), float(p('v_min'))
        self._slow = float(p('slow_down_dist'))
        self._ld_min, self._ld_gain, self._ld_max = float(p('ld_min')), float(p('ld_gain')), float(p('ld_max'))
        self._kappa_max = math.tan(min(float(p('K_steer_pos')), float(p('K_steer_neg')))) / float(p('L_eff'))
        self._goal_tol = float(p('goal_tolerance'))
        self._window_m = float(p('search_window_m'))
        self._pose_timeout = float(p('pose_timeout_s'))
        self._abort_btn = int(p('abort_button'))

        self._path = []        # [(x, y)] in map
        self._s = []           # cumulative arc length
        self._idx = 0          # nearest path index (monotonic)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Path, p('path_topic'), self._path_cb, 10)
        self.create_subscription(Joy, '/joy', self._joy_cb, 10)
        self._cmd_pub = self.create_publisher(Twist, p('cmd_topic'), 10)
        self._la_pub = self.create_publisher(PointStamped, '/pp/lookahead_point', 10)
        self._cte_pub = self.create_publisher(Float64, '/pp/cross_track_error', 10)
        self.create_timer(1.0 / float(p('rate_hz')), self._loop)
        self.get_logger().info(
            f'Pure Pursuit Ackermann: {p("path_topic")} -> {p("cmd_topic")}, v {self._v_min}-{self._v_nom} m/s, '
            f'kappa_max {self._kappa_max:.2f} 1/m (R_min {1.0 / self._kappa_max:.2f} m)')

    # ── inputs ──
    def _path_cb(self, msg: Path):
        if msg.header.frame_id and msg.header.frame_id != self._map:
            self.get_logger().error(f'Path in frame "{msg.header.frame_id}", expected "{self._map}". Ignored.')
            return
        pts = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        if len(pts) < 2:
            self.get_logger().warn('Path with fewer than 2 points. Ignored.')
            return
        s = [0.0]
        for a, b in zip(pts, pts[1:]):
            s.append(s[-1] + math.dist(a, b))
        self._path, self._s, self._idx = pts, s, 0
        self.get_logger().info(f'Path received: {len(pts)} points, {s[-1]:.2f} m.')

    def _joy_cb(self, msg: Joy):
        if (self._path and 0 <= self._abort_btn < len(msg.buttons) and msg.buttons[self._abort_btn]):
            self._path = []
            self.get_logger().warn('Joystick override button pressed: path dropped, teleop has control.')

    # ── helpers ──
    def _pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self._map, self._base, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f'No TF {self._map} -> {self._base}: {e}', throttle_duration_sec=2.0)
            return None
        age = (self.get_clock().now() - rclpy.time.Time.from_msg(tf.header.stamp)).nanoseconds * 1e-9
        if age > self._pose_timeout:
            self.get_logger().warn(f'TF {self._map} -> {self._base} is {age:.2f} s old.', throttle_duration_sec=2.0)
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return t.x, t.y, yaw

    def _finish(self, reason):
        self._path = []
        self._cmd_pub.publish(Twist())        # one neutral command; the converter then goes silent
        self.get_logger().info(reason)

    # ── control loop ──
    def _loop(self):
        if not self._path:
            return
        pose = self._pose()
        if pose is None:
            return                            # stop publishing -> converter times out to neutral
        x, y, yaw = pose
        path, s = self._path, self._s

        # nearest point, searched forward only (A* lap paths overlap themselves)
        hi = self._idx
        while hi + 1 < len(path) and s[hi + 1] - s[self._idx] <= self._window_m:
            hi += 1
        self._idx = min(range(self._idx, hi + 1), key=lambda i: math.hypot(path[i][0] - x, path[i][1] - y))

        remaining = s[-1] - s[self._idx]
        gx, gy = path[-1]
        if remaining < self._goal_tol + 0.2 and math.hypot(gx - x, gy - y) < self._goal_tol:
            self._finish('>>> Goal reached <<<')
            return

        ld = min(self._ld_max, max(self._ld_min, self._ld_min + self._ld_gain * self._v_nom))
        j = self._idx
        while j + 1 < len(path) and math.hypot(path[j][0] - x, path[j][1] - y) < ld:
            j += 1
        lx, ly = path[j]

        # lookahead point in base_link
        c, sn = math.cos(yaw), math.sin(yaw)
        dx, dy = lx - x, ly - y
        xl, yl = c * dx + sn * dy, -sn * dx + c * dy
        d2 = xl * xl + yl * yl
        if xl <= 0.0:                          # target behind: turn at full lock towards it
            kappa = math.copysign(self._kappa_max, yl if yl != 0.0 else 1.0)
        else:
            kappa = 2.0 * yl / max(d2, 1e-6)
        kappa = max(-self._kappa_max, min(self._kappa_max, kappa))

        # speed: slower in tight curves and near the goal, never below v_min
        v = self._v_nom * (1.0 - 0.5 * abs(kappa) / self._kappa_max)
        if remaining < self._slow:
            v = min(v, self._v_min + (self._v_nom - self._v_min) * remaining / self._slow)
        v = max(self._v_min, v)

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = v * kappa
        self._cmd_pub.publish(cmd)

        # debug: lookahead point and signed cross-track error to the nearest segment
        la = PointStamped()
        la.header.frame_id = self._map
        la.header.stamp = self.get_clock().now().to_msg()
        la.point.x, la.point.y = lx, ly
        self._la_pub.publish(la)
        i = min(self._idx, len(path) - 2)
        (ax, ay), (bx, by) = path[i], path[i + 1]
        seg = math.hypot(bx - ax, by - ay)
        if seg > 1e-6:
            self._cte_pub.publish(Float64(data=((bx - ax) * (y - ay) - (by - ay) * (x - ax)) / seg))

        self.get_logger().info(
            f'v {v:.2f} m/s | kappa {kappa:+.2f} 1/m | Ld {math.sqrt(d2):.2f} m | '
            f'idx {self._idx}/{len(path) - 1} | remaining {remaining:.2f} m', throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitAckermann()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()