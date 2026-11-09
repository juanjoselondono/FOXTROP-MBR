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

Small additions over the plain version (each one can be switched off by parameter;
with all three off the node behaves exactly like the plain version):
- lookahead_interp: the target is the exact point where the path crosses the circle of
    radius Ld, instead of the first waypoint beyond it (no jumps between waypoints).
- curv_span_m > 0: the speed also drops with the PATH curvature between the car and the
    lookahead point, so the car slows down before the curve, not inside it.
- max_kappa_rate > 0: limits how fast kappa may change [1/m/s]. Off by default.

Arrival at the goal:
- The last path segment is extended virtually past the goal, so the lookahead point never
    collapses onto the goal (that made the steering twitch in the last metre).
- Inside slow_down_dist the speed ramps down to v_min and Ld shrinks with it (tighter tracking).
- The run ends when the car is within goal_tolerance of the goal OR when it crosses the
    goal line (perpendicular to the final heading). It can never drive past and circle back.

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
        self.declare_parameter('slow_down_dist', 1.2)     # ramp to v_min when this close to the goal
        # lookahead Ld = ld_min + ld_gain * v, clipped to [ld_min, ld_max]
        self.declare_parameter('ld_min', 0.45)
        self.declare_parameter('ld_gain', 0.5)
        self.declare_parameter('ld_max', 1.5)
        # smoothness options (see module docstring)
        self.declare_parameter('lookahead_interp', True)
        self.declare_parameter('curv_span_m', 0.5)        # 0 = off (speed from the commanded kappa only)
        self.declare_parameter('max_kappa_rate', 0.0)     # [1/m/s], 0 = off
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
        self._interp = bool(p('lookahead_interp'))
        self._curv_span = float(p('curv_span_m'))
        self._kappa_rate = float(p('max_kappa_rate'))
        self._dt = 1.0 / float(p('rate_hz'))
        self._kappa_max = math.tan(min(float(p('K_steer_pos')), float(p('K_steer_neg')))) / float(p('L_eff'))
        self._goal_tol = float(p('goal_tolerance'))
        self._window_m = float(p('search_window_m'))
        self._pose_timeout = float(p('pose_timeout_s'))
        self._abort_btn = int(p('abort_button'))

        self._path = []        # [(x, y)] in map
        self._s = []           # cumulative arc length
        self._idx = 0          # nearest path index (monotonic)
        self._goal_dir = None  # unit vector of the final path heading
        self._kpath = []       # |path curvature| at each waypoint (empty = option off)
        self._kappa_prev = 0.0 # last commanded kappa (rate limiter)

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
        k = len(pts) - 1                          # final heading (unit vector) from the last distinct points
        while k > 0 and math.dist(pts[k - 1], pts[-1]) < 0.05:
            k -= 1
        d = math.dist(pts[max(k - 1, 0)], pts[-1])
        self._goal_dir = ((pts[-1][0] - pts[k - 1][0]) / d, (pts[-1][1] - pts[k - 1][1]) / d) if k > 0 else None
        self._kpath = self._path_curvature(pts, s) if self._curv_span > 0.0 else []
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

    def _path_curvature(self, pts, s):
        """|curvature| at every waypoint, from the points ~curv_span_m behind and ahead of it
        (Menger curvature). The span keeps A* grid corners from looking infinitely sharp."""
        n, span = len(pts), self._curv_span
        k = [0.0] * n
        a = b = 0
        for i in range(n):
            while s[i] - s[a + 1] >= span:
                a += 1
            b = max(b, i)
            while b + 1 < n and s[b] - s[i] < span:
                b += 1
            if a == i or b == i:
                continue                          # path ends: no estimate
            (ax, ay), (ix, iy), (bx, by) = pts[a], pts[i], pts[b]
            cross = abs((ix - ax) * (by - ay) - (iy - ay) * (bx - ax))
            den = math.hypot(ix - ax, iy - ay) * math.hypot(bx - ix, by - iy) * math.hypot(bx - ax, by - ay)
            k[i] = min(self._kappa_max, 2.0 * cross / den) if den > 1e-9 else self._kappa_max
        return k

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
        gdist = math.hypot(gx - x, gy - y)
        if remaining < self._goal_tol + 0.2:
            # along > 0: the car has crossed the goal line -> stop now, never circle back
            along = ((x - gx) * self._goal_dir[0] + (y - gy) * self._goal_dir[1]) if self._goal_dir else -1.0
            if gdist < self._goal_tol or along >= 0.0:
                self._finish(f'>>> Goal reached <<< ({gdist:.2f} m from the goal)')
                return

        # speed cap near the goal; the lookahead follows it (shorter Ld = tighter tracking)
        v_cap = self._v_nom
        if remaining < self._slow:
            # reaches v_min at goal_tolerance from the goal (where the run ends), not at the goal itself
            frac = (remaining - self._goal_tol) / max(self._slow - self._goal_tol, 1e-3)
            v_cap = self._v_min + (self._v_nom - self._v_min) * max(0.0, frac)
        ld = min(self._ld_max, max(self._ld_min, self._ld_min + self._ld_gain * v_cap))
        j = self._idx
        while j + 1 < len(path) and math.hypot(path[j][0] - x, path[j][1] - y) < ld:
            j += 1
        lx, ly = path[j]
        if self._interp and j > self._idx:
            # path[j-1] is inside the circle of radius ld and path[j] may be outside:
            # move the target to the exact crossing point on that segment
            ax, ay = path[j - 1]
            ux, uy = lx - ax, ly - ay
            fx, fy = ax - x, ay - y
            qa = ux * ux + uy * uy
            qb = 2.0 * (fx * ux + fy * uy)
            qc = fx * fx + fy * fy - ld * ld
            if qa > 1e-12 and qc < 0.0:
                t = (-qb + math.sqrt(qb * qb - 4.0 * qa * qc)) / (2.0 * qa)
                if t < 1.0:                       # t >= 1: the path ends inside the circle
                    lx, ly = ax + t * ux, ay + t * uy
        if self._interp and self._goal_dir and j == len(path) - 1 and gdist < ld:
            # goal inside the circle: aim at the point at distance ld on the extension of the
            # last segment, so the car arrives aligned instead of chasing a point at its nose
            ux, uy = self._goal_dir
            qb = (gx - x) * ux + (gy - y) * uy
            t = -qb + math.sqrt(qb * qb - (gdist * gdist - ld * ld))
            lx, ly = gx + t * ux, gy + t * uy

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
        if self._kappa_rate > 0.0:
            step = self._kappa_rate * self._dt
            kappa = self._kappa_prev + max(-step, min(step, kappa - self._kappa_prev))
        self._kappa_prev = kappa

        # speed: slower in tight curves and near the goal, never below v_min
        k_speed = max([abs(kappa)] + self._kpath[self._idx:j + 1])   # worst curvature up to the lookahead
        v = self._v_nom * (1.0 - 0.5 * k_speed / self._kappa_max)
        v = min(v, v_cap)
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