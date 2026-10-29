#!/usr/bin/env python3
"""
speed_estimator_node.py
=======================
Forward-speed + heading estimator for the encoder-less RC car.

Longitudinal (per dt, normalized command u = linear.x / 0.5 -- K and u_dead
were identified in NORMALIZED units, comparing them against raw linear.x puts
every command below 0.144 raw, i.e. every FTG speed, inside the dead zone):
  forward drive  u > u_dead_fwd        first order K_fwd, tau_fwd, slowed while
                                        turning by (1 - cornering_drag * |s|)
  ESC brake      u < 0 while rolling   v *= exp(-dt / tau_brake)
  reverse drive  u < -u_dead_rev       first order K_rev, tau_rev
  neutral        otherwise             constant deceleration decel_neutral
Safeguards:
  1. Impact ZUPT  -- in the dead zone, a longitudinal decel below
     -impact_decel_threshold clamps v to 0 and latches until the command
     leaves the dead zone. Needs the IMU accel bias, learned at rest.
  2. Hard cap     -- |v| <= max_speed_cap on the INTERNAL model state as well
     as the output (capping only the output still lets the hidden state wind
     up, and the late stop comes right back).
  3. Heading      -- yaw = complementary filter: integrate the yaw rate
     (Ackermann by default, or the IMU gyro) and pull toward the magnetometer
     heading with time constant tau_mag. Mag samples whose field strength is
     off by > mag_norm_tol from the at-rest reference (motor/battery
     interference) are rejected.
  4. Steering curve -- s = cmd^3 (or cmd*|cmd|) before K_steer.
Transient-gated IMU fusion of the speed (weight -> 0 while cruising) is kept.

Steering comes from /cmd_vel_stamped angular.z -- what the servo receives. The
raw /joy axis is NOT used: under FTG / manual_cruise / throttle_profile it sits
at a stale +1.0 and would make the estimate turn in circles on a straight.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu, MagneticField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, TransformStamped
from tf2_ros import TransformBroadcaster

GRAVITY = 9.81


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class SpeedEstimatorNode(Node):

    # ══════════════════════════ TUNABLE PARAMETERS ══════════════════════════
    # Every entry is a ROS parameter (override in YAML or the launch file).
    PARAMS = {
        # ── command normalization (twist_cmd_node mapping) ───────────────────
        'u_full_scale':     0.5,      # linear.x that is 100% ESC
        'steer_full_scale': 0.5,      # angular.z that is full lock
        'linear_x_min':    -0.2353,   # throttle PWM clamp 70
        'linear_x_max':     0.4941,   # throttle PWM clamp 132
        # ── longitudinal model (identified, sysid2 + VICON) ──────────────────
        'K_fwd':      5.18,          # [m/s per unit u]   val RMSE 0.33 m/s
        'tau_fwd':    0.2,           # [s]
        'u_dead_fwd': 0.144,          # [normalized]
        'K_rev':      4.04,           # ONE reverse episode -> low confidence
        'tau_rev':    0.53,
        'u_dead_rev': 0.144,
        'tau_brake':  0.33,           # ESC brake while rolling forward (2 events)
        'stop_speed': 0.05,           # [m/s] below: negative cmd = reverse, not brake
        'decel_neutral': 0.572,       # [m/s^2] coast decel (18 events)
        'cornering_drag': 0.32,       # forward gain *= 1 - drag*|s|; fit on train figure-eights (val RMSE 0.30 m/s)
        # ── safeguard 1: impact ZUPT ─────────────────────────────────────────
        'impact_decel_threshold': 1.5,  # [m/s^2] >2.5x the coast decel
        'zupt_speed': 0.05,             # in dead zone |v| below this -> 0
        # ── safeguard 2: hard cap ────────────────────────────────────────────
        'max_speed_cap': 2.0,         # [m/s] (VICON saw ~3.6 m/s at u=0.30)
        # ── safeguard 3: heading complementary filter ────────────────────────
        'yaw_rate_source': 'ackermann',  # 'ackermann' or 'gyro'
        'use_mag': True,
        'tau_mag': 2.0,               # [s] mag correction time constant
        'mag_sign': 1.0,              # -1.0 if mag heading turns opposite to yaw
        'mag_offset_x': 0.0,          # hard-iron calibration
        'mag_offset_y': 0.0,
        'mag_norm_tol': 0.25,         # reject |B| off by > 25% of the rest reference
        # ── safeguard 4: steering curve + lateral model ──────────────────────
        'steer_curve': 'cube',        # 'cube', 'square' (s*|s|) or 'linear'
        'L_eff': 0.245,               # [m] measured
        'K_steer_pos': 0.2495,        # [rad] at s=+1   val RMSE 0.164 rad/s
        'K_steer_neg': 0.3244,        # [rad] at s=-1
        # ── IMU speed fusion (transient gated) ───────────────────────────────
        'use_imu': True,
        'imu_accel_axis': 0,          # IMU axis pointing to the car's nose
        'imu_accel_sign': 1.0,
        'imu_accel_scale': 1.0,       # 9.81 if the driver reports g
        'imu_gyro_axis': 2,
        'imu_gyro_sign': 1.0,
        'imu_lpf_tau': 0.1,           # [s]
        'imu_timeout_s': 0.3,
        'bias_settle_s': 0.5,         # rest time before bias learning
        'bias_tau': 1.0,              # [s]
        'w_imu_max': 0.5,
        'accel_noise': 0.8,           # [m/s^2] |a| below -> weight 0
        'accel_full': 3.0,            # [m/s^2] |a| -> w_imu_max
        'dudt_full': 1.0,             # [1/s]
        'tau_release': 0.3,           # [s]
        'dt': 0.05,                   # [s] 20 Hz
    }
    # ════════════════════════════════════════════════════════════════════════

    def __init__(self, **kwargs):
        super().__init__('speed_estimator_node', **kwargs)
        for name, default in self.PARAMS.items():
            self.declare_parameter(name, default)
        c = {name: self.get_parameter(name).value for name in self.PARAMS}
        self.c = c
        dt = self._dt = float(c['dt'])

        self._a_fwd = math.exp(-dt / c['tau_fwd'])
        self._a_rev = math.exp(-dt / c['tau_rev'])
        self._a_brake = math.exp(-dt / c['tau_brake'])
        self._a_release = math.exp(-dt / c['tau_release'])
        self._k_mag = dt / (c['tau_mag'] + dt)
        self._k_bias = 1.0 - math.exp(-dt / c['bias_tau'])

        # state
        self._v_int = 0.0          # model state (capped)
        self._v = 0.0              # fused output
        self._w = 0.0
        self._impact_latched = False
        self._u_prev = 0.0
        self._x = self._y = self._yaw = 0.0
        self._lin_x = self._ang_z = 0.0

        # IMU
        self._a_lpf = 0.0
        self._gyro = 0.0
        self._a_norm = GRAVITY
        self._imu_last = None
        self._acc_bias = self._gyro_bias = 0.0
        self._bias_ready = False
        self._still_time = 0.0

        # magnetometer
        self._mag_heading = None   # latest calibrated heading [rad]
        self._mag_norm = None
        self._mag_norm_ref = None
        self._mag_ref = None       # heading offset aligning mag to the odom frame

        self.create_subscription(TwistStamped, '/cmd_vel_stamped', self._cmd_vel_cb, 10)
        if c['use_imu'] or c['yaw_rate_source'] == 'gyro':
            self.create_subscription(Imu, '/imu/data_raw', self._imu_cb, qos_profile_sensor_data)
        if c['use_mag']:
            self.create_subscription(MagneticField, '/imu/mag', self._mag_cb, qos_profile_sensor_data)
        self._pub_speed = self.create_publisher(Float32, '/estimated_speed', 10)
        self._pub_weight = self.create_publisher(Float32, '/estimated_speed/imu_weight', 10)
        self._pub_odom = self.create_publisher(Odometry, '/estimated_odom', 10)
        self._tf = TransformBroadcaster(self)
        self.create_timer(dt, self._update)

        self.get_logger().info(
            f'SpeedEstimator: K_fwd={c["K_fwd"]} tau={c["tau_fwd"]}s | cap {c["max_speed_cap"]} m/s | '
            f'impact ZUPT < -{c["impact_decel_threshold"]} m/s^2 | yaw rate: {c["yaw_rate_source"]}'
            f'{" + mag (tau %.1fs)" % c["tau_mag"] if c["use_mag"] else ""} | steer curve: {c["steer_curve"]}')

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _cmd_vel_cb(self, msg: TwistStamped):
        self._lin_x = float(msg.twist.linear.x)
        self._ang_z = float(msg.twist.angular.z)

    def _imu_cb(self, msg: Imu):
        c = self.c
        la, av = msg.linear_acceleration, msg.angular_velocity
        acc = [la.x * c['imu_accel_scale'], la.y * c['imu_accel_scale'], la.z * c['imu_accel_scale']]
        now = self.get_clock().now()
        h = self._dt if self._imu_last is None else max(1e-4, (now - self._imu_last).nanoseconds * 1e-9)
        self._a_lpf += (1.0 - math.exp(-h / c['imu_lpf_tau'])) * (
            c['imu_accel_sign'] * acc[c['imu_accel_axis']] - self._a_lpf)
        self._gyro = c['imu_gyro_sign'] * (av.x, av.y, av.z)[c['imu_gyro_axis']]
        self._a_norm = math.sqrt(sum(a * a for a in acc))
        self._imu_last = now

    def _mag_cb(self, msg: MagneticField):
        c = self.c
        mx = msg.magnetic_field.x - c['mag_offset_x']
        my = msg.magnetic_field.y - c['mag_offset_y']
        self._mag_norm = math.hypot(mx, my)
        self._mag_heading = c['mag_sign'] * math.atan2(my, mx)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _imu_fresh(self) -> bool:
        return (self._imu_last is not None and
                (self.get_clock().now() - self._imu_last).nanoseconds * 1e-9 <= self.c['imu_timeout_s'])

    def _accel(self):
        """Bias-corrected longitudinal accel, or None if not trustworthy yet."""
        if self.c['use_imu'] and self._bias_ready and self._imu_fresh():
            return self._a_lpf - self._acc_bias
        return None

    def _steer(self) -> float:
        s = clip(self._ang_z / self.c['steer_full_scale'], -1.0, 1.0)
        curve = self.c['steer_curve']
        if curve == 'cube':
            return s ** 3
        if curve == 'square':
            return s * abs(s)
        return s

    def _model_step(self, u: float, s: float) -> float:
        c, v = self.c, self._v_int
        if u > c['u_dead_fwd']:
            gain = c['K_fwd'] * (1.0 - c['cornering_drag'] * abs(s))
            return self._a_fwd * v + gain * (1.0 - self._a_fwd) * (u - c['u_dead_fwd'])
        if u < 0.0 and v > c['stop_speed']:
            return v * self._a_brake
        if u < -c['u_dead_rev']:
            return self._a_rev * v + c['K_rev'] * (1.0 - self._a_rev) * (u + c['u_dead_rev'])
        step = c['decel_neutral'] * self._dt
        return max(0.0, v - step) if v > 0.0 else min(0.0, v + step)

    def _update_rest_calibration(self, stopped: bool):
        """Learn accel/gyro bias and the mag field reference only while stopped."""
        if not (stopped and self._imu_fresh()):
            self._still_time = 0.0
            return
        self._still_time += self._dt
        if self._still_time < self.c['bias_settle_s']:
            return
        if not self._bias_ready:
            self._acc_bias, self._gyro_bias, self._bias_ready = self._a_lpf, self._gyro, True
            self.get_logger().info(f'IMU calibrated at rest: accel bias {self._acc_bias:+.3f} m/s^2, '
                                   f'gyro bias {self._gyro_bias:+.4f}, |a| {self._a_norm:.2f}')
            if not 7.0 < self._a_norm < 12.5:
                self.get_logger().warn('|a| at rest is not ~9.81 m/s^2: check imu_accel_scale')
        else:
            self._acc_bias += self._k_bias * (self._a_lpf - self._acc_bias)
            # A real rotation at rest (car turned by hand) is not bias: don't absorb it.
            if abs(self._gyro - self._gyro_bias) < 0.1:
                self._gyro_bias += self._k_bias * (self._gyro - self._gyro_bias)
        if self._mag_norm is not None and self._mag_norm_ref is None:
            self._mag_norm_ref = self._mag_norm

    def _mag_valid(self) -> bool:
        if self._mag_heading is None:
            return False
        if self._mag_norm_ref is None:   # no rest reference yet: accept, nothing to compare
            return True
        return abs(self._mag_norm - self._mag_norm_ref) <= self.c['mag_norm_tol'] * self._mag_norm_ref

    # ── Main loop ────────────────────────────────────────────────────────────

    def _update(self):
        c, dt = self.c, self._dt
        u = clip(self._lin_x, c['linear_x_min'], c['linear_x_max']) / c['u_full_scale']
        s = self._steer()
        in_dz = -c['u_dead_rev'] <= u <= c['u_dead_fwd']
        a = self._accel()
        cap = c['max_speed_cap']

        # model (safeguard 2: cap the internal state, not just the output)
        self._v_int = clip(self._model_step(u, s), -cap, cap)
        v_model = self._v_int

        # safeguard 1: impact ZUPT
        if not in_dz:
            self._impact_latched = False
        elif a is not None and a < -c['impact_decel_threshold'] and self._v_int > 0.0:
            self._impact_latched = True
        if self._impact_latched:
            self._v_int = v_model = 0.0

        # transient-gated IMU fusion (never inside the dead zone)
        if in_dz or a is None:
            self._w = 0.0
            v = v_model
        else:
            s_acc = clip((abs(a) - c['accel_noise']) / (c['accel_full'] - c['accel_noise']), 0.0, 1.0)
            s_cmd = clip(abs(u - self._u_prev) / dt / c['dudt_full'], 0.0, 1.0)
            self._w = max(c['w_imu_max'] * max(s_acc, s_cmd), self._w * self._a_release)
            v = (1.0 - self._w) * v_model + self._w * (self._v + a * dt)
        if v_model > 0.0:
            v = max(v, 0.0)          # never fight the commanded direction of travel
        elif v_model < 0.0:
            v = min(v, 0.0)
        if in_dz and abs(v) < c['zupt_speed']:
            v = self._v_int = 0.0
        v = clip(v, -cap, cap)
        self._v, self._u_prev = v, u
        self._update_rest_calibration(stopped=(in_dz and v == 0.0))

        # safeguard 3 + 4: heading complementary filter
        k_steer = c['K_steer_pos'] if s >= 0.0 else c['K_steer_neg']
        omega_ack = (v / c['L_eff']) * math.tan(k_steer * s)
        if c['yaw_rate_source'] == 'gyro' and self._bias_ready and self._imu_fresh():
            omega = self._gyro - self._gyro_bias
        else:
            omega = omega_ack
        self._yaw += omega * dt
        if c['use_mag'] and self._mag_valid():
            if self._mag_ref is None:
                self._mag_ref = wrap(self._mag_heading - self._yaw)
            mag_yaw = self._mag_heading - self._mag_ref
            self._yaw += self._k_mag * wrap(mag_yaw - self._yaw)

        self._x += v * math.cos(self._yaw) * dt
        self._y += v * math.sin(self._yaw) * dt
        self._publish(v, omega)

    def _publish(self, v: float, omega: float):
        now = self.get_clock().now().to_msg()
        qz, qw = math.sin(self._yaw / 2.0), math.cos(self._yaw / 2.0)
        self._pub_speed.publish(Float32(data=float(v)))
        self._pub_weight.publish(Float32(data=float(self._w)))

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = float(self._x)
        odom.pose.pose.position.y = float(self._y)
        odom.pose.pose.orientation.z = float(qz)
        odom.pose.pose.orientation.w = float(qw)
        odom.twist.twist.linear.x = float(v)
        odom.twist.twist.angular.z = float(omega)
        self._pub_odom.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x = float(self._x)
        tf.transform.translation.y = float(self._y)
        tf.transform.rotation.z = float(qz)
        tf.transform.rotation.w = float(qw)
        self._tf.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = SpeedEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
