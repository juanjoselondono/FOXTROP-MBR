#!/usr/bin/env python3
"""
manual_cruise_node.py
======================
Button-driven "cruise control" teleop for manual sysid / test runs.

Controls (indices are /joy buttons[]/axes[] -- VERIFY with
`ros2 topic echo /joy` and override via parameters, pad layouts vary):
    Up / Down (dpad, see below)           -- dial a CONSTANT speed
                                              setpoint up/down by
                                              speed_step, clamped to
                                              [0, max_speed]. Works at
                                              ANY time, running or
                                              stopped, so you can pick
                                              the target velocity for
                                              the next test before
                                              pressing A.
    A (start_button_index, default 0)     -- engage the currently dialed
                                              setpoint as-is (no ramp,
                                              no reset) and start a new
                                              `ros2 bag record` session.
    B (brake_button_index, default 1)     -- brake: stop moving (publish
                                              forces 0 regardless of the
                                              setpoint) and stop the
                                              active bag recording. The
                                              dialed setpoint is kept, so
                                              pressing A again re-runs
                                              the same speed.

Up/Down can come from either discrete buttons (increase_button_index /
decrease_button_index, default -1 == disabled) or a D-pad hat reported as
an axis (dpad_axis_index, default 7; dpad_up_positive controls sign).
Most XInput-style pads report the D-pad as a hat axis, not buttons.

Steering (angular.z) is always 0.0 (straight line).

SAFETY
  Once A is pressed the car moves on its own at the dialed setpoint and
  keeps moving until B is pressed or the node is killed. Ctrl+C
  immediately stops publishing non-zero speed and lets the RPi hardware
  watchdog (twist_cmd_node.py) force neutral within 0.5s.

Needs joy_node running (NOT teleop_node -- that would fight this node
over /cmd_vel_stamped). The bundled launch file starts joy_node for you.

Usage:
    ros2 launch sysid_2602_foxtrot manual_cruise.launch.py
    ros2 run sysid_2602_foxtrot manual_cruise_node --ros-args \\
        -p max_speed:=0.25 -p start_speed:=0.08 -p speed_step:=0.02
"""

import os
import signal
import subprocess
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Joy


class ManualCruiseNode(Node):
    """Button-driven speed ramp with automatic bag-record start/stop."""

    def __init__(self):
        super().__init__('manual_cruise_node')

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter('max_speed', 0.25)
        self.declare_parameter('start_speed', 0.08)  # initial setpoint only; Up/Down redial it anytime
        self.declare_parameter('speed_step', 0.02)
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('frame_id', 'base_link')

        self.declare_parameter('start_button_index', 0)    # A
        self.declare_parameter('brake_button_index', 1)    # B
        self.declare_parameter('increase_button_index', -1)  # -1 == disabled
        self.declare_parameter('decrease_button_index', -1)  # -1 == disabled
        self.declare_parameter('dpad_axis_index', 7)        # typical vertical hat axis
        self.declare_parameter('dpad_up_positive', True)
        self.declare_parameter('axis_threshold', 0.5)

        self.declare_parameter('record_topics', ['/joy', '/cmd_vel_stamped', '/robot1/pose'])
        self.declare_parameter('bag_output_dir', './bag_files/manual')
        self.declare_parameter('bag_storage', 'mcap')

        self._max_speed = float(self.get_parameter('max_speed').value)
        self._start_speed = float(self.get_parameter('start_speed').value)
        self._speed_step = float(self.get_parameter('speed_step').value)
        self._rate_hz = float(self.get_parameter('publish_rate').value)
        self._frame_id = str(self.get_parameter('frame_id').value)

        self._start_btn = int(self.get_parameter('start_button_index').value)
        self._brake_btn = int(self.get_parameter('brake_button_index').value)
        self._inc_btn = int(self.get_parameter('increase_button_index').value)
        self._dec_btn = int(self.get_parameter('decrease_button_index').value)
        self._dpad_axis = int(self.get_parameter('dpad_axis_index').value)
        self._dpad_up_positive = bool(self.get_parameter('dpad_up_positive').value)
        self._axis_threshold = float(self.get_parameter('axis_threshold').value)

        self._record_topics = list(self.get_parameter('record_topics').value)
        # Resolve to an absolute path NOW, at the cwd ros2 launch/run was
        # actually invoked from -- a relative path here is a common source
        # of "recordings go missing": they still get written, just not
        # where you're looking (e.g. your home dir instead of the repo).
        self._bag_output_dir = os.path.abspath(str(self.get_parameter('bag_output_dir').value))
        self._bag_storage = str(self.get_parameter('bag_storage').value)

        self._pub = self.create_publisher(TwistStamped, '/cmd_vel_stamped', 10)
        self._sub_joy = self.create_subscription(Joy, '/joy', self._joy_cb, 10)

        self._lock = threading.Lock()
        self._running = False
        self._speed = self._start_speed  # initial dialed-in setpoint, adjustable via Up/Down anytime

        self._prev_buttons = []
        self._prev_dpad_sign = 0

        self._bag_proc = None
        self._bag_lock = threading.Lock()

        self._timer = self.create_timer(1.0 / self._rate_hz, self._publish_tick)

        self.get_logger().info(
            f'Manual cruise ready: max_speed={self._max_speed} start_speed={self._start_speed} '
            f'speed_step={self._speed_step} A(start)={self._start_btn} B(brake)={self._brake_btn} '
            f'dpad_axis={self._dpad_axis}'
        )
        self.get_logger().warn(f'Bag recordings will be written under: {self._bag_output_dir}')

    # ── Joy handling ─────────────────────────────────────────────────────

    @staticmethod
    def _rising_edge(prev, cur):
        return cur and not prev

    def _joy_cb(self, msg: Joy):
        buttons = list(msg.buttons)
        prev = self._prev_buttons if len(self._prev_buttons) == len(buttons) else [0] * len(buttons)

        def pressed(idx):
            if idx < 0 or idx >= len(buttons):
                return False
            return self._rising_edge(prev[idx], buttons[idx])

        start_pressed = pressed(self._start_btn)
        brake_pressed = pressed(self._brake_btn)
        inc_pressed = pressed(self._inc_btn)
        dec_pressed = pressed(self._dec_btn)

        # D-pad reported as a hat axis (most XInput-style pads)
        dpad_inc = False
        dpad_dec = False
        if 0 <= self._dpad_axis < len(msg.axes):
            raw = msg.axes[self._dpad_axis]
            sign = 0
            if raw > self._axis_threshold:
                sign = 1
            elif raw < -self._axis_threshold:
                sign = -1
            if sign != 0 and sign != self._prev_dpad_sign:
                up = sign > 0
                if not self._dpad_up_positive:
                    up = not up
                if up:
                    dpad_inc = True
                else:
                    dpad_dec = True
            self._prev_dpad_sign = sign

        self._prev_buttons = buttons

        if brake_pressed:
            self._brake()
            return  # brake takes priority over any same-frame start/adjust

        if start_pressed:
            self._start()

        if inc_pressed or dpad_inc:
            self._adjust_speed(+self._speed_step)

        if dec_pressed or dpad_dec:
            self._adjust_speed(-self._speed_step)

    # ── State transitions ────────────────────────────────────────────────

    def _start(self):
        # Engage the currently dialed-in setpoint as-is -- do NOT reset it
        # to start_speed. Dial the constant speed with Up/Down (works at
        # any time, running or not) *before* pressing A so each test run
        # starts straight at the target velocity instead of ramping up.
        with self._lock:
            self._running = True
            speed = self._speed
        self.get_logger().warn(f'A pressed: RUNNING at constant {speed:.3f} m/s.')
        self._start_recording()

    def _brake(self):
        # Stop moving, but keep the dialed-in setpoint (self._speed)
        # untouched so the same value is ready for the next A press --
        # _publish_tick already forces 0 while not running, so this is
        # not a safety gap.
        with self._lock:
            self._running = False
            speed = self._speed
        self.get_logger().error(f'B pressed: BRAKE. Stopped (setpoint still {speed:.3f} m/s).')
        self._stop_recording()

    def _adjust_speed(self, delta: float):
        # Adjustable at any time (running or stopped) so you can dial in a
        # target speed before pressing A, not just while already moving.
        with self._lock:
            self._speed = max(0.0, min(self._max_speed, self._speed + delta))
            speed = self._speed
            running = self._running
        state = 'live' if running else 'setpoint for next A'
        self.get_logger().info(f'Speed -> {speed:.3f} m/s ({state})')

    def _publish_tick(self):
        with self._lock:
            x = self._speed if self._running else 0.0
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.twist.linear.x = x
        msg.twist.angular.z = 0.0
        self._pub.publish(msg)

    # ── Bag recording ────────────────────────────────────────────────────

    def _start_recording(self):
        with self._bag_lock:
            if self._bag_proc is not None:
                return  # already recording (e.g. A pressed twice without B)
            os.makedirs(self._bag_output_dir, exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_path = os.path.join(self._bag_output_dir, f'manual_session_{stamp}')
            cmd = ['ros2', 'bag', 'record', *self._record_topics,
                   '--storage', self._bag_storage, '--output', out_path]
            self.get_logger().warn(f'Starting bag recording -> {out_path}')
            try:
                # inherit this node's stdout/stderr so `ros2 bag record`'s
                # own [INFO]/[ERROR] lines show up right here, interleaved
                proc = subprocess.Popen(cmd)
            except (OSError, FileNotFoundError) as exc:
                self.get_logger().error(
                    f'Failed to launch `ros2 bag record`: {exc}. '
                    f'Is this shell sourced (install/setup.bash) so `ros2` is on PATH?'
                )
                return
            self._bag_proc = proc

        # give it a moment, then confirm it's actually still alive -- an
        # immediate exit (bad topic type, bad storage plugin, etc.) would
        # otherwise fail completely silently.
        threading.Timer(1.5, self._check_recording_alive, args=(proc, out_path)).start()

    def _check_recording_alive(self, proc, out_path):
        ret = proc.poll()
        if ret is not None:
            self.get_logger().error(
                f'`ros2 bag record` exited immediately (code {ret}) -- nothing was recorded to '
                f'{out_path}. Run the same `ros2 bag record ...` command by hand in a terminal '
                f'to see its error output.'
            )
            with self._bag_lock:
                if self._bag_proc is proc:
                    self._bag_proc = None

    def _stop_recording(self):
        with self._bag_lock:
            proc = self._bag_proc
            self._bag_proc = None
        if proc is None:
            return
        self.get_logger().info('Stopping bag recording...')
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.get_logger().warn('Bag recorder did not exit in time, terminating.')
            proc.terminate()

    def destroy_node(self):
        self._stop_recording()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ManualCruiseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Manual cruise node terminating.')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
