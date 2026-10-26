#!/usr/bin/env python3
"""
throttle_profile_node.py
=========================
Automated throttle-step sequencer for sysid Session A data collection.

Publishes TwistStamped commands on /cmd_vel_stamped so you don't have to
drive the throttle-level pattern by hand with the joystick:

    for each level in throttle_levels_pct:
        repeat `repeats` times:
            hold_duration seconds at that throttle level
            neutral_duration seconds at neutral (0.0)

Steering (angular.z) is always held at 0.0 (straight line), matching the
"5+ meter straight corridor" setup from the sysid procedure.

SAFETY
  This bypasses the joystick deadman switch entirely -- once started, the
  car WILL move on its own. Keep the corridor clear and be ready to
  Ctrl+C this node (which immediately starts publishing neutral, and lets
  the hardware watchdog on the RPi take over) or cut power.

  Between levels the node pauses for `pause_duration` seconds (logging a
  countdown) so you can walk the car back to the start of the corridor.
  NOTE: this is a timed pause, not a keypress -- `ros2 launch` does not
  forward terminal stdin to child processes, so waiting on Enter would
  hang forever under launch. Set interactive_pause:=false to skip the
  pause entirely and run the whole sequence unattended.

  Joystick E-STOP: holding/pressing the controller's B button (/joy
  buttons[brake_button_index], default index 1) immediately forces
  neutral and permanently aborts the rest of the sequence -- restart the
  node to run again. This needs `joy_node` running (NOT teleop_node --
  that would fight this node over /cmd_vel_stamped); the bundled launch
  file below starts joy_node for you.

Usage:
    ros2 run sysid_2602_foxtrot throttle_profile_node
    ros2 run sysid_2602_foxtrot throttle_profile_node --ros-args \\
        -p throttle_levels_pct:="[20.0, 50.0, 70.0]" \\
        -p hold_duration:=3.0 -p neutral_duration:=3.0 -p repeats:=1

Or via launch file:
    ros2 launch sysid_2602_foxtrot throttle_profile.launch.py
"""

import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Joy


class ThrottleProfileNode(Node):
    """Publishes a scripted throttle-step pattern on /cmd_vel_stamped."""

    def __init__(self):
        super().__init__('throttle_profile_node')

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter('throttle_levels_pct', [35.0, 70.0, 100.0])
        self.declare_parameter('hold_duration', 10.0)
        self.declare_parameter('neutral_duration', 3.0)
        self.declare_parameter('repeats', 1)
        # 0.11 m/s == velocidad_maxima in ftg_advanced.py: sysid must stay
        # inside the speed envelope the FTG controller will actually command.
        self.declare_parameter('max_linear_x', 0.25)
        self.declare_parameter('publish_rate', 50.0)  # Hz
        self.declare_parameter('start_delay', 3.0)    # s, time to start `ros2 bag record` first
        self.declare_parameter('interactive_pause', True)
        self.declare_parameter('pause_duration', 7.0)  # s, countdown between levels (no stdin needed)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('brake_button_index', 1)  # Xbox/Logitech-style pad: 1 == B

        self._levels_pct = [float(v) for v in self.get_parameter('throttle_levels_pct').value]
        self._hold = float(self.get_parameter('hold_duration').value)
        self._neutral = float(self.get_parameter('neutral_duration').value)
        self._repeats = int(self.get_parameter('repeats').value)
        self._max_linear_x = float(self.get_parameter('max_linear_x').value)
        self._rate_hz = float(self.get_parameter('publish_rate').value)
        self._start_delay = float(self.get_parameter('start_delay').value)
        self._interactive = bool(self.get_parameter('interactive_pause').value)
        self._pause_duration = float(self.get_parameter('pause_duration').value)
        self._frame_id = str(self.get_parameter('frame_id').value)
        self._brake_button_index = int(self.get_parameter('brake_button_index').value)

        self._pub = self.create_publisher(TwistStamped, '/cmd_vel_stamped', 10)
        self._sub_joy = self.create_subscription(Joy, '/joy', self._joy_cb, 10)

        # current command, continuously republished by the timer so the
        # RPi watchdog never sees a gap > 0.5s while the sequence runs
        self._linear_x = 0.0
        self._braked = False
        self._lock = threading.Lock()
        self._timer = self.create_timer(1.0 / self._rate_hz, self._publish_tick)

        self.get_logger().info(
            f'Throttle profile ready: levels={self._levels_pct}% '
            f'hold={self._hold}s neutral={self._neutral}s repeats={self._repeats} '
            f'interactive_pause={self._interactive} max_linear_x={self._max_linear_x} '
            f'brake_button_index={self._brake_button_index}'
        )

        self._seq_thread = threading.Thread(target=self._run_sequence, daemon=True)
        self._seq_thread.start()

    def _joy_cb(self, msg: Joy):
        if len(msg.buttons) <= self._brake_button_index:
            return
        if msg.buttons[self._brake_button_index] and not self._braked:
            self.get_logger().error('BRAKE BUTTON PRESSED: forcing neutral, aborting sequence.')
            with self._lock:
                self._braked = True
                self._linear_x = 0.0

    def _publish_tick(self):
        with self._lock:
            braked = self._braked
            x = 0.0 if braked else self._linear_x
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.twist.linear.x = x
        msg.twist.angular.z = 0.0
        self._pub.publish(msg)

    def _set_linear_x(self, value: float):
        with self._lock:
            self._linear_x = value

    def _is_braked(self) -> bool:
        with self._lock:
            return self._braked

    def _sleep(self, seconds: float) -> bool:
        """Sleep in short slices so a brake press interrupts promptly.

        Returns True if the sleep completed normally, False if it was cut
        short because the brake button was pressed.
        """
        import time
        poll = 0.05
        elapsed = 0.0
        while elapsed < seconds:
            if self._is_braked():
                return False
            step = min(poll, seconds - elapsed)
            time.sleep(step)
            elapsed += step
        return not self._is_braked()

    def _wait_for_operator(self, prompt: str):
        if not self._interactive:
            return
        # Timed countdown instead of waiting on stdin: `ros2 launch` does
        # not forward terminal input to child processes, so blocking on
        # sys.stdin.readline() here would hang forever under launch.
        self.get_logger().warn(prompt)
        remaining = int(round(self._pause_duration))
        for s in range(remaining, 0, -1):
            if self._is_braked():
                return
            self.get_logger().info(f'  repositioning window: {s}s remaining...')
            if not self._sleep(1.0):
                return

    def _run_sequence(self):
        self.get_logger().warn(
            f'Sequence starts in {self._start_delay:.1f}s -- start `ros2 bag record` NOW if you haven\'t.'
        )
        if not self._sleep(self._start_delay):
            self._abort()
            return

        for level_pct in self._levels_pct:
            if self._is_braked():
                self._abort()
                return

            cmd_x = (level_pct / 100.0) * self._max_linear_x
            self._wait_for_operator(f'About to run level {level_pct:.0f}% (linear.x={cmd_x:.3f} m/s).')
            if self._is_braked():
                self._abort()
                return

            for rep in range(1, self._repeats + 1):
                self.get_logger().info(
                    f'[{level_pct:.0f}%] rep {rep}/{self._repeats}: THROTTLE ({cmd_x:.3f} m/s) for {self._hold}s'
                )
                self._set_linear_x(cmd_x)
                if not self._sleep(self._hold):
                    self._abort()
                    return

                self.get_logger().info(
                    f'[{level_pct:.0f}%] rep {rep}/{self._repeats}: NEUTRAL for {self._neutral}s'
                )
                self._set_linear_x(0.0)
                if not self._sleep(self._neutral):
                    self._abort()
                    return

        self._set_linear_x(0.0)
        self.get_logger().warn(
            'SEQUENCE COMPLETE. Holding neutral. Stop `ros2 bag record` now (Ctrl+C), '
            'then Ctrl+C this node.'
        )

    def _abort(self):
        self._set_linear_x(0.0)
        self.get_logger().error(
            'Sequence ABORTED by brake button. Holding neutral. Restart the node to run again.'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ThrottleProfileNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Throttle profile node terminating, holding neutral.')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
