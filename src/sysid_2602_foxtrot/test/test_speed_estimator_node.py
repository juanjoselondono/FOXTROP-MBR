"""Tests for speed_estimator_node.SpeedEstimatorNode, driving its update loop directly."""

import math
import os
import sys

import pytest

rclpy = pytest.importorskip('rclpy')
from rclpy.parameter import Parameter  # noqa: E402
from geometry_msgs.msg import TwistStamped  # noqa: E402
from sensor_msgs.msg import MagneticField  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sysid_2602_foxtrot.speed_estimator_node import SpeedEstimatorNode  # noqa: E402


@pytest.fixture
def make_node():
    rclpy.init()
    nodes = []

    def factory(**overrides):
        n = SpeedEstimatorNode(parameter_overrides=[Parameter(k, value=v) for k, v in overrides.items()])
        nodes.append(n)
        return n

    yield factory
    for n in nodes:
        n.destroy_node()
    rclpy.try_shutdown()


def step(n, lin_x, ang_z=0.0, seconds=None, accel=None, gyro=0.0):
    """Advance the node `seconds` (one dt if None). accel=None means no IMU."""
    m = TwistStamped()
    m.twist.linear.x, m.twist.angular.z = float(lin_x), float(ang_z)
    n._cmd_vel_cb(m)
    for _ in range(1 if seconds is None else int(round(seconds / n._dt))):
        if accel is not None:  # inject an already-filtered IMU sample
            n._a_lpf, n._gyro, n._a_norm = accel, gyro, 9.81
            n._imu_last = n.get_clock().now()
        n._update()
    return n._v


def mag(n, heading, norm=1.0):
    msg = MagneticField()
    msg.magnetic_field.x, msg.magnetic_field.y = norm * math.cos(heading), norm * math.sin(heading)
    n._mag_cb(msg)


def test_raw_ftg_command_is_normalized_not_dead_zoned(make_node):
    n = make_node(use_mag=False)
    assert step(n, 0.09, seconds=10.0) > 0.5   # raw 0.09 = normalized 0.18 > u_dead 0.144


def test_hard_cap_holds_internal_state_and_release_stops_fast(make_node):
    n = make_node(use_mag=False)
    step(n, 0.30, seconds=20.0)
    assert n._v == pytest.approx(2.0) and n._v_int <= 2.0
    t = 0.0
    while step(n, 0.0) > 0.0:
        t += n._dt
    assert t <= 2.0 / 0.572 + 0.2       # no wound-up hidden state


def test_impact_zupt_clamps_on_decel_spike_and_latches(make_node):
    n = make_node(use_mag=False)
    step(n, 0.0, seconds=1.0, accel=0.0)  # rest -> bias calibrated
    assert n._bias_ready
    step(n, 0.09, seconds=8.0, accel=0.0)
    assert step(n, 0.0, accel=-3.0) == 0.0
    assert step(n, 0.0, seconds=1.0, accel=0.0) == 0.0


def test_coast_without_spike_is_not_clamped(make_node):
    n = make_node(use_mag=False)
    step(n, 0.0, seconds=1.0, accel=0.0)
    v0 = step(n, 0.09, seconds=8.0, accel=0.0)
    assert step(n, 0.0, accel=-0.5) == pytest.approx(v0 - 0.572 * n._dt, abs=1e-6)


def test_learned_bias_prevents_false_impact(make_node):
    n = make_node(use_mag=False)
    step(n, 0.0, seconds=1.0, accel=-2.0)          # tilted mount: -2 m/s^2 at rest
    step(n, 0.09, seconds=8.0, accel=-2.0)
    assert step(n, 0.0, accel=-2.0) > 0.0          # -2 is just the bias, not an impact


def test_reverse_generates_negative_speed(make_node):
    n = make_node(use_mag=False)
    v = step(n, -0.30, seconds=10.0)
    assert v == pytest.approx(-4.04 * (0.2353 / 0.5 - 0.144), rel=0.02)


def test_steer_curve_cube(make_node):
    n = make_node(use_mag=False)
    step(n, 0.0, ang_z=0.25)
    assert n._steer() == pytest.approx(0.125)
    step(n, 0.0, ang_z=0.5)
    assert n._steer() == pytest.approx(1.0)


def test_mag_bounds_heading_when_ackermann_rate_is_wrong(make_node):
    free = make_node(use_mag=False)
    anchored = make_node(use_mag=True, tau_mag=1.0)
    mag(anchored, 0.3)  # car truly straight, heading constant
    step(free, 0.09, ang_z=0.25, seconds=20.0)
    step(anchored, 0.09, ang_z=0.25, seconds=20.0)
    assert abs(free._yaw) > 2.0                      # blind integration spins away
    assert abs(anchored._yaw) < 0.35                 # bounded: steady lag ~ omega * tau_mag ~ 0.23 rad


def test_mag_interference_rejected(make_node):
    n = make_node(use_mag=True, tau_mag=0.5)
    mag(n, 0.0, norm=1.0)
    step(n, 0.0, seconds=1.0, accel=0.0)             # rest: learns |B| reference
    assert n._mag_norm_ref == pytest.approx(1.0)
    mag(n, 2.5, norm=2.0)                            # motor interference: |B| doubled
    step(n, 0.0, seconds=3.0, accel=0.0)
    assert abs(n._yaw) < 1e-6


def test_gyro_yaw_rate_source_removes_bias(make_node):
    n = make_node(use_mag=False, yaw_rate_source='gyro')
    step(n, 0.0, seconds=1.0, accel=0.0, gyro=0.1)   # rest: bias 0.1 rad/s
    y0 = n._yaw
    step(n, 0.09, seconds=1.0, accel=0.0, gyro=0.6)  # driving, turning at 0.5 rad/s true
    assert n._yaw - y0 == pytest.approx(0.5, rel=0.05)


def test_rotation_at_rest_is_not_absorbed_as_gyro_bias(make_node):
    n = make_node(use_mag=False, yaw_rate_source='gyro')
    step(n, 0.0, seconds=1.0, accel=0.0, gyro=0.1)
    y0 = n._yaw
    step(n, 0.0, seconds=1.0, accel=0.0, gyro=0.6)   # turned by hand while stopped
    assert n._yaw - y0 == pytest.approx(0.5, rel=0.05)
