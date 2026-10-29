"""
extraction.py
=============
Step 1: batch-read .mcap bags and synchronize topics onto the VICON clock.

Uses the pure-Python `rosbags` library -- no ROS2 install/sourcing needed
to run this offline.

IMPORTANT: on this project's real bags, /robot1/pose is published as a bare
geometry_msgs/msg/Pose (no header/stamp), NOT PoseStamped. `_pose_fields()`
below supports both so this keeps working if that ever changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

logger = logging.getLogger(__name__)


@dataclass
class RawBagData:
    """Per-topic raw samples straight out of the bag, each on its own asynchronous clock."""
    pose_t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    cmd_t: np.ndarray
    u: np.ndarray            # TwistStamped.twist.linear.x  (speed / ESC command)
    steer_cmd: np.ndarray    # TwistStamped.twist.angular.z (commanded yaw rate)
    joy_t: np.ndarray
    joy_throttle: np.ndarray
    joy_steering: np.ndarray

    @property
    def is_usable(self) -> bool:
        return len(self.pose_t) > 2 and len(self.cmd_t) > 1


@dataclass
class SessionData:
    """A fully synchronized, single-clock session (VICON time base), ready for filtering/fitting."""
    name: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    u: np.ndarray
    steer_cmd: np.ndarray
    joy_steering: np.ndarray
    dt: float
    # True where TimeSynchronizer flagged an implausible VICON jump (occlusion/dropout)
    glitch_mask: Optional[np.ndarray] = field(default=None, repr=False)
    # populated later by filtering.SavGolDifferentiator
    vx: Optional[np.ndarray] = field(default=None, repr=False)
    vy: Optional[np.ndarray] = field(default=None, repr=False)
    v: Optional[np.ndarray] = field(default=None, repr=False)
    omega: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def is_filtered(self) -> bool:
        return self.v is not None and self.omega is not None

    def __len__(self):
        return len(self.t)


class MCAPSessionReader:
    """Parses one rosbag2/mcap directory into RawBagData."""

    def __init__(self, topic_pose='/robot1/pose', topic_cmd='/cmd_vel_stamped',
                 topic_joy='/joy', joy_throttle_axis=1, joy_steering_axis=2,
                 position_scale=0.001, typestore_name=Stores.ROS2_JAZZY):
        # axis defaults match bringup_2602_foxtrot/config/joystick.yaml:
        # axis_linear.x: 1, axis_angular.yaw: 2 -- verified against real
        # bags: axes[2] has std=0.78 and range exactly matching
        # cmd_vel_stamped.twist.angular.z / 0.5 (the configured scale_angular).
        #
        # position_scale: VICON Tracker reports position in millimeters by
        # default. Verified against real bags: raw x/y values (~428, ~2474)
        # are only physically sane as mm for a room-scale capture volume
        # (0.43m/2.47m) -- in meters they'd imply an impossibly huge room.
        # 0.001 converts mm -> m; override if your Tracker is configured
        # for meters already.
        self.topic_pose = topic_pose
        self.topic_cmd = topic_cmd
        self.topic_joy = topic_joy
        self.joy_throttle_axis = joy_throttle_axis
        self.joy_steering_axis = joy_steering_axis
        self.position_scale = position_scale
        self._typestore = get_typestore(typestore_name)

    @staticmethod
    def _quat_to_yaw(o) -> float:
        return float(np.arctan2(2.0 * (o.w * o.z + o.x * o.y),
                                 1.0 - 2.0 * (o.y * o.y + o.z * o.z)))

    @staticmethod
    def _pose_fields(msg):
        """Supports both geometry_msgs/Pose and geometry_msgs/PoseStamped."""
        return (msg.pose.position, msg.pose.orientation) if hasattr(msg, 'pose') else (msg.position, msg.orientation)

    @staticmethod
    def _twist_fields(msg):
        """Supports both geometry_msgs/Twist and geometry_msgs/TwistStamped."""
        return msg.twist if hasattr(msg, 'twist') else msg

    def read(self, bag_dir: Path) -> RawBagData:
        pose_rows, cmd_rows, joy_rows = [], [], []

        with Reader(bag_dir) as reader:
            conns_by_topic: dict[str, list] = {}
            for c in reader.connections:
                conns_by_topic.setdefault(c.topic, []).append(c)

            for conn, timestamp, rawdata in reader.messages(connections=conns_by_topic.get(self.topic_pose, [])):
                msg = self._typestore.deserialize_cdr(rawdata, conn.msgtype)
                pos, orient = self._pose_fields(msg)
                pose_rows.append((timestamp * 1e-9, float(pos.x) * self.position_scale,
                                   float(pos.y) * self.position_scale, self._quat_to_yaw(orient)))

            for conn, timestamp, rawdata in reader.messages(connections=conns_by_topic.get(self.topic_cmd, [])):
                msg = self._typestore.deserialize_cdr(rawdata, conn.msgtype)
                tw = self._twist_fields(msg)
                cmd_rows.append((timestamp * 1e-9, float(tw.linear.x), float(tw.angular.z)))

            for conn, timestamp, rawdata in reader.messages(connections=conns_by_topic.get(self.topic_joy, [])):
                msg = self._typestore.deserialize_cdr(rawdata, conn.msgtype)
                axes = msg.axes
                thr = float(axes[self.joy_throttle_axis]) if len(axes) > self.joy_throttle_axis else 0.0
                strg = float(axes[self.joy_steering_axis]) if len(axes) > self.joy_steering_axis else 0.0
                joy_rows.append((timestamp * 1e-9, thr, strg))

        pose_arr = np.array(pose_rows, dtype=float) if pose_rows else np.zeros((0, 4))
        cmd_arr = np.array(cmd_rows, dtype=float) if cmd_rows else np.zeros((0, 3))
        joy_arr = np.array(joy_rows, dtype=float) if joy_rows else np.zeros((0, 3))

        # Each topic's own stream should already be time-ordered, but be defensive
        pose_arr = pose_arr[np.argsort(pose_arr[:, 0])]
        cmd_arr = cmd_arr[np.argsort(cmd_arr[:, 0])]
        joy_arr = joy_arr[np.argsort(joy_arr[:, 0])]

        return RawBagData(
            pose_t=pose_arr[:, 0], x=pose_arr[:, 1], y=pose_arr[:, 2], yaw=pose_arr[:, 3],
            cmd_t=cmd_arr[:, 0], u=cmd_arr[:, 1], steer_cmd=cmd_arr[:, 2],
            joy_t=joy_arr[:, 0], joy_throttle=joy_arr[:, 1], joy_steering=joy_arr[:, 2],
        )


class TimeSynchronizer:
    """Resamples VICON pose onto a UNIFORM time grid, then interpolates command +
    joystick streams onto that same grid via zero-order hold.

    Two different interpolation kinds are used deliberately:
      - pose (x, y, yaw): LINEAR, because VICON position is a continuous
        physical signal.
      - command / joystick: ZERO-ORDER HOLD, because a command only changes
        at the instant a new message arrives (it's piecewise-constant), so
        ZOH is the physically correct interpolation, not linear.

    The resampling step matters: real VICON bridge timestamps observed on
    this project are NOT uniformly spaced (bursts of near-duplicate
    timestamps ~20us apart interleaved with few-to-tens-of-ms gaps), so
    naively using median(diff(raw_t)) as "dt" and feeding the raw, irregular
    timestamps straight into Savitzky-Golay (which assumes uniform sampling)
    silently produces wildly wrong derivatives. Resampling onto a uniform
    grid at the bag's own AVERAGE rate (count / duration -- robust to the
    bursts, unlike the median) fixes both problems at once.
    """

    def __init__(self, interp_kind: str = 'zero', target_rate_hz: Optional[float] = None,
                 max_pose_speed_mps: float = 5.0, pose_clock: str = 'frame',
                 vicon_rate_hz: Optional[float] = None, max_frame_gap_s: float = 0.05):
        self.interp_kind = interp_kind
        self.target_rate_hz = target_rate_hz  # None == auto (count / duration of the pose stream)
        # pose_clock='frame': rebuild VICON capture times from the frame index
        # instead of trusting bag receipt stamps. Verified on real bags: VICON
        # delivers every frame at exactly 100 Hz but over the network in bursts
        # (43-54% of receipt intervals < 1 ms, receipt jitter up to ~0.5 s), so
        # receipt time is not capture time -- see _capture_times().
        if pose_clock not in ('frame', 'receipt'):
            raise ValueError("pose_clock must be 'frame' or 'receipt'")
        self.pose_clock = pose_clock
        self.vicon_rate_hz = vicon_rate_hz        # None == round(count / duration)
        self.max_frame_gap_s = max_frame_gap_s    # longer capture gaps (dropped frames) are flagged
        # VICON occlusion/dropout can teleport the reported position for a
        # frame or two (observed on real bags: implied speeds of hundreds of
        # m/s). Real motion tops out ~3 m/s on the frame clock (fast figure-
        # eights), so 5 m/s separates the two cleanly. Detection happens HERE (post-resample, uniform dt)
        # rather than on raw timestamps: the raw bridge timestamps are bursty
        # (near-duplicate timestamps ~20us apart), which makes ordinary
        # position noise divided by a near-zero raw dt look like a huge
        # glitch -- a false positive that doesn't exist once resampled onto
        # a clean, fixed-dt grid.
        #
        # We only FLAG glitches here (not patch them): a flagged sample is
        # excluded, together with a differentiation-window margin around it,
        # by filtering.SavGolDifferentiator -- which splits the session at
        # each exclusion into independently-uniform, contiguous sub-segments.
        # Linearly bridging the position value instead (an earlier version of
        # this code did that) leaves a slope DISCONTINUITY at each bridge
        # seam; Sav-Gol's polynomial window still turns that kink into a
        # derivative spike, so it doesn't actually fix anything -- excluding
        # the region outright is the only approach that's actually clean.
        self.max_pose_speed_mps = max_pose_speed_mps

    def _capture_times(self, recv_t: np.ndarray, name: str) -> np.ndarray:
        """Capture time of frame k = k*T + E[k], where E is the lower envelope of
        receipt latency: E[k] = min over j >= k of (recv_t[j] - j*T).

        Receipt = capture + network latency (>= 0), so the minimum over a run of
        frames recovers capture timing aligned to minimum latency. The envelope
        is a reverse cumulative minimum, hence non-decreasing: when frames are
        dropped, capture time really does jump by m*T and E steps up by m*T --
        observed on real bags as +40/+80/+140 ms steps -- so drops become real
        gaps instead of silently compressing time like a pure index clock would.
        """
        n = len(recv_t)
        if n < 3:
            return recv_t
        rate = self.vicon_rate_hz or float(round((n - 1) / (recv_t[-1] - recv_t[0])))
        T = 1.0 / rate
        k = np.arange(n)
        lat = recv_t - k * T
        env = np.minimum.accumulate(lat[::-1])[::-1]
        cap = k * T + env
        dropped = int(round((env[-1] - env[0]) / T))
        jitter = (lat - env) * 1e3
        logger.debug('%s: frame clock %.0f Hz, receipt jitter p50 %.0f / p95 %.0f / max %.0f ms, '
                      '~%d dropped frame(s)', name, rate, np.median(jitter),
                      np.percentile(jitter, 95), jitter.max(), dropped)
        return cap

    def _gap_mask(self, frame_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Flags resampled samples that fall inside a capture gap longer than
        max_frame_gap_s (dropped frames): there, the position is just a linear
        interpolation across the gap, not a measurement."""
        idx = np.clip(np.searchsorted(frame_t, t), 1, len(frame_t) - 1)
        gap = frame_t[idx] - frame_t[idx - 1]
        return gap > self.max_frame_gap_s

    def _detect_pose_glitches(self, x: np.ndarray, y: np.ndarray, dt: float, name: str) -> np.ndarray:
        n = len(x)
        bad = np.zeros(n, dtype=bool)
        if n < 3:
            return bad
        step_speed = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2) / dt
        bad[1:] = step_speed > self.max_pose_speed_mps
        n_bad = int(bad.sum())
        if n_bad:
            logger.warning('%s: flagging %d/%d resampled pose sample(s) as VICON glitches '
                            '(implied speed > %.1f m/s, max seen %.1f m/s)',
                            name, n_bad, n, self.max_pose_speed_mps, float(np.max(step_speed)))
        return bad

    @staticmethod
    def _dedup(t_src: np.ndarray, v_src: np.ndarray):
        """Keeps the LAST sample of each run of identical timestamps (input is
        time-sorted). Real bags contain duplicate receive stamps, which scipy's
        interp1d rejects outright ("Expect x to not have duplicates")."""
        if len(t_src) < 2:
            return t_src, v_src
        keep = np.r_[np.diff(t_src) > 0, True]
        return t_src[keep], v_src[keep]

    def _zoh(self, t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
        t_src, v_src = self._dedup(t_src, v_src)
        if len(t_src) == 0:
            return np.zeros_like(t_dst)
        if len(t_src) < 2:
            return np.full_like(t_dst, v_src[0])
        f = interp1d(t_src, v_src, kind=self.interp_kind, fill_value='extrapolate', bounds_error=False)
        return f(t_dst)

    @classmethod
    def _linear(cls, t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
        t_src, v_src = cls._dedup(t_src, v_src)
        if len(t_src) < 2:
            return np.full_like(t_dst, v_src[0] if len(v_src) else 0.0)
        f = interp1d(t_src, v_src, kind='linear', fill_value='extrapolate', bounds_error=False)
        return f(t_dst)

    def synchronize(self, name: str, raw: RawBagData) -> Optional[SessionData]:
        if not raw.is_usable:
            logger.warning('Session %s: insufficient pose/cmd data (pose=%d, cmd=%d), skipping',
                            name, len(raw.pose_t), len(raw.cmd_t))
            return None

        frame_t = self._capture_times(raw.pose_t, name) if self.pose_clock == 'frame' else raw.pose_t
        t0 = min(frame_t[0], raw.cmd_t[0])
        pose_t = frame_t - t0
        duration = float(pose_t[-1] - pose_t[0])
        n_raw = len(pose_t)
        if duration <= 0:
            logger.warning('Session %s: degenerate pose duration, skipping', name)
            return None

        auto_rate = (n_raw - 1) / duration
        rate = self.target_rate_hz or auto_rate
        dt = 1.0 / rate
        n_samples = max(4, int(duration / dt) + 1)
        # Grid spans the pose stream itself (which may start after the first
        # command), so pose is interpolated, never extrapolated.
        t = pose_t[0] + np.arange(n_samples) * dt

        yaw_unwrapped = np.unwrap(raw.yaw)
        x = self._linear(pose_t, raw.x, t)
        y = self._linear(pose_t, raw.y, t)
        yaw = self._linear(pose_t, yaw_unwrapped, t)
        glitch_mask = self._detect_pose_glitches(x, y, dt, name)
        if self.pose_clock == 'frame':
            glitch_mask |= self._gap_mask(pose_t, t)

        u = self._zoh(raw.cmd_t - t0, raw.u, t)
        steer_cmd = self._zoh(raw.cmd_t - t0, raw.steer_cmd, t)
        joy_steering = self._zoh(raw.joy_t - t0, raw.joy_steering, t)

        return SessionData(
            name=name, t=t, x=x, y=y, yaw=yaw, glitch_mask=glitch_mask,
            u=u, steer_cmd=steer_cmd, joy_steering=joy_steering, dt=dt,
        )


class BatchBagExtractor:
    """Recursively discovers every rosbag2 directory (anything containing metadata.yaml)
    under root_dir and extracts+synchronizes each independently. A failure on one bag
    is logged and skipped, never aborts the batch."""

    def __init__(self, root_dir: Path, reader: Optional[MCAPSessionReader] = None,
                 synchronizer: Optional[TimeSynchronizer] = None):
        self.root_dir = Path(root_dir)
        self.reader = reader or MCAPSessionReader()
        self.synchronizer = synchronizer or TimeSynchronizer()

    def discover(self) -> list[Path]:
        return sorted({p.parent for p in self.root_dir.rglob('metadata.yaml')})

    def extract_all(self) -> dict[str, SessionData]:
        bag_dirs = self.discover()
        logger.info('Discovered %d bag session(s) under %s', len(bag_dirs), self.root_dir)

        sessions: dict[str, SessionData] = {}
        for bag_dir in bag_dirs:
            key = str(bag_dir.relative_to(self.root_dir))
            try:
                raw = self.reader.read(bag_dir)
                session = self.synchronizer.synchronize(key, raw)
                if session is not None:
                    sessions[key] = session
                    logger.info('  %-40s  %5d VICON samples, dt=%.1fms', key, len(session), session.dt * 1000)
            except Exception:
                logger.exception('Failed to extract %s -- skipping', bag_dir)
        return sessions
