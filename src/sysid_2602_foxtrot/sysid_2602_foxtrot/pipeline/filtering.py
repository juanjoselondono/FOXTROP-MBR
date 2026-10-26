"""
filtering.py
============
Step 2: numerical differentiation of VICON position/yaw into velocity/yaw-rate
using a Savitzky-Golay filter (smooths quantization noise without the phase
lag a simple low-pass + finite-difference would introduce).

Sessions flagged by extraction.TimeSynchronizer as containing VICON
occlusion/dropout glitches are SPLIT at the glitch (plus a window margin),
not patched -- any interpolated patch still leaves a slope discontinuity
that a Sav-Gol window spanning it turns into a derivative spike. Splitting
keeps every sample that reaches the optimizer honest: each output segment
is internally contiguous and uniformly sampled, which also matters for the
longitudinal ODE integration in models.py (its dt assumes no time was cut
out from the middle of the array).
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from scipy.signal import savgol_filter

from .extraction import SessionData

logger = logging.getLogger(__name__)


class SavGolDifferentiator:
    """Computes vx, vy, signed forward speed v, and yaw rate omega from a SessionData's x/y/yaw.

    window_length / polyorder are tunable: widen the window if v(t) looks
    noisy/jittery (more smoothing), narrow it if v(t) visibly lags the real
    motion (phase lag). window_length must be odd and > polyorder.
    """

    def __init__(self, window_length: int = 21, polyorder: int = 3, min_segment_duration: float = 1.5):
        if window_length % 2 == 0:
            raise ValueError('window_length must be odd')
        if polyorder >= window_length:
            raise ValueError('polyorder must be < window_length')
        self.window_length = window_length
        self.polyorder = polyorder
        # A segment shorter than this (seconds) is dropped outright, even if
        # it's technically long enough to run Sav-Gol on: the optimizer needs
        # a full transient to identify tau, and a short fragment mostly
        # injects noise into the objective rather than useful information --
        # this matters a lot now that glitch-splitting (see differentiate())
        # can chop a session into many short pieces.
        self.min_segment_duration = min_segment_duration

    def _effective_window(self, n: int, name: str) -> int:
        wl = self.window_length
        if n >= wl:
            return wl
        wl = n if n % 2 == 1 else n - 1
        if wl <= self.polyorder:
            raise ValueError(
                f'segment {name}: only {n} samples, too short to differentiate '
                f'(need > polyorder={self.polyorder})'
            )
        logger.warning('segment %s: only %d samples, shrinking Sav-Gol window %d -> %d',
                        name, n, self.window_length, wl)
        return wl

    @staticmethod
    def _dilate(mask: np.ndarray, margin: int) -> np.ndarray:
        """Expands each True run by `margin` samples on each side -- a Sav-Gol window
        of this half-width centered anywhere in that margin would still see the glitch."""
        if margin <= 0 or not mask.any():
            return mask.copy()
        kernel = np.ones(2 * margin + 1, dtype=int)
        return np.convolve(mask.astype(int), kernel, mode='same') > 0

    @staticmethod
    def _contiguous_runs(valid: np.ndarray) -> List[np.ndarray]:
        idx = np.where(valid)[0]
        if len(idx) == 0:
            return []
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(idx) - 1]))
        return [idx[s:e + 1] for s, e in zip(starts, ends)]

    def _differentiate_segment(self, session: SessionData, sel: np.ndarray, seg_name: str) -> SessionData:
        t, x, y = session.t[sel], session.x[sel], session.y[sel]
        yaw = session.yaw[sel]
        n = len(sel)
        wl = self._effective_window(n, seg_name)

        vx = savgol_filter(x, window_length=wl, polyorder=self.polyorder, deriv=1, delta=session.dt)
        vy = savgol_filter(y, window_length=wl, polyorder=self.polyorder, deriv=1, delta=session.dt)
        omega = savgol_filter(yaw, window_length=wl, polyorder=self.polyorder, deriv=1, delta=session.dt)

        v_mag = np.sqrt(vx ** 2 + vy ** 2)
        heading_dot = vx * np.cos(yaw) + vy * np.sin(yaw)
        v_signed = np.where(heading_dot >= 0, v_mag, -v_mag)

        return SessionData(
            name=seg_name, t=t, x=x, y=y, yaw=yaw, glitch_mask=None,
            u=session.u[sel], steer_cmd=session.steer_cmd[sel], joy_steering=session.joy_steering[sel],
            dt=session.dt, vx=vx, vy=vy, v=v_signed, omega=omega,
        )

    def differentiate(self, session: SessionData) -> List[SessionData]:
        """Returns one SessionData per contiguous glitch-free stretch (usually just
        one, covering the whole input, if nothing was flagged)."""
        n = len(session)
        mask = session.glitch_mask if session.glitch_mask is not None else np.zeros(n, dtype=bool)
        margin = self.window_length // 2
        dilated = self._dilate(mask, margin)
        valid = ~dilated

        runs = self._contiguous_runs(valid)
        min_len = max(self.window_length + 1, int(np.ceil(self.min_segment_duration / session.dt)) + 1)

        segments: List[SessionData] = []
        n_too_short = 0
        multi = len(runs) > 1
        for i, sel in enumerate(runs):
            if len(sel) < min_len:
                n_too_short += 1
                continue  # below window and/or min_segment_duration -- a short fragment
                          # can't identify tau, it only adds noise to the objective
            seg_name = f'{session.name}#{i}' if multi else session.name
            try:
                segments.append(self._differentiate_segment(session, sel, seg_name))
            except ValueError as exc:
                logger.warning('Skipping %s: %s', seg_name, exc)

        if not segments:
            logger.warning('Session %s: no glitch-free segment >= %.2fs long enough to differentiate',
                            session.name, self.min_segment_duration)
        elif multi or n_too_short:
            logger.info('Session %s: split into %d usable segment(s) >= %.2fs '
                         '(dropped %d/%d sample(s) to glitches, %d segment(s) too short)',
                         session.name, len(segments), self.min_segment_duration,
                         int(dilated.sum()), n, n_too_short)
        return segments

    def differentiate_all(self, sessions: Dict[str, SessionData]) -> Dict[str, SessionData]:
        out: Dict[str, SessionData] = {}
        for name, s in sessions.items():
            for seg in self.differentiate(s):
                out[seg.name] = seg
        return out
