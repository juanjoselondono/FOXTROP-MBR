"""
regimes.py
==========
Regime-specific (Low vs Medium speed) identification from clean step segments.

Workflow:
  1. StepSegmenter   -- hysteresis-quantize the commands (removes human
                        joystick micro-jitter), cut each contiguous recording
                        into constant-command windows, drop windows shorter
                        than min_duration.
  2. RegimeClassifier -- label each window by mean |u_norm|:
                        low (<= 0.4), medium (0.4, 0.8], high (> 0.8, unused).
  3. RegimeIdentifier -- fit FirstOrderDeadZoneModel (K, tau, u_dead) and
                        AckermannLateralModel (K_steer, L_eff fixed) per regime,
                        reusing optimization.ModelFitter (L-BFGS-B).
  4. RegimeValidator  -- simulate each validation window with the model of
                        its own regime, pooled RMSE per regime.

Input: a synchronized pandas DataFrame (one row per sample) with columns
t, u, steer_cmd, v, omega and optionally session -- see ColumnMap.
sessions_to_dataframe() builds it from the main pipeline's SessionData.

Steering input is steer_cmd (cmd_vel.angular.z, normalized by
steer_full_scale = 0.5, twist_cmd_node's full servo range) -- the command the
car actually received. NOT the raw joystick axis: in sessions driven by
throttle_profile_node / manual_cruise_node it reads a stale +1.0 while the
car is commanded straight, which would masquerade as full-lock turns.

Throttle normalization: u in the bags is cmd_vel.linear.x, NOT a [-1, 1]
command. twist_cmd_node maps linear.x in [-0.5, +0.5] onto the full ESC PWM
range, so u_norm = u / u_full_scale with u_full_scale = 0.5 is the fraction
of ESC authority -- the quantity the 0.4 / 0.8 regime bounds refer to.
K and u_dead fitted here are therefore in normalized-command units
(K_norm = 0.5 * K_raw relative to the main pipeline's fit).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from .extraction import SessionData
from .models import FirstOrderDeadZoneModel, AckermannLateralModel
from .optimization import ModelFitter, FitResult

logger = logging.getLogger(__name__)

REGIMES = ('low', 'medium')


@dataclass
class ColumnMap:
    time: str = 't'
    throttle: str = 'u'
    steering: str = 'steer_cmd'
    v: str = 'v'
    omega: str = 'omega'
    session: str = 'session'


@dataclass
class RegimeConfig:
    u_full_scale: float = 0.5       # linear.x giving 100% ESC (twist_cmd_node mapping)
    steer_full_scale: float = 0.5   # angular.z giving full servo lock (twist_cmd_node mapping)
    # Hysteresis band, normalized units. 0.03 rather than 0.05: one Up/Down
    # press in manual_cruise_node changes linear.x by 0.02 = 0.04 normalized,
    # so a 0.05 band would merge consecutive real steps into one segment.
    quant_tolerance: float = 0.03
    min_duration: float = 1.2       # s -- shorter windows can't show transient + steady state
    low_max: float = 0.4
    medium_max: float = 0.8
    steer_neutral: float = 0.15     # steering windows must be an active turn to inform K_steer
    lateral_min_speed: float = 0.05 # m/s -- Ackermann is meaningless while stopped
    # Exclude the first lateral_settle_s of each steering window from fit/RMSE:
    # the servo needs ~0.4 s to swing lock-to-lock (observed: omega -2 -> +2
    # rad/s), an actuator transient the instantaneous Ackermann model cannot
    # represent. The steady-state part is what K_steer describes.
    lateral_settle_s: float = 0.5
    drop_idle: bool = True          # drop windows with neutral command AND a car at rest
    # Drop windows whose median throttle is negative. On an RC ESC a negative
    # command while rolling is a BRAKE (reverse only engages after a neutral
    # pause), and twist_cmd_node clamps it at PWM 70 (linear.x ~ -0.235), so
    # it is not the mirror image of forward drive the symmetric dead-zone model
    # assumes. Classified by |u| it would also land in "medium" and pollute it.
    forward_only: bool = True
    # Longitudinal (throttle) windows must be driven straight: |steering| <
    # steer_neutral throughout. While cornering, tire scrub makes speed swing
    # (observed on figure-eights: 1.8 -> 0.7 m/s at constant throttle), which a
    # first-order v(u) model cannot represent -- it drives K to its bound.
    longitudinal_straight_only: bool = True
    idle_speed: float = 0.02        # m/s
    gap_factor: float = 1.5         # a time step > gap_factor*dt splits a recording
    L_eff: float = 0.245            # m, measured wheelbase (fixed, not fit)
    max_steer_rad: float = 0.5236   # bound on K_steer


@dataclass
class StepSegment:
    session: str
    channel: str                    # 'throttle' or 'steering'
    regime: str                     # 'low' | 'medium' | 'high'
    level: float                    # median raw command of the segmenting channel
    mean_abs_u: float               # mean |u_norm| -- drives the regime
    dt: float
    t: np.ndarray = field(repr=False)
    u_raw: np.ndarray = field(repr=False)       # normalized, unquantized
    u_clean: np.ndarray = field(repr=False)     # normalized, constant = median of u_raw over the segment
    steer_raw: np.ndarray = field(repr=False)
    steer_clean: np.ndarray = field(repr=False)
    v: np.ndarray = field(repr=False)
    omega: np.ndarray = field(repr=False)

    @property
    def duration(self) -> float:
        return len(self.t) * self.dt


class HysteresisQuantizer:
    """Holds the current level until the input leaves a +/- tolerance band around
    it, then latches the new raw value. Jitter inside the band never creates an
    edge (unlike plain rounding, which chatters at grid boundaries). The level is
    latched raw, not snapped to a grid: snapping shifts the reference (0.08 ->
    0.09) and can hide a real step of about one tolerance."""

    def __init__(self, tolerance: float):
        self.tol = tolerance

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=float)
        if len(x) == 0:
            return out
        level = float(x[0])
        for i, xi in enumerate(x):
            if abs(xi - level) > self.tol:
                level = float(xi)
            out[i] = level
        return out


class RegimeClassifier:
    def __init__(self, low_max: float = 0.4, medium_max: float = 0.8):
        self.low_max = low_max
        self.medium_max = medium_max

    def classify(self, mean_abs_u: float) -> str:
        if mean_abs_u <= self.low_max:
            return 'low'
        if mean_abs_u <= self.medium_max:
            return 'medium'
        return 'high'


class StepSegmenter:
    def __init__(self, cfg: RegimeConfig = None, cols: ColumnMap = None):
        self.cfg = cfg or RegimeConfig()
        self.cols = cols or ColumnMap()
        self.quantizer = HysteresisQuantizer(self.cfg.quant_tolerance)
        self.classifier = RegimeClassifier(self.cfg.low_max, self.cfg.medium_max)

    def _blocks(self, df: pd.DataFrame) -> Iterator[Tuple[str, pd.DataFrame]]:
        """Contiguous, uniformly sampled recordings: split by session, then at time gaps."""
        c = self.cols
        groups = df.groupby(c.session, sort=False) if c.session in df.columns else [('data', df)]
        for name, g in groups:
            g = g.sort_values(c.time)
            t = g[c.time].to_numpy()
            if len(t) < 2:
                continue
            dt = float(np.median(np.diff(t)))
            cuts = np.flatnonzero(np.diff(t) > self.cfg.gap_factor * dt) + 1
            for k, part in enumerate(np.split(np.arange(len(g)), cuts)):
                if len(part) > 1:
                    label = str(name) if len(cuts) == 0 else f'{name}~{k}'
                    yield label, g.iloc[part]

    def segment(self, df: pd.DataFrame, channel: str) -> List[StepSegment]:
        if channel not in ('throttle', 'steering'):
            raise ValueError("channel must be 'throttle' or 'steering'")
        c, cfg = self.cols, self.cfg
        segments: List[StepSegment] = []
        n_braking = n_turning = 0

        for name, block in self._blocks(df):
            t = block[c.time].to_numpy(dtype=float)
            dt = float(np.median(np.diff(t)))
            u_raw = block[c.throttle].to_numpy(dtype=float) / cfg.u_full_scale
            s_raw = block[c.steering].to_numpy(dtype=float) / cfg.steer_full_scale
            v = block[c.v].to_numpy(dtype=float)
            omega = block[c.omega].to_numpy(dtype=float)
            u_q = self.quantizer(u_raw)
            s_q = self.quantizer(s_raw)

            # The quantized signal only locates edges. The clean command fed to
            # the model is the segment's MEDIAN raw value -- snapping to the
            # quantization grid would bias it (e.g. a real 0.08 becomes 0.10).
            key = u_q if channel == 'throttle' else s_q
            edges = np.flatnonzero(np.diff(key) != 0) + 1
            for idx in np.split(np.arange(len(key)), edges):
                if len(idx) * dt < cfg.min_duration:
                    continue
                u_level = float(np.median(u_raw[idx]))
                s_level = float(np.median(s_raw[idx]))
                level = u_level if channel == 'throttle' else s_level
                if channel == 'steering' and abs(level) < cfg.steer_neutral:
                    continue
                if cfg.drop_idle and abs(u_level) < 1e-9 and np.abs(v[idx]).max() < cfg.idle_speed:
                    continue
                if cfg.forward_only and u_level < -1e-9:
                    n_braking += 1
                    continue
                if (channel == 'throttle' and cfg.longitudinal_straight_only
                        and np.abs(s_raw[idx]).max() >= cfg.steer_neutral):
                    n_turning += 1
                    continue
                mean_abs_u = float(np.mean(np.abs(u_raw[idx])))
                n = len(idx)
                segments.append(StepSegment(
                    session=name, channel=channel,
                    regime=self.classifier.classify(mean_abs_u),
                    level=level, mean_abs_u=mean_abs_u, dt=dt,
                    t=t[idx], u_raw=u_raw[idx], u_clean=np.full(n, u_level),
                    steer_raw=s_raw[idx], steer_clean=np.full(n, s_level),
                    v=v[idx], omega=omega[idx],
                ))
        if n_braking:
            logger.info('%s segmentation: dropped %d braking/reverse window(s) (forward_only)',
                         channel, n_braking)
        if n_turning:
            logger.info('%s segmentation: dropped %d window(s) driven while steering '
                         '(longitudinal_straight_only)', channel, n_turning)
        return segments


def _lateral_mask(seg: StepSegment, cfg: RegimeConfig) -> np.ndarray:
    settled = (seg.t - seg.t[0]) >= cfg.lateral_settle_s
    return settled & (np.abs(seg.v) >= cfg.lateral_min_speed)


class RegimeIdentifier:
    """Runs the existing L-BFGS-B ModelFitter independently per regime."""

    def __init__(self, cfg: RegimeConfig = None):
        self.cfg = cfg or RegimeConfig()
        # regime -> True if at least one training segment commands past the
        # fitted dead zone. If none does, u_eff == 0 everywhere and K never
        # enters the cost: its fitted value is arbitrary, not identified.
        self.k_identifiable: Dict[str, bool] = {}

    def lateral_model(self) -> AckermannLateralModel:
        return AckermannLateralModel(L_eff=self.cfg.L_eff, max_steer_rad=self.cfg.max_steer_rad)

    def fit_longitudinal(self, segments: List[StepSegment]) -> Dict[str, FitResult]:
        fits: Dict[str, FitResult] = {}
        for regime in REGIMES:
            # Simulate with the RAW command the ESC actually received. The
            # de-jittered level only drives segmentation and regime labels:
            # inside real teleop windows the command drifts (e.g. 0.125 ->
            # 0.165), and a constant median would misstate the input.
            dataset = [({'u': s.u_raw, 'dt': s.dt, 'v0': s.v[0]}, s.v)
                       for s in segments if s.regime == regime]
            if not dataset:
                logger.warning('Longitudinal %s: no training step segments -- regime not identified', regime)
                continue
            fits[regime] = ModelFitter(FirstOrderDeadZoneModel()).fit(dataset)
            logger.info('Longitudinal %-6s (%d seg): RMSE=%.4f m/s  %s',
                         regime, len(dataset), fits[regime].rmse, fits[regime].params)
            u_dead = fits[regime].params['u_dead']
            n_active = sum(1 for inp, _ in dataset if np.abs(inp['u']).max() > u_dead)
            self.k_identifiable[regime] = n_active > 0
            if n_active == 0:
                logger.warning('Longitudinal %s: no training segment commands past u_dead=%.3f -- '
                                'K is NOT identified by this data (only tau and u_dead are)', regime, u_dead)
        return fits

    def fit_lateral(self, segments: List[StepSegment]) -> Dict[str, FitResult]:
        fits: Dict[str, FitResult] = {}
        for regime in REGIMES:
            dataset = []
            for s in segments:
                if s.regime != regime:
                    continue
                m = _lateral_mask(s, self.cfg)
                if m.sum() >= 5:
                    dataset.append(({'v': s.v[m], 'steering': s.steer_clean[m]}, s.omega[m]))
            if not dataset:
                logger.warning('Lateral %s: no training steering-step segments -- regime not identified', regime)
                continue
            fits[regime] = ModelFitter(self.lateral_model()).fit(dataset)
            logger.info('Lateral %-6s (%d seg): RMSE=%.4f rad/s  %s  (L_eff=%.3f m fixed)',
                         regime, len(dataset), fits[regime].rmse, fits[regime].params, self.cfg.L_eff)
        return fits


@dataclass
class SegmentResult:
    segment: StepSegment
    sim: np.ndarray        # full-segment simulation (for plotting)
    rmse: float            # over the scored samples
    n_scored: int


@dataclass
class RegimeScore:
    regime: str
    n_segments: int
    n_samples: int
    rmse: float            # pooled over all scored samples of the regime


class RegimeValidator:
    """Simulates each segment with the model of its OWN regime (never a cross-regime model)."""

    def __init__(self, cfg: RegimeConfig = None):
        self.cfg = cfg or RegimeConfig()

    def evaluate(self, segments: List[StepSegment], fits: Dict[str, FitResult],
                 kind: str) -> Tuple[List[SegmentResult], Dict[str, RegimeScore]]:
        long_model = FirstOrderDeadZoneModel()
        lat_model = RegimeIdentifier(self.cfg).lateral_model()
        results: List[SegmentResult] = []
        skipped = 0
        for s in segments:
            fit = fits.get(s.regime)
            if fit is None:
                skipped += 1
                continue
            if kind == 'longitudinal':
                sim = long_model.simulate(fit.raw_params, {'u': s.u_raw, 'dt': s.dt, 'v0': s.v[0]})
                err = sim - s.v
            else:
                sim = lat_model.simulate(fit.raw_params, {'v': s.v, 'steering': s.steer_clean})
                err = (sim - s.omega)[_lateral_mask(s, self.cfg)]
            if len(err) == 0:
                skipped += 1
                continue
            results.append(SegmentResult(s, sim, float(np.sqrt(np.mean(err ** 2))), len(err)))
        if skipped:
            logger.info('%s: %d segment(s) had no fitted model for their regime (or no scored samples)',
                         kind, skipped)

        scores: Dict[str, RegimeScore] = {}
        for regime in REGIMES:
            rs = [r for r in results if r.segment.regime == regime]
            if not rs:
                continue
            sq = sum(r.rmse ** 2 * r.n_scored for r in rs)
            n = sum(r.n_scored for r in rs)
            scores[regime] = RegimeScore(regime, len(rs), n, float(np.sqrt(sq / n)))
        return results, scores


def sessions_to_dataframe(sessions: Dict[str, SessionData]) -> pd.DataFrame:
    """Flattens differentiated SessionData (main pipeline output) into the synchronized DataFrame."""
    frames = [pd.DataFrame({'session': name, 't': s.t, 'u': s.u, 'steer_cmd': s.steer_cmd,
                            'joy_steering': s.joy_steering, 'v': s.v, 'omega': s.omega})
              for name, s in sessions.items() if s.is_filtered]
    if not frames:
        return pd.DataFrame(columns=['session', 't', 'u', 'steer_cmd', 'joy_steering', 'v', 'omega'])
    return pd.concat(frames, ignore_index=True)


@dataclass
class RegimeReport:
    longitudinal: Dict[str, dict]
    lateral: Dict[str, dict]
    segment_counts: Dict[str, Dict[str, int]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=float)


class RegimeAnalysis:
    """End-to-end: segment -> classify -> fit per regime -> validate -> plot -> report."""

    def __init__(self, output_dir: Path, cfg: RegimeConfig = None, cols: ColumnMap = None):
        from .visualization import RegimePlotter  # local import keeps matplotlib out of pure-analysis use
        self.cfg = cfg or RegimeConfig()
        self.cols = cols or ColumnMap()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segmenter = StepSegmenter(self.cfg, self.cols)
        self.identifier = RegimeIdentifier(self.cfg)
        self.validator = RegimeValidator(self.cfg)
        self.plotter = RegimePlotter(self.output_dir)

    @staticmethod
    def _count(segs: List[StepSegment]) -> Dict[str, int]:
        out = {r: 0 for r in ('low', 'medium', 'high')}
        for s in segs:
            out[s.regime] += 1
        return out

    @staticmethod
    def _section(fits: Dict[str, FitResult], train: Dict[str, RegimeScore],
                 val: Dict[str, RegimeScore]) -> Dict[str, dict]:
        out = {}
        for regime in REGIMES:
            out[regime] = {
                'identified': regime in fits,
                'params': fits[regime].params if regime in fits else None,
                'train_rmse': train[regime].rmse if regime in train else None,
                'val_rmse': val[regime].rmse if regime in val else None,
                'n_train_segments': train[regime].n_segments if regime in train else 0,
                'n_val_segments': val[regime].n_segments if regime in val else 0,
            }
        return out

    def _write_segment_table(self, groups: Dict[str, List[SegmentResult]]):
        rows = []
        for label, results in groups.items():
            for r in results:
                s = r.segment
                rows.append({'set': label, 'session': s.session, 'channel': s.channel, 'regime': s.regime,
                             'level': s.level, 'mean_abs_u': s.mean_abs_u, 't_start': s.t[0],
                             'duration_s': s.duration, 'rmse': r.rmse})
        pd.DataFrame(rows).to_csv(self.output_dir / 'regime_segments.csv', index=False)

    def run(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> RegimeReport:
        seg = self.segmenter
        train_long, val_long = seg.segment(train_df, 'throttle'), seg.segment(val_df, 'throttle')
        train_lat, val_lat = seg.segment(train_df, 'steering'), seg.segment(val_df, 'steering')
        counts = {
            'train_longitudinal': self._count(train_long), 'val_longitudinal': self._count(val_long),
            'train_lateral': self._count(train_lat), 'val_lateral': self._count(val_lat),
        }
        for k, v in counts.items():
            logger.info('Step segments %-19s %s', k, v)

        long_fits = self.identifier.fit_longitudinal(train_long)
        lat_fits = self.identifier.fit_lateral(train_lat)

        tl_res, tl_sc = self.validator.evaluate(train_long, long_fits, 'longitudinal')
        vl_res, vl_sc = self.validator.evaluate(val_long, long_fits, 'longitudinal')
        ta_res, ta_sc = self.validator.evaluate(train_lat, lat_fits, 'lateral')
        va_res, va_sc = self.validator.evaluate(val_lat, lat_fits, 'lateral')

        self.plotter.plot(tl_res, 'longitudinal', 'training')
        self.plotter.plot(vl_res, 'longitudinal', 'validation')
        self.plotter.plot(ta_res, 'lateral', 'training')
        self.plotter.plot(va_res, 'lateral', 'validation')
        self._write_segment_table({'train_long': tl_res, 'val_long': vl_res,
                                    'train_lat': ta_res, 'val_lat': va_res})

        long_section = self._section(long_fits, tl_sc, vl_sc)
        for regime, ok in self.identifier.k_identifiable.items():
            long_section[regime]['K_identifiable'] = ok

        report = RegimeReport(
            longitudinal=long_section,
            lateral=self._section(lat_fits, ta_sc, va_sc),
            segment_counts=counts,
        )
        (self.output_dir / 'regime_report.json').write_text(report.to_json())
        return report
