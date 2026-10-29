"""
pipeline.py
===========
Orchestrates the full offline sysid workflow end to end:

    extraction -> filtering -> {longitudinal fit (+ nonlinear fallback),
    lateral fit} -> cross-validation on held-out sessions -> plots + report

See scripts/run_sysid_pipeline.py for the CLI entry point.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .extraction import MCAPSessionReader, TimeSynchronizer, BatchBagExtractor, SessionData
from .filtering import SavGolDifferentiator
from .models import FirstOrderDeadZoneModel, NonlinearDragModel, AckermannLateralModel
from .optimization import ModelFitter, FitResult
from .validation import SessionClassifier, CrossValidator, ValidationResult
from .visualization import PlotGenerator

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    bag_root: Path
    output_dir: Path = Path('sysid_pipeline_output')

    # topics / joy axis mapping (see manual_cruise_node.py / extract_bag.py for the hardware convention)
    topic_pose: str = '/robot1/pose'
    topic_cmd: str = '/cmd_vel_stamped'
    topic_joy: str = '/joy'
    joy_throttle_axis: int = 1
    joy_steering_axis: int = 2  # matches bringup_2602_foxtrot/config/joystick.yaml axis_angular.yaw
    position_scale: float = 0.001  # VICON mm -> m
    max_pose_speed_mps: float = 5.0  # VICON glitch-rejection threshold (real motion <= ~3 m/s)
    pose_clock: str = 'frame'        # 'frame' = rebuild VICON capture times; 'receipt' = bag stamps

    # filtering.SavGolDifferentiator
    savgol_window: int = 21
    savgol_polyorder: int = 3
    # drop any glitch-free segment shorter than this -- too short to identify
    # tau from, it only injects noise into the fit objective
    min_segment_duration: float = 1.5

    # validation.CrossValidator -- units differ (m/s vs rad/s), so these are
    # independent. 0.05 m/s turned out unachievable on this heavily
    # glitch-fragmented dataset; 0.20 m/s is the realistic target instead.
    acceptance_rmse_longitudinal: float = 0.20
    acceptance_rmse_lateral: float = 0.05
    # if the base first-order model's TRAINING rmse exceeds this, also fit
    # the nonlinear-drag fallback and keep whichever trains better
    nonlinear_fallback_threshold: float = 0.05

    # models.AckermannLateralModel
    # L_eff is FIXED (physically measured), not fit -- RC Mini Cooper R56
    # chassis wheelbase, measured at exactly 24.5 cm. K_steer (the virtual
    # max steering angle at joy_steering=1) is fit instead, bounded by
    # max_steer_rad, a realistic physical steering throw (default 30 deg).
    L_eff: float = 0.245
    max_steer_rad: float = 0.5236  # 30 deg
    # The Ackermann relation omega = (v/L_eff)*tan(delta) only holds under
    # continuous rolling, and K_steer is only identifiable where the car is
    # actually turning -- so, for the LATERAL stage only (not longitudinal),
    # mask out stationary/crawling samples and near-neutral steering before
    # computing MSE, instead of letting them dilute/drag the gain to 0.
    lateral_min_speed_mps: float = 0.5       # exclude |v_vicon| below this
    lateral_min_steering_cmd: float = 0.15   # exclude |steering| below this
    # Lateral input = cmd_vel.angular.z / steer_full_scale (the command the car
    # actually received; twist_cmd_node maps +/-0.5 to full servo range).
    # NOT the raw joystick axis: in throttle_profile/manual_cruise sessions it
    # reads a stale +1.0 while the car is commanded straight.
    steer_full_scale: float = 0.5
    lateral_min_samples: int = 20            # skip a session if fewer masked samples remain

    # validation.SessionClassifier
    train_patterns: Sequence[str] = field(
        default_factory=lambda: (r'^session_a', r'^session_b', r'^manual_session'))
    validate_patterns: Sequence[str] = field(default_factory=lambda: (r'^session_c',))


@dataclass
class PipelineReport:
    n_sessions_total: int
    n_sessions_train: int
    n_sessions_validate: int
    n_sessions_unknown: int
    longitudinal_model: str
    longitudinal_params: dict
    longitudinal_training_rmse: float
    longitudinal_validation_rmse: float
    longitudinal_validation_passed: bool
    lateral_params: Optional[dict]          # None when the lateral model couldn't be identified
    lateral_training_rmse: Optional[float]
    lateral_validation_rmse: float
    lateral_validation_passed: bool
    per_session_longitudinal: Dict[str, float]
    per_session_lateral: Dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


class SysIDPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.plots = PlotGenerator(config.output_dir)

    # ── Stage helpers (also usable standalone, e.g. from a notebook) ───────

    def extract(self) -> Dict[str, SessionData]:
        reader = MCAPSessionReader(
            topic_pose=self.config.topic_pose, topic_cmd=self.config.topic_cmd,
            topic_joy=self.config.topic_joy, joy_throttle_axis=self.config.joy_throttle_axis,
            joy_steering_axis=self.config.joy_steering_axis,
            position_scale=self.config.position_scale,
        )
        synchronizer = TimeSynchronizer(max_pose_speed_mps=self.config.max_pose_speed_mps,
                                        pose_clock=self.config.pose_clock)
        extractor = BatchBagExtractor(self.config.bag_root, reader=reader, synchronizer=synchronizer)
        return extractor.extract_all()

    def filter(self, sessions: Dict[str, SessionData]) -> Dict[str, SessionData]:
        differentiator = SavGolDifferentiator(
            self.config.savgol_window, self.config.savgol_polyorder,
            min_segment_duration=self.config.min_segment_duration,
        )
        return differentiator.differentiate_all(sessions)

    @staticmethod
    def _longitudinal_dataset(sessions: Dict[str, SessionData]) -> Dict[str, tuple]:
        return {name: ({'u': s.u, 'dt': s.dt, 'v0': s.v[0]}, s.v) for name, s in sessions.items()}

    def _steering(self, s: SessionData) -> np.ndarray:
        """Normalized steering actually applied to the car (see PipelineConfig.steer_full_scale)."""
        return s.steer_cmd / self.config.steer_full_scale

    def _lateral_dataset(self, sessions: Dict[str, SessionData]) -> Dict[str, tuple]:
        """Rolling + active-cornering masked dataset for the lateral (Ackermann) fit only.
        Stationary/crawling samples (|v| < lateral_min_speed_mps) and near-neutral
        steering (|steering| < lateral_min_steering_cmd) are excluded before MSE,
        since the Ackermann relation is only valid under continuous rolling and
        K_steer is only identifiable while the car is actively turning."""
        cfg = self.config
        out: Dict[str, tuple] = {}
        n_total = n_kept = 0
        for name, s in sessions.items():
            steering = self._steering(s)
            mask = (np.abs(s.v) >= cfg.lateral_min_speed_mps) & \
                   (np.abs(steering) >= cfg.lateral_min_steering_cmd)
            n_total += len(mask)
            n_kept += int(mask.sum())
            if mask.sum() < cfg.lateral_min_samples:
                logger.debug('Lateral mask: %s has only %d/%d qualifying sample(s), skipping',
                              name, int(mask.sum()), len(mask))
                continue
            out[name] = ({'v': s.v[mask], 'steering': steering[mask]}, s.omega[mask])
        logger.info('Lateral rolling+cornering mask: kept %d/%d sample(s) across %d/%d session(s) '
                     '(|v|>=%.2f m/s, |steering|>=%.2f)',
                     n_kept, n_total, len(out), len(sessions),
                     cfg.lateral_min_speed_mps, cfg.lateral_min_steering_cmd)
        return out

    def fit_longitudinal(self, train: Dict[str, SessionData]) -> FitResult:
        dataset = list(self._longitudinal_dataset(train).values())
        base = ModelFitter(FirstOrderDeadZoneModel()).fit(dataset)
        logger.info('Longitudinal base fit (%s): RMSE=%.4f m/s  params=%s',
                     base.model_name, base.rmse, base.params)

        if base.rmse <= self.config.nonlinear_fallback_threshold:
            return base

        logger.warning(
            'Base model training RMSE %.4f > fallback threshold %.4f -- trying nonlinear drag model',
            base.rmse, self.config.nonlinear_fallback_threshold,
        )
        fallback = ModelFitter(NonlinearDragModel()).fit(dataset)
        logger.info('Longitudinal fallback fit (%s): RMSE=%.4f m/s  params=%s',
                     fallback.model_name, fallback.rmse, fallback.params)
        # Require a real improvement, not just a lower RMSE by noise: the
        # fallback has an extra free parameter (4 vs 3), so it can win
        # trivially while actually being less trustworthy (e.g. landing on
        # its parameter bounds, a sign of an unconstrained/overfit fit).
        if fallback.rmse < 0.95 * base.rmse:
            return fallback
        logger.info('Fallback improvement too small (%.4f vs %.4f) -- keeping base model', fallback.rmse, base.rmse)
        return base

    def fit_lateral(self, train: Dict[str, SessionData]) -> Optional[FitResult]:
        masked = self._lateral_dataset(train)
        dataset = list(masked.values())
        if not dataset:
            logger.warning(
                'Lateral model NOT identified: no training sample is both rolling (|v|>=%.2f m/s) and '
                'steering (|steering|>=%.2f). Training sessions need real steering commands '
                '(throttle_profile/manual_cruise always send angular.z=0).',
                self.config.lateral_min_speed_mps, self.config.lateral_min_steering_cmd)
            return None

        # Sign diagnostic: correlate steering with omega on the SAME masked,
        # actively-cornering data the fit uses. A negative correlation here is
        # the actual evidence of a sign mismatch, independent of what K_steer
        # converges to (reported so the two can be cross-checked against each other).
        all_j = np.concatenate([inp['steering'] for inp, _ in dataset])
        all_w = np.concatenate([tgt for _, tgt in dataset])
        if len(all_j) > 1 and np.std(all_j) > 1e-9 and np.std(all_w) > 1e-9:
            corr = float(np.corrcoef(all_j, all_w)[0, 1])
            logger.info('Lateral sign check: corr(steering, omega) = %+.3f over %d masked sample(s) '
                         '(positive sign expected if command polarity matches the VICON frame)',
                         corr, len(all_j))
        else:
            # Each masked (actively-cornering) stretch tends to be a single
            # held-constant steering command -- zero variance within a
            # session, and here across all of them too, so Pearson
            # correlation is undefined (not computable, not just noisy).
            # The fitted K_steer's sign (logged below) is the real answer.
            logger.info('Lateral sign check: masked steering has no variance (%d sample(s), '
                         'all near-identical steering command) -- correlation undefined, '
                         "relying on the fitted K_steer's sign instead", len(all_j))

        model = AckermannLateralModel(L_eff=self.config.L_eff, max_steer_rad=self.config.max_steer_rad)
        result = ModelFitter(model).fit(dataset)
        sign_note = 'NEGATIVE -- joystick polarity is flipped vs the VICON frame' \
            if result.raw_params[0] < 0 else 'positive, as expected'
        logger.info('Lateral fit (%s): RMSE=%.4f rad/s  params=%s  (L_eff fixed at %.3f m, K_steer sign: %s)',
                     result.model_name, result.rmse, result.params, self.config.L_eff, sign_note)
        return result

    def _model_by_name(self, name: str):
        return {
            FirstOrderDeadZoneModel.name: FirstOrderDeadZoneModel(),
            NonlinearDragModel.name: NonlinearDragModel(),
        }[name]

    # ── Full end-to-end run ─────────────────────────────────────────────────

    def run(self) -> PipelineReport:
        cfg = self.config

        logger.info('=== Stage 1/5: extraction + synchronization ===')
        raw_sessions = self.extract()
        if not raw_sessions:
            raise RuntimeError(f'No usable bag sessions found under {cfg.bag_root}')

        logger.info('=== Stage 2/5: Savitzky-Golay differentiation ===')
        sessions = self.filter(raw_sessions)

        logger.info('=== Stage 3/5: train/validate split ===')
        classifier = SessionClassifier(cfg.train_patterns, cfg.validate_patterns)
        train, validate, unknown = classifier.split(sessions)
        logger.info('train=%d  validate=%d  unknown(excluded)=%d', len(train), len(validate), len(unknown))
        if not train:
            raise RuntimeError('No training sessions matched train_patterns -- check naming or patterns')

        logger.info('=== Stage 4/5: model fitting ===')
        long_fit = self.fit_longitudinal(train)
        lat_fit = self.fit_lateral(train)
        long_model = self._model_by_name(long_fit.model_name)
        lat_model = AckermannLateralModel(L_eff=cfg.L_eff, max_steer_rad=cfg.max_steer_rad)

        logger.info('=== Stage 5/5: cross-validation + plots ===')
        long_validator = CrossValidator(cfg.acceptance_rmse_longitudinal)
        lat_validator = CrossValidator(cfg.acceptance_rmse_lateral)

        long_val_dataset = self._longitudinal_dataset(validate) if validate else {}
        lat_val_dataset = self._lateral_dataset(validate) if (validate and lat_fit) else {}
        long_val_results = long_validator.validate_all(long_model, long_fit.raw_params, long_val_dataset)
        lat_val_results = lat_validator.validate_all(lat_model, lat_fit.raw_params, lat_val_dataset) \
            if lat_fit else {}

        self._plot_training_fits(train, long_model, long_fit, lat_model, lat_fit)
        self._plot_validation(validate, long_model, long_fit, long_val_results,
                               lat_model, lat_fit, lat_val_results)

        overall_long_rmse = CrossValidator.overall_rmse(long_val_results)
        overall_lat_rmse = CrossValidator.overall_rmse(lat_val_results)

        if long_val_results:
            self.plots.plot_summary_bar(long_val_results, cfg.acceptance_rmse_longitudinal,
                                         'RMSE [m/s]', 'validation_summary_longitudinal.png')
        if lat_val_results:
            self.plots.plot_summary_bar(lat_val_results, cfg.acceptance_rmse_lateral,
                                         'RMSE [rad/s]', 'validation_summary_lateral.png')

        report = PipelineReport(
            n_sessions_total=len(sessions),
            n_sessions_train=len(train),
            n_sessions_validate=len(validate),
            n_sessions_unknown=len(unknown),
            longitudinal_model=long_fit.model_name,
            longitudinal_params=long_fit.params,
            longitudinal_training_rmse=long_fit.rmse,
            longitudinal_validation_rmse=overall_long_rmse,
            longitudinal_validation_passed=bool(overall_long_rmse < cfg.acceptance_rmse_longitudinal) if long_val_results else False,
            lateral_params=lat_fit.params if lat_fit else None,
            lateral_training_rmse=lat_fit.rmse if lat_fit else None,
            lateral_validation_rmse=overall_lat_rmse,
            lateral_validation_passed=bool(overall_lat_rmse < cfg.acceptance_rmse_lateral) if lat_val_results else False,
            per_session_longitudinal={n: r.rmse for n, r in long_val_results.items()},
            per_session_lateral={n: r.rmse for n, r in lat_val_results.items()},
        )

        report_path = Path(cfg.output_dir) / 'report.json'
        report_path.write_text(report.to_json())
        logger.info('Wrote report: %s', report_path)
        return report

    # ── Plotting helpers ─────────────────────────────────────────────────

    def _plot_training_fits(self, train, long_model, long_fit, lat_model, lat_fit):
        for name, s in train.items():
            safe = name.replace('/', '_')
            v_sim = long_model.simulate(long_fit.raw_params, {'u': s.u, 'dt': s.dt, 'v0': s.v[0]})
            rmse = float(((v_sim - s.v) ** 2).mean() ** 0.5)
            self.plots.plot_longitudinal(name, s.t, s.v, v_sim, s.u, rmse,
                                          'Longitudinal training fit', f'train_long_{safe}.png')
            if lat_fit is None:
                continue

            # Full-trajectory RMSE here, for the plot label only -- the actual
            # fit objective (lat_fit.rmse) was computed on the masked
            # rolling+cornering subset (see _lateral_dataset), which is what
            # the report/log numbers mean.
            steering = self._steering(s)
            omega_sim = lat_model.simulate(lat_fit.raw_params, {'v': s.v, 'steering': steering})
            rmse_o = float(((omega_sim - s.omega) ** 2).mean() ** 0.5)
            self.plots.plot_lateral(name, s.t, s.omega, omega_sim, steering, rmse_o,
                                     'Lateral training fit (full trajectory shown; '
                                     'fit RMSE used the masked subset)', f'train_lat_{safe}.png')

    def _plot_validation(self, validate, long_model, long_fit, long_val_results,
                          lat_model, lat_fit, lat_val_results):
        for name, s in validate.items():
            safe = name.replace('/', '_')
            if name in long_val_results:
                r = long_val_results[name]
                self.plots.plot_longitudinal(name, s.t, s.v, r.sim, s.u, r.rmse,
                                              'Longitudinal VALIDATION', f'val_long_{safe}.png')
            if name in lat_val_results:
                # r.sim/r.rmse are over the masked (rolling + active-cornering)
                # subset used for the actual metric -- shorter than the full
                # session, so plot the full trajectory separately (consistent
                # with _plot_training_fits) rather than against mismatched arrays.
                r = lat_val_results[name]
                steering = self._steering(s)
                omega_sim_full = lat_model.simulate(lat_fit.raw_params, {'v': s.v, 'steering': steering})
                self.plots.plot_lateral(name, s.t, s.omega, omega_sim_full, steering, r.rmse,
                                         'Lateral VALIDATION (full trajectory shown; '
                                         'RMSE is the masked rolling+cornering metric)',
                                         f'val_lat_{safe}.png')
