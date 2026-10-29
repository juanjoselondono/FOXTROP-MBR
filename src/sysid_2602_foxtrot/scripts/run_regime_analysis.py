#!/usr/bin/env python3
"""
run_regime_analysis.py
=======================
Regime-specific (Low / Medium speed) identification from clean step segments
(sysid_2602_foxtrot.pipeline.regimes).

Two input modes:
  1. From bags (default): runs the main pipeline's extraction + glitch-split
     differentiation on --bag-root, splits train/validate by session name
     (session_A*/B*/manual_session* = train, session_C* = validate), and also
     writes the synchronized DataFrames to train_synced.csv / val_synced.csv.
  2. From CSV: --train-csv/--val-csv, columns t,u,steer_cmd,v,omega[,session]
     (u = cmd_vel.linear.x, steer_cmd = cmd_vel.angular.z, both un-normalized).

Usage:
    python3 run_regime_analysis.py
    python3 run_regime_analysis.py --train-csv train.csv --val-csv val.csv
    python3 run_regime_analysis.py --min-duration 1.5 --quant-tolerance 0.05
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sysid_2602_foxtrot.pipeline import (SysIDPipeline, PipelineConfig, SessionClassifier,
                                          RegimeAnalysis, RegimeConfig, sessions_to_dataframe)


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bag-root', type=Path, default=here.parent / 'bag_files' / 'sysid')
    p.add_argument('--train-csv', type=Path)
    p.add_argument('--val-csv', type=Path)
    p.add_argument('--output-dir', type=Path, default=here.parent / 'sysid_pipeline_output' / 'regimes')
    p.add_argument('--u-full-scale', type=float, default=0.5,
                    help='linear.x that equals 100%% ESC command (twist_cmd_node maps ±0.5 to full PWM)')
    p.add_argument('--steer-full-scale', type=float, default=0.5,
                    help='angular.z that equals full steering lock (twist_cmd_node maps ±0.5 to full servo range)')
    p.add_argument('--quant-tolerance', type=float, default=0.03,
                    help='Hysteresis band (normalized). Keep < 0.04, one manual_cruise Up/Down step')
    p.add_argument('--min-duration', type=float, default=1.2)
    p.add_argument('--low-max', type=float, default=0.4)
    p.add_argument('--medium-max', type=float, default=0.8)
    p.add_argument('--train-pattern', action='append',
                    help='Regex on session folder name for TRAINING (repeatable). Default: session_a/b, manual_session')
    p.add_argument('--val-pattern', action='append',
                    help='Regex for VALIDATION (repeatable, takes precedence over --train-pattern). Default: session_c')
    p.add_argument('--include-braking', action='store_true',
                    help='Keep negative-throttle (brake/reverse) windows; dropped by default')
    p.add_argument('--wheelbase-m', type=float, default=0.245)
    p.add_argument('--max-steer-deg', type=float, default=30.0)
    p.add_argument('-v', '--verbose', action='store_true')
    return p.parse_args()


def load_frames(args):
    if args.train_csv or args.val_csv:
        if not (args.train_csv and args.val_csv):
            sys.exit('--train-csv and --val-csv must be given together')
        return pd.read_csv(args.train_csv), pd.read_csv(args.val_csv)

    pipe = SysIDPipeline(PipelineConfig(bag_root=args.bag_root, output_dir=args.output_dir))
    sessions = pipe.filter(pipe.extract())
    cfg = pipe.config
    train_p = args.train_pattern or cfg.train_patterns
    val_p = args.val_pattern or cfg.validate_patterns
    train, validate, _ = SessionClassifier(train_p, val_p).split(sessions)
    logging.getLogger(__name__).info('Split: %d train / %d validation segment(s)', len(train), len(validate))
    train_df, val_df = sessions_to_dataframe(train), sessions_to_dataframe(validate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.output_dir / 'train_synced.csv', index=False)
    val_df.to_csv(args.output_dir / 'val_synced.csv', index=False)
    return train_df, val_df


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    train_df, val_df = load_frames(args)
    cfg = RegimeConfig(
        u_full_scale=args.u_full_scale, steer_full_scale=args.steer_full_scale,
        quant_tolerance=args.quant_tolerance,
        min_duration=args.min_duration, low_max=args.low_max, medium_max=args.medium_max,
        L_eff=args.wheelbase_m, max_steer_rad=math.radians(args.max_steer_deg),
        forward_only=not args.include_braking,
    )
    report = RegimeAnalysis(args.output_dir, cfg).run(train_df, val_df)

    def fmt(x, unit):
        return f'{x:.4f} {unit}' if x is not None else '—'

    print('\n' + '=' * 72)
    print('REGIME-SPECIFIC IDENTIFICATION')
    print('=' * 72)
    print('Segment counts:', json.dumps(report.segment_counts))
    for title, section, unit in (('Longitudinal (K, tau, u_dead)', report.longitudinal, 'm/s'),
                                 (f'Lateral (K_steer, L_eff={args.wheelbase_m} m fixed)', report.lateral, 'rad/s')):
        print(f'\n{title}')
        for regime, r in section.items():
            if not r['identified']:
                print(f'  {regime:6s}: NOT IDENTIFIED (no training step segments in this regime)')
                continue
            params = {k: round(v, 4) for k, v in r['params'].items()}
            note = '  [K NOT identified: no segment above u_dead]' if r.get('K_identifiable') is False else ''
            print(f'  {regime:6s}: {params}{note}')
            print(f'          train RMSE {fmt(r["train_rmse"], unit)} ({r["n_train_segments"]} seg)'
                  f'  |  val RMSE {fmt(r["val_rmse"], unit)} ({r["n_val_segments"]} seg)')
    print(f'\nPlots, regime_segments.csv, regime_report.json: {args.output_dir}')


if __name__ == '__main__':
    main()
