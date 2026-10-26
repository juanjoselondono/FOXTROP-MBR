#!/usr/bin/env python3
"""
run_sysid_pipeline.py
======================
CLI entry point for the full offline sysid pipeline
(sysid_2602_foxtrot.pipeline). No ROS2 runtime needed -- only `rosbags`,
numpy, scipy, matplotlib.

Default bag_root is bag_files/sysid (NOT bag_files/, which also contains a
datosPrueba/ duplicate copy + .zip backup -- point --bag-root elsewhere if
you want those included too).

Usage:
    python3 run_sysid_pipeline.py
    python3 run_sysid_pipeline.py --bag-root ../bag_files/sysid --output-dir ../pipeline_out
    python3 run_sysid_pipeline.py --savgol-window 41 --nonlinear-fallback-threshold 0.03
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make sysid_2602_foxtrot importable without a ROS2 source

from sysid_2602_foxtrot.pipeline import SysIDPipeline, PipelineConfig


def parse_args():
    here = Path(__file__).resolve().parent
    default_bag_root = here.parent / 'bag_files' / 'sysid'

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bag-root', type=Path, default=default_bag_root,
                    help=f'Root directory to recursively search for bags (default: {default_bag_root})')
    p.add_argument('--output-dir', type=Path, default=here.parent / 'sysid_pipeline_output',
                    help='Where to write plots + report.json')
    p.add_argument('--topic-pose', default='/robot1/pose')
    p.add_argument('--topic-cmd', default='/cmd_vel_stamped')
    p.add_argument('--topic-joy', default='/joy')
    p.add_argument('--joy-throttle-axis', type=int, default=1)
    p.add_argument('--joy-steering-axis', type=int, default=2)
    p.add_argument('--position-scale', type=float, default=0.001,
                    help='Multiplier applied to raw VICON x/y (default 0.001 == assume millimeters)')
    p.add_argument('--max-pose-speed-mps', type=float, default=2.0,
                    help='Reject VICON samples implying a speed above this (occlusion/dropout glitch rejection)')
    p.add_argument('--savgol-window', type=int, default=21)
    p.add_argument('--savgol-polyorder', type=int, default=3)
    p.add_argument('--min-segment-duration', type=float, default=1.5,
                    help='Drop glitch-free segments shorter than this many seconds (too short to fit tau)')
    p.add_argument('--acceptance-rmse-longitudinal', type=float, default=0.20,
                    help='Validation RMSE [m/s] pass/fail threshold for the longitudinal model')
    p.add_argument('--acceptance-rmse-lateral', type=float, default=0.05,
                    help='Validation RMSE [rad/s] pass/fail threshold for the lateral model')
    p.add_argument('--nonlinear-fallback-threshold', type=float, default=0.05)
    p.add_argument('--wheelbase-m', type=float, default=0.245,
                    help='Fixed (measured, not fit) effective wheelbase L_eff [m] -- RC Mini Cooper R56: 0.245')
    p.add_argument('--max-steer-deg', type=float, default=30.0,
                    help='Bound on the fitted K_steer (virtual steering angle at joy_steering=±1)')
    p.add_argument('--lateral-min-speed-mps', type=float, default=0.5,
                    help='Lateral fit/validation only: exclude samples with |v_vicon| below this (rolling mask)')
    p.add_argument('--lateral-min-steering-cmd', type=float, default=0.15,
                    help='Lateral fit/validation only: exclude samples with |joy_steering| below this (active-cornering mask)')
    p.add_argument('-v', '--verbose', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    import math
    config = PipelineConfig(
        bag_root=args.bag_root,
        output_dir=args.output_dir,
        topic_pose=args.topic_pose,
        topic_cmd=args.topic_cmd,
        topic_joy=args.topic_joy,
        joy_throttle_axis=args.joy_throttle_axis,
        joy_steering_axis=args.joy_steering_axis,
        position_scale=args.position_scale,
        max_pose_speed_mps=args.max_pose_speed_mps,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
        min_segment_duration=args.min_segment_duration,
        acceptance_rmse_longitudinal=args.acceptance_rmse_longitudinal,
        acceptance_rmse_lateral=args.acceptance_rmse_lateral,
        nonlinear_fallback_threshold=args.nonlinear_fallback_threshold,
        L_eff=args.wheelbase_m,
        max_steer_rad=math.radians(args.max_steer_deg),
        lateral_min_speed_mps=args.lateral_min_speed_mps,
        lateral_min_steering_cmd=args.lateral_min_steering_cmd,
    )

    report = SysIDPipeline(config).run()

    print('\n' + '=' * 72)
    print('SYSID PIPELINE SUMMARY')
    print('=' * 72)
    print(f'Sessions: {report.n_sessions_train} train / {report.n_sessions_validate} validate '
          f'/ {report.n_sessions_unknown} unknown(excluded) / {report.n_sessions_total} total')
    print(f'\nLongitudinal model : {report.longitudinal_model}')
    print(f'  params           : {report.longitudinal_params}')
    print(f'  training RMSE    : {report.longitudinal_training_rmse:.4f} m/s')
    print(f'  validation RMSE  : {report.longitudinal_validation_rmse:.4f} m/s  '
          f'(acceptance < {args.acceptance_rmse_longitudinal:.2f})  '
          f'[{"PASS" if report.longitudinal_validation_passed else "FAIL"}]')
    print(f'\nLateral model (Ackermann, L_eff fixed at {args.wheelbase_m:.3f} m)')
    print(f'  params           : {report.lateral_params}')
    print(f'  training RMSE    : {report.lateral_training_rmse:.4f} rad/s')
    print(f'  validation RMSE  : {report.lateral_validation_rmse:.4f} rad/s  '
          f'(acceptance < {args.acceptance_rmse_lateral:.2f})  '
          f'[{"PASS" if report.lateral_validation_passed else "FAIL"}]')
    print(f'\nFull report + plots: {args.output_dir}')


if __name__ == '__main__':
    main()
