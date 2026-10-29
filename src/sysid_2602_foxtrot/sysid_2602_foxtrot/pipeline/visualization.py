"""
visualization.py
=================
Step 5b: comparison plots (v_sim vs v_vicon, omega_sim vs omega_vicon,
raw command input), saved automatically to an output directory.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # headless-safe for unattended batch runs
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np

from .validation import ValidationResult

logger = logging.getLogger(__name__)

# Reference categorical slots 1-2 (blue, orange) in fixed order; ink/surface tokens.
REGIME_COLORS = {'low': '#2a78d6', 'medium': '#eb6834'}
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
GRID_INK = '#e4e3df'
SURFACE = '#fcfcfb'


class PlotGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_longitudinal(self, session_name: str, t: np.ndarray, v_true: np.ndarray,
                           v_sim: np.ndarray, u: np.ndarray, rmse: float,
                           title_prefix: str, filename: str) -> Path:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        ax1.plot(t, v_true, label='VICON ground truth', color='steelblue', lw=1.5)
        ax1.plot(t, v_sim, label=f'Model  RMSE={rmse:.3f} m/s', color='tomato', ls='--', lw=1.5)
        ax1.set_ylabel('Speed [m/s]'); ax1.legend(); ax1.grid(True)

        ax2.step(t, u, where='post', color='orange', label='command u')
        ax2.set_ylabel('u [-1, 1]'); ax2.set_xlabel('Time [s]'); ax2.legend(); ax2.grid(True)

        fig.suptitle(f'{title_prefix}: {session_name}')
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_lateral(self, session_name: str, t: np.ndarray, omega_true: np.ndarray,
                      omega_sim: np.ndarray, joy_steering: np.ndarray, rmse: float,
                      title_prefix: str, filename: str) -> Path:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        ax1.plot(t, omega_true, label='VICON ground truth', color='steelblue', lw=1.5)
        ax1.plot(t, omega_sim, label=f'Ackermann model  RMSE={rmse:.3f} rad/s', color='tomato', ls='--', lw=1.5)
        ax1.set_ylabel('Yaw rate [rad/s]'); ax1.legend(); ax1.grid(True)

        ax2.step(t, joy_steering, where='post', color='orange', label='steering command (angular.z / 0.5)')
        ax2.set_ylabel('steering [-1, 1]'); ax2.set_xlabel('Time [s]'); ax2.legend(); ax2.grid(True)

        fig.suptitle(f'{title_prefix}: {session_name}')
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_summary_bar(self, results: Dict[str, ValidationResult], acceptance_rmse: float,
                          ylabel: str, filename: str) -> Optional[Path]:
        if not results:
            return None
        names = list(results.keys())
        rmses = [r.rmse for r in results.values()]
        colors = ['seagreen' if r.passed else 'crimson' for r in results.values()]

        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(names)), 5))
        ax.bar(names, rmses, color=colors)
        ax.axhline(acceptance_rmse, color='black', ls='--', lw=1, label=f'acceptance ({acceptance_rmse})')
        ax.set_ylabel(ylabel)
        ax.set_title('Validation RMSE per held-out session')
        ax.legend()
        ax.grid(True, axis='y')
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


class RegimePlotter:
    """Small multiples of clean step segments: per segment, the response (ground
    truth + regime-colored simulation) on top and the command (raw + quantized)
    in its own panel below -- separate panels, never a second y-axis."""

    def __init__(self, output_dir: Path, per_page: int = 12, ncols: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.per_page = per_page
        self.ncols = ncols

    @staticmethod
    def _style(ax):
        ax.set_facecolor(SURFACE)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(INK_SECONDARY)
        ax.tick_params(colors=INK_SECONDARY, labelsize=7)
        ax.grid(True, color=GRID_INK, lw=0.6)
        ax.set_axisbelow(True)

    def plot(self, results: List, kind: str, split: str) -> List[Path]:
        """results: List[regimes.SegmentResult]; kind: 'longitudinal' | 'lateral'."""
        if not results:
            logger.info('No %s %s segments to plot', kind, split)
            return []
        results = sorted(results, key=lambda r: (r.segment.regime, r.segment.session, r.segment.t[0]))
        if kind == 'longitudinal':
            y_label, unit, cmd_label = 'v [m/s]', 'm/s', 'u_norm'
        else:
            y_label, unit, cmd_label = 'ω [rad/s]', 'rad/s', 'steering'

        paths = []
        n_pages = math.ceil(len(results) / self.per_page)
        for page in range(n_pages):
            chunk = results[page * self.per_page:(page + 1) * self.per_page]
            ncols = min(self.ncols, len(chunk))
            nrows = math.ceil(len(chunk) / ncols)
            height = 3.8 * nrows + 1.0
            fig = plt.figure(figsize=(4.2 * ncols, height), facecolor=SURFACE)
            # Outer grid spaces the segments apart; each cell's inner grid keeps
            # its response and command panels tight together.
            outer = GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.3,
                             top=1 - 0.95 / height, bottom=0.45 / height)

            for i, r in enumerate(chunk):
                s = r.segment
                row, col = divmod(i, ncols)
                inner = outer[row, col].subgridspec(2, 1, height_ratios=[3, 1.2], hspace=0.12)
                ax_y = fig.add_subplot(inner[0])
                ax_u = fig.add_subplot(inner[1], sharex=ax_y)
                t = s.t - s.t[0]
                truth = s.v if kind == 'longitudinal' else s.omega
                cmd_raw = s.u_raw if kind == 'longitudinal' else s.steer_raw
                cmd_clean = s.u_clean if kind == 'longitudinal' else s.steer_clean

                ax_y.plot(t, truth, color=INK_PRIMARY, lw=1.5)
                ax_y.plot(t, r.sim, color=REGIME_COLORS.get(s.regime, INK_SECONDARY), lw=2, ls='--')
                ax_u.plot(t, cmd_raw, color=INK_SECONDARY, lw=1, alpha=0.45)
                ax_u.step(t, cmd_clean, where='post', color=INK_SECONDARY, lw=1.6)

                ax_y.set_title(f'{s.session}\n{s.regime.capitalize()} · |u|={s.mean_abs_u:.2f} '
                               f'· RMSE {r.rmse:.3f} {unit}', fontsize=8, color=INK_PRIMARY)
                ax_y.set_ylabel(y_label, fontsize=7, color=INK_SECONDARY)
                ax_u.set_ylabel(cmd_label, fontsize=7, color=INK_SECONDARY)
                ax_u.set_xlabel('t [s]', fontsize=7, color=INK_SECONDARY)
                plt.setp(ax_y.get_xticklabels(), visible=False)
                self._style(ax_y)
                self._style(ax_u)

            handles = [
                Line2D([], [], color=INK_PRIMARY, lw=1.5, label='VICON ground truth'),
                Line2D([], [], color=REGIME_COLORS['low'], lw=2, ls='--', label='Model · Low regime'),
                Line2D([], [], color=REGIME_COLORS['medium'], lw=2, ls='--', label='Model · Medium regime'),
                Line2D([], [], color=INK_SECONDARY, lw=1, alpha=0.45, label='Command (raw)'),
                Line2D([], [], color=INK_SECONDARY, lw=1.6, label='Command (segment level, de-jittered)'),
            ]
            fig.suptitle(f'{kind.capitalize()} step segments — {split}'
                         + (f' (page {page + 1}/{n_pages})' if n_pages > 1 else ''),
                         fontsize=11, color=INK_PRIMARY, y=1 - 0.12 / height)
            fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, fontsize=8,
                       labelcolor=INK_PRIMARY, bbox_to_anchor=(0.5, 1 - 0.38 / height))
            path = self.output_dir / f'regime_{kind}_{split}' \
                f'{f"_p{page + 1}" if n_pages > 1 else ""}.png'
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=SURFACE)
            plt.close(fig)
            paths.append(path)
        return paths
