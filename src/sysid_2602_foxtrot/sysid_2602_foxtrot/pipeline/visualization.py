"""
visualization.py
=================
Step 5b: comparison plots (v_sim vs v_vicon, omega_sim vs omega_vicon,
raw command input), saved automatically to an output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use('Agg')  # headless-safe for unattended batch runs
import matplotlib.pyplot as plt
import numpy as np

from .validation import ValidationResult


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

        ax2.step(t, joy_steering, where='post', color='orange', label='joystick steering')
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
