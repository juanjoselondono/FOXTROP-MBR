#!/usr/bin/env python3
"""
compute_velocity.py
===================
Derive v(t) and omega(t) from VICON pose data using Savitzky-Golay differentiation.

Input:  pose_raw.csv  (produced by extract_bag.py)
Output: vicon_velocity.csv  [t, x, y, yaw, vx, vy, v, omega]

Usage:
    python3 compute_velocity.py [pose_raw.csv]

Tuning:
    Increase WINDOW if the signal looks noisy.
    Decrease WINDOW if the velocity lags the real motion.
    WINDOW must be an odd integer.

    VICON @ 100 Hz → WINDOW=21 ≈ 0.21 s smoothing window
    VICON @ 200 Hz → WINDOW=41 ≈ 0.21 s smoothing window
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

INPUT_CSV  = sys.argv[1] if len(sys.argv) > 1 else 'pose_raw.csv'
OUTPUT_CSV = 'vicon_velocity.csv'

# ── Savitzky-Golay filter parameters ─────────────────────────────────────────
WINDOW = 21   # samples (odd number; increase for VICON @ 200 Hz)
POLY   = 3    # polynomial order

# ─────────────────────────────────────────────────────────────────────────────

df  = pd.read_csv(INPUT_CSV)
t   = df['t'].values
x   = df['x'].values
y   = df['y'].values
yaw = np.unwrap(df['yaw'].values)   # prevent ±π discontinuities

# Median timestep (assumed roughly uniform after sorting)
dt = float(np.median(np.diff(t)))

print(f'Loaded {len(df)} pose samples, median dt={dt*1000:.2f} ms')
print(f'Using Savitzky-Golay: window={WINDOW}, polyorder={POLY}')

vx    = savgol_filter(x,   window_length=WINDOW, polyorder=POLY, deriv=1, delta=dt)
vy    = savgol_filter(y,   window_length=WINDOW, polyorder=POLY, deriv=1, delta=dt)
omega = savgol_filter(yaw, window_length=WINDOW, polyorder=POLY, deriv=1, delta=dt)

# Signed forward speed in the vehicle body frame
v_mag    = np.sqrt(vx**2 + vy**2)
heading_dot = vx * np.cos(yaw) + vy * np.sin(yaw)
v_signed = np.where(heading_dot >= 0, v_mag, -v_mag)

# ── Save ─────────────────────────────────────────────────────────────────────
df_out = pd.DataFrame({
    't':     t,
    'x':     x,
    'y':     y,
    'yaw':   yaw,
    'vx':    vx,
    'vy':    vy,
    'v':     v_signed,
    'omega': omega,
})
df_out.to_csv(OUTPUT_CSV, index=False)

print(f'\nSaved: {OUTPUT_CSV}')
print(f'  v range   : [{v_signed.min():.3f}, {v_signed.max():.3f}] m/s')
print(f'  omega range: [{omega.min():.3f}, {omega.max():.3f}] rad/s')

# ── Quick plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(t, x, label='x [m]', color='steelblue')
axes[0].plot(t, y, label='y [m]', color='tomato')
axes[0].set_ylabel('Position [m]')
axes[0].legend(); axes[0].grid(True)

axes[1].plot(t, np.degrees(yaw), label='yaw [deg]', color='purple')
axes[1].set_ylabel('Heading [°]')
axes[1].legend(); axes[1].grid(True)

axes[2].plot(t, v_signed, label='v_signed [m/s]', color='green')
axes[2].set_ylabel('Speed [m/s]')
axes[2].set_xlabel('Time [s]')
axes[2].legend(); axes[2].grid(True)

plt.suptitle('VICON Ground-Truth Velocity (Savitzky-Golay)')
plt.tight_layout()
plt.savefig('vicon_velocity_plot.png', dpi=150)
plt.show()
print('Saved: vicon_velocity_plot.png')
