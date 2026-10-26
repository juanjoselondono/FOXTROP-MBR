#!/usr/bin/env python3
"""
validate_model.py
=================
Cross-validate the fitted model on a *different* bag (validation set).

Workflow:
  1. Run extract_bag.py on your second bag (sysid_run_02) to get new CSVs.
  2. Run compute_velocity.py on the new pose_raw.csv.
  3. Paste the fitted parameters from fitted_params.txt into the block below.
  4. Run: python3 validate_model.py

Acceptance criterion: Validation RMSE < 0.05 m/s  (or < 5 % of max speed).

Input files (must exist in current directory, from validation bag):
    vicon_velocity.csv
    esc_command_raw.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ── Paste fitted parameters here ──────────────────────────────────────────────
K_fit      = 1.0     # m/s per unit command  ← UPDATE after running fit_model.py
tau_fit    = 0.3     # s
u_dead_fit = 0.05    # normalised

# ─────────────────────────────────────────────────────────────────────────────

def apply_deadzone(u_arr, u_dead):
    u_eff = np.zeros_like(u_arr)
    u_eff[u_arr >  u_dead] = u_arr[u_arr >  u_dead] - u_dead
    u_eff[u_arr < -u_dead] = u_arr[u_arr < -u_dead] + u_dead
    return u_eff


def simulate_model(K, tau, u_dead, u_arr, dt):
    u_eff = apply_deadzone(u_arr, u_dead)
    v = np.zeros(len(u_arr))
    for i in range(1, len(u_arr)):
        dvdt = (K * u_eff[i - 1] - v[i - 1]) / tau
        v[i] = v[i - 1] + dvdt * dt
    return v


# ── Load validation data ──────────────────────────────────────────────────────
df_v   = pd.read_csv('vicon_velocity.csv')
df_cmd = pd.read_csv('esc_command_raw.csv')

t_vicon = df_v['t'].values
v_vicon = df_v['v'].values
t_cmd   = df_cmd['t'].values
u_raw   = df_cmd['u'].values

u_interp_fn = interp1d(t_cmd, u_raw, kind='zero', fill_value='extrapolate')
u  = u_interp_fn(t_vicon)
dt = float(np.median(np.diff(t_vicon)))

# ── Simulate ──────────────────────────────────────────────────────────────────
v_val = simulate_model(K_fit, tau_fit, u_dead_fit, u, dt)
rmse  = float(np.sqrt(np.mean((v_val - v_vicon) ** 2)))
max_v = float(np.max(np.abs(v_vicon)))

print(f'Validation RMSE : {rmse:.4f} m/s')
print(f'Max VICON speed : {max_v:.4f} m/s')
print(f'Relative error  : {100*rmse/max_v:.2f} %')
if rmse < 0.05:
    print('✓  PASS — RMSE < 0.05 m/s')
else:
    print('✗  FAIL — RMSE >= 0.05 m/s — consider refitting or using Model 3 (nonlinear drag)')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

ax1.plot(t_vicon, v_vicon, label='VICON ground truth', color='steelblue', lw=1.5)
ax1.plot(t_vicon, v_val,
         label=f'Model prediction  RMSE={rmse:.3f} m/s',
         color='tomato', linestyle='--', lw=1.5)
ax1.set_ylabel('Speed [m/s]')
ax1.legend(); ax1.grid(True)

ax2.step(t_vicon, u, label='ESC command u', color='orange', where='post')
ax2.set_ylabel('ESC command [-1, +1]')
ax2.set_xlabel('Time [s]')
ax2.legend(); ax2.grid(True)

plt.suptitle('Vehicle Speed Model — Validation')
plt.tight_layout()
plt.savefig('model_validation.png', dpi=150)
plt.show()
print('Saved: model_validation.png')
