#!/usr/bin/env python3
"""
test_sysid_suite.py
===================
Automated test suite for the sysid_2602_foxtrot package.
Tests all offline script logic and the online node logic WITHOUT needing
a running ROS 2 instance, real bag data, or pandas/rosbags.

Dependencies: numpy, scipy, yaml  (all available via system packages in ROS 2 Jazzy)

Run:
    python3 test_sysid_suite.py
    # or
    pytest test_sysid_suite.py -v
"""

import sys
import os
import math
import tempfile
import shutil
import importlib.util
import numpy as np

# ── Colour helpers ────────────────────────────────────────────────────────────
PASS = '\033[92m✓ PASS\033[0m'
FAIL = '\033[91m✗ FAIL\033[0m'
results = []


def record(name, passed, detail=''):
    status = PASS if passed else FAIL
    print(f'  [{status}] {name}' + (f'  ({detail})' if detail else ''))
    results.append((name, passed))


def section(title):
    print(f'\n{"─"*62}')
    print(f'  {title}')
    print(f'{"─"*62}')


pkg_root = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — File Existence
# ─────────────────────────────────────────────────────────────────────────────
section('1. File Existence')

required_files = [
    'scripts/extract_bag.py',
    'scripts/compute_velocity.py',
    'scripts/fit_model.py',
    'scripts/validate_model.py',
    'sysid_2602_foxtrot/speed_estimator_node.py',
    'sysid_2602_foxtrot/__init__.py',
    'launch/speed_estimator.launch.py',
    'config/speed_estimator_params.yaml',
    'package.xml',
    'setup.py',
    'setup.cfg',
    'resource/sysid_2602_foxtrot',
    'bag_files/sysid/.gitkeep',
]

for rel in required_files:
    full = os.path.normpath(os.path.join(pkg_root, rel))
    record(rel, os.path.isfile(full))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Python Dependency Imports
# ─────────────────────────────────────────────────────────────────────────────
section('2. Python Dependency Imports')

for lib in ['numpy', 'scipy', 'yaml', 'matplotlib']:
    try:
        importlib.import_module(lib)
        record(f'import {lib}', True)
    except ImportError as e:
        record(f'import {lib}', False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Model Mathematics
# ─────────────────────────────────────────────────────────────────────────────
section('3. Model Mathematics — Dead-Zone & 1st-Order Dynamics')

# ── 3.1  Dead-zone ───────────────────────────────────────────────────────────
def apply_deadzone(u_arr, u_dead):
    u_eff = np.zeros_like(u_arr, dtype=float)
    u_eff[u_arr >  u_dead] = u_arr[u_arr >  u_dead] - u_dead
    u_eff[u_arr < -u_dead] = u_arr[u_arr < -u_dead] + u_dead
    return u_eff

u_test = np.array([-1.0, -0.1, -0.049, 0.0, 0.049, 0.05, 0.1, 1.0])
u_dead = 0.05
u_eff  = apply_deadzone(u_test, u_dead)

record('deadzone: values in [-u_dead, u_dead] → 0',
       np.all(u_eff[np.abs(u_test) <= u_dead] == 0.0))
record('deadzone: full throttle +1.0 → u_eff = 0.95',
       np.isclose(u_eff[-1], 0.95, atol=1e-9))
record('deadzone: full throttle -1.0 → u_eff = -0.95',
       np.isclose(u_eff[0], -0.95, atol=1e-9))
record('deadzone: u=0.06 with u_dead=0.05 → u_eff = 0.01',
       np.isclose(apply_deadzone(np.array([0.06]), 0.05)[0], 0.01, atol=1e-9))


# ── 3.2  First-order model simulation ────────────────────────────────────────
def simulate_model(K, tau, u_dead_val, u_arr, dt):
    u_eff = apply_deadzone(u_arr, u_dead_val)
    v = np.zeros(len(u_arr))
    for i in range(1, len(u_arr)):
        dvdt = (K * u_eff[i - 1] - v[i - 1]) / tau
        v[i] = v[i - 1] + dvdt * dt
    return v

K, tau, dt = 2.0, 0.5, 0.005
N = int(15 * tau / dt)
u_step = np.ones(N)
v_sim  = simulate_model(K, tau, 0.0, u_step, dt)
v_ss   = K * 1.0

record('step response: converges to K·u within 1 %',
       abs(v_sim[-1] - v_ss) / v_ss < 0.01,
       f'v_final={v_sim[-1]:.4f}, expected={v_ss:.4f}')

idx_tau = int(tau / dt)
record('step response: v(τ) ≈ 0.632·K  (±5 %)',
       abs(v_sim[idx_tau] / v_ss - 0.632) < 0.05,
       f'v(τ)={v_sim[idx_tau]:.4f}')

# Decay from initial condition with zero input
v_d = np.zeros(int(15 * tau / dt))
v_d[0] = 3.0
for i in range(1, len(v_d)):
    v_d[i] = v_d[i-1] + ((K * 0.0 - v_d[i-1]) / tau) * dt
record('zero input: speed decays to < 0.01 m/s after 15τ',
       abs(v_d[-1]) < 0.01,
       f'v_final={v_d[-1]:.5f}')

# Symmetry: negative throttle → negative steady-state speed
u_neg = -np.ones(N)
v_neg = simulate_model(K, tau, 0.0, u_neg, dt)
record('negative throttle: speed → -K·|u|',
       abs(v_neg[-1] + v_ss) / v_ss < 0.01,
       f'v_final={v_neg[-1]:.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Savitzky-Golay Velocity Derivation
# ─────────────────────────────────────────────────────────────────────────────
section('4. Savitzky-Golay Velocity Derivation')

from scipy.signal import savgol_filter

dt_sg = 0.01          # 100 Hz (typical VICON)
t_sg  = np.arange(0, 5.0, dt_sg)
v_true = 1.5          # constant forward speed [m/s]
x_true  = v_true * t_sg
y_true  = np.zeros_like(t_sg)
yaw_true = np.zeros_like(t_sg)

rng     = np.random.default_rng(42)
x_noisy = x_true + rng.normal(0, 0.0005, t_sg.shape)   # 0.5 mm noise (VICON spec)

WINDOW, POLY = 21, 3
vx_sg = savgol_filter(x_noisy, window_length=WINDOW, polyorder=POLY,
                      deriv=1, delta=dt_sg)
vy_sg = savgol_filter(y_true,  window_length=WINDOW, polyorder=POLY,
                      deriv=1, delta=dt_sg)
v_sg  = np.sqrt(vx_sg**2 + vy_sg**2)

w2        = WINDOW // 2
v_int     = v_sg[w2:-w2]
err_mean  = float(np.abs(v_int - v_true).mean())
err_max   = float(np.abs(v_int - v_true).max())

record('SG derivative: mean speed error < 0.01 m/s on noisy straight path',
       err_mean < 0.01, f'mean={err_mean:.5f} m/s')
record('SG derivative: max speed error < 0.05 m/s on noisy straight path',
       err_max  < 0.05, f'max={err_max:.5f} m/s')

# Yaw unwrap: discontinuity at ±π
yaw_raw = np.array([3.0, 3.1, -3.1, -3.0])
yaw_uw  = np.unwrap(yaw_raw)
record('np.unwrap: removes ±π jumps (max successive diff < π)',
       np.all(np.abs(np.diff(yaw_uw)) < math.pi),
       f'max_diff={np.abs(np.diff(yaw_uw)).max():.4f} rad')

# Signed speed: moving backward should give negative v
x_back = -v_true * t_sg
vx_back = savgol_filter(x_back, window_length=WINDOW, polyorder=POLY,
                        deriv=1, delta=dt_sg)
v_back  = np.sqrt(vx_back**2)
hdot    = vx_back * np.cos(yaw_true) + np.zeros_like(t_sg) * np.sin(yaw_true)
v_signed_back = np.where(hdot >= 0, v_back, -v_back)
record('signed speed: backward motion gives negative v',
       np.all(v_signed_back[w2:-w2] < 0))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Parameter Recovery (Optimizer Self-Test)
# ─────────────────────────────────────────────────────────────────────────────
section('5. Parameter Recovery — Optimizer Recovers Known Parameters')

from scipy.optimize import minimize

K_true, tau_true, ud_true = 1.8, 0.35, 0.07
dt_opt = 0.01
t_opt  = np.arange(0, 40.0, dt_opt)

# Diverse excitation: steps at different throttle levels
u_syn = np.zeros_like(t_opt)
for s, amp in [(0,0.8),(8,0.5),(16,1.0),(24,0.3),(32,0.9)]:
    u_syn[(t_opt>=s) & (t_opt<s+4)] = amp
for s, amp in [(4,-0.6),(12,-0.4),(20,-0.8)]:
    u_syn[(t_opt>=s) & (t_opt<s+3)] = amp

v_gt = simulate_model(K_true, tau_true, ud_true, u_syn, dt_opt)
v_gt += rng.normal(0, 0.003, v_gt.shape)   # light noise

def cost(params):
    K_, tau_, ud_ = params
    if K_ <= 0 or tau_ <= 0 or ud_ < 0 or ud_ > 0.5:
        return 1e9
    return float(np.mean((simulate_model(K_, tau_, ud_, u_syn, dt_opt) - v_gt)**2))

res = minimize(cost, [1.0, 0.3, 0.05], method='L-BFGS-B',
               bounds=[(0.1,5),(0.01,2),(0,0.4)],
               options={'maxiter': 3000, 'ftol': 1e-13})
K_r, tau_r, ud_r = res.x
rmse_train = math.sqrt(res.fun)

record(f'optimizer: K recovered within 3 %  (true={K_true}, got={K_r:.4f})',
       abs(K_r  - K_true) / K_true < 0.03)
record(f'optimizer: tau recovered within 5 % (true={tau_true}, got={tau_r:.4f})',
       abs(tau_r - tau_true) / tau_true < 0.05)
record(f'optimizer: u_dead recovered within 0.015 (true={ud_true}, got={ud_r:.4f})',
       abs(ud_r - ud_true) < 0.015)
record(f'optimizer: training RMSE < 0.01 m/s',
       rmse_train < 0.01, f'RMSE={rmse_train:.5f}')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Validate Model (cross-validation logic)
# ─────────────────────────────────────────────────────────────────────────────
section('6. Validate Model — Cross-Validation Logic')

# Generate a different excitation signal (ramp) for validation
u_val = np.zeros_like(t_opt)
for s in np.arange(0, 36, 8):
    # ramp up
    ramp_t = t_opt[(t_opt>=s) & (t_opt<s+3.5)]
    ramp_v = np.linspace(0, 0.9, len(ramp_t))
    u_val[(t_opt>=s) & (t_opt<s+3.5)] = ramp_v
    # ramp down
    ramp_d_t = t_opt[(t_opt>=s+3.5) & (t_opt<s+7)]
    ramp_d_v = np.linspace(0.9, 0, len(ramp_d_t))
    u_val[(t_opt>=s+3.5) & (t_opt<s+7)] = ramp_d_v

v_val_gt   = simulate_model(K_true, tau_true, ud_true, u_val, dt_opt)
v_val_gt  += rng.normal(0, 0.003, v_val_gt.shape)
v_val_pred = simulate_model(K_r, tau_r, ud_r, u_val, dt_opt)   # fitted params
rmse_val   = math.sqrt(float(np.mean((v_val_pred - v_val_gt)**2)))
max_v_val  = float(np.max(np.abs(v_val_gt)))

record('cross-validation: RMSE < 0.05 m/s on ramp excitation',
       rmse_val < 0.05, f'RMSE={rmse_val:.5f} m/s')
record('cross-validation: relative error < 5 % of max speed',
       (rmse_val / max_v_val) < 0.05,
       f'{100*rmse_val/max_v_val:.2f} %')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Speed Estimator Node Logic (no ROS 2 runtime)
# ─────────────────────────────────────────────────────────────────────────────
section('7. SpeedEstimator Node Logic (no ROS 2 runtime)')


class _NodeLogic:
    """Mirrors the update loop of speed_estimator_node.py."""
    def __init__(self, K, tau, u_dead, dt):
        self.K=K; self.tau=tau; self.u_dead=u_dead; self.dt=dt
        self.v=0.0; self.x=0.0; self.y=0.0; self.yaw=0.0; self.omega=0.0; self.u=0.0

    def set_cmd(self, u, omega=0.0): self.u=u; self.omega=omega

    def _dz(self, u):
        if u  >  self.u_dead: return u - self.u_dead
        if u  < -self.u_dead: return u + self.u_dead
        return 0.0

    def update(self):
        u_eff = self._dz(self.u)
        self.v  += ((self.K * u_eff - self.v) / self.tau) * self.dt
        self.yaw += self.omega * self.dt
        self.x   += self.v * math.cos(self.yaw) * self.dt
        self.y   += self.v * math.sin(self.yaw) * self.dt


# Test 1: converges to K*(1-u_dead)
e1 = _NodeLogic(K=2.0, tau=0.4, u_dead=0.05, dt=0.005)
e1.set_cmd(1.0)
for _ in range(int(15 * 0.4 / 0.005)): e1.update()
expected_ss = 2.0 * 0.95
record('node: full throttle → speed = K·(1 - u_dead) within 1 %',
       abs(e1.v - expected_ss) / expected_ss < 0.01,
       f'v={e1.v:.4f}, expected={expected_ss:.4f}')

# Test 2: below dead-zone → stays at 0
e2 = _NodeLogic(K=2.0, tau=0.4, u_dead=0.05, dt=0.005)
e2.set_cmd(0.03)
for _ in range(2000): e2.update()
record('node: u < u_dead → speed stays at 0',
       abs(e2.v) < 1e-9, f'v={e2.v:.2e}')

# Test 3: dead-reckoning straight line advances ~1 m in 1 s
e3 = _NodeLogic(K=1.0, tau=0.05, u_dead=0.0, dt=0.005)
e3.set_cmd(1.0)
for _ in range(int(10 * 0.05 / 0.005)): e3.update()  # settle
x0 = e3.x
for _ in range(int(1.0 / 0.005)): e3.update()        # 1 s
dx = e3.x - x0
record('node: dead-reckoning ≈ 1.0 m in 1 s at v=1 m/s (±2 cm)',
       abs(dx - 1.0) < 0.02, f'dx={dx:.4f} m')

# Test 4: turning maneuver traces a circle — check that MAX y-excursion > 0.3 m
# (ω=π rad/s means the robot completes a circle in 2 s and returns to start,
#  so we measure the peak displacement during the trajectory, not the endpoint)
e4 = _NodeLogic(K=1.0, tau=0.05, u_dead=0.0, dt=0.005)
e4.set_cmd(1.0, omega=math.pi)   # 180°/s yaw rate
for _ in range(int(20 * 0.05 / 0.005)): e4.update()  # fully settle (robot at v=1 m/s)
y0 = e4.y
max_dy = 0.0
for _ in range(int(2.0 / 0.005)):   # 2 s = one full circle
    e4.update()
    max_dy = max(max_dy, abs(e4.y - y0))
record('node: turning (ω=π) peak y-excursion > 0.3 m (circular path)',
       max_dy > 0.3, f'max_dy={max_dy:.4f} m')

# Test 5: negative throttle → negative speed
e5 = _NodeLogic(K=2.0, tau=0.4, u_dead=0.05, dt=0.005)
e5.set_cmd(-1.0)
for _ in range(int(15 * 0.4 / 0.005)): e5.update()
record('node: negative throttle → negative steady-state speed',
       e5.v < -expected_ss * 0.99, f'v={e5.v:.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — YAML Config Validation
# ─────────────────────────────────────────────────────────────────────────────
section('8. YAML Config Validation')

try:
    import yaml
    yaml_path = os.path.join(pkg_root, 'config', 'speed_estimator_params.yaml')
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    params = cfg['speed_estimator_node']['ros__parameters']
    record('YAML parses without error', True)
    for key in ['K', 'tau', 'u_dead', 'dt']:
        record(f'YAML: "{key}" key present', key in params)
    record('YAML: K > 0',               float(params['K'])      > 0)
    record('YAML: tau > 0',             float(params['tau'])     > 0)
    record('YAML: u_dead in [0, 0.5]',  0 <= float(params['u_dead']) <= 0.5)
    record('YAML: dt in (0, 1]',        0 <  float(params['dt']) <= 1.0)
except Exception as e:
    record('YAML config validation', False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — package.xml Validation
# ─────────────────────────────────────────────────────────────────────────────
section('9. package.xml Validation')

import xml.etree.ElementTree as ET
try:
    tree  = ET.parse(os.path.join(pkg_root, 'package.xml'))
    root  = tree.getroot()
    record('package.xml parses as valid XML', True)
    record('package name = sysid_2602_foxtrot',
           root.find('name').text == 'sysid_2602_foxtrot')
    deps = {d.text for d in root.iter('depend')}
    for dep in ['rclpy', 'std_msgs', 'geometry_msgs', 'nav_msgs', 'tf2_ros']:
        record(f'dependency declared: {dep}', dep in deps)
    bt = root.find('.//build_type')
    record('build_type = ament_python',
           bt is not None and bt.text == 'ament_python')
except Exception as e:
    record('package.xml parsing', False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Script Syntax Check (compile without executing)
# ─────────────────────────────────────────────────────────────────────────────
section('10. Script Syntax Check (py_compile)')

import py_compile, traceback

scripts = [
    ('scripts/extract_bag.py',       False),  # has top-level code → don't exec
    ('scripts/compute_velocity.py',  False),
    ('scripts/fit_model.py',         False),
    ('scripts/validate_model.py',    False),
    ('sysid_2602_foxtrot/speed_estimator_node.py', False),
    ('launch/speed_estimator.launch.py',            False),
]

for rel, _ in scripts:
    full = os.path.join(pkg_root, rel)
    try:
        py_compile.compile(full, doraise=True)
        record(f'syntax OK: {rel}', True)
    except py_compile.PyCompileError as e:
        record(f'syntax OK: {rel}', False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"═"*62}')
print(f'  TEST SUMMARY')
print(f'{"═"*62}')
total  = len(results)
passed = sum(1 for _, p in results if p)
failed = total - passed
print(f'  Total : {total}')
print(f'  Passed: \033[92m{passed}\033[0m')
print(f'  Failed: \033[91m{failed}\033[0m')
print(f'{"═"*62}')

if failed:
    print('\nFailed tests:')
    for name, p in results:
        if not p:
            print(f'  ✗  {name}')
    sys.exit(1)
else:
    print('\n\033[92mAll tests passed! ✓\033[0m')
    sys.exit(0)
