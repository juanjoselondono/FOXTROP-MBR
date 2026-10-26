"""
models.py
=========
Step 3: parametric vehicle models.

Longitudinal:
  - FirstOrderDeadZoneModel (base):   v_dot = (K * u_eff - v) / tau
  - NonlinearDragModel (fallback):    v_dot = K_drive * u_eff - b*v - c*v*|v|
    (mass folded into K_drive/b/c, i.e. m == 1; v*|v| keeps drag opposing
    motion correctly in reverse too)

Lateral (Ackermann kinematics):
  - AckermannLateralModel:            omega = (v / L_eff) * tan(delta)
    delta = joy_steering * max_steer_rad (a fixed, assumed calibration --
    only L_eff is identified).

All models share the ParametricModel interface so ModelFitter (optimization.py)
can fit any of them without special-casing: simulate(params, inputs) -> array.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class ParametricModel(ABC):
    name: str
    param_names: Sequence[str]
    bounds: Sequence[tuple]
    x0: Sequence[float]

    @abstractmethod
    def simulate(self, params: Sequence[float], inputs: dict) -> np.ndarray:
        """Returns the simulated output array for this model given `inputs` (keys depend on the model)."""

    def param_dict(self, params: Sequence[float]) -> dict:
        return dict(zip(self.param_names, params))


def apply_deadzone(u: np.ndarray, u_dead: float) -> np.ndarray:
    u_eff = np.zeros_like(u)
    u_eff[u > u_dead] = u[u > u_dead] - u_dead
    u_eff[u < -u_dead] = u[u < -u_dead] + u_dead
    return u_eff


class FirstOrderDeadZoneModel(ParametricModel):
    """v_dot = (K * u_eff - v) / tau,  u_eff = deadzone(u, u_dead). inputs: {'u', 'dt', 'v0'(optional)}."""

    name = 'first_order_deadzone'
    param_names = ['K', 'tau', 'u_dead']
    bounds = [(0.1, 5.0), (0.01, 2.0), (0.0, 0.4)]
    x0 = [1.0, 0.3, 0.05]

    def simulate(self, params, inputs):
        K, tau, u_dead = params
        u, dt = inputs['u'], inputs['dt']
        u_eff = apply_deadzone(u, u_dead)
        v = np.zeros(len(u))
        # Warm-start from the segment's own measured initial speed, not 0 --
        # matters when a session was split around a VICON glitch (see
        # filtering.py): the car's real velocity was continuous across the
        # gap even though the observation wasn't, so forcing v[0]=0 would
        # bias every split segment with a fake startup transient.
        if len(v) and 'v0' in inputs:
            v[0] = inputs['v0']
        # Exact zero-order-hold discretization of the LTI ODE, unconditionally
        # stable for any dt > 0. Forward Euler (v += dt*(K*u_eff-v)/tau) is
        # only stable for dt < 2*tau -- a real risk here since tau's lower
        # bound (0.01s) sits right at the VICON sample period, so the
        # optimizer can wander into Euler's divergent region and blow up.
        alpha = np.exp(-dt / tau)
        for i in range(1, len(u)):
            v[i] = v[i - 1] * alpha + K * u_eff[i - 1] * (1.0 - alpha)
        return v


class NonlinearDragModel(ParametricModel):
    """m*v_dot = K_drive*u_eff - b*v - c*v*|v|  (m folded into the other gains). inputs: {'u', 'dt'}.

    Use when FirstOrderDeadZoneModel's training RMSE is too high -- e.g. the
    car has noticeable quadratic (aero/rolling) drag at higher speeds that a
    single time constant can't capture.
    """

    name = 'nonlinear_drag'
    param_names = ['K_drive', 'b', 'c', 'u_dead']
    bounds = [(0.1, 10.0), (0.01, 5.0), (0.0, 5.0), (0.0, 0.4)]
    x0 = [1.0, 1.0, 0.1, 0.05]

    def __init__(self, n_substeps: int = 5):
        # Nonlinear, so there's no exact discretization like the first-order
        # model's exp(-dt/tau). Sub-stepping the explicit Euler integration
        # keeps it stable even when fitted (b, c) imply a time constant
        # short relative to the VICON sample period.
        self.n_substeps = max(1, n_substeps)

    def simulate(self, params, inputs):
        K_drive, b, c, u_dead = params
        u, dt = inputs['u'], inputs['dt']
        u_eff = apply_deadzone(u, u_dead)
        h = dt / self.n_substeps
        v = np.zeros(len(u))
        if len(v) and 'v0' in inputs:  # see FirstOrderDeadZoneModel.simulate for why
            v[0] = inputs['v0']
        for i in range(1, len(u)):
            vp = v[i - 1]
            ui = u_eff[i - 1]
            for _ in range(self.n_substeps):
                v_dot = K_drive * ui - b * vp - c * vp * abs(vp)
                vp = vp + h * v_dot
            v[i] = vp
        return v


class AckermannLateralModel(ParametricModel):
    """omega = (v / L_eff) * tan(delta), delta = K_steer * joy_steering.
    inputs: {'v', 'joy_steering'}.

    L_eff is a FIXED, physically measured constant (not fit) -- for this
    chassis (RC Mini Cooper R56), 0.245 m measured wheelbase. Letting the
    optimizer treat L_eff as free let it just run to its upper bound instead
    of converging, since L_eff and the steering-to-angle gain are only
    identifiable as a ratio from omega/v/joy_steering data alone (the model
    is omega = (K_steer/L_eff) * v * joy_steering to first order, so one of
    the two must be pinned). K_steer (the virtual max steering angle at
    joy_steering=1) is what actually gets fit.

    Bounds are SYMMETRIC (+/- max_steer_rad), not one-sided: this is the
    sign check for whether the joystick's polarity matches the kinematic
    convention (positive joy_steering -> positive yaw rate in the VICON
    frame). Pinning K_steer >= 0 would silently hide a sign mismatch by
    forcing a bad fit instead of just reporting it; letting the optimizer
    choose the sign freely is the actual verification -- if it converges
    negative, the joystick axis and the VICON frame disagree on handedness.
    """

    name = 'ackermann_lateral'
    param_names = ['K_steer']

    def __init__(self, L_eff: float = 0.245, max_steer_rad: float = 0.5236):
        self.L_eff = L_eff
        self.bounds = [(-max_steer_rad, max_steer_rad)]
        self.x0 = [0.01]  # near zero but nonzero, so L-BFGS-B isn't started on a flat/symmetric point

    def simulate(self, params, inputs):
        (K_steer,) = params
        v, joy_steering = inputs['v'], inputs['joy_steering']
        delta = K_steer * joy_steering
        return (v / self.L_eff) * np.tan(delta)
