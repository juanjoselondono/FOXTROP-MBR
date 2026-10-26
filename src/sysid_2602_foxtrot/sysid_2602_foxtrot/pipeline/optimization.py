"""
optimization.py
================
Step 4: nonlinear least-squares fitting of a ParametricModel's parameters
against one or more training sessions, via scipy.optimize.minimize
(L-BFGS-B, bounded).

Model-agnostic: works for any ParametricModel (longitudinal or lateral)
since the simulate(params, inputs) -> array interface is uniform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .models import ParametricModel

Dataset = Sequence[Tuple[dict, np.ndarray]]  # [(inputs, target), ...] one pair per training session


@dataclass
class FitResult:
    model_name: str
    params: dict
    raw_params: list
    mse: float
    rmse: float
    success: bool
    n_sessions: int
    message: str = field(default='', repr=False)


class ModelFitter:
    """Fits a single ParametricModel across possibly-many sessions at once.

    The cost is the mean of each session's own MSE (not the MSE over the
    concatenation of all sessions), so a long bag doesn't dominate a short
    one just by sample count.
    """

    def __init__(self, model: ParametricModel, method: str = 'L-BFGS-B',
                 maxiter: int = 2000, ftol: float = 1e-12):
        self.model = model
        self.method = method
        self.options = {'maxiter': maxiter, 'ftol': ftol}

    def _session_mse(self, params, inputs: dict, target: np.ndarray) -> float:
        sim = self.model.simulate(params, inputs)
        if sim is None or len(sim) != len(target) or not np.all(np.isfinite(sim)):
            return 1e9
        return float(np.mean((sim - target) ** 2))

    def _cost(self, params, dataset: Dataset) -> float:
        return float(np.mean([self._session_mse(params, inputs, target) for inputs, target in dataset]))

    def fit(self, dataset: Dataset) -> FitResult:
        if not dataset:
            raise ValueError(f'{self.model.name}: no training sessions provided')

        result = minimize(
            self._cost, self.model.x0, args=(dataset,),
            method=self.method, bounds=self.model.bounds, options=self.options,
        )
        mse = float(result.fun)
        return FitResult(
            model_name=self.model.name,
            params=self.model.param_dict(result.x),
            raw_params=list(result.x),
            mse=mse,
            rmse=float(np.sqrt(mse)) if mse >= 0 else float('nan'),
            success=bool(result.success),
            n_sessions=len(dataset),
            message=str(result.message),
        )
