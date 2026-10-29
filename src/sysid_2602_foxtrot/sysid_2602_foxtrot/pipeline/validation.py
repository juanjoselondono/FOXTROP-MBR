"""
validation.py
=============
Step 5a: strict train/validate session partitioning + cross-validation RMSE.

SessionClassifier splits sessions by folder-name convention so fitting never
sees the sessions it's graded on: 'session_a*'/'session_b*' (step/ramp
excitation, per the sysid procedure) and 'manual_session*' (constant-speed
runs from manual_cruise_node) are training data; 'session_c*' (circles /
mixed maneuvers) are validation-only. Override the regexes if your naming
differs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np

from .extraction import SessionData
from .models import ParametricModel

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    session_name: str
    rmse: float
    max_target: float
    relative_error_pct: float
    passed: bool
    sim: np.ndarray = field(default=None, repr=False)


class SessionClassifier:
    def __init__(self,
                 train_patterns: Sequence[str] = (r'^session_a', r'^session_b', r'^manual_session'),
                 validate_patterns: Sequence[str] = (r'^session_c',)):
        self.train_re = [re.compile(p, re.IGNORECASE) for p in train_patterns]
        self.validate_re = [re.compile(p, re.IGNORECASE) for p in validate_patterns]

    def classify(self, name: str) -> str:
        base = name.split('/')[-1]  # match on the leaf folder name, ignore any parent path
        # Validation wins: a broad train pattern must never swallow a held-out session.
        if any(r.search(base) for r in self.validate_re):
            return 'validate'
        if any(r.search(base) for r in self.train_re):
            return 'train'
        return 'unknown'

    def split(self, sessions: Dict[str, SessionData]) -> Tuple[Dict, Dict, Dict]:
        train, validate, unknown = {}, {}, {}
        for name, s in sessions.items():
            bucket = self.classify(name)
            (train if bucket == 'train' else validate if bucket == 'validate' else unknown)[name] = s
        if unknown:
            logger.warning('%d session(s) matched neither train nor validate pattern, excluded: %s',
                            len(unknown), list(unknown.keys()))
        return train, validate, unknown


class CrossValidator:
    """Simulates an already-fitted model against held-out sessions it never trained on."""

    def __init__(self, acceptance_rmse: float = 0.05):
        self.acceptance_rmse = acceptance_rmse

    def validate_session(self, model: ParametricModel, raw_params: Sequence[float],
                          inputs: dict, target: np.ndarray, name: str) -> ValidationResult:
        sim = model.simulate(raw_params, inputs)
        rmse = float(np.sqrt(np.mean((sim - target) ** 2)))
        max_t = float(np.max(np.abs(target))) if len(target) else 0.0
        rel = 100.0 * rmse / max_t if max_t > 1e-6 else float('nan')
        return ValidationResult(name, rmse, max_t, rel, rmse < self.acceptance_rmse, sim=sim)

    def validate_all(self, model: ParametricModel, raw_params: Sequence[float],
                      dataset: Dict[str, Tuple[dict, np.ndarray]]) -> Dict[str, ValidationResult]:
        return {
            name: self.validate_session(model, raw_params, inputs, target, name)
            for name, (inputs, target) in dataset.items()
        }

    @staticmethod
    def overall_rmse(results: Dict[str, ValidationResult]) -> float:
        if not results:
            return float('nan')
        return float(np.sqrt(np.mean([r.rmse ** 2 for r in results.values()])))
