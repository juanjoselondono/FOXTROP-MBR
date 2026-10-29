"""Offline, ROS2-runtime-free system identification pipeline for the RC car.

Reads .mcap bags with the pure-Python `rosbags` library (no `rclpy`/sourced
workspace needed), synchronizes VICON + command + joystick streams,
differentiates ground-truth velocity/yaw-rate, fits parametric longitudinal
and lateral models by nonlinear least squares, and cross-validates on
held-out sessions.

See pipeline.SysIDPipeline for the end-to-end orchestrator, or use the
extraction/filtering/models/optimization/validation/visualization modules
directly for custom workflows.
"""

from .extraction import MCAPSessionReader, TimeSynchronizer, BatchBagExtractor, RawBagData, SessionData
from .filtering import SavGolDifferentiator
from .models import ParametricModel, FirstOrderDeadZoneModel, NonlinearDragModel, AckermannLateralModel
from .optimization import ModelFitter, FitResult
from .validation import SessionClassifier, CrossValidator, ValidationResult
from .visualization import PlotGenerator
from .pipeline import SysIDPipeline, PipelineConfig, PipelineReport
from .visualization import RegimePlotter
from .regimes import (ColumnMap, RegimeConfig, StepSegment, HysteresisQuantizer, RegimeClassifier,
                      StepSegmenter, RegimeIdentifier, RegimeValidator, RegimeAnalysis, RegimeReport,
                      sessions_to_dataframe)

__all__ = [
    'MCAPSessionReader', 'TimeSynchronizer', 'BatchBagExtractor', 'RawBagData', 'SessionData',
    'SavGolDifferentiator',
    'ParametricModel', 'FirstOrderDeadZoneModel', 'NonlinearDragModel', 'AckermannLateralModel',
    'ModelFitter', 'FitResult',
    'SessionClassifier', 'CrossValidator', 'ValidationResult',
    'PlotGenerator',
    'SysIDPipeline', 'PipelineConfig', 'PipelineReport',
    'RegimePlotter', 'ColumnMap', 'RegimeConfig', 'StepSegment', 'HysteresisQuantizer', 'RegimeClassifier',
    'StepSegmenter', 'RegimeIdentifier', 'RegimeValidator', 'RegimeAnalysis', 'RegimeReport',
    'sessions_to_dataframe',
]
