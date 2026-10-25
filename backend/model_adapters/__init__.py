"""
Model Adapters package for RehabVision AI.
Provides model-agnostic interfaces for machine-learning and deep-learning integration.
"""

from model_adapters.base import (
    BaseModelAdapter,
    ModelMetadata,
    ModelValidationResult,
    ModelPredictionResult,
)
from model_adapters.registry import ModelRegistry, model_registry
from model_adapters.existing_mocap_lstm import ExistingMocapLSTMAdapter
from model_adapters.example_teammate_adapter import ExampleTeammateModelAdapter

__all__ = [
    "BaseModelAdapter",
    "ModelMetadata",
    "ModelValidationResult",
    "ModelPredictionResult",
    "ModelRegistry",
    "model_registry",
    "ExistingMocapLSTMAdapter",
    "ExampleTeammateModelAdapter",
]
