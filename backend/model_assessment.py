"""
Deep-Learning Model Assessment Engine for RehabVision AI.

Provides a dedicated model-assessment interface that accepts validated movement sequences,
checks input schema compatibility, verifies numerical integrity (finite/non-NaN),
measures inference latency, and returns structured predictions or explicit rejection reasons.

Delegates execution to registered model adapters in model_adapters.registry.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Union
import time
import numpy as np

from model_adapters import model_registry, ModelRegistry

@dataclass
class ModelAssessmentResult:
    model_id: str
    model_version: str
    architecture: str
    input_schema: str
    input_quality_status: str  # "VALID" | "INCOMPATIBLE_REPRESENTATION" | "INVALID_SHAPE" | "NUMERICAL_ANOMALY" | "MODEL_UNAVAILABLE" | "EXERCISE_MISMATCH"
    predicted_class: Optional[int] = None
    class_label: Optional[str] = None  # "correct" | "incorrect" | None
    probabilities: Optional[Dict[str, float]] = None  # {"incorrect": float, "correct": float}
    probability_calibration_status: str = "uncalibrated_softmax"
    inference_latency_ms: float = 0.0
    rejection_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelAssessmentEngine:
    """
    Dedicated assessment interface for deep learning rehabilitation form models.
    Supports model-agnostic execution by delegating to registered model adapters in model_registry.
    Strictly isolates verified motion-capture inputs from live uncalibrated webcam inputs.
    """
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or model_registry
        self.model_id = "rehab-lstm-rehab24-6-v1"
        self.model_version = "1.0.0"
        self.architecture = "PyTorch 2-Layer LSTM (unidirectional, hidden=64)"
        self.supported_input_schema = "mocap_3d_64x80"

    def evaluate_sequence(
        self,
        sequence: Union[List[List[float]], np.ndarray],
        input_source: str = "verified_mocap",
        exercise_id: str = "standing-right-arm-abduction",
        model_id: Optional[str] = None,
        repetition_index: int = 1
    ) -> ModelAssessmentResult:
        """
        Evaluate a movement sequence using the registered model adapter for the given exercise.
        Guards against domain shift, missing values, NaN/Inf, shape mismatches, and exercise mismatches.
        """
        # 1. Resolve adapter from registry
        if model_id:
            adapter = self.registry.get(model_id)
        else:
            adapter = self.registry.get_default_for_exercise(exercise_id)

        if adapter is None:
            return ModelAssessmentResult(
                model_id=model_id or f"unregistered-{exercise_id}",
                model_version="0.0.0",
                architecture="none",
                input_schema="none",
                input_quality_status="MODEL_UNAVAILABLE",
                rejection_reason=f"No machine learning model is registered for exercise '{exercise_id}'. Statistical biomechanical assessment remains operational.",
                warnings=[f"ML model unavailable for exercise '{exercise_id}'; statistical assessment active."]
            )

        meta = adapter.get_metadata()
        pred_res = adapter.evaluate_input(
            input_data=sequence,
            input_source=input_source,
            exercise_id=exercise_id
        )

        return ModelAssessmentResult(
            model_id=pred_res.model_id,
            model_version=pred_res.model_version,
            architecture=meta.framework,
            input_schema=meta.input_schema,
            input_quality_status=pred_res.quality_status,
            predicted_class=pred_res.predicted_class,
            class_label=pred_res.class_label,
            probabilities=pred_res.probabilities,
            probability_calibration_status=pred_res.probability_calibration_status,
            inference_latency_ms=pred_res.inference_latency_ms,
            rejection_reason=pred_res.rejection_reason,
            warnings=pred_res.warnings
        )


model_assessment_engine = ModelAssessmentEngine()
