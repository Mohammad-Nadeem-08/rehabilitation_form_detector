"""
Model-Agnostic Adapter Base Contract for RehabVision AI.

Defines a strict, typed, uniform interface that any machine-learning
or deep-learning model (LSTM, CNN, Transformer, GCN, ONNX, PyTorch)
must fulfill to integrate seamlessly into RehabVision AI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Union
import time


@dataclass
class ModelMetadata:
    model_id: str
    model_name: str
    model_version: str
    supported_exercise_ids: List[str]
    framework: str  # e.g., "PyTorch", "ONNX", "TorchScript", "Scikit-Learn"
    input_schema: str  # e.g., "mocap_3d_64x80", "mediapipe_2d_series"
    input_dimensions: List[int]  # e.g., [64, 80]
    coordinate_system: str  # e.g., "pelvis_centered_metric_3d"
    feature_order: List[str]  # Detailed ordering of features
    preprocessing_requirements: str  # Description of z-score, normalization, etc.
    class_mapping: Dict[int, str]  # e.g., {0: "incorrect", 1: "correct"}
    output_semantics: str  # e.g., "binary_classification_softmax"
    is_calibrated: bool  # Whether probabilities are calibrated or raw softmax
    device_requirement: str  # "cpu", "cuda", etc.
    validation_status: str  # "VALIDATED", "PRELIMINARY", "TEMPLATE_MOCK"
    author_provenance: str
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelValidationResult:
    is_valid: bool
    quality_status: str  # "VALID" | "INCOMPATIBLE_REPRESENTATION" | "INVALID_SHAPE" | "NUMERICAL_ANOMALY" | "MODEL_UNAVAILABLE" | "EXERCISE_MISMATCH"
    rejection_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelPredictionResult:
    model_id: str
    model_version: str
    predicted_class: Optional[int]
    class_label: Optional[str]
    probabilities: Optional[Dict[str, float]]
    probability_calibration_status: str  # "uncalibrated_softmax", "platt_calibrated", "raw_score"
    raw_logits: Optional[List[float]] = None
    inference_latency_ms: float = 0.0
    quality_status: str = "VALID"
    rejection_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseModelAdapter(ABC):
    """
    Abstract Model Adapter Contract.

    Guarantees that any model added to the system adheres to:
      1. get_metadata(): Full transparency of inputs, outputs, frameworks, and supported exercises.
      2. validate_input(): Early rejection of incompatible shapes, NaNs, domain shifts, or wrong exercise.
      3. preprocess(): Explicit, tested transformations without silent feature reinterpretation.
      4. predict(): Safe inference producing typed ModelPredictionResult.
      5. health_check(): Verifies model weights are loaded and host runtime is ready.
    """

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Returns complete, structured metadata describing this model adapter."""
        pass

    @abstractmethod
    def validate_input(
        self,
        input_data: Any,
        input_source: str,
        exercise_id: str
    ) -> ModelValidationResult:
        """
        Validates input against declared schema, shape, numeric integrity, and exercise compatibility.
        Must NEVER silently reshape or reorder incompatible inputs.
        """
        pass

    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """
        Applies documented preprocessing (e.g., z-score normalization, sequence extraction).
        Assumes input has passed validate_input().
        """
        pass

    @abstractmethod
    def predict(self, preprocessed_data: Any) -> ModelPredictionResult:
        """
        Executes model inference on preprocessed data and returns standardized prediction result.
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Verifies that model artifacts are present on disk, loaded in memory, and responsive.
        """
        pass

    def evaluate_input(
        self,
        input_data: Any,
        input_source: str = "verified_mocap",
        exercise_id: str = "standing-right-arm-abduction"
    ) -> ModelPredictionResult:
        """
        Template method orchestrating validation, preprocessing, and inference with timing.
        Safely captures exceptions and handles rejection without crashing.
        """
        meta = self.get_metadata()
        start_time = time.perf_counter()

        # 1. Validation check
        val = self.validate_input(input_data, input_source, exercise_id)
        if not val.is_valid:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ModelPredictionResult(
                model_id=meta.model_id,
                model_version=meta.model_version,
                predicted_class=None,
                class_label=None,
                probabilities=None,
                probability_calibration_status="unavailable",
                inference_latency_ms=round(elapsed_ms, 2),
                quality_status=val.quality_status,
                rejection_reason=val.rejection_reason,
                warnings=val.warnings
            )

        # 2. Preprocess & Predict
        try:
            preprocessed = self.preprocess(input_data)
            pred = self.predict(preprocessed)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            pred.inference_latency_ms = round(elapsed_ms, 2)
            return pred
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ModelPredictionResult(
                model_id=meta.model_id,
                model_version=meta.model_version,
                predicted_class=None,
                class_label=None,
                probabilities=None,
                probability_calibration_status="error",
                inference_latency_ms=round(elapsed_ms, 2),
                quality_status="INFERENCE_ERROR",
                rejection_reason=f"Runtime error during model evaluation: {str(e)}",
                warnings=[f"Exception raised in adapter {meta.model_id}: {str(e)}"]
            )
