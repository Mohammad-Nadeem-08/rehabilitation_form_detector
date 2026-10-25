"""
Model Adapter for the existing REHAB24-6 3D Motion-Capture PyTorch LSTM.

Wraps the trained PyTorch LSTM checkpoint (rehab_lstm.pth) and preprocessing parameters
(preprocessing.npz) under the model-agnostic BaseModelAdapter contract.
"""

import os
import sys
import numpy as np
import torch
from typing import Dict, List, Any, Optional

# Ensure backend root is available for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from model_adapters.base import (
    BaseModelAdapter, ModelMetadata, ModelValidationResult, ModelPredictionResult
)
from inference import lstm_engine


class ExistingMocapLSTMAdapter(BaseModelAdapter):
    """
    Adapter for the PyTorch 2-Layer LSTM trained on REHAB24-6 motion-capture data.
    Restricted to standing right-arm shoulder abduction.
    Strictly isolates 3D mocap input from live uncalibrated monocular MediaPipe webcam inputs.
    """

    def __init__(self):
        self.model_id = "rehab-lstm-rehab24-6-v1"
        self.model_name = "PyTorch 2-Layer LSTM (REHAB24-6 Shoulder Abduction)"
        self.model_version = "1.0.0"
        self.supported_exercise_ids = ["standing-right-arm-abduction"]
        self.framework = "PyTorch"
        self.input_schema = "mocap_3d_64x80"
        self.input_dimensions = [64, 80]
        self.coordinate_system = "pelvis_centered_metric_3d"
        self.feature_order = [
            f"joint_{j}_{axis}" for j in range(26) for axis in ["x", "y", "z"]
        ] + ["primary_elevation_ratio_78", "secondary_elbow_ratio_79"]
        self.class_mapping = {0: "incorrect", 1: "correct"}

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            model_name=self.model_name,
            model_version=self.model_version,
            supported_exercise_ids=self.supported_exercise_ids,
            framework=self.framework,
            input_schema=self.input_schema,
            input_dimensions=self.input_dimensions,
            coordinate_system=self.coordinate_system,
            feature_order=self.feature_order,
            preprocessing_requirements="Z-score normalization with training mean & std from preprocessing.npz: (x - mu) / (sigma + 1e-8)",
            class_mapping=self.class_mapping,
            output_semantics="binary_classification_softmax_probabilities",
            is_calibrated=False,
            device_requirement="cpu",
            validation_status="PRELIMINARY_MOCAP_BENCHMARK",
            author_provenance="Trained on REHAB24-6 motion-capture dataset (9 participants, 178 repetitions).",
            notes="Requires 3D optical motion capture joint positions. Incompatible with uncalibrated 2D monocular MediaPipe webcam."
        )

    def validate_input(
        self,
        input_data: Any,
        input_source: str,
        exercise_id: str
    ) -> ModelValidationResult:
        warnings: List[str] = []

        # 1. Model Availability Guard
        if not lstm_engine.is_loaded or lstm_engine.model is None:
            return ModelValidationResult(
                is_valid=False,
                quality_status="MODEL_UNAVAILABLE",
                rejection_reason=f"Model engine not ready: {lstm_engine.load_error}",
                warnings=["PyTorch LSTM checkpoint not loaded on host."]
            )

        # 2. Exercise Compatibility Guard
        if exercise_id not in self.supported_exercise_ids:
            return ModelValidationResult(
                is_valid=False,
                quality_status="EXERCISE_MISMATCH",
                rejection_reason=(
                    f"Model '{self.model_id}' only supports exercises {self.supported_exercise_ids}, "
                    f"not '{exercise_id}'."
                ),
                warnings=[f"Exercise '{exercise_id}' is unsupported by this model adapter."]
            )

        # 3. Domain Shift Guard (Webcam MediaPipe Protection)
        if any(w in input_source.lower() for w in ["webcam", "mediapipe", "camera", "monocular"]):
            return ModelValidationResult(
                is_valid=False,
                quality_status="INCOMPATIBLE_REPRESENTATION",
                rejection_reason=(
                    "Domain shift: PyTorch LSTM was trained on 3D motion-capture marker coordinates (REHAB24-6). "
                    "Live monocular MediaPipe landmarks lack calibrated 3D depth and metric scale. "
                    "To prevent fabricated predictions, live webcam inference is safely reserved."
                ),
                warnings=["Live deep-learning classification unavailable: input representation requires validation."]
            )

        # 4. Shape & Dimensionality Guard
        if input_data is None:
            return ModelValidationResult(
                is_valid=False,
                quality_status="INVALID_SHAPE",
                rejection_reason="Input data is None. Expected (64, 80) sequence array.",
                warnings=["No input tensor provided."]
            )

        try:
            seq_np = np.array(input_data, dtype=np.float32)
        except Exception as e:
            return ModelValidationResult(
                is_valid=False,
                quality_status="INVALID_SHAPE",
                rejection_reason=f"Failed to parse input as numeric array: {str(e)}",
                warnings=[str(e)]
            )

        if seq_np.ndim != 2 or seq_np.shape != (64, 80):
            return ModelValidationResult(
                is_valid=False,
                quality_status="INVALID_SHAPE",
                rejection_reason=f"Sequence shape {seq_np.shape} is incompatible. Expected exactly (64, 80): 64 timesteps x 80 features.",
                warnings=[f"Dimension mismatch: expected (64, 80), received {seq_np.shape}."]
            )

        # 5. Numerical Integrity Guard (NaN & Inf Check)
        if not np.isfinite(seq_np).all():
            nan_count = int(np.isnan(seq_np).sum())
            inf_count = int(np.isinf(seq_np).sum())
            return ModelValidationResult(
                is_valid=False,
                quality_status="NUMERICAL_ANOMALY",
                rejection_reason=f"Input sequence contains non-finite values ({nan_count} NaNs, {inf_count} Infs).",
                warnings=["Non-finite floating point numbers detected in input tensor."]
            )

        return ModelValidationResult(is_valid=True, quality_status="VALID")

    def preprocess(self, input_data: Any) -> torch.Tensor:
        seq_np = np.array(input_data, dtype=np.float32)
        batch_input = np.expand_dims(seq_np, axis=0)  # (1, 64, 80)
        norm_seq = (batch_input - lstm_engine.feature_mean) / (lstm_engine.feature_std + 1e-8)
        return torch.tensor(norm_seq, dtype=torch.float32)

    def predict(self, preprocessed_data: torch.Tensor) -> ModelPredictionResult:
        with torch.no_grad():
            logits = lstm_engine.model(preprocessed_data)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))

        prob_inc = round(float(probs[0]), 4)
        prob_cor = round(float(probs[1]), 4)

        if not np.isfinite(prob_cor) or not np.isfinite(prob_inc) or abs((prob_cor + prob_inc) - 1.0) > 1e-3:
            return ModelPredictionResult(
                model_id=self.model_id,
                model_version=self.model_version,
                predicted_class=None,
                class_label=None,
                probabilities=None,
                probability_calibration_status="error",
                quality_status="NUMERICAL_ANOMALY",
                rejection_reason="Degenerate model probability distribution (fails unit sum invariant).",
                warnings=["Softmax sum failed unit probability invariant."]
            )

        return ModelPredictionResult(
            model_id=self.model_id,
            model_version=self.model_version,
            predicted_class=pred_class,
            class_label=self.class_mapping[pred_class],
            probabilities={"incorrect": prob_inc, "correct": prob_cor},
            probability_calibration_status="uncalibrated_softmax",
            raw_logits=[round(float(l), 4) for l in logits.cpu().numpy()[0]],
            quality_status="VALID"
        )

    def health_check(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "status": "ready" if lstm_engine.is_loaded else "unavailable",
            "model_path": lstm_engine.model_path,
            "artifacts_present": os.path.exists(lstm_engine.model_path) and os.path.exists(lstm_engine.prep_path),
            "load_error": lstm_engine.load_error
        }
