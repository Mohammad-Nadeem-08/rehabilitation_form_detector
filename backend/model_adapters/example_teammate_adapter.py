"""
Example Model Adapter Template for Teammate Integration.

This module provides a concrete reference template for a teammate to integrate their
own trained rehabilitation form detection model (PyTorch, ONNX, TorchScript, etc.)
without editing existing exercise strategies, the existing LSTM, or the frontend.

INSTRUCTIONS FOR TEAMMATE:
1. Copy or rename this file (e.g., `bicep_curl_transformer.py`).
2. Place your trained model weights (e.g., `bicep_model.pth` or `bicep_model.onnx`)
   in `backend/models/`.
3. Update `self.supported_exercise_ids` with your exercise ID (e.g., 'seated-right-bicep-curl').
4. Define your model's exact input shape, coordinate requirements, and feature list in get_metadata().
5. Implement validate_input() with your schema checks.
6. Replace `_mock_inference()` with your real PyTorch/ONNX forward pass.
7. Register your adapter in `backend/model_adapters/registry.py`.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from model_adapters.base import (
    BaseModelAdapter, ModelMetadata, ModelValidationResult, ModelPredictionResult
)


class ExampleTeammateModelAdapter(BaseModelAdapter):
    """
    Template adapter illustrating how a teammate integrates an ML model.

    NOTE: This class includes an explicit mock inference engine solely for
    software interface validation and regression testing. It must be replaced
    with real model artifacts before clinical deployment.
    """

    def __init__(
        self,
        model_id: str = "teammate-bicep-curl-v1",
        model_name: str = "Teammate Bicep Curl Classifier (Template/Mock)",
        model_version: str = "0.1.0-template",
        model_artifact_path: Optional[str] = None
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.model_version = model_version
        self.supported_exercise_ids = ["seated-right-bicep-curl"]
        self.framework = "PyTorch / ONNX (Template)"
        self.input_schema = "angular_kinematic_time_series_32x4"
        self.input_dimensions = [32, 4]  # 32 timesteps, 4 features: [elbow, shoulder, wrist_vel, trunk_lean]
        self.coordinate_system = "joint_angle_degrees_and_angular_velocity"
        self.feature_order = [
            "elbow_flexion_degrees",
            "shoulder_flexion_degrees",
            "wrist_angular_velocity_deg_s",
            "trunk_compensatory_lean_deg"
        ]
        self.class_mapping = {
            0: "incorrect",
            1: "correct"
        }
        self.model_artifact_path = model_artifact_path or os.path.join(
            backend_dir, "models", "teammate_model.pth"
        )
        self.is_real_weights_loaded = False
        self._load_model_if_present()

    def _load_model_if_present(self):
        """
        Safe loader: Attempts to load real weights if the file exists on disk.
        Does NOT execute untrusted arbitrary code. Restricts to torch.load or onnxruntime.
        """
        if os.path.exists(self.model_artifact_path):
            try:
                # Real loader example:
                # import torch
                # self.model = torch.load(self.model_artifact_path, map_location="cpu")
                # self.model.eval()
                # self.is_real_weights_loaded = True
                pass
            except Exception as e:
                self.is_real_weights_loaded = False
                print(f"Warning: Could not load teammate model from {self.model_artifact_path}: {e}")

    def get_metadata(self) -> ModelMetadata:
        val_status = "VALIDATED_PRODUCTION" if self.is_real_weights_loaded else "TEMPLATE_MOCK_REQUIRES_TRAINED_WEIGHTS"
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
            preprocessing_requirements="Min-Max scaling to [0, 1] range based on anatomical joint limits.",
            class_mapping=self.class_mapping,
            output_semantics="binary_classification_softmax_probabilities",
            is_calibrated=False,
            device_requirement="cpu",
            validation_status=val_status,
            author_provenance="Teammate Machine Learning Model Integration",
            notes="Template adapter. Set model_artifact_path to real weights to activate live inference."
        )

    def validate_input(
        self,
        input_data: Any,
        input_source: str,
        exercise_id: str
    ) -> ModelValidationResult:
        warnings: List[str] = []

        # 1. Exercise Compatibility Guard
        if exercise_id not in self.supported_exercise_ids:
            return ModelValidationResult(
                is_valid=False,
                quality_status="EXERCISE_MISMATCH",
                rejection_reason=(
                    f"Model '{self.model_id}' supports {self.supported_exercise_ids}, "
                    f"not '{exercise_id}'."
                ),
                warnings=[f"Exercise '{exercise_id}' is unsupported by this model adapter."]
            )

        # 2. Input null guard
        if input_data is None:
            return ModelValidationResult(
                is_valid=False,
                quality_status="INVALID_SHAPE",
                rejection_reason="Input data is None. Expected (32, 4) array.",
                warnings=["Null input provided."]
            )

        try:
            arr = np.array(input_data, dtype=np.float32)
        except Exception as e:
            return ModelValidationResult(
                is_valid=False,
                quality_status="INVALID_SHAPE",
                rejection_reason=f"Failed to parse input as numeric array: {str(e)}",
                warnings=[str(e)]
            )

        # 3. Shape validation
        expected_shape = tuple(self.input_dimensions)
        if arr.shape != expected_shape:
            return ModelValidationResult(
                is_valid=False,
                quality_status="INVALID_SHAPE",
                rejection_reason=f"Shape mismatch: expected {expected_shape}, got {arr.shape}.",
                warnings=[f"Expected {expected_shape}, received {arr.shape}."]
            )

        # 4. Numerical integrity
        if not np.isfinite(arr).all():
            return ModelValidationResult(
                is_valid=False,
                quality_status="NUMERICAL_ANOMALY",
                rejection_reason="Input contains NaN or infinite values.",
                warnings=["Non-finite floats detected in input."]
            )

        return ModelValidationResult(is_valid=True, quality_status="VALID")

    def preprocess(self, input_data: Any) -> np.ndarray:
        arr = np.array(input_data, dtype=np.float32)
        # Example preprocessing: normalize elbow (0..180) and shoulder (0..180) to [0, 1]
        norm = np.copy(arr)
        norm[:, 0] = np.clip(norm[:, 0] / 180.0, 0.0, 1.0)
        norm[:, 1] = np.clip(norm[:, 1] / 180.0, 0.0, 1.0)
        return norm

    def predict(self, preprocessed_data: np.ndarray) -> ModelPredictionResult:
        # If real model loaded, run real inference:
        # if self.is_real_weights_loaded:
        #     return self._real_predict(preprocessed_data)

        # Deterministic template/mock inference for interface verification:
        # Computes mean elbow angle across the sequence:
        mean_elbow = float(np.mean(preprocessed_data[:, 0]))
        # In normalized units, full bicep flexion reaches < 0.35 (i.e. < 63 degrees)
        is_correct = 1 if mean_elbow < 0.50 else 0
        prob_corr = 0.88 if is_correct == 1 else 0.18
        prob_inc = round(1.0 - prob_corr, 4)

        return ModelPredictionResult(
            model_id=self.model_id,
            model_version=self.model_version,
            predicted_class=is_correct,
            class_label=self.class_mapping[is_correct],
            probabilities={"incorrect": prob_inc, "correct": prob_corr},
            probability_calibration_status="mock_template_output",
            quality_status="VALID",
            warnings=["Using template mock predictor; integrate trained weights for clinical inference."]
        )

    def health_check(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "status": "ready (mock_template)" if not self.is_real_weights_loaded else "ready (production_weights)",
            "is_real_weights_loaded": self.is_real_weights_loaded,
            "model_artifact_path": self.model_artifact_path,
            "supported_exercises": self.supported_exercise_ids
        }
