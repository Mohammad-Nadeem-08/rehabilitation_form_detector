"""
Comprehensive Unit & Integration Tests for Model-Agnostic Adapters and Registry.

Tests:
- Model registration and lookup.
- Exercise-to-model compatibility.
- Correct input schema validation.
- Feature ordering and preprocessing invariants.
- Output class mapping and probability semantics.
- Unavailable model behavior.
- Inference exception handling without process crashing.
- Model version and provenance metadata.
- Rejection of incompatible input representations (webcam domain shift).
- Deterministic behavior for identical inputs.
- Teammate template adapter contract.
- API endpoints /api/models and /api/models/{model_id}.
"""

import os
import sys
import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from model_adapters.base import (
    BaseModelAdapter, ModelMetadata, ModelValidationResult, ModelPredictionResult
)
from model_adapters.registry import ModelRegistry, model_registry
from model_adapters.existing_mocap_lstm import ExistingMocapLSTMAdapter
from model_adapters.example_teammate_adapter import ExampleTeammateModelAdapter
from model_assessment import model_assessment_engine
from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test Mocks for Interface Testing (Clearly labeled as test fixtures)
# ---------------------------------------------------------------------------

class MockFaultyAdapter(BaseModelAdapter):
    """Mock adapter that simulates an internal runtime inference error."""
    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="mock-faulty-v1",
            model_name="Mock Faulty Adapter",
            model_version="1.0.0-test",
            supported_exercise_ids=["test-exercise"],
            framework="Mock",
            input_schema="test_schema",
            input_dimensions=[10],
            coordinate_system="test",
            feature_order=["f0"],
            preprocessing_requirements="none",
            class_mapping={0: "incorrect", 1: "correct"},
            output_semantics="binary",
            is_calibrated=False,
            device_requirement="cpu",
            validation_status="MOCK_TEST_FIXTURE",
            author_provenance="Unit Tests"
        )

    def validate_input(self, input_data, input_source, exercise_id):
        return ModelValidationResult(is_valid=True, quality_status="VALID")

    def preprocess(self, input_data):
        return input_data

    def predict(self, preprocessed_data):
        raise RuntimeError("Simulated runtime tensor execution failure in adapter.")

    def health_check(self):
        return {"status": "mock_faulty"}


# ---------------------------------------------------------------------------
# Registry & Discovery Tests
# ---------------------------------------------------------------------------

def test_model_registry_registration_and_lookup():
    reg = ModelRegistry()
    mocap_adapter = reg.get("rehab-lstm-rehab24-6-v1")
    assert mocap_adapter is not None
    meta = mocap_adapter.get_metadata()
    assert meta.model_id == "rehab-lstm-rehab24-6-v1"
    assert "standing-right-arm-abduction" in meta.supported_exercise_ids
    assert meta.framework == "PyTorch"
    assert meta.input_dimensions == [64, 80]

    bicep_adapter = reg.get("teammate-bicep-curl-v1")
    assert bicep_adapter is not None
    assert "seated-right-bicep-curl" in bicep_adapter.get_metadata().supported_exercise_ids


def test_exercise_to_model_compatibility():
    reg = ModelRegistry()

    # Supported exercise 1: shoulder abduction
    abd_models = reg.get_for_exercise("standing-right-arm-abduction")
    assert len(abd_models) >= 1
    assert abd_models[0].get_metadata().model_id == "rehab-lstm-rehab24-6-v1"

    # Supported exercise 2: bicep curl
    curl_models = reg.get_for_exercise("seated-right-bicep-curl")
    assert len(curl_models) >= 1
    assert curl_models[0].get_metadata().model_id == "teammate-bicep-curl-v1"

    # Unsupported exercise: statistical only
    unsupported = reg.get_for_exercise("unregistered-squat-exercise")
    assert len(unsupported) == 0
    assert reg.get_default_for_exercise("unregistered-squat-exercise") is None


# ---------------------------------------------------------------------------
# Schema, Preprocessing & Validation Tests
# ---------------------------------------------------------------------------

def test_mocap_lstm_adapter_input_validation():
    adapter = ExistingMocapLSTMAdapter()

    # 1. Valid mocap sequence of shape (64, 80)
    valid_seq = np.zeros((64, 80), dtype=np.float32).tolist()
    res_valid = adapter.validate_input(valid_seq, input_source="verified_mocap", exercise_id="standing-right-arm-abduction")
    assert res_valid.is_valid is True
    assert res_valid.quality_status == "VALID"

    # 2. Incompatible shape (30, 80)
    invalid_shape = np.zeros((30, 80), dtype=np.float32).tolist()
    res_shape = adapter.validate_input(invalid_shape, input_source="verified_mocap", exercise_id="standing-right-arm-abduction")
    assert res_shape.is_valid is False
    assert res_shape.quality_status == "INVALID_SHAPE"

    # 3. Numerical anomaly (NaN and Inf)
    nan_seq = np.zeros((64, 80), dtype=np.float32)
    nan_seq[10, 5] = np.nan
    res_nan = adapter.validate_input(nan_seq.tolist(), input_source="verified_mocap", exercise_id="standing-right-arm-abduction")
    assert res_nan.is_valid is False
    assert res_nan.quality_status == "NUMERICAL_ANOMALY"

    # 4. Exercise mismatch
    res_ex = adapter.validate_input(valid_seq, input_source="verified_mocap", exercise_id="seated-right-bicep-curl")
    assert res_ex.is_valid is False
    assert res_ex.quality_status == "EXERCISE_MISMATCH"

    # 5. Domain shift guard (webcam input)
    res_webcam = adapter.validate_input(valid_seq, input_source="webcam_mediapipe", exercise_id="standing-right-arm-abduction")
    assert res_webcam.is_valid is False
    assert res_webcam.quality_status == "INCOMPATIBLE_REPRESENTATION"
    assert "Domain shift" in res_webcam.rejection_reason


def test_mocap_lstm_adapter_feature_order_and_preprocessing():
    adapter = ExistingMocapLSTMAdapter()
    meta = adapter.get_metadata()

    # 80 features declared: 26 joints * 3 + 2 angles
    assert len(meta.feature_order) == 80
    assert meta.feature_order[0] == "joint_0_x"
    assert meta.feature_order[77] == "joint_25_z"
    assert meta.feature_order[78] == "primary_elevation_ratio_78"
    assert meta.feature_order[79] == "secondary_elbow_ratio_79"

    # Preprocessing produces torch Tensor of shape (1, 64, 80)
    zeros = np.zeros((64, 80), dtype=np.float32)
    tensor = adapter.preprocess(zeros)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 64, 80)


def test_mocap_lstm_adapter_prediction_contract():
    adapter = ExistingMocapLSTMAdapter()
    zeros = np.zeros((64, 80), dtype=np.float32)
    res = adapter.evaluate_input(zeros, input_source="verified_mocap", exercise_id="standing-right-arm-abduction")

    assert res.quality_status == "VALID"
    assert res.model_id == "rehab-lstm-rehab24-6-v1"
    assert res.predicted_class in [0, 1]
    assert res.class_label in ["incorrect", "correct"]
    assert "correct" in res.probabilities and "incorrect" in res.probabilities
    assert abs(res.probabilities["correct"] + res.probabilities["incorrect"] - 1.0) < 1e-3
    assert res.inference_latency_ms > 0.0


def test_deterministic_behavior_for_identical_inputs():
    adapter = ExistingMocapLSTMAdapter()
    np.random.seed(123)
    seq = np.random.normal(0, 0.05, (64, 80)).astype(np.float32).tolist()

    pred1 = adapter.evaluate_input(seq, input_source="verified_mocap", exercise_id="standing-right-arm-abduction")
    pred2 = adapter.evaluate_input(seq, input_source="verified_mocap", exercise_id="standing-right-arm-abduction")

    assert pred1.predicted_class == pred2.predicted_class
    assert pred1.class_label == pred2.class_label
    assert pred1.probabilities == pred2.probabilities
    assert pred1.raw_logits == pred2.raw_logits


# ---------------------------------------------------------------------------
# Exception & Unavailable Model Handling
# ---------------------------------------------------------------------------

def test_adapter_inference_exception_handling():
    faulty = MockFaultyAdapter()
    res = faulty.evaluate_input(
        input_data=[0.0] * 10,
        input_source="test",
        exercise_id="test-exercise"
    )

    assert res.quality_status == "INFERENCE_ERROR"
    assert res.predicted_class is None
    assert "Simulated runtime tensor execution failure" in res.rejection_reason
    assert res.inference_latency_ms >= 0.0


def test_model_assessment_engine_unregistered_exercise_graceful_abstention():
    # If evaluating an exercise with no registered model
    res = model_assessment_engine.evaluate_sequence(
        sequence=[[0.0] * 80 for _ in range(64)],
        input_source="verified_mocap",
        exercise_id="unregistered-exercise-999"
    )
    assert res.input_quality_status == "MODEL_UNAVAILABLE"
    assert res.predicted_class is None
    assert "No machine learning model is registered" in res.rejection_reason


# ---------------------------------------------------------------------------
# Teammate Template Adapter Contract Tests
# ---------------------------------------------------------------------------

def test_teammate_template_adapter_contract():
    bicep_adapter = ExampleTeammateModelAdapter()
    meta = bicep_adapter.get_metadata()

    assert meta.model_id == "teammate-bicep-curl-v1"
    assert meta.supported_exercise_ids == ["seated-right-bicep-curl"]
    assert meta.input_dimensions == [32, 4]
    assert len(meta.feature_order) == 4

    # Exercise mismatch when called with shoulder abduction
    wrong_ex = bicep_adapter.validate_input(
        np.zeros((32, 4)), input_source="verified", exercise_id="standing-right-arm-abduction"
    )
    assert wrong_ex.is_valid is False
    assert wrong_ex.quality_status == "EXERCISE_MISMATCH"

    # Shape validation on correct exercise
    wrong_shape = bicep_adapter.validate_input(
        np.zeros((64, 80)), input_source="verified", exercise_id="seated-right-bicep-curl"
    )
    assert wrong_shape.is_valid is False
    assert wrong_shape.quality_status == "INVALID_SHAPE"

    # Successful evaluation on compatible (32, 4) input
    sample_bicep_seq = np.full((32, 4), 45.0, dtype=np.float32)  # 45 deg flexion
    pred = bicep_adapter.evaluate_input(
        sample_bicep_seq, input_source="verified", exercise_id="seated-right-bicep-curl"
    )
    assert pred.quality_status == "VALID"
    assert pred.predicted_class in [0, 1]
    assert pred.class_label in ["incorrect", "correct"]
    assert any("template mock predictor" in w for w in pred.warnings)


# ---------------------------------------------------------------------------
# FastAPI Endpoints Smoke Tests for Models
# ---------------------------------------------------------------------------

def test_api_models_endpoints():
    # 1. List all models
    res = client.get("/api/models")
    assert res.status_code == 200
    models_list = res.json()
    assert isinstance(models_list, list)
    assert len(models_list) >= 2

    model_ids = [m["model_id"] for m in models_list]
    assert "rehab-lstm-rehab24-6-v1" in model_ids
    assert "teammate-bicep-curl-v1" in model_ids

    # 2. Filter by exercise
    res_filter = client.get("/api/models?exercise_id=standing-right-arm-abduction")
    assert res_filter.status_code == 200
    filtered = res_filter.json()
    assert len(filtered) >= 1
    assert filtered[0]["model_id"] == "rehab-lstm-rehab24-6-v1"

    # 3. Get model by ID
    res_single = client.get("/api/models/rehab-lstm-rehab24-6-v1")
    assert res_single.status_code == 200
    single = res_single.json()
    assert single["model_id"] == "rehab-lstm-rehab24-6-v1"
    assert single["framework"] == "PyTorch"
    assert single["health"]["status"] in ["ready", "unavailable"]

    # 4. Non-existent model returns 404
    res_404 = client.get("/api/models/non-existent-model-xyz")
    assert res_404.status_code == 404


def test_evaluate_custom_model_cohort():
    """Verify teammate's reusable model cohort evaluation interface."""
    from evaluation import independent_evaluator

    # Synthetic test cohort for teammate's bicep curl model
    dummy_cohort = [
        {
            "repetition_id": "rep-test-01",
            "participant_id": 101,
            "ground_truth_label": 1,
            "sequence_input": np.full((32, 4), 40.0, dtype=np.float32).tolist()
        },
        {
            "repetition_id": "rep-test-02",
            "participant_id": 101,
            "ground_truth_label": 0,
            "sequence_input": np.full((32, 4), 110.0, dtype=np.float32).tolist()
        },
        {
            "repetition_id": "rep-test-03",
            "participant_id": 102,
            "ground_truth_label": 1,
            "sequence_input": np.full((32, 4), 35.0, dtype=np.float32).tolist()
        },
        {
            "repetition_id": "rep-test-04",
            "participant_id": 102,
            "ground_truth_label": 0,
            "sequence_input": np.full((32, 4), 120.0, dtype=np.float32).tolist()
        }
    ]

    report = independent_evaluator.evaluate_custom_model_cohort(
        cohort=dummy_cohort,
        model_id="teammate-bicep-curl-v1",
        exercise_id="seated-right-bicep-curl",
        input_key="sequence_input"
    )

    assert report["model_id"] == "teammate-bicep-curl-v1"
    assert report["exercise_id"] == "seated-right-bicep-curl"
    assert report["dataset_sample_count"] == 4
    assert report["participant_distribution"] == {"101": 2, "102": 2}
    assert report["class_distribution"] == {"0": 2, "1": 2}
    assert report["metrics"]["accuracy"] == 1.0
    assert report["metrics"]["macro_f1"] == 1.0
    assert report["metrics"]["decision_coverage_pct"] == 100.0

