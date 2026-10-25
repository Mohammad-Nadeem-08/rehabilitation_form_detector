# RehabVision AI — Model Integration Guide for Teammates

## Overview

RehabVision AI uses a **model-agnostic adapter architecture** located in `backend/model_adapters/`. This design ensures that you can integrate your own trained machine-learning or deep-learning model (LSTM, CNN, Transformer, Temporal GCN, ONNX, TorchScript) for any rehabilitation exercise **without editing the frontend, modifying existing exercise strategies, or altering the existing shoulder-abduction LSTM**.

```
backend/
├── model_adapters/
│   ├── __init__.py                  # Public exports
│   ├── base.py                      # BaseModelAdapter contract & typed results
│   ├── registry.py                  # Central ModelRegistry singleton
│   ├── existing_mocap_lstm.py       # Adapter for REHAB24-6 mocap LSTM
│   └── example_teammate_adapter.py  # Reference template adapter for teammates
├── models/
│   ├── rehab_lstm.pth               # Existing mocap LSTM weights
│   ├── preprocessing.npz            # Existing normalization parameters
│   └── metadata.json                # Existing metadata
```

---

## 1. The Core Model Adapter Contract

Every model adapter must inherit from `BaseModelAdapter` in [`backend/model_adapters/base.py`](../backend/model_adapters/base.py) and implement five mandatory methods:

```python
from model_adapters.base import (
    BaseModelAdapter, ModelMetadata, ModelValidationResult, ModelPredictionResult
)

class MyCustomModelAdapter(BaseModelAdapter):
    def get_metadata(self) -> ModelMetadata:
        """Declares model ID, version, supported exercises, schema, and units."""
        ...

    def validate_input(self, input_data: Any, input_source: str, exercise_id: str) -> ModelValidationResult:
        """Validates input shape, numeric integrity (NaN/Inf), exercise compatibility, and source."""
        ...

    def preprocess(self, input_data: Any) -> Any:
        """Applies normalization, feature scaling, or tensor conversions."""
        ...

    def predict(self, preprocessed_data: Any) -> ModelPredictionResult:
        """Executes forward pass and returns standardized ModelPredictionResult."""
        ...

    def health_check(self) -> Dict[str, Any]:
        """Verifies model file exists on disk and is loaded in memory."""
        ...
```

### The Standardized Execution Pipeline
When an assessment runs, `adapter.evaluate_input(...)` automatically handles the workflow:
1. Calls `validate_input(...)`. If invalid, immediately returns a structured rejection (`INVALID_SHAPE`, `NUMERICAL_ANOMALY`, `INCOMPATIBLE_REPRESENTATION`, `EXERCISE_MISMATCH`, or `MODEL_UNAVAILABLE`) with an explainable reason.
2. Calls `preprocess(...)` only on valid inputs.
3. Calls `predict(...)` and measures inference latency via high-resolution timer.
4. Catches unexpected runtime exceptions safely and returns `INFERENCE_ERROR` without crashing the FastAPI application.

---

## 2. Step-by-Step Integration Workflow

### Step 1: Place Your Model Artifacts
Place your model weight files in `backend/models/`:
- PyTorch State Dict / TorchScript: `backend/models/my_exercise_model.pth`
- ONNX Runtime: `backend/models/my_exercise_model.onnx`
- Normalization parameters: `backend/models/my_exercise_prep.npz` or `.json`

> **Security & Safe Loading:**
> - Never use `pickle.load()` on untrusted input.
> - For PyTorch checkpoints, use `torch.load(path, map_location="cpu", weights_only=True)` when supported.
> - For ONNX models, initialize `onnxruntime.InferenceSession(path)`.

### Step 2: Implement Your Model Adapter
Create `backend/model_adapters/my_exercise_adapter.py`. You can copy [`backend/model_adapters/example_teammate_adapter.py`](../backend/model_adapters/example_teammate_adapter.py) as a starting template:

```python
import os
import numpy as np
import torch
from model_adapters.base import (
    BaseModelAdapter, ModelMetadata, ModelValidationResult, ModelPredictionResult
)

class MyBicepCurlModelAdapter(BaseModelAdapter):
    def __init__(self, artifact_path: str = None):
        self.model_id = "bicep-curl-transformer-v1"
        self.model_version = "1.0.0"
        self.supported_exercise_ids = ["seated-right-bicep-curl"]
        self.artifact_path = artifact_path or os.path.join(
            os.path.dirname(__file__), "..", "models", "my_bicep_model.pth"
        )
        self.model = None
        self._load_weights()

    def _load_weights(self):
        if os.path.exists(self.artifact_path):
            self.model = torch.load(self.artifact_path, map_location="cpu")
            self.model.eval()

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            model_name="Seated Bicep Curl Transformer",
            model_version=self.model_version,
            supported_exercise_ids=self.supported_exercise_ids,
            framework="PyTorch",
            input_schema="joint_angles_32x4",
            input_dimensions=[32, 4],
            coordinate_system="anatomical_degrees",
            feature_order=["elbow_angle", "shoulder_angle", "wrist_velocity", "trunk_lean"],
            preprocessing_requirements="Min-Max scaling to [0, 1] range",
            class_mapping={0: "incorrect", 1: "correct"},
            output_semantics="binary_classification_softmax_probabilities",
            is_calibrated=False,
            device_requirement="cpu",
            validation_status="VALIDATED_PRODUCTION" if self.model else "UNLOADED",
            author_provenance="Teammate Capstone Exercise Model"
        )

    def validate_input(self, input_data, input_source, exercise_id):
        if exercise_id not in self.supported_exercise_ids:
            return ModelValidationResult(
                is_valid=False,
                quality_status="EXERCISE_MISMATCH",
                rejection_reason=f"Exercise '{exercise_id}' not supported."
            )
        if input_data is None:
            return ModelValidationResult(is_valid=False, quality_status="INVALID_SHAPE", rejection_reason="Input data is None.")
        
        arr = np.array(input_data, dtype=np.float32)
        if arr.shape != (32, 4):
            return ModelValidationResult(
                is_valid=False,
                quality_status="INVALID_SHAPE",
                rejection_reason=f"Expected shape (32, 4), got {arr.shape}."
            )
        if not np.isfinite(arr).all():
            return ModelValidationResult(
                is_valid=False,
                quality_status="NUMERICAL_ANOMALY",
                rejection_reason="Input contains NaN or Inf values."
            )
        return ModelValidationResult(is_valid=True, quality_status="VALID")

    def preprocess(self, input_data):
        arr = np.array(input_data, dtype=np.float32)
        norm = np.copy(arr)
        norm[:, 0] /= 180.0  # Elbow normalized
        norm[:, 1] /= 180.0  # Shoulder normalized
        return torch.tensor(norm, dtype=torch.float32).unsqueeze(0)

    def predict(self, preprocessed_data):
        with torch.no_grad():
            logits = self.model(preprocessed_data)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_class = int(np.argmax(probs))

        return ModelPredictionResult(
            model_id=self.model_id,
            model_version=self.model_version,
            predicted_class=pred_class,
            class_label="correct" if pred_class == 1 else "incorrect",
            probabilities={"incorrect": float(probs[0]), "correct": float(probs[1])},
            probability_calibration_status="uncalibrated_softmax",
            quality_status="VALID"
        )

    def health_check(self):
        return {
            "model_id": self.model_id,
            "status": "ready" if self.model is not None else "unavailable",
            "artifact_present": os.path.exists(self.artifact_path)
        }
```

### Step 3: Register Your Adapter in `model_adapters/registry.py`
In [`backend/model_adapters/registry.py`](../backend/model_adapters/registry.py), import and register your adapter:

```python
from model_adapters.my_exercise_adapter import MyBicepCurlModelAdapter

# Inside _register_default_adapters():
bicep_adapter = MyBicepCurlModelAdapter()
self.register(bicep_adapter, is_default=True)
```

That's it! Once registered:
- `GET /api/models` automatically lists your model.
- `GET /api/models/{model_id}` returns its full metadata and health.
- `POST /api/assessment/evaluate` automatically routes repetitions of your exercise to your adapter.
- The hybrid ensemble engine incorporates your model's predictions according to consensus policy.
- If your model is missing or uncalibrated, the system gracefully falls back to statistical biomechanical assessment and returns `NEEDS_REVIEW`.

---

## 3. Supported Model Frameworks

| Framework | Loading Method | Preprocessing Output | Inference Call |
| :--- | :--- | :--- | :--- |
| **PyTorch (`.pth`)** | `torch.load(..., map_location="cpu")` | `torch.Tensor` | `model(tensor)` |
| **TorchScript (`.pt`)** | `torch.jit.load(...)` | `torch.Tensor` | `model(tensor)` |
| **ONNX Runtime (`.onnx`)** | `onnxruntime.InferenceSession(...)` | `np.ndarray` | `session.run(...)` |
| **Scikit-Learn (`.joblib`)** | `joblib.load(...)` | `np.ndarray` | `clf.predict_proba(...)` |

---

## 4. How the Model Interacts with the Hybrid Ensemble

1. **Exercise Compatibility:** When a session runs for exercise `X`, the engine queries `model_registry.get_default_for_exercise(X)`.
2. **Quality Gating:** If the adapter returns `quality_status != "VALID"` (e.g. `INCOMPATIBLE_REPRESENTATION`, `MODEL_UNAVAILABLE`), the ensemble engine sets `decision: "NEEDS_REVIEW"`, includes the explanation in the response, and preserves the statistical score.
3. **Branch Agreement:** When the model is valid:
   - $P_{\text{combined}} = w_{\text{stat}} \cdot P_{\text{stat}} + w_{\text{model}} \cdot P_{\text{model}}$ (Default: $0.50 / 0.50$)
   - Consensus $\ge 0.70 \implies \text{CORRECT\_ESTIMATE}$
   - Consensus $\le 0.40 \implies \text{INCORRECT\_ESTIMATE}$
   - Discrepancy or ambiguous range $\implies \text{NEEDS\_REVIEW}$

---

## 5. Verification Checklist for Your Integration
- [ ] Model weights placed in `backend/models/`.
- [ ] Adapter class inherits from `BaseModelAdapter`.
- [ ] `validate_input` rejects unexpected shapes and NaN values.
- [ ] Preprocessing normalization matches the training pipeline.
- [ ] Adapter registered in `model_registry`.
- [ ] Added unit tests in `backend/tests/test_model_adapters.py`.
- [ ] Ran `pytest backend/tests -v` and verified all tests pass.
- [ ] Tested via `GET /api/models` and confirmed healthy status.
