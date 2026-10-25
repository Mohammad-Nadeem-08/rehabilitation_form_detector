# Changelog

All notable changes to the RehabVision AI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-10-25

### Added
- **Model-Agnostic Adapter Subsystem (`backend/model_adapters/`):**
  - `BaseModelAdapter`: Formal abstract interface declaring `ModelMetadata`, `ModelValidationResult`, `ModelPredictionResult`, `validate_input()`, `preprocess()`, `predict()`, and `health_check()`.
  - `ModelRegistry`: Central registry singleton (`model_registry`) providing decoupled registration, retrieval, and exercise-to-model compatibility validation.
  - `ExistingMocapLSTMAdapter`: Encapsulates the frozen REHAB24-6 motion-capture LSTM (`(64, 80)` tensor contract) under the adapter interface with strict domain-shift safeguards.
  - `ExampleTeammateAdapter`: Reference adapter template demonstrating how a teammate can integrate a custom model architecture (e.g., for `seated-right-bicep-curl` with shape `(32, 4)`) without modifying existing backend modules.
- **Model Registry REST Endpoints:**
  - `GET /api/models`: Lists all registered models and their operational statuses.
  - `GET /api/models/{model_id}`: Retrieves comprehensive metadata, feature ordering, coordinate systems, and operational boundaries for a specific model.
- **Teammate Integration & Evaluation Documentation:**
  - `docs/MODEL_INTEGRATION_GUIDE.md`: Step-by-step instructions for adding custom models (PyTorch, TorchScript, ONNX) with secure weight loading.
  - `docs/MODEL_EVALUATION_GUIDE.md`: Framework for evaluating custom models across participants, computing confusion matrices, per-class F1, and Brier scores.
  - `CONTRIBUTING.md`: Contribution guidelines, development environment setup, and verification requirements.
  - `.env.example`: Standardized environment variable template.
- **Decoupled Evaluation Metrics & Cohort Benchmarking:**
  - Added `cohort_accuracy`, `decided_accuracy`, `brier_score_decided`, and `brier_score_cohort` to `evaluation.py`.
  - Added `evaluate_custom_model_cohort(...)` to benchmark custom teammate models alongside statistical and ensemble baselines.
- **Comprehensive Adapter Test Suite (`backend/tests/test_model_adapters.py`):**
  - 11 new automated unit and integration tests covering registration, lookup, exercise-to-model compatibility, input validation, deterministic inference, exception isolation, and cohort evaluation.
- **Frontend Model Awareness:**
  - Updated `AssessmentPanel.tsx` to dynamically query and display registered model IDs, architectures, and input feature shapes.

### Changed
- **Ensemble Assessment Engine (`backend/model_assessment.py`):**
  - Refactored `evaluate_sequence(...)` to dynamically delegate inference to the registered adapter corresponding to `(exercise_id, model_id)`.
  - Added graceful fallback to `statistical_only` when an exercise has no compatible model registered.
- **Scientific Audit & Accuracy Clarification:**
  - Documented that the preliminary 10% LSTM score reported in synthetic tests was caused by zero-motion Gaussian noise inputs falling near the model's initialization boundary.
  - Clarified that the REHAB24-6 raw mocap training files were absent from the repository and reported the frozen checkpoint's validation as unverified on live camera feeds.
  - Separated Decided Accuracy (100% on 24/24 decided test samples) from Full Cohort Accuracy (60% on 40 samples with 40% abstention rate).
- **Backend Test Suite:**
  - Expanded total backend tests from 51 to 62 tests (100% pass rate).

### Security
- Enforced safe model loading restrictions, rejecting arbitrary code execution and untrusted serialization formats.
- Updated `.gitignore` to prevent committing local databases, environment variables, and temporary scratch scripts.

---

## [1.0.0] - 2026-10-24

### Added
- In-browser MediaPipe PoseLandmarker (33 3D landmarks) running at 30+ FPS.
- Pluggable `ExerciseStrategy` framework with `standing-right-arm-abduction` and `seated-right-bicep-curl`.
- Finite State Machine (FSM) repetition counter with hysteresis guarding and tracking-loss safety reset.
- Dynamic Time Warping (DTW) with canonical repetition-cycle segmentation.
- Dual reference standard classes: Universal Standard References and Patient-Specific Clinician-Approved References.
- SQLite database persistence (`rehabvision.db`) with patient data isolation.
- Publication-quality clinical PDF reports with ReportLab and embedded Matplotlib trajectory charts.
- PyTorch LSTM Model Lab for motion-capture sequences.
