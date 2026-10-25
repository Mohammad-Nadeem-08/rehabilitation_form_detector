# RehabVision AI — API Integration Guide

This guide describes the REST API contract for integrating the **RehabVision AI** backend with teammate rehabilitation platforms, external frontends, and mobile clinical clients.

---

## 1. Connection & Protocol Details

- **Default Base URL:** `http://127.0.0.1:8000`
- **Default Frontend Port:** `http://127.0.0.1:5173`
- **CORS Policy:** Permissive by default in development (`allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]`).
- **Data Exchange Format:** `application/json` (except `/api/reports/{id}` which streams `application/pdf`).
- **Interactive Documentation:** 
  - Swagger UI: `http://127.0.0.1:8000/docs`
  - ReDoc: `http://127.0.0.1:8000/redoc`

---

## 2. API Endpoints Contract

### 2.1 System Health
`GET /api/health`

Verifies server health, loaded PyTorch model state, and list of supported exercises from the strategy registry.

**Response `200 OK`:**
```json
{
  "status": "healthy",
  "app": "RehabVision AI",
  "version": "1.0.0",
  "model_loaded": true,
  "model_error": null,
  "supported_exercises": [
    "standing-right-arm-abduction",
    "seated-right-bicep-curl"
  ],
  "notice": "Educational prototype only. Not validated for clinical decision-making or determining exercise safety."
}
```

---

### 2.2 Exercise Strategy Registry

#### List Exercises
`GET /api/exercises`

Returns all registered rehabilitation exercises, their primary and secondary kinematic metric definitions, target ROM bounds, FSM threshold configurations, and landmark requirements.

**Response `200 OK`:**
```json
[
  {
    "id": "standing-right-arm-abduction",
    "name": "Standing Right-Arm Shoulder Abduction",
    "target_joint": "Right Shoulder & Elbow",
    "description": "Coronal plane shoulder elevation from 0° resting arm up to 90°-100° with extended elbow (~170°-180°), controlled eccentric descent.",
    "target_rom_min": 80.0,
    "target_rom_max": 105.0,
    "created_at": "2026-10-25T10:00:00",
    "primary_metric_name": "Arm Elevation",
    "secondary_metric_name": "Elbow Extension",
    "fsm_thresholds": {
      "start_threshold": 30.0,
      "raised_threshold": 80.0,
      "return_threshold": 25.0,
      "hysteresis": 5.0,
      "min_duration_sec": 0.5,
      "confirmation_frames": 2
    },
    "required_landmarks": [
      { "index": 12, "name": "Right Shoulder", "role": "Vertex for elevation" },
      { "index": 14, "name": "Right Elbow", "role": "Arm segment endpoint" },
      { "index": 24, "name": "Right Hip", "role": "Torso vertical baseline" },
      { "index": 23, "name": "Left Hip", "role": "Fallback torso baseline" },
      { "index": 16, "name": "Right Wrist", "role": "Elbow angle extension" }
    ]
  },
  {
    "id": "seated-right-bicep-curl",
    "name": "Seated Right-Arm Bicep Curl",
    "target_joint": "Right Biceps Brachii & Elbow Flexor",
    "description": "Seated elbow flexion through full active range with humeral stabilization...",
    "target_rom_min": 95.0,
    "target_rom_max": 135.0,
    "created_at": "2026-10-25T10:00:00",
    "primary_metric_name": "Elbow Flexion",
    "secondary_metric_name": "Humeral Stability",
    "fsm_thresholds": {
      "start_threshold": 35.0,
      "raised_threshold": 110.0,
      "return_threshold": 25.0,
      "hysteresis": 5.0,
      "min_duration_sec": 0.6,
      "confirmation_frames": 2
    },
    "required_landmarks": [
      { "index": 12, "name": "Right Shoulder", "role": "Proximal anchor" },
      { "index": 14, "name": "Right Elbow", "role": "Hinge joint vertex" },
      { "index": 16, "name": "Right Wrist", "role": "Distal lever arm" },
      { "index": 24, "name": "Right Hip", "role": "Torso alignment baseline" }
    ]
  }
]
```

#### Get Single Exercise
`GET /api/exercises/{exercise_id}`

**Parameters:** `exercise_id` (e.g. `standing-right-arm-abduction` or `seated-right-bicep-curl`).  
**Response `200 OK`:** Single `ExerciseSchema` object.  
**Response `404 Not Found`:** If `exercise_id` is not registered.

---

### 2.3 Reference Library & Provenance

#### List References
`GET /api/references`

**Optional Query Parameters:**
- `exercise_id`: Filter by exercise string (e.g. `standing-right-arm-abduction`).
- `reference_type`: Filter by standard (`standard`) vs patient-specific (`patient_specific`).
- `patient_id`: If specified, applies patient isolation: returns references belonging to `patient_id` **plus** all reusable `standard` baselines, strictly filtering out references belonging to other patients.

**Response `200 OK`:**
```json
[
  {
    "id": "ref-patient-101-approved",
    "exercise_id": "standing-right-arm-abduction",
    "exercise_name": "Standing Right-Arm Shoulder Abduction",
    "notes": "Clinician-approved tailored reference for patient-101 (85° target elevation; scapular accommodation).",
    "duration_seconds": 3.2,
    "frame_count": 64,
    "joint_angles": [
      { "timestamp": 0.0, "elevation": 15.0, "elbow": 170.0, "confidence": 0.95 },
      { "timestamp": 0.05, "elevation": 18.5, "elbow": 169.8, "confidence": 0.95 }
    ],
    "landmarks_summary": {
      "avg_confidence": 0.95,
      "peak_elevation": 85.0
    },
    "created_at": "2026-10-24T11:00:00",
    "is_demo": 1,
    "reference_type": "patient_specific",
    "version": "1.1",
    "provenance": "Approved by Dr. Sarah Jenkins, PT on 2026-10-24",
    "patient_id": "patient-101",
    "is_active": 1
  },
  {
    "id": "ref-gold-standard-demo",
    "exercise_id": "standing-right-arm-abduction",
    "exercise_name": "Standing Right-Arm Shoulder Abduction",
    "notes": "Clinical baseline standard: Controlled 95° abduction with stabilized trunk and extended elbow (~170°).",
    "duration_seconds": 3.2,
    "frame_count": 64,
    "joint_angles": [ ... ],
    "landmarks_summary": { ... },
    "created_at": "2026-10-23T10:00:00",
    "is_demo": 1,
    "reference_type": "standard",
    "version": "1.0",
    "provenance": "Physiotherapy Clinical Protocol v1.0 (Hospital Baseline)",
    "patient_id": null,
    "is_active": 1
  }
]
```

#### Create Reference Standard
`POST /api/references`

Saves a captured kinematic repetition as a baseline comparison standard.

**Request Payload:**
```json
{
  "exercise_id": "standing-right-arm-abduction",
  "exercise_name": "Standing Right-Arm Shoulder Abduction",
  "notes": "Clinician guided repetition targeting 85 degrees elevation.",
  "duration_seconds": 3.2,
  "frame_count": 64,
  "joint_angles": [
    { "timestamp": 0.0, "elevation": 15.0, "elbow": 170.0, "confidence": 0.95 }
  ],
  "landmarks_summary": {
    "peak_elevation": 85.0,
    "min_elbow": 168.0
  },
  "reference_type": "patient_specific",
  "version": "1.1",
  "provenance": "Approved by Dr. Sarah Jenkins, PT on 2026-10-24",
  "patient_id": "patient-101",
  "is_active": 1
}
```

---

### 2.4 Movement Comparison (DTW)
`POST /api/compare`

Compares a patient workout trajectory against a selected reference standard using Dynamic Time Warping (DTW) with automatic repetition-cycle segmentation.

**Request Payload:**
```json
{
  "reference_id": "ref-patient-101-approved",
  "home_angles": [
    { "timestamp": 0.0, "elevation": 15.0, "elbow": 170.0 },
    { "timestamp": 0.1, "elevation": 45.0, "elbow": 170.0 },
    { "timestamp": 0.8, "elevation": 86.0, "elbow": 169.0 },
    { "timestamp": 1.6, "elevation": 18.0, "elbow": 171.0 }
  ]
}
```

**Response `200 OK`:**
```json
{
  "status": "success",
  "comparison_score": 93.4,
  "mean_elevation_deviation": 5.2,
  "max_elevation_deviation": 9.8,
  "mean_elbow_deviation": 3.8,
  "rmse": 5.9,
  "aligned_reference_elevation": [15.0, 48.0, 85.0, 15.0],
  "aligned_home_elevation": [15.0, 45.0, 86.0, 18.0],
  "aligned_reference_elbow": [170.0, 169.0, 168.0, 170.0],
  "aligned_home_elbow": [170.0, 170.0, 169.0, 171.0],
  "warnings": []
}
```

---

### 2.5 Session History & Reports

#### Record Completed Session
`POST /api/sessions`

Stores a patient exercise session, repetition count, biomechanical angle statistics, DTW scores, and full reference provenance.

**Request Payload:**
```json
{
  "exercise_id": "standing-right-arm-abduction",
  "exercise_name": "Standing Right-Arm Shoulder Abduction",
  "reference_id": "ref-patient-101-approved",
  "reference_type": "patient_specific",
  "reference_version": "1.1",
  "reference_provenance": "Approved by Dr. Sarah Jenkins, PT on 2026-10-24",
  "patient_id": "patient-101",
  "repetitions_completed": 4,
  "duration_seconds": 15.0,
  "angle_stats": {
    "peak_elevation": 86.2,
    "mean_elevation": 46.5,
    "min_elevation": 14.0,
    "mean_elbow": 171.0,
    "rom": 72.2
  },
  "comparison_score": 93.4,
  "joint_deviations": {
    "mean_elevation_deviation": 5.2,
    "max_elevation_deviation": 9.8,
    "mean_elbow_deviation": 3.8,
    "rmse": 5.9
  },
  "model_status": "input_requires_validation",
  "lstm_prediction": null,
  "warnings": [],
  "notes": "Patient completed 4 controlled repetitions.",
  "raw_trajectory": [ ... ],
  "is_demo": 0
}
```

#### List Sessions
`GET /api/sessions`

**Optional Query Parameter:** `patient_id` (e.g. `?patient_id=patient-101`) to retrieve sessions for that patient only.

#### Download Clinical PDF Report
`GET /api/reports/{sess_id}`

Generates a publication-grade multi-table clinical assessment PDF report including:
- Mandatory Educational & Clinical Disclaimer
- Session Metadata Table (Session ID, Timestamp, Target Exercise, Duration, Repetitions, Reference ID, Reference Type, Version, Clinician Provenance, Patient Identifier)
- Biomechanical Kinematic Measurements Table
- Embedded Matplotlib Kinematic Elevation vs Elbow Trajectory Chart
- PyTorch LSTM Status Section
- Observations & Physiotherapist Notes Box
- Clinician Signature & Date Block

---

### 2.6 PyTorch Deep Learning Model Lab

#### Verified Sequence Inference
`POST /api/model/predict`

Enforces verified motion-capture tensor dimensions of shape `(64, 80)`.

**Request Payload:**
```json
{
  "sequence": [[ ... 80 float features ... ], ... 64 timesteps ...],
  "source_description": "REHAB24-6 mocap test repetition"
}
```

**Response `200 OK`:**
```json
{
  "status": "success",
  "predicted_class": 1,
  "class_label": "correct",
  "probabilities": {
    "incorrect": 0.12,
    "correct": 0.88
  },
  "input_shape": [1, 64, 80],
  "warning": "Model trained on REHAB24-6 3D motion capture dataset (9 participants, 178 total repetitions). Held-out participant accuracy: 82.5%, Macro F1: 81.95%. Results are preliminary and educational only. Not validated for clinical decision-making or determining exercise safety.",
  "educational_note": "Inference executed via verified PyTorch LSTM (2 layers, hidden_size=64, bidirectional=True)."
}
```

**Response `400 Bad Request` (Invalid Shape):**
```json
{
  "detail": "Input sequence must have shape (64, 80). Received (30, 80)."
}
```

#### Synthetic Sequence Fetcher
`GET /api/model/sample-sequence?form=correct`

Returns a valid synthetic `(64, 80)` sequence suitable for sending to `/api/model/predict` to demonstrate genuine PyTorch neural network execution without manual tensor construction.

---

### 2.7 Hybrid Ensemble Assessment Endpoints

#### Evaluate Repetition (Tri-Branch Assessment)
`POST /api/assessment/evaluate`

Evaluates a completed repetition simultaneously through the Statistical Branch, Deep-Learning Branch, and Hybrid Ensemble Decision Engine.

**Request Payload:**
```json
{
  "exercise_id": "standing-right-arm-abduction",
  "reference_id": "ref-gold-standard-demo",
  "trajectory": [
    { "timestamp": 0.0, "elevation": 15.0, "elbow": 172.0, "confidence": 0.95 },
    { "timestamp": 0.8, "elevation": 88.5, "elbow": 170.0, "confidence": 0.94 },
    { "timestamp": 1.6, "elevation": 16.0, "elbow": 172.0, "confidence": 0.96 }
  ],
  "mocap_sequence": null,
  "input_source": "webcam_mediapipe",
  "duration_sec": 1.6,
  "repetition_index": 1
}
```

**Response `200 OK` (Webcam Input $\to$ Safe Abstention):**
```json
{
  "status": "success",
  "statistical_assessment": {
    "score": 91.2,
    "classification_estimate": "CORRECT",
    "quality_status": "VALID",
    "primary_metric_name": "Arm Elevation",
    "secondary_metric_name": "Elbow Extension",
    "raw_measurements": {
      "peak_primary_angle": 88.5,
      "active_rom": 73.5,
      "mean_secondary_angle": 171.3,
      "rmse": 5.4
    },
    "sub_scores": {
      "rom_criterion": 1.0,
      "trajectory_conformity": 0.88,
      "secondary_posture": 1.0,
      "cadence_timing": 1.0
    },
    "warnings": []
  },
  "model_assessment": {
    "model_id": "rehab-lstm-rehab24-6-v1",
    "model_version": "1.0.0",
    "architecture": "PyTorch 2-Layer LSTM (unidirectional, hidden=64)",
    "input_schema": "mediapipe_monocular_uncalibrated",
    "input_quality_status": "INCOMPATIBLE_REPRESENTATION",
    "predicted_class": null,
    "class_label": null,
    "probabilities": null,
    "rejection_reason": "Domain shift: PyTorch LSTM was trained on 3D motion-capture marker coordinates (REHAB24-6). Live monocular MediaPipe landmarks lack calibrated 3D depth and metric scale."
  },
  "ensemble_decision": {
    "decision": "NEEDS_REVIEW",
    "combined_score": null,
    "confidence": 0.50,
    "branch_agreement": null,
    "explanation": "Ensemble abstains (NEEDS_REVIEW): Deep-learning branch input requires validation (MediaPipe monocular vs mocap domain shift). Statistical biomechanical branch scored 91.2/100 (CORRECT). Under the ensemble clinical safety policy, an autonomous decision requires both branches to be validated.",
    "reasons": [
      "Model branch abstained: Domain shift (MediaPipe monocular vs 3D mocap)."
    ],
    "weights_applied": {
      "statistical": 0.50,
      "model": 0.50
    }
  },
  "educational_disclaimer": "RehabVision AI is an educational research prototype. Assessments estimate movement similarity and do not represent clinical diagnoses or substitute for physiotherapy."
}
```

#### Cohort Evaluation Across Partitions
`GET /api/assessment/evaluate-cohort?split=test`

Runs independent performance evaluations on participant-partitioned test repetitions for all three systems:
- System A: Statistical-only
- System B: LSTM-only
- System C: Hybrid Ensemble

**Response `200 OK`:**
```json
{
  "split": "test",
  "participants": [2, 8],
  "sample_count": 40,
  "statistical_only": {
    "system_name": "Statistical / Biomechanical Evaluator",
    "total_samples": 40,
    "decided_samples": 40,
    "decision_coverage_pct": 100.0,
    "accuracy": 0.85,
    "macro_f1": 0.8291,
    "average_latency_ms": 9.61
  },
  "lstm_only": {
    "system_name": "PyTorch LSTM Evaluator",
    "total_samples": 40,
    "decided_samples": 40,
    "decision_coverage_pct": 100.0,
    "average_latency_ms": 3.25
  },
  "ensemble": {
    "system_name": "Hybrid Ensemble Decision Evaluator",
    "total_samples": 40,
    "decided_samples": 24,
    "abstained_samples": 16,
    "decision_coverage_pct": 60.0,
    "abstention_rate_pct": 40.0,
    "accuracy": 1.0,
    "macro_f1": 1.0,
    "brier_score": 0.0674,
    "average_latency_ms": 14.62
  },
  "evaluation_notes": [
    "Participant-wise holdout partition strictly maintained with zero participant leakage.",
    "Ensemble correctly transitions conflicting or degraded inputs to NEEDS_REVIEW abstention state."
  ]
}
```

---

### 2.8 Pluggable Model Registry Endpoints

#### List Registered Models
`GET /api/models`

**Optional Query Parameter:** `exercise_id` (e.g. `?exercise_id=standing-right-arm-abduction`) to filter models by supported exercise.

**Response `200 OK`:**
```json
[
  {
    "model_id": "existing-mocap-lstm",
    "model_version": "1.0.0",
    "exercise_ids": ["standing-right-arm-abduction"],
    "architecture": "PyTorch 2-Layer LSTM (hidden_size=64)",
    "input_sequence_length": 64,
    "input_feature_dim": 80,
    "input_schema_description": "64 timesteps x 80 features (78 normalized 3D mocap coordinates + 1 elevation + 1 trunk angle)",
    "coordinate_system": "3D Motion Capture normalized meters",
    "class_labels": { "0": "incorrect", "1": "correct" },
    "is_calibrated": true,
    "operational_status": "PRELIMINARY_MOCAP_BENCHMARK",
    "device_requirements": "CPU / CUDA",
    "provenance_notes": "Trained on REHAB24-6 motion-capture dataset. Strict domain guard rejects uncalibrated webcam coordinates."
  },
  {
    "model_id": "teammate-bicep-curl-model",
    "model_version": "0.1.0-template",
    "exercise_ids": ["seated-right-bicep-curl"],
    "architecture": "Pluggable Architecture (Teammate Model Adapter)",
    "input_sequence_length": 32,
    "input_feature_dim": 4,
    "input_schema_description": "32 timesteps x 4 features (elbow_flexion, shoulder_sway, trunk_lean, movement_cadence)",
    "coordinate_system": "2D MediaPipe Normalized Image Coordinates + Kinematic Angles (deg)",
    "class_labels": { "0": "incorrect", "1": "correct" },
    "is_calibrated": false,
    "operational_status": "TEMPLATE_MOCK_REQUIRES_TRAINED_WEIGHTS",
    "device_requirements": "CPU",
    "provenance_notes": "Reference adapter template for teammate model integration. Place trained weights in backend/models/ to activate."
  }
]
```

#### Get Model Metadata
`GET /api/models/{model_id}`

Retrieves the complete declaration and feature schema for a specific model ID (e.g., `/api/models/existing-mocap-lstm`). Returns `404 Not Found` if the model ID is not registered.

---

## 3. Client Integration Code Examples

### 3.1 TypeScript / Fetch
```typescript
import { ReferenceData, CompareResult } from './types';

// Fetch references isolated for patient-101
async function getPatientReferences(patientId: string): Promise<ReferenceData[]> {
  const response = await fetch(`http://127.0.0.1:8000/api/references?patient_id=${encodeURIComponent(patientId)}`);
  if (!response.ok) throw new Error(`API error: ${response.statusText}`);
  return response.json();
}

// Compare patient movement
async function compareWorkout(refId: string, trajectory: Array<{timestamp: number, elevation: number, elbow: number}>): Promise<CompareResult> {
  const response = await fetch('http://127.0.0.1:8000/api/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      reference_id: refId,
      home_angles: trajectory
    })
  });
  return response.json();
}
```

### 3.2 Python / Requests
```python
import requests

BASE_URL = "http://127.0.0.1:8000"

# 1. Health check
health = requests.get(f"{BASE_URL}/api/health").json()
print("System Status:", health["status"], "| Supported Exercises:", health["supported_exercises"])

# 2. Query references for patient-101
refs = requests.get(f"{BASE_URL}/api/references", params={"patient_id": "patient-101"}).json()
for r in refs:
    print(f"Ref: {r['id']} | Type: {r['reference_type']} | Prov: {r['provenance']}")

# 3. Download session PDF report
sess_id = "sess-demo-001"
pdf_res = requests.get(f"{BASE_URL}/api/reports/{sess_id}")
with open(f"Report_{sess_id}.pdf", "wb") as f:
    f.write(pdf_res.content)
print(f"Report saved: Report_{sess_id}.pdf")
```
