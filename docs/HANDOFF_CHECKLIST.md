# RehabVision AI — Teammate Integration & Handoff Checklist

This document provides a concise, step-by-step integration checklist for your teammate integrating **RehabVision AI** into the broader rehabilitation application.

---

## 1. Quickstart & Verification Checklist

- [ ] **Step 1: Clone / Locate Workspace**
  - Project root: `RehabVision-AI/` (e.g. `git clone https://github.com/Mohammad-Nadeem-08/rehabilitation_form_detector.git`)
- [ ] **Step 2: Python Environment Verification**
  - Python version: 3.10 to 3.14 (Verified on Python 3.14.3).
  - Install dependencies: `pip install -r requirements.txt` (FastAPI, Uvicorn, PyTorch, ReportLab, Matplotlib, Pydantic, NumPy, Pytest).
- [ ] **Step 3: Node.js & Frontend Verification**
  - Node.js: v18+ / v20+.
  - In `frontend/`: `npm install`
- [ ] **Step 4: Run Automated Verification Test Suites**
  ```powershell
  # Backend unit, adapter, and regression tests (62 tests):
  python -m pytest backend/tests -v

  # Frontend repetition tracker & kinematic tests (13 tests):
  cd frontend
  npx tsx src/services/repetitionTracker.test.ts

  # Production bundle build verification:
  npm run build
  ```
  Expected output: **62 passed in backend**, **13 passed in frontend**, **0 build errors**.
- [ ] **Step 5: Launch Local Services**
  - Execute `start.bat` in the project root, or launch manually in separate terminals:
    - Terminal 1 (Backend): `cd backend && uvicorn main:app --host 127.0.0.1 --port 8000`
    - Terminal 2 (Frontend): `cd frontend && npm run dev`
  - Verify API health: `curl http://127.0.0.1:8000/api/health`
  - Verify Model Registry: `curl http://127.0.0.1:8000/api/models`
  - Verify Cohort Evaluation: `curl http://127.0.0.1:8000/api/assessment/evaluate-cohort`
  - Open UI: `http://127.0.0.1:5173`

---

## 2. Integration Touchpoints for Teammate Application

### Touchpoint A: Connecting an External Frontend to the REST API
If your teammate is building or hosting their own frontend:
1. Point their HTTP client to `http://127.0.0.1:8000` (or configure a production reverse proxy).
2. Follow the detailed request/response schemas in [docs/API_INTEGRATION_GUIDE.md](docs/API_INTEGRATION_GUIDE.md).
3. Ensure requests specify `Content-Type: application/json`.
4. To download reports, use `GET /api/reports/{session_id}` (returns binary PDF).

### Touchpoint B: Patient Data Isolation
1. When a patient logs in, pass `patient_id` (e.g. `patient-101`) to:
   - `GET /api/references?patient_id=patient-101` (retrieves patient's customized references + universal standards; excludes other patients).
   - `GET /api/sessions?patient_id=patient-101` (retrieves session history strictly for that patient).
   - `POST /api/sessions` (includes `patient_id` in the session payload).

### Touchpoint C: Registering Additional Rehabilitation Exercises
1. To add a new exercise without duplicating core code:
   - Follow [docs/EXERCISE_CONFIGURATION_GUIDE.md](docs/EXERCISE_CONFIGURATION_GUIDE.md).
   - Create a strategy class in `backend/exercise_registry.py` inheriting from `ExerciseStrategy`.
   - Register it with `exercise_registry.register(YourStrategy())`.
   - Mirror the thresholds in `frontend/src/services/exerciseRegistry.ts`.
   - The DTW dynamic time warping, repetition FSM, SQLite schema, and PDF reports immediately work with the new exercise.

### Touchpoint D: Managing Reference Types & Clinician Provenance
1. **Standard References:** For clinic-wide or hospital baselines, set `reference_type: "standard"` and provide protocol documentation in `provenance`.
2. **Patient-Specific References:** When a clinician tailors an exercise arc for a specific patient, set `reference_type: "patient_specific"`, specify `patient_id: "patient-101"`, and include the approving clinician in `provenance` (e.g., `"Approved by Dr. Sarah Jenkins, PT on 2026-10-24"`).
3. Both reference types are explicitly identified with their version and provenance badges in the UI, SQLite database, and generated PDF reports.

### Touchpoint E: Integrating Teammate's Trained ML/DL Model
1. To plug in a trained machine-learning model (LSTM, CNN, Transformer, GCN, ONNX) for a new exercise:
   - Follow the detailed walkthrough in [docs/MODEL_INTEGRATION_GUIDE.md](docs/MODEL_INTEGRATION_GUIDE.md).
   - Create an adapter subclass in `backend/model_adapters/` inheriting from `BaseModelAdapter`.
   - Declare the model metadata, input shape, coordinate system, and class index mapping.
   - Register it in `backend/model_adapters/registry.py` with `model_registry.register(YourAdapter())`.
   - Use safe model loading (e.g., `weights_only=True` for PyTorch) and verify inference with [docs/MODEL_EVALUATION_GUIDE.md](docs/MODEL_EVALUATION_GUIDE.md).
   - Zero modifications are needed in existing assessment or database code!

### Touchpoint F: Tri-Branch Hybrid Ensemble Assessment Pipeline
1. **Call Evaluation Endpoint:** `POST /api/assessment/evaluate`
   - Accepts trajectory angles, optional mocap/model sequence, exercise ID, and optional `model_id`.
   - Computes Statistical Branch, Deep-Learning Branch, and Hybrid Ensemble Decision.
   - Enforces conflict abstention: returns `NEEDS_REVIEW` whenever branches conflict or webcam input lacks calibrated 3D representation.
2. **Retrieve Model Evaluation Metrics:** `GET /api/assessment/evaluate-cohort?split=test`
   - Returns independent performance metrics across the 40 held-out test repetitions.
3. **UI Integration:** Use `<AssessmentPanel />` component (`frontend/src/components/AssessmentPanel.tsx`) to display the 3-card architecture (Statistical Score, DL Model Status, Ensemble Decision).

---

## 3. Key Files & Directory Sitemap

```
RehabVision-AI/
├── README.md                           # Master project guide & architecture overview
├── CONTRIBUTING.md                     # Code standards and contribution guide
├── CHANGELOG.md                        # Version history and release notes
├── .env.example                        # Template environment variables
├── start.bat                           # Single-click launcher for FastAPI & Vite
├── requirements.txt                    # Python dependencies
│
├── backend/
│   ├── main.py                         # FastAPI routes and middleware
│   ├── model_adapters/                 # Model-agnostic adapter framework
│   │   ├── base.py                     # BaseModelAdapter abstract interface & dataclasses
│   │   ├── registry.py                 # ModelRegistry singleton & exercise mapping
│   │   ├── existing_mocap_lstm.py      # Adapter for REHAB24-6 (64, 80) mocap LSTM
│   │   └── example_teammate_adapter.py # Reference adapter template for teammate model
│   ├── statistical_assessment.py       # Branch 1: Statistical & biomechanical assessment engine
│   ├── model_assessment.py             # Branch 2: PyTorch LSTM assessment engine & compatibility guard
│   ├── ensemble.py                     # Branch 3: Hybrid ensemble decision engine & abstention policy
│   ├── evaluation.py                   # Evaluation framework (decided vs cohort accuracy, Brier scores)
│   ├── exercise_registry.py            # Pluggable ExerciseStrategy base class & registry
│   ├── movement_comparison.py          # Repetition cycle segmentation & DTW alignment
│   ├── pose_features.py                # 2D/3D joint angles, landmark scoring, FSM
│   ├── database.py                     # SQLite initialization, schema migrations, seed data
│   ├── reporting.py                    # ReportLab PDF report generation with ensemble cards
│   ├── inference.py                    # PyTorch LSTM loader, compatibility guard, and sample generator
│   ├── models/
│   │   ├── rehab_lstm.pth              # Trained PyTorch weights (unidirectional 2-layer LSTM, hidden=64)
│   │   ├── preprocessing.npz           # Feature scalers and normalization stats (1, 1, 80)
│   │   └── metadata.json               # Training metadata (REHAB24-6, 82.5% accuracy)
│   └── tests/
│       ├── test_all.py                 # Pytest test suite (51 unit, regression, and ensemble tests)
│       └── test_model_adapters.py      # Adapter tests (11 model registry & contract tests)
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                     # Main layout and tab navigation
│   │   ├── types.ts                    # TypeScript domain interfaces
│   │   ├── services/
│   │   │   ├── api.ts                  # REST API client (saveSession, evaluateHybridAssessment, etc.)
│   │   │   ├── exerciseRegistry.ts     # Client exercise configurations and thresholds
│   │   │   ├── repetitionTracker.ts    # Client FSM and repetition engine
│   │   │   ├── repetitionTracker.test.ts # Frontend unit test suite (13 tests)
│   │   │   └── mediapipeService.ts     # MediaPipe Tasks Vision wrapper
│   │   ├── pages/
│   │   │   ├── LiveAssessmentPage.tsx  # Real-time CV camera assessment with AssessmentPanel
│   │   │   ├── ExerciseReferencePage.tsx # Reference capture, provenance, and library
│   │   │   ├── SessionHistoryPage.tsx  # Session log, provenance columns, PDF download
│   │   │   ├── ModelLabPage.tsx        # PyTorch LSTM interactive sequence tester
│   │   │   └── ProgressDashboardPage.tsx # Aggregate statistics & telemetry
│   │   └── components/
│   │       ├── AssessmentPanel.tsx     # Dedicated 3-card hybrid ensemble results panel
│   │       ├── CameraView.tsx          # Video canvas, skeleton overlay, FSM diagnostic HUD
│   │       ├── KinematicChart.tsx      # SVG live trajectory plotting
│   │       └── ReportModal.tsx         # Detailed session report dialog with AssessmentPanel
│   └── package.json
│
└── docs/
    ├── MODEL_INTEGRATION_GUIDE.md      # Pluggable model adapter integration walkthrough
    ├── MODEL_EVALUATION_GUIDE.md       # Benchmarking framework and evaluation metrics
    ├── ENSEMBLE_ASSESSMENT.md          # Comprehensive tri-branch ensemble specification & metrics
    ├── ARCHITECTURE.md                 # System architecture, data flow, FSM, DTW, and LSTM boundaries
    ├── API_INTEGRATION_GUIDE.md        # Complete OpenAPI / REST contract and client examples
    ├── EXERCISE_CONFIGURATION_GUIDE.md # Tutorial for adding new exercises with zero duplication
    ├── FINAL_TEST_REPORT.md            # Detailed results of real webcam and automated test suites
    └── HANDOFF_CHECKLIST.md            # This integration and handoff document
```

---

## 4. Maintenance & Support Contact

- **Project:** RehabVision AI (7th-Semester Deep Learning Capstone)
- **Status:** All 10 phases of the Hybrid Deep Learning + Statistical Ensemble Assessment are complete, verified, and passing 100% of automated tests.

