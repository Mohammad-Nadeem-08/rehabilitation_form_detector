# Contributing to RehabVision AI

Welcome to the RehabVision AI project! This document outlines coding standards, workflow conventions, and steps for adding new exercises, reference movements, and machine-learning models.

---

## 1. Project Organization

```
RehabVision-AI/
├── backend/
│   ├── model_adapters/      # Model-agnostic ML adapters & registry
│   ├── models/              # Checkpoint files (.pth, .onnx, preprocessing.npz)
│   ├── tests/               # Pytest test suite (62 tests passing)
│   ├── database.py          # SQLite schema & persistence migrations
│   ├── ensemble.py          # Hybrid ensemble decision engine
│   ├── evaluation.py        # Independent evaluation & benchmarking framework
│   ├── exercise_registry.py # Exercise movement strategy registry
│   ├── inference.py         # PyTorch LSTM inference engine
│   ├── main.py              # FastAPI REST API application
│   ├── model_assessment.py  # Model assessment engine interface
│   ├── movement_comparison.py # DTW trajectory & kinematic comparisons
│   ├── pose_features.py     # MediaPipe landmark geometry & FSM tracker
│   ├── reporting.py         # PDF clinical report generator
│   ├── schemas.py           # Pydantic request/response schemas
│   └── statistical_assessment.py # Biomechanical rule-based evaluator
├── frontend/
│   ├── src/
│   │   ├── components/      # UI components (AssessmentPanel, CameraView, etc.)
│   │   ├── pages/           # Route views (LiveAssessment, ModelLab, Sessions)
│   │   ├── services/        # API clients & TypeScript repetition tracker FSM
│   │   └── types.ts         # TypeScript interfaces
├── docs/                    # Technical architecture & integration documentation
└── requirements.txt         # Python dependencies
```

---

## 2. Adding a New Exercise

To add a new rehabilitation exercise (e.g., seated shoulder press, knee extension):
1. **Implement Exercise Strategy:**
   - In [`backend/exercise_registry.py`](backend/exercise_registry.py), subclass `ExerciseStrategy`.
   - Define `primary_metric_name`, `secondary_metric_name`, `fsm_thresholds`, `required_landmarks`, and custom angle calculation rules.
2. **Register the Strategy:**
   - Call `exercise_registry.register(MyNewExerciseStrategy())`.
3. **Register in Frontend:**
   - In `frontend/src/services/exerciseRegistry.ts`, add the exercise configuration and landmark mappings.
4. **Add Unit Tests:**
   - Add kinematic geometry and FSM threshold tests in `backend/tests/test_all.py`.

---

## 3. Adding a New Machine Learning Model

To integrate your trained machine-learning model without touching existing code:
1. Review [`docs/MODEL_INTEGRATION_GUIDE.md`](docs/MODEL_INTEGRATION_GUIDE.md).
2. Place your model file in `backend/models/`.
3. Subclass `BaseModelAdapter` in `backend/model_adapters/`.
4. Register your adapter in `backend/model_adapters/registry.py`.
5. Add unit tests in `backend/tests/test_model_adapters.py`.

---

## 4. Development Workflow & Running Tests

### Backend Setup & Tests
```powershell
# Install dependencies
pip install -r requirements.txt

# Run complete pytest test suite (62 tests)
pytest backend/tests -v

# Start FastAPI backend daemon
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend Setup & Tests
```powershell
cd frontend

# Install packages
npm ci

# Run TypeScript repetition tracker test suite (13 tests)
npx tsx src/services/repetitionTracker.test.ts

# Run production build
npm run build

# Start local Vite dev server
npm run dev
```

---

## 5. Coding & Contribution Rules

1. **Model-Agnostic Integrity:** Never hardcode exercise-specific logic into `main.py` or `ensemble.py`. Use the strategy and model registries.
2. **Clinical Safety Guards:** If an input representation is uncalibrated (such as live monocular webcam landmarks fed to 3D mocap models), the adapter must reject the input with `INCOMPATIBLE_REPRESENTATION`, and the ensemble must abstain (`NEEDS_REVIEW`).
3. **Preserve Passing Tests:** All 62 backend tests and 13 frontend tests must pass before submitting changes.
4. **Documentation Sync:** When adding an API route or adapter, update `docs/API_INTEGRATION_GUIDE.md` and `docs/MODEL_INTEGRATION_GUIDE.md`.
