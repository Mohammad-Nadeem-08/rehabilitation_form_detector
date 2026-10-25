import os
import json
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from database import init_db, get_db_connection
from schemas import (
    ExerciseSchema, ReferenceCreate, ReferenceResponse,
    SessionCreate, SessionResponse, CompareRequest, CompareResponse,
    ModelPredictRequest, ModelPredictResponse, ProgressResponse,
    AngleStats, JointDeviations, LSTMPrediction,
    StatisticalAssessmentSchema, ModelAssessmentSchema, EnsembleDecisionSchema,
    HybridEvaluationRequest, HybridEvaluationResponse, CohortEvaluationResponse,
    ModelMetadataSchema
)
from movement_comparison import compare_angle_trajectories
from inference import lstm_engine
from reporting import generate_session_pdf
from statistical_assessment import StatisticalAssessmentEngine
from model_assessment import ModelAssessmentEngine
from ensemble import EnsembleDecisionEngine
from evaluation import evaluate_cohort
from model_adapters import model_registry

# Initialize SQLite database schema and seed data
init_db()

# Initialize assessment singleton engines
statistical_assessment_engine = StatisticalAssessmentEngine()
model_assessment_engine = ModelAssessmentEngine()
ensemble_decision_engine = EnsembleDecisionEngine()

app = FastAPI(
    title="RehabVision AI Backend",
    description="Personalized AI-Based Rehabilitation Form Detection Using LSTM and Human Pose Estimation (Educational Prototype)",
    version="1.0.0"
)

# Enable CORS for local Vite development frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from exercise_registry import exercise_registry

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "app": "RehabVision AI",
        "version": "1.0.0",
        "model_loaded": lstm_engine.is_loaded,
        "model_error": lstm_engine.load_error,
        "supported_exercises": [s.id for s in exercise_registry.list_all()],
        "notice": "Educational prototype only. Not validated for clinical decision-making or determining exercise safety."
    }

@app.get("/api/exercises", response_model=List[ExerciseSchema])
def get_exercises():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM exercises")
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        row_dict = dict(r)
        strategy = exercise_registry.get(row_dict["id"])
        if strategy:
            row_dict["primary_metric_name"] = strategy.primary_metric_name
            row_dict["secondary_metric_name"] = strategy.secondary_metric_name
            row_dict["fsm_thresholds"] = strategy.fsm_thresholds
            row_dict["required_landmarks"] = strategy.required_landmarks
        results.append(ExerciseSchema(**row_dict))
    return results

@app.get("/api/exercises/{exercise_id}", response_model=ExerciseSchema)
def get_exercise_by_id(exercise_id: str):
    strategy = exercise_registry.get(exercise_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found in registry")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM exercises WHERE id = ?", (exercise_id,))
    row = cursor.fetchone()
    conn.close()

    base_dict = dict(row) if row else strategy.to_dict()
    base_dict["primary_metric_name"] = strategy.primary_metric_name
    base_dict["secondary_metric_name"] = strategy.secondary_metric_name
    base_dict["fsm_thresholds"] = strategy.fsm_thresholds
    base_dict["required_landmarks"] = strategy.required_landmarks
    return ExerciseSchema(**base_dict)

@app.post("/api/references", response_model=ReferenceResponse)
def create_reference(ref_in: ReferenceCreate):
    ref_id = f"ref-{uuid.uuid4().hex[:8]}"
    created_at = datetime.now().isoformat()
    angles_json = json.dumps([p.dict() for p in ref_in.joint_angles])
    summary_json = json.dumps(ref_in.landmarks_summary or {})

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO references_table (
        id, exercise_id, exercise_name, notes, duration_seconds,
        frame_count, joint_angles, landmarks_summary, created_at, is_demo,
        reference_type, version, provenance, patient_id, is_active
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
    """, (
        ref_id, ref_in.exercise_id, ref_in.exercise_name, ref_in.notes,
        ref_in.duration_seconds, ref_in.frame_count, angles_json,
        summary_json, created_at,
        ref_in.reference_type, ref_in.version, ref_in.provenance,
        ref_in.patient_id, ref_in.is_active
    ))
    conn.commit()
    conn.close()

    return ReferenceResponse(
        id=ref_id,
        exercise_id=ref_in.exercise_id,
        exercise_name=ref_in.exercise_name,
        notes=ref_in.notes,
        duration_seconds=ref_in.duration_seconds,
        frame_count=ref_in.frame_count,
        joint_angles=ref_in.joint_angles,
        landmarks_summary=ref_in.landmarks_summary,
        created_at=created_at,
        is_demo=0,
        reference_type=ref_in.reference_type,
        version=ref_in.version,
        provenance=ref_in.provenance,
        patient_id=ref_in.patient_id,
        is_active=ref_in.is_active
    )

@app.get("/api/references", response_model=List[ReferenceResponse])
def list_references(
    exercise_id: Optional[str] = None,
    patient_id: Optional[str] = None,
    reference_type: Optional[str] = None
):
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM references_table WHERE 1=1"
    params = []

    if exercise_id:
        query += " AND exercise_id = ?"
        params.append(exercise_id)
    if reference_type:
        query += " AND reference_type = ?"
        params.append(reference_type)
    if patient_id:
        # Include patient-specific references for this patient OR standard reusable references
        query += " AND (patient_id = ? OR reference_type = 'standard')"
        params.append(patient_id)

    query += " ORDER BY created_at DESC"
    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        row_dict = dict(r)
        row_dict["joint_angles"] = json.loads(row_dict["joint_angles"])
        row_dict["landmarks_summary"] = json.loads(row_dict["landmarks_summary"])
        # Handle default fallback values for older rows
        row_dict.setdefault("reference_type", "standard")
        row_dict.setdefault("version", "1.0")
        row_dict.setdefault("provenance", "Clinical Protocol Baseline")
        row_dict.setdefault("patient_id", None)
        row_dict.setdefault("is_active", 1)
        result.append(ReferenceResponse(**row_dict))
    return result

@app.get("/api/references/{ref_id}", response_model=ReferenceResponse)
def get_reference(ref_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM references_table WHERE id = ?", (ref_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Reference '{ref_id}' not found")
    
    row_dict = dict(row)
    row_dict["joint_angles"] = json.loads(row_dict["joint_angles"])
    row_dict["landmarks_summary"] = json.loads(row_dict["landmarks_summary"])
    row_dict.setdefault("reference_type", "standard")
    row_dict.setdefault("version", "1.0")
    row_dict.setdefault("provenance", "Clinical Protocol Baseline")
    row_dict.setdefault("patient_id", None)
    row_dict.setdefault("is_active", 1)
    return ReferenceResponse(**row_dict)

@app.post("/api/compare", response_model=CompareResponse)
def compare_movement(req: CompareRequest):
    ref_angles = req.reference_angles
    if not ref_angles and req.reference_id:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT joint_angles FROM references_table WHERE id = ?", (req.reference_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            raw_angles = json.loads(row["joint_angles"])
            ref_angles = raw_angles
        else:
            raise HTTPException(status_code=404, detail=f"Reference '{req.reference_id}' not found")
    
    if not ref_angles:
        # Load default demo reference
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT joint_angles FROM references_table ORDER BY created_at ASC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        if row:
            ref_angles = json.loads(row["joint_angles"])
        else:
            raise HTTPException(status_code=400, detail="No reference sequence available to compare against")

    ref_dicts = [a if isinstance(a, dict) else a.dict() for a in ref_angles]
    home_dicts = [a if isinstance(a, dict) else a.dict() for a in req.home_angles]

    comparison_result = compare_angle_trajectories(ref_dicts, home_dicts)
    return CompareResponse(**comparison_result)

@app.post("/api/assessment/evaluate", response_model=HybridEvaluationResponse)
def evaluate_hybrid_movement(req: HybridEvaluationRequest):
    """
    Evaluates a completed movement repetition through the 3-branch assessment architecture:
    1. Statistical/biomechanical assessment (ROM, DTW, posture, cadence).
    2. Deep learning model assessment (strict mocap schema guard, input_requires_validation for webcam).
    3. Hybrid ensemble decision engine (abstained NEEDS_REVIEW if either branch incompatible or low quality).
    """
    # 1. Resolve reference angles
    ref_angles = req.reference_angles
    if not ref_angles and req.reference_id:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT joint_angles FROM references_table WHERE id = ?", (req.reference_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            ref_angles = json.loads(row["joint_angles"])

    # 2. Statistical Assessment Branch
    stat_result = statistical_assessment_engine.evaluate_repetition(
        trajectory=req.trajectory,
        exercise_id=req.exercise_id,
        reference_angles=ref_angles,
        duration_sec=req.duration_sec,
        repetition_index=req.repetition_index,
        clinician_criteria=req.clinician_criteria,
    )

    # 3. Model Assessment Branch
    model_result = model_assessment_engine.evaluate_sequence(
        sequence=req.mocap_sequence,
        input_source=req.input_source,
        exercise_id=req.exercise_id,
        model_id=req.model_id,
        repetition_index=req.repetition_index,
    )

    # 4. Hybrid Ensemble Decision Engine
    ensemble_result = ensemble_decision_engine.evaluate(
        statistical_result=stat_result,
        model_result=model_result,
        input_quality_metadata={
            "source": req.input_source,
            "trajectory_points": len(req.trajectory) if req.trajectory else 0,
            "quality_status": stat_result.quality_status
        },
        exercise_id=req.exercise_id,
        reference_id=req.reference_id,
        repetition_index=req.repetition_index,
    )

    return HybridEvaluationResponse(
        status="success",
        statistical_assessment=stat_result.to_dict(),
        model_assessment=model_result.to_dict(),
        ensemble_decision=ensemble_result.to_dict(),
        educational_disclaimer="RehabVision AI is an educational research prototype. Assessments estimate movement similarity and do not represent clinical diagnoses or substitute for physiotherapy."
    )

@app.get("/api/models", response_model=List[ModelMetadataSchema])
def list_registered_models(exercise_id: Optional[str] = Query(None)):
    """
    Returns all registered machine-learning model adapters, their declared input schemas,
    coordinate systems, compatibility status, and health.
    """
    if exercise_id:
        adapters = model_registry.get_for_exercise(exercise_id)
        results = []
        for a in adapters:
            m = a.get_metadata().to_dict()
            m["health"] = a.health_check()
            results.append(m)
        return results
    return model_registry.list_all()

@app.get("/api/models/{model_id}", response_model=ModelMetadataSchema)
def get_model_by_id(model_id: str):
    """
    Returns declared metadata, input schema, class mapping, and health details
    for a specific registered model adapter.
    """
    adapter = model_registry.get(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Model adapter '{model_id}' not found in registry.")
    meta = adapter.get_metadata().to_dict()
    meta["health"] = adapter.health_check()
    return meta

@app.get("/api/assessment/evaluate-cohort", response_model=CohortEvaluationResponse)
def get_cohort_evaluation(split: str = Query("test", enum=["train", "val", "test"])):
    """
    Evaluates all 3 assessment pipelines (Statistical-Only, LSTM-Only, Hybrid Ensemble)
    independently across the verified REHAB24-6 participant partitions.
    """
    metrics = evaluate_cohort(split=split)
    return CohortEvaluationResponse(**metrics)

@app.post("/api/sessions", response_model=SessionResponse)
def record_session(sess_in: SessionCreate):
    sess_id = f"sess-{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now().isoformat()

    # Automatically resolve reference provenance if reference_id provided but provenance empty
    ref_type = sess_in.reference_type or "standard"
    ref_ver = sess_in.reference_version or "1.0"
    ref_prov = sess_in.reference_provenance

    if sess_in.reference_id and not ref_prov:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT reference_type, version, provenance, patient_id FROM references_table WHERE id = ?", (sess_in.reference_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            ref_type = row["reference_type"] or ref_type
            ref_ver = row["version"] or ref_ver
            ref_prov = row["provenance"]

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO sessions (
        id, timestamp, exercise_id, exercise_name, reference_id,
        repetitions_completed, duration_seconds, angle_stats, comparison_score,
        joint_deviations, model_status, lstm_prediction, warnings, notes, raw_trajectory, is_demo,
        reference_type, reference_version, reference_provenance, patient_id,
        statistical_assessment, model_assessment, ensemble_decision
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        sess_id,
        timestamp,
        sess_in.exercise_id,
        sess_in.exercise_name,
        sess_in.reference_id,
        sess_in.repetitions_completed,
        sess_in.duration_seconds,
        sess_in.angle_stats.json(),
        sess_in.comparison_score,
        sess_in.joint_deviations.json(),
        sess_in.model_status,
        sess_in.lstm_prediction.json() if sess_in.lstm_prediction else None,
        json.dumps(sess_in.warnings),
        sess_in.notes,
        json.dumps([p.dict() for p in sess_in.raw_trajectory]) if sess_in.raw_trajectory else None,
        sess_in.is_demo,
        ref_type,
        ref_ver,
        ref_prov,
        sess_in.patient_id,
        json.dumps(sess_in.statistical_assessment) if sess_in.statistical_assessment else None,
        json.dumps(sess_in.model_assessment) if sess_in.model_assessment else None,
        json.dumps(sess_in.ensemble_decision) if sess_in.ensemble_decision else None
    ))
    conn.commit()
    conn.close()

    return SessionResponse(
        id=sess_id,
        timestamp=timestamp,
        exercise_id=sess_in.exercise_id,
        exercise_name=sess_in.exercise_name,
        reference_id=sess_in.reference_id,
        reference_type=ref_type,
        reference_version=ref_ver,
        reference_provenance=ref_prov,
        patient_id=sess_in.patient_id,
        repetitions_completed=sess_in.repetitions_completed,
        duration_seconds=sess_in.duration_seconds,
        angle_stats=sess_in.angle_stats,
        comparison_score=sess_in.comparison_score,
        joint_deviations=sess_in.joint_deviations,
        model_status=sess_in.model_status,
        lstm_prediction=sess_in.lstm_prediction,
        statistical_assessment=sess_in.statistical_assessment,
        model_assessment=sess_in.model_assessment,
        ensemble_decision=sess_in.ensemble_decision,
        warnings=sess_in.warnings,
        notes=sess_in.notes,
        raw_trajectory=sess_in.raw_trajectory,
        is_demo=sess_in.is_demo
    )

@app.get("/api/sessions", response_model=List[SessionResponse])
def list_sessions(patient_id: Optional[str] = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if patient_id:
        cursor.execute("SELECT * FROM sessions WHERE patient_id = ? ORDER BY timestamp DESC", (patient_id,))
    else:
        cursor.execute("SELECT * FROM sessions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        row_dict = dict(r)
        row_dict["angle_stats"] = json.loads(row_dict["angle_stats"])
        row_dict["joint_deviations"] = json.loads(row_dict["joint_deviations"])
        row_dict["warnings"] = json.loads(row_dict["warnings"])
        if row_dict.get("lstm_prediction"):
            row_dict["lstm_prediction"] = json.loads(row_dict["lstm_prediction"])
        if row_dict.get("statistical_assessment"):
            row_dict["statistical_assessment"] = json.loads(row_dict["statistical_assessment"])
        if row_dict.get("model_assessment"):
            row_dict["model_assessment"] = json.loads(row_dict["model_assessment"])
        if row_dict.get("ensemble_decision"):
            row_dict["ensemble_decision"] = json.loads(row_dict["ensemble_decision"])
        if row_dict.get("raw_trajectory"):
            row_dict["raw_trajectory"] = json.loads(row_dict["raw_trajectory"])
        row_dict.setdefault("reference_type", "standard")
        row_dict.setdefault("reference_version", "1.0")
        row_dict.setdefault("reference_provenance", None)
        row_dict.setdefault("patient_id", None)
        result.append(SessionResponse(**row_dict))
    return result

@app.get("/api/sessions/{sess_id}", response_model=SessionResponse)
def get_session(sess_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (sess_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Session '{sess_id}' not found")

    row_dict = dict(row)
    row_dict["angle_stats"] = json.loads(row_dict["angle_stats"])
    row_dict["joint_deviations"] = json.loads(row_dict["joint_deviations"])
    row_dict["warnings"] = json.loads(row_dict["warnings"])
    if row_dict.get("lstm_prediction"):
        row_dict["lstm_prediction"] = json.loads(row_dict["lstm_prediction"])
    if row_dict.get("statistical_assessment"):
        row_dict["statistical_assessment"] = json.loads(row_dict["statistical_assessment"])
    if row_dict.get("model_assessment"):
        row_dict["model_assessment"] = json.loads(row_dict["model_assessment"])
    if row_dict.get("ensemble_decision"):
        row_dict["ensemble_decision"] = json.loads(row_dict["ensemble_decision"])
    if row_dict.get("raw_trajectory"):
        row_dict["raw_trajectory"] = json.loads(row_dict["raw_trajectory"])
    return SessionResponse(**row_dict)

@app.post("/api/model/predict", response_model=ModelPredictResponse)
def predict_lstm(req: ModelPredictRequest):
    """
    Genuine PyTorch inference endpoint for verified 80-feature motion-capture sequences.
    Strictly enforces input shape (64, 80).
    """
    res = lstm_engine.predict_compatible_sequence(req.sequence)
    if res.get("status") != "success":
        raise HTTPException(status_code=400, detail=res.get("message", "Prediction failed"))
    
    return ModelPredictResponse(**res)

@app.get("/api/model/sample-sequence")
def get_sample_sequence(form: str = Query("correct", enum=["correct", "incorrect"])):
    """
    Helper to fetch a synthetic compatible (64, 80) sequence for demonstrating the real PyTorch LSTM.
    """
    is_correct = (form == "correct")
    seq = lstm_engine.generate_sample_mocap_repetition(is_correct=is_correct)
    return {
        "form": form,
        "shape": [len(seq), len(seq[0])],
        "sequence": seq,
        "note": "Compatible 80-feature sequence (64 timesteps x 80 features). Suitable for /api/model/predict."
    }

@app.get("/api/progress", response_model=ProgressResponse)
def get_progress():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM references_table")
    total_refs = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*), SUM(repetitions_completed), AVG(comparison_score) FROM sessions")
    row = cursor.fetchone()
    total_sess = row[0] or 0
    total_reps = row[1] or 0
    avg_score = round(float(row[2] or 0.0), 1)

    cursor.execute("SELECT * FROM sessions ORDER BY timestamp DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()

    recent = []
    for r in rows:
        row_dict = dict(r)
        row_dict["angle_stats"] = json.loads(row_dict["angle_stats"])
        row_dict["joint_deviations"] = json.loads(row_dict["joint_deviations"])
        row_dict["warnings"] = json.loads(row_dict["warnings"])
        if row_dict.get("lstm_prediction"):
            row_dict["lstm_prediction"] = json.loads(row_dict["lstm_prediction"])
        if row_dict.get("statistical_assessment"):
            row_dict["statistical_assessment"] = json.loads(row_dict["statistical_assessment"])
        if row_dict.get("model_assessment"):
            row_dict["model_assessment"] = json.loads(row_dict["model_assessment"])
        if row_dict.get("ensemble_decision"):
            row_dict["ensemble_decision"] = json.loads(row_dict["ensemble_decision"])
        if row_dict.get("raw_trajectory"):
            row_dict["raw_trajectory"] = json.loads(row_dict["raw_trajectory"])
        recent.append(SessionResponse(**row_dict))

    return ProgressResponse(
        total_references=total_refs,
        total_sessions=total_sess,
        total_repetitions=total_reps,
        average_comparison_score=avg_score,
        recent_sessions=recent
    )

@app.get("/api/reports/{sess_id}")
def download_report(sess_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (sess_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Session '{sess_id}' not found")

    row_dict = dict(row)
    row_dict["angle_stats"] = json.loads(row_dict["angle_stats"])
    row_dict["joint_deviations"] = json.loads(row_dict["joint_deviations"])
    row_dict["warnings"] = json.loads(row_dict["warnings"])
    if row_dict.get("lstm_prediction"):
        row_dict["lstm_prediction"] = json.loads(row_dict["lstm_prediction"])
    if row_dict.get("statistical_assessment"):
        row_dict["statistical_assessment"] = json.loads(row_dict["statistical_assessment"])
    if row_dict.get("model_assessment"):
        row_dict["model_assessment"] = json.loads(row_dict["model_assessment"])
    if row_dict.get("ensemble_decision"):
        row_dict["ensemble_decision"] = json.loads(row_dict["ensemble_decision"])
    if row_dict.get("raw_trajectory"):
        row_dict["raw_trajectory"] = json.loads(row_dict["raw_trajectory"])

    pdf_bytes = generate_session_pdf(row_dict)
    filename = f"RehabVision_Report_{sess_id}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
