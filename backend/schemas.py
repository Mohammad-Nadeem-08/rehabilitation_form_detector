from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ExerciseSchema(BaseModel):
    id: str
    name: str
    target_joint: str
    description: Optional[str] = None
    target_rom_min: float = 80.0
    target_rom_max: float = 105.0
    created_at: Optional[str] = None
    primary_metric_name: Optional[str] = "Arm Elevation"
    secondary_metric_name: Optional[str] = "Elbow Extension"
    fsm_thresholds: Optional[Dict[str, float]] = None
    required_landmarks: Optional[List[Dict[str, Any]]] = None

class AnglePoint(BaseModel):
    timestamp: float
    elevation: float
    elbow: float
    confidence: Optional[float] = 1.0

class ReferenceCreate(BaseModel):
    exercise_id: str = "standing-right-arm-abduction"
    exercise_name: str = "Standing Right-Arm Shoulder Abduction"
    notes: Optional[str] = ""
    duration_seconds: float
    frame_count: int
    joint_angles: List[AnglePoint]
    landmarks_summary: Optional[Dict[str, Any]] = None
    reference_type: str = "standard"  # "standard" | "patient_specific"
    version: str = "1.0"
    provenance: Optional[str] = "Clinical Standard Protocol"
    patient_id: Optional[str] = None
    is_active: int = 1

class ReferenceResponse(BaseModel):
    id: str
    exercise_id: str
    exercise_name: str
    notes: Optional[str] = None
    duration_seconds: float
    frame_count: int
    joint_angles: List[AnglePoint]
    landmarks_summary: Optional[Dict[str, Any]] = None
    created_at: str
    is_demo: int = 0
    reference_type: str = "standard"
    version: str = "1.0"
    provenance: Optional[str] = "Clinical Standard Protocol"
    patient_id: Optional[str] = None
    is_active: int = 1

class AngleStats(BaseModel):
    peak_elevation: float
    mean_elevation: float
    min_elevation: float
    mean_elbow: float
    rom: float

class JointDeviations(BaseModel):
    mean_elevation_deviation: float
    max_elevation_deviation: float
    mean_elbow_deviation: float
    rmse: float

class LSTMPrediction(BaseModel):
    source: str
    predicted_class: int
    class_label: str
    probabilities: Dict[str, float]
    accuracy_context: str

class SessionCreate(BaseModel):
    exercise_id: str = "standing-right-arm-abduction"
    exercise_name: str = "Standing Right-Arm Shoulder Abduction"
    reference_id: Optional[str] = None
    reference_type: Optional[str] = "standard"
    reference_version: Optional[str] = "1.0"
    reference_provenance: Optional[str] = None
    patient_id: Optional[str] = None
    repetitions_completed: int
    duration_seconds: float
    angle_stats: AngleStats
    comparison_score: Optional[float] = None
    joint_deviations: JointDeviations
    model_status: str = "input_requires_validation"
    lstm_prediction: Optional[LSTMPrediction] = None
    statistical_assessment: Optional[Dict[str, Any]] = None
    model_assessment: Optional[Dict[str, Any]] = None
    ensemble_decision: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    notes: Optional[str] = ""
    raw_trajectory: Optional[List[AnglePoint]] = None
    is_demo: int = 0

class SessionResponse(BaseModel):
    id: str
    timestamp: str
    exercise_id: str
    exercise_name: str
    reference_id: Optional[str] = None
    reference_type: Optional[str] = "standard"
    reference_version: Optional[str] = "1.0"
    reference_provenance: Optional[str] = None
    patient_id: Optional[str] = None
    repetitions_completed: int
    duration_seconds: float
    angle_stats: AngleStats
    comparison_score: Optional[float] = None
    joint_deviations: JointDeviations
    model_status: str
    lstm_prediction: Optional[LSTMPrediction] = None
    statistical_assessment: Optional[Dict[str, Any]] = None
    model_assessment: Optional[Dict[str, Any]] = None
    ensemble_decision: Optional[Dict[str, Any]] = None
    warnings: List[str]
    notes: Optional[str] = None
    raw_trajectory: Optional[List[AnglePoint]] = None
    is_demo: int = 0

class CompareRequest(BaseModel):
    reference_id: Optional[str] = None
    reference_angles: Optional[List[AnglePoint]] = None
    home_angles: List[AnglePoint]

class CompareResponse(BaseModel):
    status: str
    comparison_score: float
    mean_elevation_deviation: float
    max_elevation_deviation: float
    mean_elbow_deviation: float
    rmse: float
    aligned_reference_elevation: List[float]
    aligned_home_elevation: List[float]
    aligned_reference_elbow: List[float]
    aligned_home_elbow: List[float]
    warnings: List[str]

class ModelPredictRequest(BaseModel):
    sequence: List[List[float]] # shape (64, 80)
    source_description: Optional[str] = "External compatible motion-capture sequence"

class ModelPredictResponse(BaseModel):
    status: str
    predicted_class: int
    class_label: str
    probabilities: Dict[str, float]
    input_shape: List[int]
    warning: str
    educational_note: str

class StatisticalAssessmentSchema(BaseModel):
    score: float
    classification_estimate: str
    quality_status: str
    primary_metric_name: str
    secondary_metric_name: str
    raw_measurements: Dict[str, float]
    sub_scores: Dict[str, float]
    metric_directions: Dict[str, str]
    warnings: List[str]
    exercise_id: str
    repetition_index: int = 1

class ModelAssessmentSchema(BaseModel):
    model_id: str
    model_version: str
    architecture: str
    input_schema: str
    input_quality_status: str
    predicted_class: Optional[int] = None
    class_label: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    probability_calibration_status: str = "uncalibrated_softmax"
    inference_latency_ms: float = 0.0
    rejection_reason: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

class EnsembleDecisionSchema(BaseModel):
    decision: str
    combined_score: Optional[float] = None
    confidence: float
    branch_agreement: Optional[bool] = None
    explanation: str
    reasons: List[str]
    warnings: List[str]
    statistical_branch: Optional[Dict[str, Any]] = None
    model_branch: Optional[Dict[str, Any]] = None
    ensemble_policy: str = "experimental_weighted_agreement_v1"
    weights_applied: Dict[str, float] = Field(default_factory=lambda: {"statistical": 0.50, "model": 0.50})
    exercise_id: str = "standing-right-arm-abduction"
    reference_id: Optional[str] = None
    repetition_index: int = 1

class HybridEvaluationRequest(BaseModel):
    exercise_id: str = "standing-right-arm-abduction"
    reference_id: Optional[str] = None
    reference_angles: Optional[List[Dict[str, Any]]] = None
    trajectory: List[Dict[str, Any]]
    mocap_sequence: Optional[List[List[float]]] = None
    model_id: Optional[str] = None
    input_source: str = "webcam_mediapipe"
    duration_sec: Optional[float] = None
    repetition_index: int = 1
    clinician_criteria: Optional[Dict[str, Any]] = None

class ModelMetadataSchema(BaseModel):
    model_id: str
    model_name: str
    model_version: str
    supported_exercise_ids: List[str]
    framework: str
    input_schema: str
    input_dimensions: List[int]
    coordinate_system: str
    feature_order: List[str]
    preprocessing_requirements: str
    class_mapping: Dict[int, str]
    output_semantics: str
    is_calibrated: bool
    device_requirement: str
    validation_status: str
    author_provenance: str
    notes: Optional[str] = None
    is_default_for: Optional[List[str]] = None
    health: Optional[Dict[str, Any]] = None

class HybridEvaluationResponse(BaseModel):
    status: str
    statistical_assessment: Dict[str, Any]
    model_assessment: Dict[str, Any]
    ensemble_decision: Dict[str, Any]
    educational_disclaimer: str

class CohortEvaluationResponse(BaseModel):
    split: str
    participants: List[int]
    sample_count: int
    statistical_only: Dict[str, Any]
    lstm_only: Dict[str, Any]
    ensemble: Dict[str, Any]
    evaluation_notes: List[str]

class ProgressResponse(BaseModel):
    total_references: int
    total_sessions: int
    total_repetitions: int
    average_comparison_score: float
    recent_sessions: List[SessionResponse]

