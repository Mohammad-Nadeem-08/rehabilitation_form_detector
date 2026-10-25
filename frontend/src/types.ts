export interface AnglePoint {
  timestamp: number;
  elevation: number;
  elbow: number;
  confidence?: number;
}

export interface AngleStats {
  peak_elevation: number;
  mean_elevation: number;
  min_elevation: number;
  mean_elbow: number;
  rom: number;
}

export interface JointDeviations {
  mean_elevation_deviation: number;
  max_elevation_deviation: number;
  mean_elbow_deviation: number;
  rmse: number;
}

export interface LSTMPrediction {
  source: string;
  predicted_class: number;
  class_label: string;
  probabilities: {
    incorrect: number;
    correct: number;
  };
  accuracy_context: string;
}

export interface ReferenceData {
  id: string;
  exercise_id: string;
  exercise_name: string;
  notes?: string;
  duration_seconds: number;
  frame_count: number;
  joint_angles: AnglePoint[];
  landmarks_summary?: Record<string, any>;
  created_at: string;
  is_demo?: number;
  reference_type?: 'standard' | 'patient_specific';
  version?: string;
  provenance?: string;
  patient_id?: string | null;
  is_active?: number;
}

export interface SessionData {
  id: string;
  timestamp: string;
  exercise_id: string;
  exercise_name: string;
  reference_id?: string;
  reference_type?: 'standard' | 'patient_specific';
  reference_version?: string;
  reference_provenance?: string;
  patient_id?: string | null;
  repetitions_completed: number;
  duration_seconds: number;
  angle_stats: AngleStats;
  comparison_score?: number;
  joint_deviations: JointDeviations;
  model_status: string;
  lstm_prediction?: LSTMPrediction | null;
  statistical_assessment?: StatisticalAssessmentResult | null;
  model_assessment?: ModelAssessmentResult | null;
  ensemble_decision?: EnsembleDecisionResult | null;
  warnings: string[];
  notes?: string;
  raw_trajectory?: AnglePoint[];
  is_demo?: number;
}

export interface ExerciseStrategyConfig {
  id: string;
  name: string;
  target_joint: string;
  description?: string;
  target_rom_min: number;
  target_rom_max: number;
  primary_metric_name: string;
  secondary_metric_name: string;
  fsm_thresholds: {
    start_threshold: number;
    raised_threshold: number;
    return_threshold: number;
    hysteresis: number;
    min_duration_sec: number;
    confirmation_frames: number;
  };
  required_landmarks: Array<{
    index: number;
    name: string;
    role: string;
  }>;
}

export interface CompareResult {
  status: string;
  comparison_score: number;
  mean_elevation_deviation: number;
  max_elevation_deviation: number;
  mean_elbow_deviation: number;
  rmse: number;
  aligned_reference_elevation: number[];
  aligned_home_elevation: number[];
  aligned_reference_elbow: number[];
  aligned_home_elbow: number[];
  warnings: string[];
}

export interface ModelPredictResponse {
  status: string;
  predicted_class: number;
  class_label: string;
  probabilities: {
    incorrect: number;
    correct: number;
  };
  input_shape: number[];
  warning: string;
  educational_note: string;
}

export interface ProgressSummary {
  total_references: number;
  total_sessions: number;
  total_repetitions: number;
  average_comparison_score: number;
  recent_sessions: SessionData[];
}

export type RepState = 'RESTING' | 'RAISING' | 'RAISED' | 'LOWERING';

export interface LandmarkTrackingDetails {
  shoulderScore: number;
  elbowScore: number;
  wristScore: number;
  rHipScore: number;
  lHipScore: number;
  hipScore: number;
  threshold: number;
  hasRequired: boolean;
  wristValid: boolean;
  invalidReason?: string;
}

export interface FSMDiagnostics {
  measuredAngle: number;
  trackingValid: boolean;
  thresholds: {
    start: number;
    raised: number;
    returnThresh: number;
    hysteresis: number;
  };
  previousState: RepState;
  currentState: RepState;
  transitionReason: string;
  trackingDetails?: LandmarkTrackingDetails;
  repCount: number;
  lastUpdated: number;
}

export interface LiveTelemetry {
  elevation: number;
  elbow: number;
  trunkLean: number;
  confidence: number;
  state: RepState;
  repCount: number;
  fps: number;
}

export interface StatisticalAssessmentResult {
  score: number;
  classification_estimate: string;
  quality_status: string;
  primary_metric_name: string;
  secondary_metric_name: string;
  raw_measurements: Record<string, number>;
  sub_scores: Record<string, number>;
  metric_directions: Record<string, string>;
  warnings: string[];
  exercise_id: string;
  repetition_index?: number;
}

export interface ModelAssessmentResult {
  model_id: string;
  model_version: string;
  architecture: string;
  input_schema: string;
  input_quality_status: string;
  predicted_class?: number | null;
  class_label?: string | null;
  probabilities?: {
    incorrect: number;
    correct: number;
  } | null;
  probability_calibration_status: string;
  inference_latency_ms: number;
  rejection_reason?: string | null;
  warnings?: string[];
}

export interface EnsembleDecisionResult {
  decision: 'CORRECT_ESTIMATE' | 'INCORRECT_ESTIMATE' | 'NEEDS_REVIEW';
  combined_score?: number | null;
  confidence: number;
  branch_agreement?: boolean | null;
  explanation: string;
  reasons: string[];
  warnings: string[];
  statistical_branch?: StatisticalAssessmentResult | null;
  model_branch?: ModelAssessmentResult | null;
  ensemble_policy: string;
  weights_applied: {
    statistical: number;
    model: number;
  };
  exercise_id: string;
  reference_id?: string | null;
  repetition_index: number;
}

export interface HybridEvaluationResponse {
  status: string;
  statistical_assessment: StatisticalAssessmentResult;
  model_assessment: ModelAssessmentResult;
  ensemble_decision: EnsembleDecisionResult;
  educational_disclaimer: string;
}

export interface CohortEvaluationMetrics {
  system_name: string;
  total_samples: number;
  decided_samples: number;
  abstained_samples: number;
  decision_coverage_pct: number;
  abstention_rate_pct: number;
  accuracy: number;
  precision_correct: number;
  recall_correct: number;
  f1_correct: number;
  precision_incorrect: number;
  recall_incorrect: number;
  f1_incorrect: number;
  macro_f1: number;
  confusion_matrix: { tp: number; tn: number; fp: number; fn: number };
  brier_score: number;
  roc_auc?: number | null;
  average_latency_ms: number;
}

export interface CohortEvaluationResponse {
  split: string;
  participants: number[];
  sample_count: number;
  statistical_only: CohortEvaluationMetrics;
  lstm_only: CohortEvaluationMetrics;
  ensemble: CohortEvaluationMetrics;
  evaluation_notes: string[];
}

export interface RegisteredModelMetadata {
  model_id: string;
  model_name: string;
  model_version: string;
  supported_exercise_ids: string[];
  framework: string;
  input_schema: string;
  input_dimensions: number[];
  coordinate_system: string;
  feature_order: string[];
  preprocessing_requirements: string;
  class_mapping: Record<string, string>;
  output_semantics: string;
  is_calibrated: boolean;
  device_requirement: string;
  validation_status: string;
  author_provenance: string;
  notes?: string | null;
  is_default_for?: string[];
  health?: Record<string, any>;
}



