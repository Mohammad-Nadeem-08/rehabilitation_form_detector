import {
  ReferenceData, SessionData, CompareResult,
  ModelPredictResponse, ProgressSummary, AnglePoint, AngleStats, JointDeviations, LSTMPrediction
} from '../types';

const API_BASE = '/api';

export async function checkHealth(): Promise<{
  status: string;
  app: string;
  version: string;
  model_loaded: boolean;
  model_error?: string | null;
  supported_exercise: string;
  notice: string;
}> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Backend health check failed');
  return res.json();
}

export async function fetchExercises(): Promise<any[]> {
  const res = await fetch(`${API_BASE}/exercises`);
  if (!res.ok) throw new Error('Failed to fetch exercises');
  return res.json();
}

export async function fetchReferences(): Promise<ReferenceData[]> {
  const res = await fetch(`${API_BASE}/references`);
  if (!res.ok) throw new Error('Failed to fetch references');
  return res.json();
}

export async function fetchReference(refId: string): Promise<ReferenceData> {
  const res = await fetch(`${API_BASE}/references/${refId}`);
  if (!res.ok) throw new Error(`Failed to fetch reference ${refId}`);
  return res.json();
}

export async function saveReference(payload: {
  exercise_id?: string;
  exercise_name?: string;
  notes?: string;
  duration_seconds: number;
  frame_count: number;
  joint_angles: AnglePoint[];
  landmarks_summary?: Record<string, any>;
  reference_type?: 'standard' | 'patient_specific';
  version?: string;
  provenance?: string;
  patient_id?: string | null;
  is_active?: number;
}): Promise<ReferenceData> {
  const res = await fetch(`${API_BASE}/references`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to save reference' }));
    throw new Error(err.detail || 'Failed to save reference');
  }
  return res.json();
}

export async function fetchSessions(patientId?: string): Promise<SessionData[]> {
  const url = patientId ? `${API_BASE}/sessions?patient_id=${encodeURIComponent(patientId)}` : `${API_BASE}/sessions`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch sessions');
  return res.json();
}

export async function fetchSession(sessionId: string): Promise<SessionData> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}`);
  if (!res.ok) throw new Error(`Failed to fetch session ${sessionId}`);
  return res.json();
}

export async function saveSession(payload: {
  exercise_id?: string;
  exercise_name?: string;
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
  statistical_assessment?: any;
  model_assessment?: any;
  ensemble_decision?: any;
  warnings: string[];
  notes?: string;
  raw_trajectory?: AnglePoint[];
  is_demo?: number;
}): Promise<SessionData> {
  const res = await fetch(`${API_BASE}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to record session' }));
    throw new Error(err.detail || 'Failed to record session');
  }
  return res.json();
}

export async function compareMovement(payload: {
  reference_id?: string;
  reference_angles?: AnglePoint[];
  home_angles: AnglePoint[];
}): Promise<CompareResult> {
  const res = await fetch(`${API_BASE}/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Movement comparison failed' }));
    throw new Error(err.detail || 'Movement comparison failed');
  }
  return res.json();
}

export async function evaluateHybridAssessment(payload: {
  exercise_id?: string;
  reference_id?: string;
  reference_angles?: any[];
  trajectory: any[];
  mocap_sequence?: number[][];
  input_source?: string;
  duration_sec?: number;
  repetition_index?: number;
  clinician_criteria?: Record<string, any>;
}): Promise<{
  status: string;
  statistical_assessment: any;
  model_assessment: any;
  ensemble_decision: any;
  educational_disclaimer: string;
}> {
  const res = await fetch(`${API_BASE}/assessment/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Hybrid assessment evaluation failed' }));
    throw new Error(err.detail || 'Hybrid assessment evaluation failed');
  }
  return res.json();
}

export async function fetchCohortEvaluation(split: 'train' | 'val' | 'test' = 'test'): Promise<any> {
  const res = await fetch(`${API_BASE}/assessment/evaluate-cohort?split=${split}`);
  if (!res.ok) throw new Error('Failed to fetch cohort evaluation metrics');
  return res.json();
}

export async function predictCompatibleLSTM(sequence: number[][]): Promise<ModelPredictResponse> {
  const res = await fetch(`${API_BASE}/model/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sequence }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'LSTM prediction request failed' }));
    throw new Error(err.detail || 'LSTM prediction request failed');
  }
  return res.json();
}

export async function fetchSampleSequence(form: 'correct' | 'incorrect' = 'correct'): Promise<{
  form: string;
  shape: number[];
  sequence: number[][];
  note: string;
}> {
  const res = await fetch(`${API_BASE}/model/sample-sequence?form=${form}`);
  if (!res.ok) throw new Error('Failed to fetch sample sequence');
  return res.json();
}

export async function fetchProgress(): Promise<ProgressSummary> {
  const res = await fetch(`${API_BASE}/progress`);
  if (!res.ok) throw new Error('Failed to fetch progress summary');
  return res.json();
}

export function getReportDownloadUrl(sessionId: string): string {
  return `${API_BASE}/reports/${sessionId}`;
}

export async function fetchRegisteredModels(exerciseId?: string): Promise<any[]> {
  const url = exerciseId ? `${API_BASE}/models?exercise_id=${encodeURIComponent(exerciseId)}` : `${API_BASE}/models`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch registered models');
  return res.json();
}

