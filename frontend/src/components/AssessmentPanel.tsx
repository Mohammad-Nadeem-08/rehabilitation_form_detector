import React from 'react';
import { StatisticalAssessmentResult, ModelAssessmentResult, EnsembleDecisionResult } from '../types';

interface AssessmentPanelProps {
  statistical?: StatisticalAssessmentResult | null;
  model?: ModelAssessmentResult | null;
  ensemble?: EnsembleDecisionResult | null;
  className?: string;
}

export const AssessmentPanel: React.FC<AssessmentPanelProps> = ({
  statistical,
  model,
  ensemble,
  className = ''
}) => {
  if (!statistical && !model && !ensemble) {
    return (
      <div className={`p-4 bg-slate-900/60 border border-slate-800 rounded-xl text-slate-400 text-sm text-center ${className}`}>
        Complete at least one full repetition to view the Hybrid Deep Learning + Statistical Ensemble Assessment.
      </div>
    );
  }

  // Helper for ensemble badge color & label
  const getDecisionBadge = (decision?: string) => {
    switch (decision) {
      case 'CORRECT_ESTIMATE':
        return {
          bg: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
          label: 'CORRECT FORM ESTIMATE',
          icon: '✓',
          description: 'Both branches confirm movement within clinical reference tolerances.'
        };
      case 'INCORRECT_ESTIMATE':
        return {
          bg: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
          label: 'FAULT PATTERN DETECTED',
          icon: '✕',
          description: 'Movement deviations or compensations identified.'
        };
      case 'NEEDS_REVIEW':
      default:
        return {
          bg: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
          label: 'NEEDS CLINICAL REVIEW',
          icon: '⚠',
          description: 'Single branch, input requires validation, or branches conflict.'
        };
    }
  };

  const decisionBadge = getDecisionBadge(ensemble?.decision);

  return (
    <div className={`flex flex-col gap-4 bg-slate-900/80 border border-slate-800/80 rounded-2xl p-5 shadow-xl backdrop-blur-md ${className}`}>
      {/* Top Header & Ensemble Summary Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-slate-800 gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-cyan-400 bg-cyan-950/60 px-2.5 py-0.5 rounded-full border border-cyan-800/50">
              Hybrid Tri-Branch Pipeline
            </span>
            <span className="text-xs text-slate-400">
              Rep #{ensemble?.repetition_index || statistical?.repetition_index || 1}
            </span>
          </div>
          <h3 className="text-lg font-bold text-white mt-1">
            Rehabilitation Movement Ensemble Assessment
          </h3>
        </div>

        {/* Big Decision Badge */}
        <div className={`flex items-center gap-2 px-4 py-2 rounded-xl border font-bold text-sm shadow-inner ${decisionBadge.bg}`}>
          <span className="text-base">{decisionBadge.icon}</span>
          <span>{decisionBadge.label}</span>
          {ensemble?.combined_score != null && (
            <span className="ml-1 text-xs opacity-80 font-mono">
              ({ensemble.combined_score.toFixed(1)}/100)
            </span>
          )}
        </div>
      </div>

      {/* Rationale & Educational Warning */}
      <div className="bg-slate-950/60 border border-slate-800/60 rounded-xl p-3 text-xs flex flex-col gap-1.5">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-slate-300">Ensemble Policy Rationale:</span>
          <span className="font-mono text-slate-400">
            Agreement: {ensemble?.branch_agreement === true ? 'AGREE' : (ensemble?.branch_agreement === false ? 'CONFLICT' : 'SINGLE_BRANCH')}
          </span>
        </div>
        <p className="text-slate-300 leading-relaxed">
          {ensemble?.explanation || 'Awaiting multi-branch validation.'}
        </p>
        {ensemble?.reasons && ensemble.reasons.length > 0 && (
          <ul className="list-disc list-inside text-slate-400 space-y-0.5 pt-1">
            {ensemble.reasons.map((r, idx) => (
              <li key={idx}>{r}</li>
            ))}
          </ul>
        )}
      </div>

      {/* 3-Card Architecture Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Branch 1: Statistical / Biomechanical */}
        <div className="bg-slate-950/40 border border-slate-800/70 rounded-xl p-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-sky-400 uppercase tracking-wider">
                1. Statistical Branch
              </span>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                statistical?.quality_status === 'VALID' ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30' : 'bg-amber-500/20 text-amber-300'
              }`}>
                {statistical?.quality_status || 'AWAITING'}
              </span>
            </div>

            <div className="flex items-baseline gap-2 mb-3">
              <span className="text-2xl font-extrabold text-white">
                {statistical ? `${statistical.score.toFixed(1)}` : '--'}
              </span>
              <span className="text-xs text-slate-400 font-medium">/ 100 Biomechanical Score</span>
            </div>

            {/* Sub-scores */}
            {statistical?.sub_scores && (
              <div className="space-y-1.5 text-xs">
                <div className="flex justify-between text-slate-300">
                  <span>Range of Motion (ROM):</span>
                  <span className="font-mono text-sky-300 font-bold">
                    {(statistical.sub_scores.rom * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span>DTW Trajectory Similarity:</span>
                  <span className="font-mono text-sky-300 font-bold">
                    {(statistical.sub_scores.dtw_similarity * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span>Posture Alignment:</span>
                  <span className="font-mono text-sky-300 font-bold">
                    {(statistical.sub_scores.posture_compliance * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span>Cadence & Tempo:</span>
                  <span className="font-mono text-sky-300 font-bold">
                    {(statistical.sub_scores.cadence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            )}

            {/* Key Measurements */}
            {statistical?.raw_measurements && (
              <div className="mt-3 pt-2.5 border-t border-slate-800/80 text-[11px] text-slate-400 space-y-1">
                <div>Peak Elevation: <b className="text-slate-200">{statistical.raw_measurements.peak_primary_metric?.toFixed(1)}°</b></div>
                <div>Active ROM: <b className="text-slate-200">{statistical.raw_measurements.active_rom?.toFixed(1)}°</b></div>
                <div>Mean Elbow: <b className="text-slate-200">{statistical.raw_measurements.mean_secondary_metric?.toFixed(1)}°</b></div>
                <div>DTW RMSE: <b className="text-slate-200">{statistical.raw_measurements.rmse?.toFixed(1)}°</b></div>
              </div>
            )}
          </div>
          <div className="mt-3 pt-2 text-[10px] text-slate-500 italic">
            Deterministic kinematic reference & DTW curve comparison.
          </div>
        </div>

        {/* Branch 2: Deep Learning Model Branch */}
        <div className="bg-slate-950/40 border border-slate-800/70 rounded-xl p-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-indigo-400 uppercase tracking-wider">
                2. Deep Learning Branch
              </span>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                model?.input_quality_status === 'VALID' ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30' : 'bg-slate-800 text-slate-400'
              }`}>
                {model?.input_quality_status || 'INPUT_REQUIRES_VALIDATION'}
              </span>
            </div>

            {model?.input_quality_status === 'VALID' && model.class_label ? (
              <>
                <div className="flex items-baseline gap-2 mb-3">
                  <span className="text-2xl font-extrabold uppercase text-white">
                    {model.class_label}
                  </span>
                  <span className="text-xs text-slate-400 font-mono">
                    (Class {model.predicted_class})
                  </span>
                </div>

                {model.probabilities && (
                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between text-slate-300">
                      <span>Correct Form Probability:</span>
                      <span className="font-mono text-emerald-400 font-bold">
                        {(model.probabilities.correct * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-slate-300">
                      <span>Fault Form Probability:</span>
                      <span className="font-mono text-rose-400 font-bold">
                        {(model.probabilities.incorrect * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-slate-300">
                      <span>Inference Latency:</span>
                      <span className="font-mono text-indigo-300">
                        {model.inference_latency_ms.toFixed(1)} ms
                      </span>
                    </div>
                    <div className="flex justify-between text-slate-400 text-[11px]">
                      <span>Calibration:</span>
                      <span>{model.probability_calibration_status}</span>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="space-y-2 py-2">
                <div className="text-xs text-amber-400/90 font-medium">
                  {model?.rejection_reason || 'Monocular live webcam input requires calibrated 3D representation.'}
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Deep-learning model integration enforces domain-shift safeguards. Uncalibrated monocular webcam inputs safely abstain from ML inference until a calibrated representation is supplied.
                </p>
                <div className="text-[10px] text-slate-500 font-mono mt-1">
                  Schema: {model?.input_schema || 'mocap_3d_64x80'} | Architecture: {model?.architecture || 'PyTorch LSTM'}
                </div>
              </div>
            )}
          </div>

          <div className="mt-3 pt-2 text-[10px] text-slate-500 italic">
            Registered Model: {model?.model_id || 'rehab-lstm-rehab24-6-v1'} {model?.model_version ? `(v${model.model_version})` : ''}
          </div>
        </div>

        {/* Branch 3: Ensemble Decision Engine */}
        <div className="bg-slate-950/40 border border-slate-800/70 rounded-xl p-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">
                3. Ensemble Engine
              </span>
              <span className="text-[10px] font-mono text-slate-400">
                Weights: 50% / 50%
              </span>
            </div>

            <div className="flex items-baseline gap-2 mb-3">
              <span className="text-2xl font-extrabold text-white">
                {ensemble?.combined_score != null ? `${ensemble.combined_score.toFixed(1)}` : 'Abstained'}
              </span>
              <span className="text-xs text-slate-400 font-medium">
                {ensemble?.combined_score != null ? '/ 100 Combined' : '(NEEDS_REVIEW)'}
              </span>
            </div>

            <div className="space-y-1.5 text-xs">
              <div className="flex justify-between text-slate-300">
                <span>Decision State:</span>
                <span className="font-bold text-slate-200">
                  {ensemble?.decision || 'NEEDS_REVIEW'}
                </span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Confidence:</span>
                <span className="font-mono text-emerald-300">
                  {ensemble ? `${(ensemble.confidence * 100).toFixed(0)}%` : '--'}
                </span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Agreement Status:</span>
                <span className="font-mono text-slate-300">
                  {ensemble?.branch_agreement === true ? 'Consensus' : (ensemble?.branch_agreement === false ? 'Conflict' : 'Single-Branch')}
                </span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Policy:</span>
                <span className="font-mono text-slate-400 text-[10px]">
                  {ensemble?.ensemble_policy || 'experimental_v1'}
                </span>
              </div>
            </div>
          </div>

          <div className="mt-3 pt-2 text-[10px] text-amber-500/80">
            Educational Prototype: Not validated for clinical diagnosis.
          </div>
        </div>
      </div>
    </div>
  );
};
