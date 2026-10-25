import React from 'react';
import {
  X, Download, FileText, CheckCircle, AlertTriangle, Cpu,
  Activity, ShieldAlert
} from 'lucide-react';
import { SessionData } from '../types';
import { getReportDownloadUrl } from '../services/api';
import { KinematicChart } from './KinematicChart';
import { AssessmentPanel } from './AssessmentPanel';

interface ReportModalProps {
  session: SessionData | null;
  onClose: () => void;
}

export const ReportModal: React.FC<ReportModalProps> = ({ session, onClose }) => {
  if (!session) return null;

  const downloadUrl = getReportDownloadUrl(session.id);

  // Prepare chart data if raw trajectory is available
  const chartData = session.raw_trajectory?.map((pt, idx) => ({
    time: pt.timestamp,
    elevation: pt.elevation,
    elbow: pt.elbow,
  })) || [];

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      backgroundColor: 'rgba(7, 11, 25, 0.85)',
      backdropFilter: 'blur(8px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 100,
      padding: '20px',
    }}>
      <div className="glass-panel" style={{
        maxWidth: '840px',
        width: '100%',
        maxHeight: '90vh',
        overflowY: 'auto',
        padding: '28px',
        position: 'relative',
        background: '#0F172A',
        border: '1px solid var(--border-active)',
      }}>
        {/* Close Button */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            background: 'transparent',
            border: 'none',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
          }}
        >
          <X size={22} />
        </button>

        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
          <div style={{
            padding: '10px',
            borderRadius: '10px',
            background: 'rgba(0, 229, 255, 0.1)',
            color: 'var(--teal-primary)',
          }}>
            <FileText size={24} />
          </div>
          <div>
            <h2 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#F8FAFC' }}>
              Assessment Report — {session.exercise_name}
            </h2>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Session ID: <code style={{ color: '#00E5FF' }}>{session.id}</code> • {new Date(session.timestamp).toLocaleString()}
              {session.is_demo ? <span className="badge badge-amber" style={{ marginLeft: '8px' }}>Demo Record</span> : null}
            </div>
          </div>
        </div>

        {/* Educational Prototype Notice */}
        <div style={{
          background: 'rgba(239, 68, 68, 0.08)',
          border: '1px solid rgba(239, 68, 68, 0.25)',
          borderRadius: '8px',
          padding: '10px 14px',
          fontSize: '0.8rem',
          color: '#FCA5A5',
          marginBottom: '20px',
          display: 'flex',
          gap: '8px',
        }}>
          <AlertTriangle size={16} color="#EF4444" style={{ flexShrink: 0, marginTop: '2px' }} />
          <span>
            <strong>Educational Prototype:</strong> Automated metrics are for engineering validation only. 
            Held-out participant accuracy (82.5%) is preliminary. Not validated for clinical diagnosis.
          </span>
        </div>

        {/* Key Metrics Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', marginBottom: '20px' }}>
          <div className="hud-card">
            <div className="hud-value" style={{ color: '#00E5FF' }}>
              {session.repetitions_completed}
            </div>
            <div className="hud-label">Repetitions Completed</div>
          </div>

          <div className="hud-card">
            <div className="hud-value" style={{ color: (session.comparison_score ?? 0) >= 80 ? '#10B981' : '#F59E0B' }}>
              {session.comparison_score ? `${session.comparison_score.toFixed(1)}%` : 'N/A'}
            </div>
            <div className="hud-label">Reference Similarity Index</div>
          </div>

          <div className="hud-card">
            <div className="hud-value" style={{ color: '#F8FAFC' }}>
              {session.angle_stats?.peak_elevation ? `${session.angle_stats.peak_elevation.toFixed(1)}°` : 'N/A'}
            </div>
            <div className="hud-label">Peak Elevation (Target: 90°)</div>
          </div>

          <div className="hud-card">
            <div className="hud-value" style={{ color: session.angle_stats?.mean_elbow >= 160 ? '#10B981' : '#F59E0B' }}>
              {session.angle_stats?.mean_elbow ? `${session.angle_stats.mean_elbow.toFixed(1)}°` : 'N/A'}
            </div>
            <div className="hud-label">Mean Elbow Extension</div>
          </div>
        </div>

        {/* Reference Standard & Provenance Info */}
        <div className="glass-panel" style={{ padding: '14px 16px', marginBottom: '16px', background: 'rgba(15, 23, 42, 0.6)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Evaluated Against Reference: </span>
              <strong style={{ color: '#00E5FF', fontSize: '0.85rem' }}>{session.reference_id || 'ref-gold-standard-demo'}</strong>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginLeft: '6px' }}>v{session.reference_version || '1.0'}</span>
              {session.reference_type === 'patient_specific' ? (
                <span className="badge" style={{ background: 'rgba(168, 85, 247, 0.2)', color: '#C084FC', border: '1px solid #A855F7', fontSize: '0.65rem', marginLeft: '8px' }}>
                  Patient-Specific (Clinician Approved)
                </span>
              ) : (
                <span className="badge badge-teal" style={{ fontSize: '0.65rem', marginLeft: '8px' }}>
                  Standard Protocol Baseline
                </span>
              )}
              {session.patient_id && (
                <span className="badge badge-blue" style={{ fontSize: '0.65rem', marginLeft: '6px' }}>
                  Patient: {session.patient_id}
                </span>
              )}
            </div>
            {session.reference_provenance && (
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                Provenance: {session.reference_provenance}
              </div>
            )}
          </div>
        </div>

        {/* Biomechanical Deviation Analysis */}
        <div className="glass-panel" style={{ padding: '16px', marginBottom: '20px' }}>
          <h4 style={{ fontSize: '0.9rem', fontWeight: 700, color: '#F8FAFC', marginBottom: '12px' }}>
            Biomechanical Kinematics & DTW Deviations
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '0.85rem' }}>
            <div>
              <span style={{ color: 'var(--text-secondary)' }}>Mean Elevation Deviation: </span>
              <strong>{session.joint_deviations?.mean_elevation_deviation.toFixed(1)}°</strong>
            </div>
            <div>
              <span style={{ color: 'var(--text-secondary)' }}>Maximum Deviation: </span>
              <strong>{session.joint_deviations?.max_elevation_deviation.toFixed(1)}°</strong>
            </div>
            <div>
              <span style={{ color: 'var(--text-secondary)' }}>Mean Elbow Posture Deviation: </span>
              <strong>{session.joint_deviations?.mean_elbow_deviation.toFixed(1)}°</strong>
            </div>
            <div>
              <span style={{ color: 'var(--text-secondary)' }}>Trajectory RMSE: </span>
              <strong>{session.joint_deviations?.rmse.toFixed(1)}°</strong>
            </div>
          </div>
        </div>

        {/* Trajectory Chart */}
        {chartData.length > 0 && (
          <div style={{ marginBottom: '20px' }}>
            <KinematicChart data={chartData} mode="live" title="Recorded Session Kinematics" />
          </div>
        )}

        {/* Dedicated Assessment Panel */}
        {session.statistical_assessment || session.model_assessment || session.ensemble_decision ? (
          <div style={{ marginBottom: '20px' }}>
            <AssessmentPanel
              statistical={session.statistical_assessment}
              model={session.model_assessment}
              ensemble={session.ensemble_decision}
            />
          </div>
        ) : (
          /* Deep Learning LSTM Classification Status Fallback */
          <div className="glass-panel" style={{ padding: '16px', marginBottom: '20px', borderLeft: '4px solid #00E5FF' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <Cpu size={18} color="#00E5FF" />
              <h4 style={{ fontSize: '0.9rem', fontWeight: 700, color: '#F8FAFC' }}>
                PyTorch Deep Learning Form Model Status
              </h4>
            </div>

            {session.lstm_prediction ? (
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                <div>Form Classification: <strong style={{ color: session.lstm_prediction.predicted_class === 1 ? '#10B981' : '#EF4444' }}>{session.lstm_prediction.class_label.toUpperCase()}</strong></div>
                <div>Probabilities: Correct: {(session.lstm_prediction.probabilities.correct * 100).toFixed(1)}% | Incorrect: {(session.lstm_prediction.probabilities.incorrect * 100).toFixed(1)}%</div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '4px' }}>{session.lstm_prediction.accuracy_context}</div>
              </div>
            ) : (
              <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                <div style={{ color: '#FCD34D', fontWeight: 600, marginBottom: '4px' }}>
                  Live LSTM prediction unavailable: input representation requires validation.
                </div>
                <div>
                  The PyTorch LSTM was trained on 3D motion-capture joint coordinates (80 features from 26 joints), 
                  whereas MediaPipe produces 33 normalized landmarks. Live webcam feedback is governed by dynamic reference DTW alignment 
                  to avoid fabricated AI predictions. Verified 80-feature mocap sequences can be evaluated in the <strong>Model Lab</strong>.
                </div>
              </div>
            )}
          </div>
        )}

        {/* Warnings & Clinical Caveats */}
        {session.warnings && session.warnings.length > 0 && (
          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#F87171', marginBottom: '6px' }}>
              Recorded Biomechanical Warnings
            </h4>
            <ul style={{ paddingLeft: '20px', fontSize: '0.82rem', color: '#CBD5E1' }}>
              {session.warnings.map((w, idx) => (
                <li key={idx} style={{ marginBottom: '4px' }}>{w}</li>
              ))}
            </ul>
          </div>
        )}

        {/* User Notes */}
        {session.notes && (
          <div style={{ marginBottom: '24px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            <strong>Session Notes:</strong> {session.notes}
          </div>
        )}

        {/* Download Action Footer */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '12px' }}>
          <button onClick={onClose} className="btn btn-secondary">
            Close
          </button>
          <a
            href={downloadUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-primary"
            style={{ textDecoration: 'none' }}
          >
            <Download size={16} />
            Download PDF Report
          </a>
        </div>
      </div>
    </div>
  );
};
