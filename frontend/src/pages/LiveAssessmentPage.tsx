import React, { useState } from 'react';
import { Cpu, ArrowRight } from 'lucide-react';
import { CameraView } from '../components/CameraView';
import { KinematicChart } from '../components/KinematicChart';
import { ReportModal } from '../components/ReportModal';
import { AssessmentPanel } from '../components/AssessmentPanel';
import { compareMovement, saveSession, fetchReference, evaluateHybridAssessment } from '../services/api';
import { AnglePoint, CompareResult, SessionData, ReferenceData } from '../types';
import { CompletedRepInfo } from '../services/repetitionTracker';

interface LiveAssessmentPageProps {
  activeReferenceId: string | null;
  setActiveTab: (tab: string) => void;
  onSessionSaved?: (session: SessionData) => void;
}

export const LiveAssessmentPage: React.FC<LiveAssessmentPageProps> = ({
  activeReferenceId,
  setActiveTab,
  onSessionSaved,
}) => {
  const [liveChartPoints, setLiveChartPoints] = useState<any[]>([]);
  const [analysisResult, setAnalysisResult] = useState<CompareResult | null>(null);
  const [savedSession, setSavedSession] = useState<SessionData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [showReportModal, setShowReportModal] = useState<boolean>(false);
  const [activeRefDetails, setActiveRefDetails] = useState<ReferenceData | null>(null);

  React.useEffect(() => {
    const targetId = activeReferenceId || 'ref-gold-standard-demo';
    fetchReference(targetId)
      .then((data) => setActiveRefDetails(data))
      .catch(() => setActiveRefDetails(null));
  }, [activeReferenceId]);

  // Keep sliding window of last 60 points for smooth live visualization
  const handleFrameMetrics = (metrics: any, timestampSec: number) => {
    setLiveChartPoints((prev) => {
      const updated = [
        ...prev,
        {
          time: Math.round(timestampSec * 10) / 10,
          elevation: metrics.elevation,
          elbow: metrics.elbow,
        },
      ];
      return updated.length > 80 ? updated.slice(updated.length - 80) : updated;
    });
  };

  const handleSessionComplete = async (
    trajectory: AnglePoint[],
    repHistory: CompletedRepInfo[],
    durationSec: number
  ) => {
    if (trajectory.length < 15) {
      alert('Recorded session is too short (< 15 frames) for valid movement analysis.');
      return;
    }

    try {
      setIsAnalyzing(true);

      // 1. Send trajectory to backend for DTW movement comparison against active reference
      const comparison = await compareMovement({
        reference_id: activeReferenceId || undefined,
        home_angles: trajectory,
      });

      setAnalysisResult(comparison);

      // 1b. Call genuine tri-branch hybrid assessment pipeline
      let hybridRes: any = null;
      try {
        hybridRes = await evaluateHybridAssessment({
          exercise_id: 'standing-right-arm-abduction',
          reference_id: activeReferenceId || activeRefDetails?.id || undefined,
          trajectory: trajectory,
          duration_sec: durationSec,
          input_source: 'webcam_mediapipe',
        });
      } catch (hErr) {
        console.warn('Hybrid assessment call error:', hErr);
      }

      // 2. Compute aggregate biomechanical stats from genuine exercise repetitions
      // Filter out non-exercise reach/collapse movements (e.g. reaching for mouse with elbow collapsed < 40°)
      const validReps = repHistory.filter((r) => r.minElbow >= 40.0 && r.peakElevation >= 70.0);
      const repsToUse = validReps.length > 0 ? validReps : repHistory;

      let peakElev: number;
      let minElev: number;
      let meanElev: number;
      let meanElbow: number;
      let rom: number;

      if (repsToUse.length > 0) {
        // Compute stats from genuine exercise repetitions
        peakElev = Math.max(...repsToUse.map((r) => r.peakElevation));
        const firstRepStart = repsToUse[0].completedAt - repsToUse[0].duration;
        const lastRepEnd = repsToUse[repsToUse.length - 1].completedAt;

        // Active exercise frames within repetition bounds
        const exerciseFrames = trajectory.filter(
          (p) => p.timestamp >= firstRepStart - 0.2 && p.timestamp <= lastRepEnd + 0.2
        );
        const framesToUse = exerciseFrames.length >= 10 ? exerciseFrames : trajectory;
        const elevations = framesToUse.map((p) => p.elevation);
        const elbows = framesToUse.map((p) => p.elbow);
        minElev = Math.min(...elevations);
        meanElev = elevations.reduce((a, b) => a + b, 0) / elevations.length;
        meanElbow = elbows.reduce((a, b) => a + b, 0) / elbows.length;
        rom = Math.round((peakElev - minElev) * 10) / 10;
      } else {
        const elevations = trajectory.map((p) => p.elevation);
        const elbows = trajectory.map((p) => p.elbow);
        peakElev = Math.max(...elevations);
        minElev = Math.min(...elevations);
        meanElev = elevations.reduce((a, b) => a + b, 0) / elevations.length;
        meanElbow = elbows.reduce((a, b) => a + b, 0) / elbows.length;
        rom = Math.round((peakElev - minElev) * 10) / 10;
      }

      // 3. Save completed session into SQLite database with full provenance
      const newSession = await saveSession({
        exercise_id: 'standing-right-arm-abduction',
        exercise_name: 'Standing Right-Arm Shoulder Abduction',
        reference_id: activeReferenceId || activeRefDetails?.id || 'ref-gold-standard-demo',
        reference_type: activeRefDetails?.reference_type || 'standard',
        reference_version: activeRefDetails?.version || '1.0',
        reference_provenance: activeRefDetails?.provenance || 'Physiotherapy Clinical Protocol v1.0',
        patient_id: activeRefDetails?.patient_id || null,
        repetitions_completed: repsToUse.length,
        duration_seconds: durationSec,
        angle_stats: {
          peak_elevation: Math.round(peakElev * 10) / 10,
          mean_elevation: Math.round(meanElev * 10) / 10,
          min_elevation: Math.round(minElev * 10) / 10,
          mean_elbow: Math.round(meanElbow * 10) / 10,
          rom,
        },
        comparison_score: comparison.comparison_score,
        joint_deviations: {
          mean_elevation_deviation: comparison.mean_elevation_deviation,
          max_elevation_deviation: comparison.max_elevation_deviation,
          mean_elbow_deviation: comparison.mean_elbow_deviation,
          rmse: comparison.rmse,
        },
        model_status: hybridRes?.model_assessment?.input_quality_status || 'input_requires_validation',
        lstm_prediction: null,
        statistical_assessment: hybridRes?.statistical_assessment || null,
        model_assessment: hybridRes?.model_assessment || null,
        ensemble_decision: hybridRes?.ensemble_decision || null,
        warnings: comparison.warnings,
        notes: `Live camera workout. Repetitions: ${repsToUse.length}. Peak elevation: ${peakElev.toFixed(1)}°. Reference: ${activeReferenceId || 'ref-gold-standard-demo'}.`,
        raw_trajectory: trajectory,
        is_demo: 0,
      });

      setSavedSession(newSession);
      if (onSessionSaved) onSessionSaved(newSession);
      setShowReportModal(true);
    } catch (err: any) {
      console.error('Session analysis failed:', err);
      alert(`Session analysis error: ${err.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header Info */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <span className="badge badge-teal">Live Movement Assessment</span>
              <span className="badge badge-green">Exercise: Standing Right-Arm Shoulder Abduction</span>
            </div>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: '#F8FAFC' }}>
              Real-Time Computer Vision & Biomechanics
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', maxWidth: '780px' }}>
              MediaPipe Pose landmark detection tracks right shoulder, elbow, wrist, and torso angles in real-time.
              Repetitions are tracked by a finite state machine and compared via Dynamic Time Warping (DTW) to the selected reference standard.
            </p>
          </div>

          <div style={{ textAlign: 'right', background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Active Comparison Standard:</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', justifyContent: 'flex-end', marginTop: '2px' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#00E5FF', fontSize: '0.92rem' }}>
                {activeReferenceId || activeRefDetails?.id || 'ref-gold-standard-demo'}
              </span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                v{activeRefDetails?.version || '1.0'}
              </span>
              {activeRefDetails?.reference_type === 'patient_specific' ? (
                <span className="badge" style={{ background: 'rgba(168, 85, 247, 0.2)', color: '#C084FC', border: '1px solid #A855F7', fontSize: '0.62rem' }}>
                  Patient-Specific
                </span>
              ) : (
                <span className="badge badge-teal" style={{ fontSize: '0.62rem' }}>
                  Standard
                </span>
              )}
            </div>
            {activeRefDetails?.provenance && (
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '3px', fontStyle: 'italic', maxWidth: '250px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {activeRefDetails.provenance}
              </div>
            )}
            <button
              onClick={() => setActiveTab('reference')}
              className="btn btn-secondary"
              style={{ marginTop: '8px', fontSize: '0.75rem', padding: '4px 10px' }}
            >
              Change Reference Standard
            </button>
          </div>
        </div>
      </div>

      {/* Main Assessment Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(460px, 1fr))', gap: '24px' }}>
        {/* Left: Camera & Skeleton Feed */}
        <div>
          <CameraView
            modeTitle="Standing Right-Arm Shoulder Abduction"
            onFrameMetrics={handleFrameMetrics}
            onSessionComplete={handleSessionComplete}
          />
        </div>

        {/* Right: Live Telemetry & Kinematics */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Live Chart */}
          <KinematicChart
            title="Real-Time Arm Elevation & Elbow Telemetry"
            data={liveChartPoints}
            mode="live"
          />

          {/* Model Status Card */}
          <div className="glass-panel" style={{ padding: '20px', borderLeft: '4px solid #F59E0B' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <Cpu size={18} color="#F59E0B" />
              <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#F8FAFC' }}>
                PyTorch LSTM Deployment Protocol
              </h4>
              <span className="badge badge-amber" style={{ fontSize: '0.65rem' }}>
                Input Validation Guard
              </span>
            </div>

            <p style={{ fontSize: '0.83rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '12px' }}>
              <strong>Notice:</strong> Live LSTM prediction unavailable: input representation requires validation.
              The trained PyTorch model expects (64, 80) sequences from 26 3D motion-capture joints. 
              MediaPipe webcam landmarks are evaluated through kinematic DTW curves to avoid fabricated AI outputs.
            </p>

            <button
              onClick={() => setActiveTab('model-lab')}
              className="btn btn-secondary"
              style={{ fontSize: '0.8rem', padding: '8px 14px' }}
            >
              Test Real PyTorch LSTM in Model Lab <ArrowRight size={14} />
            </button>
          </div>

          {/* Biomechanical Protocol Guidelines */}
          <div className="glass-panel" style={{ padding: '20px' }}>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#F8FAFC', marginBottom: '10px' }}>
              Movement Execution Protocol
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              <div>• <strong>Start Posture:</strong> Upright, right arm at side (~0°–15°).</div>
              <div>• <strong>Ascent:</strong> Smooth elevation in the coronal plane to ~90°.</div>
              <div>• <strong>Elbow Form:</strong> Keep elbow extended (~170°–180°).</div>
              <div>• <strong>Trunk Stability:</strong> Avoid leaning sideways or hunching shoulder.</div>
            </div>
          </div>
        </div>
      </div>

      {/* Dedicated Hybrid Assessment Panel (Statistical + Deep Learning + Ensemble) */}
      {savedSession && (
        <AssessmentPanel
          statistical={savedSession.statistical_assessment}
          model={savedSession.model_assessment}
          ensemble={savedSession.ensemble_decision}
        />
      )}

      {/* Report Modal */}
      {showReportModal && (
        <ReportModal
          session={savedSession}
          onClose={() => setShowReportModal(false)}
        />
      )}
    </div>
  );
};
