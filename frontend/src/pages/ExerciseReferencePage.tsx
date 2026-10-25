import React, { useState, useEffect } from 'react';
import {
  FileText, CheckCircle, Save, RefreshCw, AlertCircle, Bookmark, Star,
  ShieldCheck, UserCheck, Layers, Filter
} from 'lucide-react';
import { CameraView } from '../components/CameraView';
import { KinematicChart } from '../components/KinematicChart';
import { fetchReferences, saveReference } from '../services/api';
import { ReferenceData, AnglePoint } from '../types';

interface ExerciseReferencePageProps {
  activeReferenceId: string | null;
  setActiveReferenceId: (id: string) => void;
}

export const ExerciseReferencePage: React.FC<ExerciseReferencePageProps> = ({
  activeReferenceId,
  setActiveReferenceId,
}) => {
  const [references, setReferences] = useState<ReferenceData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [notes, setNotes] = useState<string>('');
  
  // Reference Type & Provenance metadata
  const [refType, setRefType] = useState<'standard' | 'patient_specific'>('standard');
  const [refVersion, setRefVersion] = useState<string>('1.0');
  const [provenance, setProvenance] = useState<string>('Physiotherapy Clinical Protocol v1.0');
  const [patientId, setPatientId] = useState<string>('');
  const [filterType, setFilterType] = useState<'all' | 'standard' | 'patient_specific'>('all');

  const [recordedTrajectory, setRecordedTrajectory] = useState<AnglePoint[]>([]);
  const [recordedDuration, setRecordedDuration] = useState<number>(0);
  const [recordedRepHistory, setRecordedRepHistory] = useState<any[]>([]);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveSuccessMsg, setSaveSuccessMsg] = useState<string | null>(null);

  useEffect(() => {
    loadReferences();
  }, []);

  const loadReferences = async () => {
    try {
      setLoading(true);
      const list = await fetchReferences();
      setReferences(list);
      if (!activeReferenceId && list.length > 0) {
        setActiveReferenceId(list[0].id);
      }
    } catch (err) {
      console.error('Failed to load references:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSessionComplete = (
    trajectory: AnglePoint[],
    repHistory: any[],
    durationSec: number
  ) => {
    setRecordedTrajectory(trajectory);
    setRecordedRepHistory(repHistory);
    setRecordedDuration(durationSec);
  };

  const handleSaveReference = async () => {
    if (recordedTrajectory.length < 15) {
      alert('Recorded reference is too short (< 15 frames). Please perform a full repetition.');
      return;
    }

    try {
      setIsSaving(true);
      const validReps = recordedRepHistory.filter((r) => r.minElbow >= 40.0 && r.peakElevation >= 70.0);
      const repsToUse = validReps.length > 0 ? validReps : recordedRepHistory;
      const peak = repsToUse.length > 0 ? Math.max(...repsToUse.map((r: any) => r.peakElevation)) : Math.max(...recordedTrajectory.map((p) => p.elevation));
      const minElb = repsToUse.length > 0 ? Math.min(...repsToUse.map((r: any) => r.minElbow)) : Math.min(...recordedTrajectory.map((p) => p.elbow));

      const newRef = await saveReference({
        exercise_id: 'standing-right-arm-abduction',
        exercise_name: 'Standing Right-Arm Shoulder Abduction',
        notes: notes || (refType === 'patient_specific' ? `Clinician-approved tailored reference for ${patientId || 'patient'}.` : 'Standard clinical exercise reference baseline.'),
        duration_seconds: recordedDuration,
        frame_count: recordedTrajectory.length,
        joint_angles: recordedTrajectory,
        landmarks_summary: {
          peak_elevation: Math.round(peak * 10) / 10,
          min_elbow: Math.round(minElb * 10) / 10,
        },
        reference_type: refType,
        version: refVersion || '1.0',
        provenance: provenance || (refType === 'patient_specific' ? 'Clinician Verified' : 'Standard Protocol'),
        patient_id: refType === 'patient_specific' ? (patientId.trim() || null) : null,
        is_active: 1,
      });

      setSaveSuccessMsg(`Reference saved as ${newRef.id} (${refType === 'patient_specific' ? 'Patient-Specific' : 'Standard'})!`);
      setActiveReferenceId(newRef.id);
      setRecordedTrajectory([]);
      setNotes('');
      await loadReferences();
      setTimeout(() => setSaveSuccessMsg(null), 4000);
    } catch (err: any) {
      alert(`Save failed: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  const previewChartData = recordedTrajectory.map((p) => ({
    time: p.timestamp,
    elevation: p.elevation,
    elbow: p.elbow,
  }));

  const filteredReferences = references.filter((ref) => {
    if (filterType === 'all') return true;
    return (ref.reference_type || 'standard') === filterType;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header Info */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
          <span className="badge badge-teal">Two Reference Standards</span>
          <span className="badge badge-green">Target: Standing Right-Arm Shoulder Abduction</span>
          <span className="badge badge-blue">Pluggable Exercise Registry</span>
        </div>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: '#F8FAFC', marginBottom: '8px' }}>
          Reference Movement Management & Library
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', maxWidth: '820px' }}>
          Manage two distinct reference categories: <strong>Reusable Standard Clinical References</strong> (general protocol baselines) and <strong>Patient-Specific Clinician-Approved References</strong> (individualized movement goals with full clinician provenance and versioning).
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '24px' }}>
        {/* Left: Camera View */}
        <div>
          <CameraView
            modeTitle="Record Reference Repetition"
            onSessionComplete={handleSessionComplete}
          />
        </div>

        {/* Right: Review & Save Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div className="glass-panel" style={{ padding: '20px' }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#F8FAFC', marginBottom: '12px' }}>
              Reference Repetition Configuration & Save
            </h3>

            {recordedTrajectory.length === 0 ? (
              <div style={{
                padding: '30px',
                border: '1px dashed var(--border-subtle)',
                borderRadius: '8px',
                textAlign: 'center',
                color: 'var(--text-muted)',
                fontSize: '0.85rem',
              }}>
                No reference recorded in this session yet. Click <strong>Start Camera</strong> and then <strong>Start Recording</strong>,
                perform one full, controlled shoulder abduction to 90° with extended elbow, and click <strong>Stop & Analyze</strong>.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <KinematicChart
                  title="Recorded Reference Kinematics"
                  data={previewChartData}
                  mode="live"
                />

                <div style={{ display: 'flex', gap: '10px' }}>
                  <div className="hud-card" style={{ flex: 1 }}>
                    <div className="hud-value" style={{ fontSize: '1.3rem' }}>{recordedDuration.toFixed(1)}s</div>
                    <div className="hud-label">Duration</div>
                  </div>
                  <div className="hud-card" style={{ flex: 1 }}>
                    <div className="hud-value" style={{ fontSize: '1.3rem' }}>{recordedTrajectory.length}</div>
                    <div className="hud-label">Captured Frames</div>
                  </div>
                  <div className="hud-card" style={{ flex: 1 }}>
                    <div className="hud-value" style={{ fontSize: '1.3rem' }}>
                      {Math.max(...recordedTrajectory.map((p) => p.elevation)).toFixed(1)}°
                    </div>
                    <div className="hud-label">Peak Elevation</div>
                  </div>
                </div>

                {/* Reference Type Selection */}
                <div>
                  <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                    Reference Classification:
                  </label>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                    <button
                      type="button"
                      onClick={() => {
                        setRefType('standard');
                        setProvenance('Physiotherapy Clinical Protocol v1.0');
                      }}
                      className={`btn ${refType === 'standard' ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ fontSize: '0.8rem', padding: '10px' }}
                    >
                      <Layers size={14} /> Reusable Standard
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setRefType('patient_specific');
                        setProvenance('Approved by Clinician (Custom)');
                      }}
                      className={`btn ${refType === 'patient_specific' ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ fontSize: '0.8rem', padding: '10px' }}
                    >
                      <UserCheck size={14} /> Patient-Specific
                    </button>
                  </div>
                </div>

                {/* Patient ID (if patient specific) */}
                {refType === 'patient_specific' && (
                  <div>
                    <label style={{ display: 'block', fontSize: '0.8rem', color: '#38BDF8', marginBottom: '6px' }}>
                      Patient Identifier (Isolation Tag):
                    </label>
                    <input
                      type="text"
                      value={patientId}
                      onChange={(e) => setPatientId(e.target.value)}
                      placeholder="e.g., patient-101 or PAT-0042"
                      style={{
                        width: '100%',
                        padding: '9px 12px',
                        background: 'rgba(10, 17, 40, 0.8)',
                        border: '1px solid #38BDF8',
                        borderRadius: '8px',
                        color: '#F8FAFC',
                        fontSize: '0.85rem',
                        outline: 'none',
                      }}
                    />
                  </div>
                )}

                {/* Version and Provenance */}
                <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: '10px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                      Version:
                    </label>
                    <input
                      type="text"
                      value={refVersion}
                      onChange={(e) => setRefVersion(e.target.value)}
                      placeholder="1.0"
                      style={{
                        width: '100%',
                        padding: '9px 12px',
                        background: 'rgba(10, 17, 40, 0.8)',
                        border: '1px solid var(--border-subtle)',
                        borderRadius: '8px',
                        color: '#F8FAFC',
                        fontSize: '0.85rem',
                        outline: 'none',
                      }}
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                      Clinician Provenance / Protocol Authority:
                    </label>
                    <input
                      type="text"
                      value={provenance}
                      onChange={(e) => setProvenance(e.target.value)}
                      placeholder="e.g. Approved by Dr. Sarah Jenkins, PT"
                      style={{
                        width: '100%',
                        padding: '9px 12px',
                        background: 'rgba(10, 17, 40, 0.8)',
                        border: '1px solid var(--border-subtle)',
                        borderRadius: '8px',
                        color: '#F8FAFC',
                        fontSize: '0.85rem',
                        outline: 'none',
                      }}
                    />
                  </div>
                </div>

                {/* Notes */}
                <div>
                  <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                    Clinical Guidance & Target Notes:
                  </label>
                  <input
                    type="text"
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder="e.g., 90 deg target, scapular stabilization, controlled descent."
                    style={{
                      width: '100%',
                      padding: '10px 14px',
                      background: 'rgba(10, 17, 40, 0.8)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: '8px',
                      color: '#F8FAFC',
                      fontSize: '0.85rem',
                      outline: 'none',
                    }}
                  />
                </div>

                <button
                  onClick={handleSaveReference}
                  disabled={isSaving}
                  className="btn btn-primary"
                  style={{ width: '100%', padding: '12px' }}
                >
                  {isSaving ? <RefreshCw size={16} className="animate-spin" /> : <Save size={16} />}
                  Save as {refType === 'patient_specific' ? 'Approved Patient Reference' : 'Standard Reusable Baseline'}
                </button>

                {saveSuccessMsg && (
                  <div style={{ color: '#10B981', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <CheckCircle size={15} />
                    {saveSuccessMsg}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Reference Library List with Category Tabs */}
          <div className="glass-panel" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#F8FAFC' }}>
                Saved Reference Standards ({filteredReferences.length})
              </h3>
              <div style={{ display: 'flex', gap: '6px' }}>
                <button
                  onClick={() => setFilterType('all')}
                  className={`btn ${filterType === 'all' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ padding: '4px 8px', fontSize: '0.72rem' }}
                >
                  All
                </button>
                <button
                  onClick={() => setFilterType('standard')}
                  className={`btn ${filterType === 'standard' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ padding: '4px 8px', fontSize: '0.72rem' }}
                >
                  Standard
                </button>
                <button
                  onClick={() => setFilterType('patient_specific')}
                  className={`btn ${filterType === 'patient_specific' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ padding: '4px 8px', fontSize: '0.72rem' }}
                >
                  Patient-Specific
                </button>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {filteredReferences.map((ref) => {
                const isActive = activeReferenceId === ref.id;
                const isPatientSpecific = ref.reference_type === 'patient_specific';
                return (
                  <div
                    key={ref.id}
                    onClick={() => setActiveReferenceId(ref.id)}
                    className="glass-card-interactive"
                    style={{
                      padding: '12px 16px',
                      background: isActive ? 'rgba(0, 229, 255, 0.08)' : 'rgba(15, 23, 42, 0.5)',
                      border: isActive ? '1px solid #00E5FF' : '1px solid var(--border-subtle)',
                      borderRadius: '8px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '12px',
                    }}
                  >
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                        <span style={{ fontWeight: 700, fontSize: '0.88rem', color: '#F8FAFC' }}>
                          {ref.id}
                        </span>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                          v{ref.version || '1.0'}
                        </span>
                        {isPatientSpecific ? (
                          <span className="badge" style={{ background: 'rgba(168, 85, 247, 0.2)', color: '#C084FC', border: '1px solid #A855F7', fontSize: '0.65rem' }}>
                            <UserCheck size={10} /> Patient-Specific
                          </span>
                        ) : (
                          <span className="badge badge-teal" style={{ fontSize: '0.65rem' }}>
                            <ShieldCheck size={10} /> Standard Baseline
                          </span>
                        )}
                        {ref.patient_id && (
                          <span className="badge badge-blue" style={{ fontSize: '0.65rem' }}>
                            Patient: {ref.patient_id}
                          </span>
                        )}
                        {isActive && (
                          <span className="badge badge-green" style={{ fontSize: '0.65rem' }}>
                            <Star size={10} fill="#10B981" /> Active Target
                          </span>
                        )}
                      </div>
                      <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                        {ref.duration_seconds.toFixed(1)}s • {ref.frame_count} frames • {ref.notes || 'No notes'}
                      </div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px', fontStyle: 'italic' }}>
                        Provenance: {ref.provenance || 'Standard Protocol Baseline'}
                      </div>
                    </div>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setActiveReferenceId(ref.id);
                      }}
                      className={`btn ${isActive ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ padding: '6px 12px', fontSize: '0.75rem', whiteSpace: 'nowrap' }}
                    >
                      {isActive ? 'Active' : 'Select'}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

