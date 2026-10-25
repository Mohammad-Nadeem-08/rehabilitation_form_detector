import React, { useState } from 'react';
import {
  Cpu, Play, RefreshCw, CheckCircle, AlertTriangle, FileCode,
  ShieldAlert, Info, Layers
} from 'lucide-react';
import { predictCompatibleLSTM, fetchSampleSequence } from '../services/api';
import { ModelPredictResponse } from '../types';

export const ModelLabPage: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<ModelPredictResponse | null>(null);
  const [activeForm, setActiveForm] = useState<'correct' | 'incorrect'>('correct');
  const [sampleShape, setSampleShape] = useState<number[]>([64, 80]);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const testSample = async (form: 'correct' | 'incorrect') => {
    try {
      setLoading(true);
      setActiveForm(form);
      setStatusMsg(`Fetching compatible (64, 80) sequence for '${form}' repetition...`);

      // 1. Fetch sample compatible 80-feature sequence
      const sample = await fetchSampleSequence(form);
      setSampleShape(sample.shape);

      setStatusMsg(`Executing PyTorch LSTM inference via /api/model/predict...`);

      // 2. Run real PyTorch inference
      const result = await predictCompatibleLSTM(sample.sequence);
      setPrediction(result);
      setStatusMsg(null);
    } catch (err: any) {
      console.error('Inference error:', err);
      alert(`Model evaluation error: ${err.message}`);
      setStatusMsg(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header Info */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <span className="badge badge-teal">Phase 7 PyTorch LSTM Deployment</span>
          <span className="badge badge-green">Trained Checkpoint: rehab_lstm.pth</span>
        </div>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: '#F8FAFC', marginBottom: '8px' }}>
          Model Lab — PyTorch LSTM Form Classification
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', maxWidth: '820px' }}>
          Direct testbench for the 2-layer PyTorch LSTM model trained on the REHAB24-6 rehabilitation dataset. 
          Evaluates sequences of shape (64, 80) incorporating 78 normalized coordinates from 26 3D joints, arm elevation, and elbow angle.
        </p>
      </div>

      {/* Critical Architecture & Compatibility Notice */}
      <div className="glass-panel" style={{ padding: '20px', borderLeft: '4px solid #00E5FF' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
          <ShieldAlert size={20} color="#00E5FF" />
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#F8FAFC' }}>
            Biomechanical Coordinate Representation Guard
          </h3>
        </div>
        <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          <strong>Why live MediaPipe landmarks are separated from the LSTM:</strong> The trained model expects exact 3D motion-capture joint 
          vectors (80 dimensions: 26 joints × 3 coords + 2 angles) normalized with training distribution mean & std. 
          MediaPipe in the browser produces 33 landmarks in a camera-perspective coordinate frame. 
          To preserve scientific integrity and prevent fabricated AI predictions, live webcam tracking is analyzed using 
          dynamic time warping (DTW) against genuine baseline curves, while this Model Lab demonstrates the true PyTorch LSTM on validated sequences.
        </p>
      </div>

      {/* Architecture Spec Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
        <div className="glass-panel" style={{ padding: '16px' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Architecture</div>
          <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#F8FAFC', marginTop: '4px' }}>PyTorch LSTM (2-Layer)</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Hidden Size: 64 • Batch First</div>
        </div>

        <div className="glass-panel" style={{ padding: '16px' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Input Tensor Shape</div>
          <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#00E5FF', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>(64, 80)</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>64 timesteps × 80 features</div>
        </div>

        <div className="glass-panel" style={{ padding: '16px' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Dataset Context</div>
          <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#F8FAFC', marginTop: '4px' }}>REHAB24-6 (9 Subjects)</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>178 Total Repetitions</div>
        </div>

        <div className="glass-panel" style={{ padding: '16px' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Reported Test Metrics</div>
          <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#10B981', marginTop: '4px' }}>82.5% Accuracy</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Macro F1: 81.95% (Preliminary)</div>
        </div>
      </div>

      {/* Interactive Inference Testbench */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#F8FAFC', marginBottom: '14px' }}>
          Run Verified PyTorch Checkpoint Inference
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '20px' }}>
          Select a sample repetition sequence to execute real-time forward pass through the trained LSTM network:
        </p>

        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '24px' }}>
          <button
            onClick={() => testSample('correct')}
            disabled={loading}
            className="btn btn-primary"
            style={{ padding: '12px 20px' }}
          >
            {loading && activeForm === 'correct' ? <RefreshCw size={16} className="animate-spin" /> : <Play size={16} />}
            Evaluate "Correct" Movement Sample (Class 1)
          </button>

          <button
            onClick={() => testSample('incorrect')}
            disabled={loading}
            className="btn btn-secondary"
            style={{ padding: '12px 20px', borderColor: 'rgba(239, 68, 68, 0.4)' }}
          >
            {loading && activeForm === 'incorrect' ? <RefreshCw size={16} className="animate-spin" /> : <Play size={16} />}
            Evaluate "Restricted Form" Sample (Class 0)
          </button>
        </div>

        {statusMsg && (
          <div style={{ color: '#00E5FF', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <RefreshCw size={15} className="animate-spin" />
            {statusMsg}
          </div>
        )}

        {/* Prediction Results Display */}
        {prediction && (
          <div style={{
            background: 'rgba(10, 17, 40, 0.85)',
            border: '1px solid var(--border-active)',
            borderRadius: '12px',
            padding: '24px',
            display: 'flex',
            flexDirection: 'column',
            gap: '18px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                  LSTM Classification Output
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '4px' }}>
                  <span style={{
                    fontSize: '1.75rem',
                    fontWeight: 800,
                    color: prediction.predicted_class === 1 ? '#10B981' : '#EF4444',
                    fontFamily: 'var(--font-mono)',
                  }}>
                    {prediction.class_label.toUpperCase()} (Class {prediction.predicted_class})
                  </span>
                  <span className={`badge ${prediction.predicted_class === 1 ? 'badge-green' : 'badge-red'}`}>
                    {prediction.predicted_class === 1 ? 'Correct Abduction Form' : 'Movement Form Deviation'}
                  </span>
                </div>
              </div>

              <div style={{ textAlign: 'right' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Input Tensor Shape</span>
                <div style={{ fontFamily: 'var(--font-mono)', color: '#00E5FF', fontWeight: 700 }}>
                  ({sampleShape.join(', ')})
                </div>
              </div>
            </div>

            {/* Softmax Probability Bars */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                <span>Class 1 (Correct Form Probability):</span>
                <strong style={{ color: '#10B981' }}>{(prediction.probabilities.correct * 100).toFixed(2)}%</strong>
              </div>
              <div style={{ height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden', marginBottom: '12px' }}>
                <div style={{
                  height: '100%',
                  width: `${prediction.probabilities.correct * 100}%`,
                  background: '#10B981',
                  transition: 'width 0.4s ease',
                }} />
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                <span>Class 0 (Incorrect Form Probability):</span>
                <strong style={{ color: '#EF4444' }}>{(prediction.probabilities.incorrect * 100).toFixed(2)}%</strong>
              </div>
              <div style={{ height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${prediction.probabilities.incorrect * 100}%`,
                  background: '#EF4444',
                  transition: 'width 0.4s ease',
                }} />
              </div>
            </div>

            {/* Warning & Context Banner */}
            <div style={{
              background: 'rgba(239, 68, 68, 0.08)',
              border: '1px solid rgba(239, 68, 68, 0.25)',
              borderRadius: '8px',
              padding: '12px',
              fontSize: '0.8rem',
              color: '#FCA5A5',
              lineHeight: 1.5,
            }}>
              <strong>Clinical & Evaluation Disclaimer:</strong> {prediction.warning}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
