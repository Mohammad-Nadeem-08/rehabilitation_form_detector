import React, { useRef, useState, useEffect, useCallback } from 'react';
import {
  Camera, CameraOff, Play, Pause, Square, RefreshCw,
  AlertCircle, CheckCircle, ShieldAlert, Info
} from 'lucide-react';
import {
  getPoseLandmarker, extractRightArmMetrics, drawPoseSkeleton,
  LandmarkSmoother, ComputedPoseMetrics
} from '../services/mediapipeService';
import {
  ClientRepetitionTracker, CompletedRepInfo, RepState, FSMDiagnostics
} from '../services/repetitionTracker';
import { AnglePoint } from '../types';

interface CameraViewProps {
  onFrameMetrics?: (metrics: ComputedPoseMetrics, timestampSec: number) => void;
  onRepetitionComplete?: (repInfo: CompletedRepInfo) => void;
  onSessionComplete?: (trajectory: AnglePoint[], repHistory: CompletedRepInfo[], durationSec: number) => void;
  modeTitle?: string;
  isRecording?: boolean;
}

export const CameraView: React.FC<CameraViewProps> = ({
  onFrameMetrics,
  onRepetitionComplete,
  onSessionComplete,
  modeTitle = "Standing Right-Arm Shoulder Abduction",
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animFrameIdRef = useRef<number | null>(null);

  // Smoothers & Repetition Tracker
  const elevationSmootherRef = useRef(new LandmarkSmoother(0.65));
  const elbowSmootherRef = useRef(new LandmarkSmoother(0.65));
  const repTrackerRef = useRef(new ClientRepetitionTracker());

  // Mutable refs to prevent stale closure bugs inside requestAnimationFrame
  const isRecordingRef = useRef<boolean>(false);
  const isPausedRef = useRef<boolean>(false);
  const onFrameMetricsRef = useRef(onFrameMetrics);
  const onRepetitionCompleteRef = useRef(onRepetitionComplete);
  const onSessionCompleteRef = useRef(onSessionComplete);

  useEffect(() => {
    onFrameMetricsRef.current = onFrameMetrics;
  }, [onFrameMetrics]);

  useEffect(() => {
    onRepetitionCompleteRef.current = onRepetitionComplete;
  }, [onRepetitionComplete]);

  useEffect(() => {
    onSessionCompleteRef.current = onSessionComplete;
  }, [onSessionComplete]);

  // State
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [modelLoading, setModelLoading] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);

  // Live Telemetry & FSM
  const [elevation, setElevation] = useState<number>(0);
  const [elbow, setElbow] = useState<number>(180);
  const [trunkLean, setTrunkLean] = useState<number>(0);
  const [confidence, setConfidence] = useState<number>(0);
  const [repState, setRepState] = useState<RepState>('RESTING');
  const [repCount, setRepCount] = useState<number>(0);
  const [hasLandmarks, setHasLandmarks] = useState<boolean>(false);
  const [lastRepFeedback, setLastRepFeedback] = useState<string | null>(null);
  const [liveInstruction, setLiveInstruction] = useState<string>(
    'Stand upright with arm at your side. Begin by raising right arm outward.'
  );
  const [diagnostics, setDiagnostics] = useState<FSMDiagnostics | null>(null);
  const [showDiagnosticsPanel, setShowDiagnosticsPanel] = useState<boolean>(true);

  // Trajectory buffer during recording
  const recordedTrajectoryRef = useRef<AnglePoint[]>([]);
  const recordingStartTimeRef = useRef<number>(0);

  // Stop camera tracks cleanly
  const stopCameraStream = useCallback(() => {
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
      animFrameIdRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    isRecordingRef.current = false;
    isPausedRef.current = false;
    setCameraActive(false);
    setIsRecording(false);
    setIsPaused(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCameraStream();
    };
  }, [stopCameraStream]);

  // Request camera and start tracking
  const startCamera = async () => {
    setCameraError(null);
    setModelLoading(true);

    try {
      // 1. Initialize PoseLandmarker
      const landmarker = await getPoseLandmarker();
      if (!landmarker) {
        throw new Error('Could not initialize PoseLandmarker.');
      }

      // 2. Request user media stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraActive(true);
      setModelLoading(false);

      // 3. Start processing loop
      let lastTimestampMs = -1;
      const processFrame = () => {
        if (!videoRef.current || !canvasRef.current || videoRef.current.paused || videoRef.current.ended) {
          animFrameIdRef.current = requestAnimationFrame(processFrame);
          return;
        }

        const nowMs = performance.now();
        if (nowMs > lastTimestampMs && videoRef.current.readyState >= 2) {
          lastTimestampMs = nowMs;

          try {
            const results = landmarker.detectForVideo(videoRef.current, nowMs);
            const nowSec = nowMs / 1000.0;

            if (results && results.landmarks && results.landmarks.length > 0) {
              const rawLandmarks = results.landmarks[0];
              const metrics = extractRightArmMetrics(
                rawLandmarks,
                elevationSmootherRef.current,
                elbowSmootherRef.current
              );

              setElevation(metrics.elevation);
              setElbow(metrics.elbow);
              setTrunkLean(metrics.trunkLean);
              setConfidence(metrics.confidence);
              setHasLandmarks(metrics.hasRequiredLandmarks);

              if (onFrameMetricsRef.current) {
                onFrameMetricsRef.current(metrics, nowSec);
              }

              // Run Repetition Tracker FSM on EVERY valid frame
              const fsmResult = repTrackerRef.current.update(
                metrics.elevation,
                metrics.elbow,
                nowSec,
                metrics.hasRequiredLandmarks,
                metrics.trackingDetails
              );

              setRepCount(fsmResult.repCount);
              setRepState(fsmResult.state);
              setLiveInstruction(fsmResult.instruction);
              setDiagnostics(fsmResult.diagnostics);

              // Record point into trajectory if recording is active and not paused
              if (isRecordingRef.current && !isPausedRef.current && metrics.hasRequiredLandmarks) {
                recordedTrajectoryRef.current.push({
                  timestamp: Math.round((nowSec - recordingStartTimeRef.current) * 100) / 100,
                  elevation: metrics.elevation,
                  elbow: metrics.elbow,
                  confidence: metrics.confidence,
                });
              }

              if (fsmResult.completedRep) {
                const rep = fsmResult.completedRep;
                setLastRepFeedback(`Rep #${rep.repNumber} Complete: ${rep.peakElevation}° peak in ${rep.duration}s`);
                if (onRepetitionCompleteRef.current) {
                  onRepetitionCompleteRef.current(rep);
                }
                setTimeout(() => setLastRepFeedback(null), 3500);
              }

              // Draw skeleton overlay
              const canvas = canvasRef.current;
              const ctx = canvas?.getContext('2d');
              if (ctx && canvas) {
                drawPoseSkeleton(ctx, rawLandmarks, canvas.width, canvas.height, metrics.elevation);
              }
            } else {
              setHasLandmarks(false);

              // Signal tracking loss to FSM
              const fsmResult = repTrackerRef.current.update(0, 180, nowSec, false, {
                shoulderScore: 0,
                elbowScore: 0,
                wristScore: 0,
                rHipScore: 0,
                lHipScore: 0,
                hipScore: 0,
                threshold: 0.30,
                hasRequired: false,
                wristValid: false,
                invalidReason: 'No pose detected in camera frame',
              });
              setRepState(fsmResult.state);
              setLiveInstruction(fsmResult.instruction);
              setDiagnostics(fsmResult.diagnostics);

              const canvas = canvasRef.current;
              const ctx = canvas?.getContext('2d');
              if (ctx && canvas) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
              }
            }
          } catch (detectionErr) {
            console.error('Frame detection error:', detectionErr);
          }
        }

        animFrameIdRef.current = requestAnimationFrame(processFrame);
      };

      animFrameIdRef.current = requestAnimationFrame(processFrame);
    } catch (err: any) {
      console.error('Failed to start camera/model:', err);
      setCameraError(err.message || 'Camera permission denied or camera device unavailable.');
      setModelLoading(false);
      stopCameraStream();
    }
  };

  const startRecording = () => {
    recordedTrajectoryRef.current = [];
    recordingStartTimeRef.current = performance.now() / 1000.0;
    repTrackerRef.current.reset();
    setRepCount(0);
    setRepState('RESTING');
    isRecordingRef.current = true;
    isPausedRef.current = false;
    setIsRecording(true);
    setIsPaused(false);
    setLastRepFeedback("Recording started: Stand facing camera and elevate right arm.");
    setTimeout(() => setLastRepFeedback(null), 3000);
  };

  const togglePause = () => {
    const nextPaused = !isPausedRef.current;
    isPausedRef.current = nextPaused;
    setIsPaused(nextPaused);
  };

  const stopRecording = () => {
    isRecordingRef.current = false;
    isPausedRef.current = false;
    setIsRecording(false);
    setIsPaused(false);
    const duration = Math.max(0.5, (performance.now() / 1000.0) - recordingStartTimeRef.current);
    if (onSessionCompleteRef.current) {
      onSessionCompleteRef.current(
        [...recordedTrajectoryRef.current],
        [...repTrackerRef.current.repHistory],
        Math.round(duration * 10) / 10
      );
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Real-time Movement Instruction Bar */}
      {cameraActive && (
        <div style={{
          background: 'rgba(15, 23, 42, 0.95)',
          border: '1px solid rgba(0, 229, 255, 0.4)',
          borderRadius: '10px',
          padding: '12px 18px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div className={`badge ${
              repState === 'RAISED' ? 'badge-green' :
              repState === 'RAISING' ? 'badge-teal' :
              repState === 'LOWERING' ? 'badge-amber' : 'badge-teal'
            }`} style={{ fontSize: '0.8rem', fontWeight: 800, padding: '4px 12px' }}>
              FSM: {repState}
            </div>
            <span style={{ fontSize: '0.92rem', color: '#F8FAFC', fontWeight: 600 }}>
              {liveInstruction}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Completed Reps:</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '1.5rem', fontWeight: 800, color: '#00E5FF' }}>
              {repCount}
            </span>
          </div>
        </div>
      )}

      {/* Video & Canvas Stream Container */}
      <div
        className="glass-panel"
        style={{
          position: 'relative',
          overflow: 'hidden',
          borderRadius: '12px',
          aspectRatio: '4/3',
          maxHeight: '480px',
          background: '#030712',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <video
          ref={videoRef}
          playsInline
          muted
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            transform: 'scaleX(-1)', // Mirror mode for intuitive self-feedback
          }}
        />

        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            transform: 'scaleX(-1)', // Match mirrored video
          }}
        />

        {/* Live Rep Feedback Banner */}
        {lastRepFeedback && (
          <div style={{
            position: 'absolute',
            top: '20px',
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(0, 229, 255, 0.95)',
            color: '#070B19',
            padding: '8px 20px',
            borderRadius: '20px',
            fontWeight: 800,
            fontSize: '0.88rem',
            boxShadow: '0 0 25px rgba(0, 229, 255, 0.6)',
            zIndex: 30,
            animation: 'pulse 1s infinite alternate',
          }}>
            {lastRepFeedback}
          </div>
        )}

        {/* Not Active Placeholder */}
        {!cameraActive && (
          <div style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(7, 11, 25, 0.95)',
            padding: '20px',
            textAlign: 'center',
          }}>
            <div style={{
              width: '64px',
              height: '64px',
              borderRadius: '50%',
              background: 'rgba(0, 229, 255, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '16px',
              border: '1px solid rgba(0, 229, 255, 0.3)',
            }}>
              <Camera size={32} color="#00E5FF" />
            </div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '8px', color: '#F8FAFC' }}>
              {modeTitle}
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', maxWidth: '400px', marginBottom: '20px' }}>
              Position yourself 2–3 meters from the webcam with your entire upper body and right arm visible.
            </p>
            <button
              onClick={startCamera}
              disabled={modelLoading}
              className="btn btn-primary"
              style={{ padding: '12px 24px', fontSize: '0.95rem' }}
            >
              {modelLoading ? (
                <>
                  <RefreshCw size={18} className="animate-spin" />
                  Initializing MediaPipe Vision...
                </>
              ) : (
                <>
                  <Camera size={18} />
                  Start Camera
                </>
              )}
            </button>
            {cameraError && (
              <div style={{ marginTop: '16px', color: '#F87171', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <AlertCircle size={14} />
                {cameraError}
              </div>
            )}
          </div>
        )}

        {/* Live Camera HUD Overlay */}
        {cameraActive && (
          <div style={{
            position: 'absolute',
            bottom: '12px',
            left: '12px',
            right: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            pointerEvents: 'none',
          }}>
            <div style={{ display: 'flex', gap: '8px' }}>
              <div className="badge badge-teal" style={{ background: 'rgba(7, 11, 25, 0.85)' }}>
                {hasLandmarks ? <CheckCircle size={12} color="#10B981" /> : <ShieldAlert size={12} color="#EF4444" />}
                {hasLandmarks ? `Tracking (${Math.round(confidence * 100)}%)` : 'Acquiring Pose...'}
              </div>

              {isRecording ? (
                <div className={`badge ${isPaused ? 'badge-amber' : 'badge-green'}`} style={{ background: 'rgba(7, 11, 25, 0.85)' }}>
                  {isPaused ? 'Paused' : `Session Recording Active`}
                </div>
              ) : (
                <div className="badge badge-teal" style={{ background: 'rgba(7, 11, 25, 0.85)' }}>
                  Live Practice Mode
                </div>
              )}
            </div>

            <div className={`badge ${
              repState === 'RAISED' ? 'badge-green' :
              repState === 'RAISING' ? 'badge-teal' :
              repState === 'LOWERING' ? 'badge-amber' : 'badge-teal'
            }`} style={{ background: 'rgba(7, 11, 25, 0.9)' }}>
              State: {repState}
            </div>
          </div>
        )}
      </div>

      {/* Primary Action Controls */}
      {cameraActive && (
        <div className="glass-panel" style={{ padding: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {!isRecording ? (
              <button onClick={startRecording} className="btn btn-primary">
                <Play size={16} />
                Start Recording Session
              </button>
            ) : (
              <>
                <button onClick={togglePause} className="btn btn-secondary">
                  {isPaused ? <Play size={16} /> : <Pause size={16} />}
                  {isPaused ? 'Resume' : 'Pause'}
                </button>
                <button onClick={stopRecording} className="btn btn-danger">
                  <Square size={16} />
                  Stop & Analyze
                </button>
              </>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button
              onClick={() => setShowDiagnosticsPanel((prev) => !prev)}
              className="btn btn-secondary"
              style={{ fontSize: '0.8rem', padding: '8px 12px' }}
            >
              <Info size={14} />
              {showDiagnosticsPanel ? 'Hide Diagnostics' : 'Show Diagnostics'}
            </button>
            <button onClick={stopCameraStream} className="btn btn-secondary" style={{ color: '#94A3B8' }}>
              <CameraOff size={16} />
              Close Camera
            </button>
          </div>
        </div>
      )}

      {/* Rate-Limited Live Diagnostics Panel */}
      {cameraActive && showDiagnosticsPanel && diagnostics && (
        <div style={{
          background: 'rgba(7, 11, 25, 0.92)',
          border: '1px solid rgba(0, 229, 255, 0.3)',
          borderRadius: '8px',
          padding: '10px 14px',
          fontSize: '0.75rem',
          color: 'var(--text-secondary)',
          display: 'flex',
          flexDirection: 'column',
          gap: '6px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              <span className="badge badge-teal" style={{ fontSize: '0.62rem', padding: '2px 6px' }}>FSM Diagnostic Stream</span>
              <span>Angle: <strong style={{ color: '#00E5FF' }}>{diagnostics.measuredAngle}°</strong></span>
              <span>• Tracking: <strong style={{ color: diagnostics.trackingValid ? '#10B981' : '#EF4444' }}>
                {diagnostics.trackingValid ? 'VALID' : 'LOST'}
              </strong></span>
              {diagnostics.trackingDetails && (
                <span style={{ fontFamily: 'monospace', fontSize: '0.72rem', color: '#94A3B8' }}>
                  [Sh: <span style={{ color: diagnostics.trackingDetails.shoulderScore >= diagnostics.trackingDetails.threshold ? '#10B981' : '#EF4444' }}>{diagnostics.trackingDetails.shoulderScore}</span>
                  , Elb: <span style={{ color: diagnostics.trackingDetails.elbowScore >= diagnostics.trackingDetails.threshold ? '#10B981' : '#EF4444' }}>{diagnostics.trackingDetails.elbowScore}</span>
                  , Hip: <span style={{ color: diagnostics.trackingDetails.hipScore >= diagnostics.trackingDetails.threshold ? '#10B981' : '#EF4444' }}>{diagnostics.trackingDetails.hipScore}</span>
                  , Wri: <span style={{ color: diagnostics.trackingDetails.wristValid ? '#10B981' : '#F59E0B' }}>{diagnostics.trackingDetails.wristScore}</span>
                  {diagnostics.trackingDetails.wristValid ? '' : ' (flexion only)'}]
                </span>
              )}
              <span>• Reps: <strong style={{ color: '#00E5FF' }}>{diagnostics.repCount}</strong></span>
            </div>
            <div>
              Cycle: <strong style={{ color: '#F8FAFC' }}>{diagnostics.previousState} → {diagnostics.currentState}</strong>
            </div>
          </div>
          {diagnostics.trackingDetails?.invalidReason && (
            <div style={{ color: '#F87171', fontSize: '0.72rem' }}>
              <strong>Tracking Invalid Reason:</strong> {diagnostics.trackingDetails.invalidReason}
            </div>
          )}
          <div style={{ color: '#CBD5E1', fontSize: '0.72rem', borderTop: '1px solid rgba(51, 65, 85, 0.4)', paddingTop: '4px' }}>
            <span>Last Reason: <strong>{diagnostics.transitionReason}</strong></span>
          </div>
        </div>
      )}

      {/* Telemetry Metric Meters */}
      {cameraActive && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '12px' }}>
          <div className="hud-card">
            <div className="hud-value" style={{ color: elevation >= 80 && elevation <= 105 ? '#10B981' : '#00E5FF' }}>
              {elevation}°
            </div>
            <div className="hud-label">Right Arm Elevation</div>
            <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', marginTop: '6px', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${Math.min(100, (elevation / 120) * 100)}%`, background: '#00E5FF', transition: 'width 0.1s linear' }} />
            </div>
          </div>

          <div className="hud-card">
            <div className="hud-value" style={{ color: elbow >= 160 ? '#10B981' : '#F59E0B' }}>
              {elbow}°
            </div>
            <div className="hud-label">Elbow Angle {elbow < 150 ? '(Flexed)' : '(Straight)'}</div>
            <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', marginTop: '6px', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${Math.min(100, (elbow / 180) * 100)}%`, background: elbow >= 160 ? '#10B981' : '#F59E0B' }} />
            </div>
          </div>

          <div className="hud-card">
            <div className="hud-value" style={{ color: trunkLean <= 10 ? '#10B981' : '#EF4444' }}>
              {trunkLean}°
            </div>
            <div className="hud-label">Trunk Lean {trunkLean > 10 ? '(Compensating)' : '(Stable)'}</div>
          </div>

          <div className="hud-card">
            <div className="hud-value" style={{ color: '#00E5FF' }}>
              {repCount}
            </div>
            <div className="hud-label">Repetitions Counted</div>
          </div>
        </div>
      )}
    </div>
  );
};
