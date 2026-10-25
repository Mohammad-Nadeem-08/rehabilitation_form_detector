import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';

let landmarkerInstance: PoseLandmarker | null = null;
let isInitializing = false;

export async function getPoseLandmarker(): Promise<PoseLandmarker> {
  if (landmarkerInstance) {
    return landmarkerInstance;
  }
  if (isInitializing) {
    // Wait for ongoing initialization
    while (isInitializing) {
      await new Promise((r) => setTimeout(r, 100));
    }
    if (landmarkerInstance) return landmarkerInstance;
  }

  isInitializing = true;
  try {
    let vision;
    try {
      vision = await FilesetResolver.forVisionTasks('/wasm');
    } catch (err) {
      console.warn('Local WASM resolver failed, falling back to CDN...', err);
      vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );
    }

    try {
      landmarkerInstance = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: '/models/pose_landmarker_lite.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    } catch (gpuErr) {
      console.warn('GPU delegate or local model failed, retrying with CPU...', gpuErr);
      landmarkerInstance = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: '/models/pose_landmarker_lite.task',
          delegate: 'CPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    }

    return landmarkerInstance;
  } finally {
    isInitializing = false;
  }
}

import { LandmarkTrackingDetails } from '../types';

export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
  visibility?: number;
  presence?: number;
}

export interface ComputedPoseMetrics {
  elevation: number;
  elbow: number;
  trunkLean: number;
  confidence: number;
  hasRequiredLandmarks: boolean;
  landmarks: LandmarkPoint[];
  trackingDetails?: LandmarkTrackingDetails;
}

export function getLandmarkScore(landmark?: LandmarkPoint | null): number {
  if (!landmark) return 0;
  const hasVis = typeof landmark.visibility === 'number' && !isNaN(landmark.visibility);
  const hasPres = typeof landmark.presence === 'number' && !isNaN(landmark.presence);
  if (hasVis && hasPres) {
    return Math.min(landmark.visibility!, landmark.presence!);
  }
  if (hasVis) return landmark.visibility!;
  if (hasPres) return landmark.presence!;
  return 1.0;
}

// 2D Angle calculation between 3 points: vertex is p2
export function calculate2DAngle(
  p1: { x: number; y: number },
  p2: { x: number; y: number },
  p3: { x: number; y: number }
): number {
  const v1x = p1.x - p2.x;
  const v1y = p1.y - p2.y;
  const v2x = p3.x - p2.x;
  const v2y = p3.y - p2.y;

  const dot = v1x * v2x + v1y * v2y;
  const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
  const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);

  if (mag1 < 1e-6 || mag2 < 1e-6) return 0;
  const cosTheta = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
  return (Math.acos(cosTheta) * 180) / Math.PI;
}

export class LandmarkSmoother {
  private alpha: number;
  private lastVal: number | null = null;

  constructor(alpha: number = 0.65) {
    this.alpha = alpha;
  }

  update(val: number): number {
    if (this.lastVal === null) {
      this.lastVal = val;
      return val;
    }
    const smoothed = this.alpha * val + (1 - this.alpha) * this.lastVal;
    this.lastVal = smoothed;
    return smoothed;
  }

  reset() {
    this.lastVal = null;
  }
}

export function extractRightArmMetrics(
  landmarks: LandmarkPoint[],
  elevationSmoother?: LandmarkSmoother,
  elbowSmoother?: LandmarkSmoother,
  minVisibilityThreshold: number = 0.30
): ComputedPoseMetrics {
  // Required landmarks for right-arm elevation:
  // 12: Right Shoulder (Required)
  // 14: Right Elbow (Required)
  // 24: Right Hip (Preferred torso anchor)
  // 23: Left Hip (Torso anchor fallback)
  // Auxiliary landmarks:
  // 16: Right Wrist (Used for elbow angle only; non-critical for arm elevation)
  // 11: Left Shoulder (Used for trunk lean)

  if (!landmarks || landmarks.length < 25) {
    const trackingDetails: LandmarkTrackingDetails = {
      shoulderScore: 0,
      elbowScore: 0,
      wristScore: 0,
      rHipScore: 0,
      lHipScore: 0,
      hipScore: 0,
      threshold: minVisibilityThreshold,
      hasRequired: false,
      wristValid: false,
      invalidReason: 'Insufficient pose landmarks (< 25 points detected)',
    };

    return {
      elevation: 0,
      elbow: 180,
      trunkLean: 0,
      confidence: 0,
      hasRequiredLandmarks: false,
      landmarks: landmarks || [],
      trackingDetails,
    };
  }

  const rShoulder = landmarks[12];
  const rElbow = landmarks[14];
  const rWrist = landmarks[16];
  const rHip = landmarks[24];
  const lShoulder = landmarks[11];
  const lHip = landmarks[23];

  const shoulderScore = getLandmarkScore(rShoulder);
  const elbowScore = getLandmarkScore(rElbow);
  const rHipScore = getLandmarkScore(rHip);
  const lHipScore = getLandmarkScore(lHip);
  const wristScore = getLandmarkScore(rWrist);

  // Torso reference: prefer right hip (24), fallback to left hip (23) if right hip is occluded
  const hipScore = Math.max(rHipScore, lHipScore);
  const hipRef = rHipScore >= minVisibilityThreshold ? rHip : (lHipScore >= minVisibilityThreshold ? lHip : rHip);

  // Check required landmarks specifically for right arm elevation
  const invalidReasons: string[] = [];
  if (shoulderScore < minVisibilityThreshold) {
    invalidReasons.push(`Right shoulder (12) score ${shoulderScore.toFixed(2)} < ${minVisibilityThreshold}`);
  }
  if (elbowScore < minVisibilityThreshold) {
    invalidReasons.push(`Right elbow (14) score ${elbowScore.toFixed(2)} < ${minVisibilityThreshold}`);
  }
  if (hipScore < minVisibilityThreshold) {
    invalidReasons.push(`Torso hip (24/23) score ${hipScore.toFixed(2)} < ${minVisibilityThreshold}`);
  }

  const isElevationValid = invalidReasons.length === 0;
  const wristValid = wristScore >= minVisibilityThreshold;

  // Elevation tracking confidence is bounded by required landmarks
  const elevationConfidence = isElevationValid
    ? Math.min(shoulderScore, elbowScore, hipScore)
    : 0;

  const trackingDetails: LandmarkTrackingDetails = {
    shoulderScore: Math.round(shoulderScore * 100) / 100,
    elbowScore: Math.round(elbowScore * 100) / 100,
    wristScore: Math.round(wristScore * 100) / 100,
    rHipScore: Math.round(rHipScore * 100) / 100,
    lHipScore: Math.round(lHipScore * 100) / 100,
    hipScore: Math.round(hipScore * 100) / 100,
    threshold: minVisibilityThreshold,
    hasRequired: isElevationValid,
    wristValid,
    invalidReason: invalidReasons.length > 0 ? invalidReasons.join('; ') : undefined,
  };

  if (!isElevationValid) {
    // If required landmarks are missing or below threshold, tracking is invalid
    return {
      elevation: 0,
      elbow: 180,
      trunkLean: 0,
      confidence: Math.round(elevationConfidence * 100) / 100,
      hasRequiredLandmarks: false,
      landmarks,
      trackingDetails,
    };
  }

  // 1. Right arm elevation relative to torso line (Hip -> Right Shoulder -> Right Elbow)
  const rawElevation = calculate2DAngle(hipRef, rShoulder, rElbow);

  // 2. Right elbow angle (extension vs flexion)
  // If wrist is valid, calculate from wrist; if wrist is occluded, default to 180 or last smoothed value
  let rawElbow = 180;
  if (wristValid && rWrist) {
    rawElbow = calculate2DAngle(rShoulder, rElbow, rWrist);
  }

  // 3. Trunk lean: vector from mid-hip to mid-shoulder vs vertical
  let trunkLean = 0;
  if (lHip && lShoulder) {
    const midHip = { x: (rHip.x + lHip.x) / 2, y: (rHip.y + lHip.y) / 2 };
    const midShoulder = { x: (rShoulder.x + lShoulder.x) / 2, y: (rShoulder.y + lShoulder.y) / 2 };
    const dx = midHip.x - midShoulder.x;
    const dy = midHip.y - midShoulder.y;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len > 1e-6) {
      const cosTheta = Math.max(-1, Math.min(1, dy / len));
      trunkLean = (Math.acos(cosTheta) * 180) / Math.PI;
    }
  }

  // Apply smoothers if provided
  const elevation = elevationSmoother ? elevationSmoother.update(rawElevation) : rawElevation;
  const elbow = (wristValid && elbowSmoother) ? elbowSmoother.update(rawElbow) : (elbowSmoother ? (elbowSmoother as any).lastVal ?? 180 : rawElbow);

  return {
    elevation: Math.round(elevation * 10) / 10,
    elbow: Math.round(elbow * 10) / 10,
    trunkLean: Math.round(trunkLean * 10) / 10,
    confidence: Math.round(elevationConfidence * 100) / 100,
    hasRequiredLandmarks: true,
    landmarks,
    trackingDetails,
  };
}

// Draw skeleton overlay on HTML5 Canvas
export function drawPoseSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: LandmarkPoint[],
  canvasWidth: number,
  canvasHeight: number,
  activeElevation: number
) {
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  if (!landmarks || landmarks.length === 0) return;

  const POSE_CONNECTIONS: [number, number][] = [
    // Torso
    [11, 12], [12, 24], [24, 23], [23, 11],
    // Right arm (Target)
    [12, 14], [14, 16],
    // Left arm
    [11, 13], [13, 15],
    // Legs
    [24, 26], [26, 28], [23, 25], [25, 27],
  ];

  // Draw connections
  POSE_CONNECTIONS.forEach(([i1, i2]) => {
    const p1 = landmarks[i1];
    const p2 = landmarks[i2];
    if (!p1 || !p2) return;
    if (getLandmarkScore(p1) < 0.25 || getLandmarkScore(p2) < 0.25) return;

    const isTargetArm = (i1 === 12 && i2 === 14) || (i1 === 14 && i2 === 16);

    ctx.beginPath();
    ctx.moveTo(p1.x * canvasWidth, p1.y * canvasHeight);
    ctx.lineTo(p2.x * canvasWidth, p2.y * canvasHeight);

    if (isTargetArm) {
      // Glow effect for target right arm
      ctx.lineWidth = 5;
      ctx.strokeStyle = activeElevation > 75 ? '#00E5FF' : '#00B4D8';
      ctx.shadowColor = '#00E5FF';
      ctx.shadowBlur = 12;
    } else {
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.45)';
      ctx.shadowBlur = 0;
    }

    ctx.stroke();
    ctx.shadowBlur = 0; // reset
  });

  // Draw landmark points
  landmarks.forEach((p, idx) => {
    if (getLandmarkScore(p) < 0.25) return;
    const isTargetJoint = idx === 12 || idx === 14 || idx === 16 || idx === 24;

    const cx = p.x * canvasWidth;
    const cy = p.y * canvasHeight;

    ctx.beginPath();
    ctx.arc(cx, cy, isTargetJoint ? 7 : 3.5, 0, 2 * Math.PI);

    if (isTargetJoint) {
      ctx.fillStyle = '#00E5FF';
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2.5;
      ctx.shadowColor = '#00E5FF';
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.stroke();
      ctx.shadowBlur = 0;
    } else {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
      ctx.fill();
    }
  });
}
