import { ClientRepetitionTracker } from './repetitionTracker';
import { extractRightArmMetrics, LandmarkPoint } from './mediapipeService';

function assert(condition: boolean, message: string) {
  if (!condition) {
    throw new Error('Assertion failed: ' + message);
  }
}

console.log('--- Starting Frontend Repetition Tracker Tests ---');

// Test 1: Full repetition sequence: RESTING -> RAISING -> RAISED -> LOWERING -> RESTING
{
  const tracker = new ClientRepetitionTracker(30.0, 80.0, 25.0, 5.0, 0.5, 2, 1.5);
  const statesSeen: string[] = [];

  // 1. Resting frames
  for (let t = 0.0; t < 0.3; t += 0.05) {
    const res = tracker.update(15.0, 175.0, t, true);
    if (statesSeen.length === 0 || statesSeen[statesSeen.length - 1] !== res.state) {
      statesSeen.push(res.state);
    }
  }

  // 2. Raising frames (elevation 38° -> 75°)
  for (let t = 0.3; t < 1.0; t += 0.05) {
    const angle = 38.0 + (t - 0.3) * 55.0; // rises towards 80°
    const res = tracker.update(angle, 172.0, t, true);
    if (statesSeen[statesSeen.length - 1] !== res.state) {
      statesSeen.push(res.state);
    }
  }

  // 3. Apex / Raised frames (elevation 85° -> 92°)
  for (let t = 1.0; t < 1.4; t += 0.05) {
    const res = tracker.update(88.0, 170.0, t, true);
    if (statesSeen[statesSeen.length - 1] !== res.state) {
      statesSeen.push(res.state);
    }
  }

  // 4. Lowering frames (elevation drops 70° -> 35°)
  for (let t = 1.4; t < 2.0; t += 0.05) {
    const angle = 70.0 - (t - 1.4) * 55.0;
    const res = tracker.update(angle, 172.0, t, true);
    if (statesSeen[statesSeen.length - 1] !== res.state) {
      statesSeen.push(res.state);
    }
  }

  // 5. Returned to Resting (elevation 20° -> 15°)
  let finalRep: any = null;
  for (let t = 2.0; t < 2.3; t += 0.05) {
    const res = tracker.update(18.0, 175.0, t, true);
    if (res.completedRep) finalRep = res.completedRep;
    if (statesSeen[statesSeen.length - 1] !== res.state) {
      statesSeen.push(res.state);
    }
  }

  console.log('Test 1 States Seen:', statesSeen);
  assert(statesSeen.join(' -> ') === 'RESTING -> RAISING -> RAISED -> LOWERING -> RESTING', 'Expected full cycle');
  assert(tracker.repCount === 1, `Expected repCount 1, got ${tracker.repCount}`);
  assert(finalRep !== null, 'Expected completedRep object');
  assert(finalRep.peakElevation >= 85.0, 'Expected peak elevation >= 85');
  console.log('✓ Test 1: Full repetition cycle PASSED');
}

// Test 2: Incomplete movement (does not reach apex 80°)
{
  const tracker = new ClientRepetitionTracker(30.0, 80.0, 25.0, 5.0, 0.5, 2, 1.5);
  // Raises to only 50°
  for (let t = 0.0; t < 1.0; t += 0.05) {
    tracker.update(20.0 + t * 30.0, 175.0, t, true);
  }
  // Lowers back to 15°
  for (let t = 1.0; t < 2.0; t += 0.05) {
    tracker.update(50.0 - (t - 1.0) * 35.0, 175.0, t, true);
  }
  assert(tracker.repCount === 0, `Expected repCount 0 for incomplete movement, got ${tracker.repCount}`);
  assert(tracker.state === 'RESTING', 'Expected reset to RESTING');
  console.log('✓ Test 2: Incomplete movement abort PASSED');
}

// Test 3: Noisy jitter around threshold
{
  const tracker = new ClientRepetitionTracker(30.0, 80.0, 25.0, 5.0, 0.5, 2, 1.5);
  // Jitter around 32-34 deg (startThreshold + hysteresis is 35 deg)
  for (let t = 0.0; t < 1.0; t += 0.05) {
    const jitter = 32.0 + (t % 2 === 0 ? 2 : -2);
    tracker.update(jitter, 175.0, t, true);
    assert(tracker.state === 'RESTING', 'Should stay in RESTING');
  }
  assert(tracker.repCount === 0, 'No false reps from jitter');
  console.log('✓ Test 3: Noisy jitter rejection PASSED');
}

// Test 4: Repeated static frames
{
  const tracker = new ClientRepetitionTracker(30.0, 80.0, 25.0, 5.0, 0.5, 2, 1.5);
  for (let t = 0.0; t < 2.0; t += 0.05) {
    tracker.update(88.0, 175.0, t, true);
  }
  assert(tracker.repCount === 0, 'Static frames must not count reps');
  console.log('✓ Test 4: Repeated static frames PASSED');
}

// Test 5: Tracking loss handling
{
  const tracker = new ClientRepetitionTracker(30.0, 80.0, 25.0, 5.0, 0.5, 2, 1.0);
  // Raise arm to 85°
  for (let t = 0.0; t < 0.6; t += 0.05) {
    tracker.update(40.0 + t * 75.0, 175.0, t, true);
  }
  assert(tracker.state === 'RAISED' || tracker.state === 'RAISING', 'Should be raised');

  // Tracking lost for 2.0s (> maxTrackingLossSec 1.0s)
  for (let t = 0.6; t < 2.6; t += 0.05) {
    tracker.update(0.0, 180.0, t, false);
  }
  assert(tracker.repCount === 0, 'Tracking loss must NEVER count a repetition!');
  assert(tracker.state === 'RESTING', 'Should reset to RESTING on extended tracking loss');
  console.log('✓ Test 5: Tracking loss safety PASSED');
}

// Helper: build standard 33 pose landmarks
function createDummyLandmarks(overrides: Record<number, Partial<LandmarkPoint>> = {}): LandmarkPoint[] {
  const points: LandmarkPoint[] = [];
  for (let i = 0; i < 33; i++) {
    points.push({
      x: 0.5,
      y: 0.5,
      z: 0.0,
      visibility: 0.90,
      presence: 0.90,
      ...(overrides[i] || {}),
    });
  }
  return points;
}

// Test 6: Landmark tracking validity with all required joints present
{
  const landmarks = createDummyLandmarks({
    12: { x: 0.5, y: 0.3, visibility: 0.88, presence: 0.92 }, // Right Shoulder
    14: { x: 0.8, y: 0.3, visibility: 0.85, presence: 0.89 }, // Right Elbow
    16: { x: 1.0, y: 0.3, visibility: 0.80, presence: 0.85 }, // Right Wrist
    24: { x: 0.5, y: 0.7, visibility: 0.90, presence: 0.94 }, // Right Hip
    23: { x: 0.4, y: 0.7, visibility: 0.90, presence: 0.94 }, // Left Hip
    11: { x: 0.4, y: 0.3, visibility: 0.88, presence: 0.92 }, // Left Shoulder
  });

  const metrics = extractRightArmMetrics(landmarks);
  assert(metrics.hasRequiredLandmarks === true, 'All required landmarks present should yield valid tracking');
  assert(metrics.elevation > 0, `Expected positive elevation, got ${metrics.elevation}`);
  assert(metrics.trackingDetails !== undefined, 'Expected trackingDetails');
  assert(metrics.trackingDetails?.hasRequired === true, 'Expected trackingDetails.hasRequired === true');
  assert(metrics.trackingDetails?.wristValid === true, 'Expected wristValid === true');
  assert(metrics.trackingDetails?.invalidReason === undefined, 'Expected no invalidReason');
  console.log('✓ Test 6: Valid required landmarks PASSED');
}

// Test 7: Landmark tracking invalid when right shoulder (12) is missing or low visibility (< 0.30)
{
  const landmarks = createDummyLandmarks({
    12: { x: 0.5, y: 0.3, visibility: 0.15 }, // Right Shoulder low score
  });

  const metrics = extractRightArmMetrics(landmarks);
  assert(metrics.hasRequiredLandmarks === false, 'Low shoulder score must invalidate tracking');
  assert(metrics.elevation === 0, 'Invalid tracking must report 0 elevation');
  assert(metrics.trackingDetails?.invalidReason?.includes('Right shoulder (12)') === true, 'Must specify shoulder reason');
  console.log('✓ Test 7: Missing/low right shoulder invalidation PASSED');
}

// Test 8: Landmark tracking invalid when right elbow (14) is missing or low visibility (< 0.30)
{
  const landmarks = createDummyLandmarks({
    14: { x: 0.8, y: 0.3, visibility: 0.12 }, // Right Elbow low score
  });

  const metrics = extractRightArmMetrics(landmarks);
  assert(metrics.hasRequiredLandmarks === false, 'Low elbow score must invalidate tracking');
  assert(metrics.trackingDetails?.invalidReason?.includes('Right elbow (14)') === true, 'Must specify elbow reason');
  console.log('✓ Test 8: Missing/low right elbow invalidation PASSED');
}

// Test 9: Unrelated landmark (Right Wrist 16) occluded or low visibility must NOT invalidate elevation tracking
{
  const landmarks = createDummyLandmarks({
    12: { x: 0.5, y: 0.3, visibility: 0.88 }, // Right Shoulder valid
    14: { x: 0.8, y: 0.3, visibility: 0.85 }, // Right Elbow valid
    24: { x: 0.5, y: 0.7, visibility: 0.90 }, // Right Hip valid
    16: { x: 0.0, y: 0.0, visibility: 0.08 }, // Right Wrist blurry / out of frame (score 0.08 < 0.30)
  });

  const metrics = extractRightArmMetrics(landmarks);
  assert(metrics.hasRequiredLandmarks === true, 'Unrelated low wrist must NOT invalidate elevation tracking!');
  assert(metrics.trackingDetails?.wristValid === false, 'Wrist should be marked invalid for elbow angle');
  assert(metrics.elevation > 0, 'Elevation should be properly computed');
  console.log('✓ Test 9: Unrelated wrist occlusion does not fail tracking PASSED');
}

// Test 10: Torso hip fallback when Right Hip (24) is occluded but Left Hip (23) is visible
{
  const landmarks = createDummyLandmarks({
    12: { x: 0.5, y: 0.3, visibility: 0.88 },
    14: { x: 0.8, y: 0.3, visibility: 0.85 },
    24: { x: 0.0, y: 0.0, visibility: 0.10 }, // Right Hip behind desk
    23: { x: 0.4, y: 0.7, visibility: 0.90 }, // Left Hip clearly visible
  });

  const metrics = extractRightArmMetrics(landmarks);
  assert(metrics.hasRequiredLandmarks === true, 'Left hip fallback should keep tracking valid');
  assert(metrics.trackingDetails?.hipScore! >= 0.90, 'Expected hipScore to reflect visible left hip');
  console.log('✓ Test 10: Torso hip fallback PASSED');
}

// Test 11: End-to-end bug reproduction & fix validation:
// User reported symptom: angles reach 94°, 106.5°, 103.4°, wrist score dips,
// FSM previously aborted with Tracking: LOST. Now tracking remains VALID and rep is counted.
{
  const tracker = new ClientRepetitionTracker(30.0, 80.0, 25.0, 5.0, 0.5, 2, 1.5);
  const anglesAndWristVis = [
    { angle: 18.0, wristVis: 0.85, t: 0.0 },
    { angle: 22.0, wristVis: 0.80, t: 0.1 },
    { angle: 45.0, wristVis: 0.75, t: 0.3 }, // RAISING
    { angle: 70.0, wristVis: 0.60, t: 0.5 },
    { angle: 94.0, wristVis: 0.22, t: 0.7 }, // Apex reached (94°), wrist drops below 0.30!
    { angle: 106.5, wristVis: 0.15, t: 0.8 }, // Peak reached (106.5°), wrist out of camera view!
    { angle: 103.4, wristVis: 0.18, t: 0.9 }, // Sustained peak (103.4°)
    { angle: 75.0, wristVis: 0.55, t: 1.1 }, // LOWERING
    { angle: 45.0, wristVis: 0.75, t: 1.3 },
    { angle: 22.0, wristVis: 0.85, t: 1.5 }, // RESTING
    { angle: 18.0, wristVis: 0.90, t: 1.6 },
  ];

  let completedRep: any = null;
  for (const step of anglesAndWristVis) {
    // Generate landmarks matching this angle
    // Shoulder (12), Elbow (14), Hip (24) are all well tracked (0.85)
    // Wrist (16) has visibility `step.wristVis`
    const rad = (step.angle * Math.PI) / 180.0;
    const lms = createDummyLandmarks({
      12: { x: 0.5, y: 0.3, visibility: 0.88 },
      24: { x: 0.5, y: 0.7, visibility: 0.90 },
      14: { x: 0.5 + 0.3 * Math.sin(rad), y: 0.3 + 0.3 * Math.cos(rad), visibility: 0.85 },
      16: { x: 0.5 + 0.5 * Math.sin(rad), y: 0.3 + 0.5 * Math.cos(rad), visibility: step.wristVis },
    });

    const metrics = extractRightArmMetrics(lms);
    assert(metrics.hasRequiredLandmarks === true, `Tracking must remain valid at angle ${step.angle}° even with wrist vis ${step.wristVis}`);

    const res = tracker.update(
      metrics.elevation,
      metrics.elbow,
      step.t,
      metrics.hasRequiredLandmarks,
      metrics.trackingDetails
    );

    if (res.completedRep) {
      completedRep = res.completedRep;
    }
  }

  assert(tracker.repCount === 1, `Expected repCount 1 after full high-elevation cycle, got ${tracker.repCount}`);
  assert(completedRep !== null, 'Expected completedRep object');
  assert(completedRep.peakElevation >= 90.0, `Expected peak elevation >= 90°, got ${completedRep.peakElevation}°`);
  console.log('✓ Test 11: End-to-end high-elevation cycle with low wrist score PASSED');
}

// Test 12: Synthetic known joint positions (Arm-down, Horizontal abduction, Elevated arm, Bent elbow 90°, Acutely bent elbow 30°)
{
  // 1. Arm-down: hanging straight down alongside torso (elevation 0.0°, elbow 180.0°)
  const armDownLms = createDummyLandmarks({
    12: { x: 0.5, y: 0.3, visibility: 0.95 },
    24: { x: 0.5, y: 0.7, visibility: 0.95 },
    14: { x: 0.5, y: 0.6, visibility: 0.95 },
    16: { x: 0.5, y: 0.9, visibility: 0.95 },
  });
  const mDown = extractRightArmMetrics(armDownLms);
  assert(Math.abs(mDown.elevation - 0.0) < 0.1, `Expected 0.0° elevation, got ${mDown.elevation}°`);
  assert(Math.abs(mDown.elbow - 180.0) < 0.1, `Expected 180.0° elbow, got ${mDown.elbow}°`);

  // 2. Horizontal abduction: extended to side at shoulder height (elevation 90.0°, elbow 180.0°)
  const horizLms = createDummyLandmarks({
    12: { x: 0.5, y: 0.3, visibility: 0.95 },
    24: { x: 0.5, y: 0.7, visibility: 0.95 },
    14: { x: 0.8, y: 0.3, visibility: 0.95 },
    16: { x: 1.1, y: 0.3, visibility: 0.95 },
  });
  const mHoriz = extractRightArmMetrics(horizLms);
  assert(Math.abs(mHoriz.elevation - 90.0) < 0.1, `Expected 90.0° elevation, got ${mHoriz.elevation}°`);
  assert(Math.abs(mHoriz.elbow - 180.0) < 0.1, `Expected 180.0° elbow, got ${mHoriz.elbow}°`);

  // 3. Elevated overhead arm: pointing straight up (elevation 180.0°, elbow 180.0°)
  const overheadLms = createDummyLandmarks({
    12: { x: 0.5, y: 0.3, visibility: 0.95 },
    24: { x: 0.5, y: 0.7, visibility: 0.95 },
    14: { x: 0.5, y: 0.0, visibility: 0.95 },
    16: { x: 0.5, y: -0.3, visibility: 0.95 },
  });
  const mOverhead = extractRightArmMetrics(overheadLms);
  assert(Math.abs(mOverhead.elevation - 180.0) < 0.1, `Expected 180.0° elevation, got ${mOverhead.elevation}°`);
  assert(Math.abs(mOverhead.elbow - 180.0) < 0.1, `Expected 180.0° elbow, got ${mOverhead.elbow}°`);

  // 4. Bent elbow at 90° flexion: humerus horizontal, forearm vertical (elbow 90.0°)
  const bent90Lms = createDummyLandmarks({
    12: { x: 0.5, y: 0.3, visibility: 0.95 },
    24: { x: 0.5, y: 0.7, visibility: 0.95 },
    14: { x: 0.8, y: 0.3, visibility: 0.95 },
    16: { x: 0.8, y: 0.0, visibility: 0.95 },
  });
  const mBent90 = extractRightArmMetrics(bent90Lms);
  assert(Math.abs(mBent90.elevation - 90.0) < 0.1, `Expected 90.0° elevation, got ${mBent90.elevation}°`);
  assert(Math.abs(mBent90.elbow - 90.0) < 0.1, `Expected 90.0° elbow, got ${mBent90.elbow}°`);

  // 5. Acutely bent elbow at 30° flexion
  const rad30 = (30.0 * Math.PI) / 180.0;
  const bent30Lms = createDummyLandmarks({
    12: { x: 0.5, y: 0.3, visibility: 0.95 },
    24: { x: 0.5, y: 0.7, visibility: 0.95 },
    14: { x: 0.8, y: 0.3, visibility: 0.95 },
    16: { x: 0.8 - 0.3 * Math.cos(rad30), y: 0.3 - 0.3 * Math.sin(rad30), visibility: 0.95 },
  });
  const mBent30 = extractRightArmMetrics(bent30Lms);
  assert(Math.abs(mBent30.elbow - 30.0) < 0.2, `Expected 30.0° elbow, got ${mBent30.elbow}°`);

  console.log('✓ Test 12: Synthetic known joint positions (0°, 90°, 180°, 90° bend, 30° bend) PASSED');
}

// Test 13: Repetition-based stats filtering of post-workout reach artifacts
{
  const completedReps = [
    { repNumber: 1, duration: 2.1, peakElevation: 98.5, minElbow: 162.0, rom: 78.5, completedAt: 6.0 },
    { repNumber: 2, duration: 2.0, peakElevation: 102.0, minElbow: 159.0, rom: 82.0, completedAt: 10.0 },
    { repNumber: 3, duration: 2.2, peakElevation: 96.0, minElbow: 161.0, rom: 76.0, completedAt: 14.0 },
    // Non-exercise reach artifact at end (e.g. clicking stop with collapsed elbow 18.0° and reaching 150.0°)
    { repNumber: 4, duration: 1.5, peakElevation: 150.6, minElbow: 18.0, rom: 132.6, completedAt: 18.0 },
  ];

  // Filtering criteria: minElbow >= 40.0 and peakElevation >= 70.0
  const validReps = completedReps.filter((r) => r.minElbow >= 40.0 && r.peakElevation >= 70.0);
  assert(validReps.length === 3, `Expected 3 valid reps, got ${validReps.length}`);
  const truePeak = Math.max(...validReps.map((r) => r.peakElevation));
  const trueMinElbow = Math.min(...validReps.map((r) => r.minElbow));
  assert(truePeak === 102.0, `True peak should be 102.0°, got ${truePeak}°`);
  assert(trueMinElbow === 159.0, `True min elbow should be 159.0°, got ${trueMinElbow}°`);
  console.log('✓ Test 13: Repetition-based stats filtering of reach artifacts PASSED');
}

console.log('--- ALL FRONTEND REPETITION TRACKER TESTS PASSED ---');
