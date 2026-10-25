import { RepState, FSMDiagnostics, LandmarkTrackingDetails } from '../types';

export interface CompletedRepInfo {
  repNumber: number;
  duration: number;
  peakElevation: number;
  minElbow: number;
  rom: number;
  completedAt: number;
}

export { type RepState, type FSMDiagnostics, type LandmarkTrackingDetails };

export class ClientRepetitionTracker {
  // Configurable thresholds (degrees)
  public startThreshold: number;
  public raisedThreshold: number;
  public returnThreshold: number;
  public hysteresis: number;
  public minDurationSec: number;
  public confirmationFrames: number;
  public maxTrackingLossSec: number;

  // FSM State
  public state: RepState = 'RESTING';
  public previousState: RepState = 'RESTING';
  public repCount: number = 0;
  public currentRepStartTime: number | null = null;
  public currentRepPeakElevation: number = 0;
  public currentRepMinElbow: number = 180;
  public repHistory: CompletedRepInfo[] = [];

  // Stability & Confirmation buffers
  private pendingState: RepState | null = null;
  private pendingFrames: number = 0;
  private lastValidTrackingTimestamp: number | null = null;
  private trackingLossStartTimestamp: number | null = null;

  // Diagnostics & logging
  public lastTransitionReason: string = 'Initial state';
  private lastLogTimestamp: number = 0;
  public diagnostics: FSMDiagnostics;

  constructor(
    startThreshold = 30.0,
    raisedThreshold = 80.0,
    returnThreshold = 25.0,
    hysteresis = 5.0,
    minDurationSec = 0.6,
    confirmationFrames = 2,
    maxTrackingLossSec = 1.5
  ) {
    this.startThreshold = startThreshold;
    this.raisedThreshold = raisedThreshold;
    this.returnThreshold = returnThreshold;
    this.hysteresis = hysteresis;
    this.minDurationSec = minDurationSec;
    this.confirmationFrames = confirmationFrames;
    this.maxTrackingLossSec = maxTrackingLossSec;

    this.diagnostics = {
      measuredAngle: 0,
      trackingValid: false,
      thresholds: {
        start: this.startThreshold,
        raised: this.raisedThreshold,
        returnThresh: this.returnThreshold,
        hysteresis: this.hysteresis,
      },
      previousState: this.previousState,
      currentState: this.state,
      transitionReason: this.lastTransitionReason,
      repCount: this.repCount,
      lastUpdated: 0,
    };
  }

  public getInstruction(): string {
    switch (this.state) {
      case 'RESTING':
        return 'Stand upright with arm at your side. Begin by raising right arm outward.';
      case 'RAISING':
        return 'Raising: Continue lifting arm toward shoulder height (90°).';
      case 'RAISED':
        return 'Apex reached! Hold briefly, then lower arm slowly.';
      case 'LOWERING':
        return 'Lowering: Return arm smoothly back down to your side.';
      default:
        return 'Follow movement guidelines.';
    }
  }

  private transitionTo(newState: RepState, reason: string, timestampSec: number) {
    this.previousState = this.state;
    this.state = newState;
    this.lastTransitionReason = reason;
    this.pendingState = null;
    this.pendingFrames = 0;

    // Immediate log on state transition
    this.logDiagnostic(timestampSec, true);
  }

  private checkCandidateTransition(
    candidate: RepState,
    reason: string,
    timestampSec: number
  ): boolean {
    if (this.pendingState === candidate) {
      this.pendingFrames += 1;
    } else {
      this.pendingState = candidate;
      this.pendingFrames = 1;
    }

    if (this.pendingFrames >= this.confirmationFrames) {
      this.transitionTo(candidate, reason, timestampSec);
      return true;
    }
    return false;
  }

  public update(
    elevation: number,
    elbow: number,
    timestampSec: number,
    hasTracking: boolean = true,
    trackingDetails?: LandmarkTrackingDetails
  ): {
    repCount: number;
    state: RepState;
    completedRep: CompletedRepInfo | null;
    instruction: string;
    diagnostics: FSMDiagnostics;
  } {
    let completedRep: CompletedRepInfo | null = null;

    // 1. Handle Lost Tracking
    if (!hasTracking) {
      if (this.trackingLossStartTimestamp === null) {
        this.trackingLossStartTimestamp = timestampSec;
      }
      const lostDuration = timestampSec - this.trackingLossStartTimestamp;

      // If tracking lost for > maxTrackingLossSec during an active rep, safely reset to RESTING
      if (this.state !== 'RESTING' && lostDuration >= this.maxTrackingLossSec) {
        this.transitionTo(
          'RESTING',
          `Tracking lost for ${lostDuration.toFixed(1)}s; safely reset to RESTING without counting.`,
          timestampSec
        );
        this.currentRepStartTime = null;
        this.currentRepPeakElevation = 0;
        this.currentRepMinElbow = 180;
      }

      this.updateDiagnostics(elevation, false, timestampSec, trackingDetails);
      this.rateLimitedLog(timestampSec);

      return {
        repCount: this.repCount,
        state: this.state,
        completedRep: null,
        instruction: 'Pose tracking unavailable. Ensure your upper body and right arm are in view.',
        diagnostics: this.diagnostics,
      };
    }

    // Tracking is valid
    this.trackingLossStartTimestamp = null;
    this.lastValidTrackingTimestamp = timestampSec;

    // 2. FSM Cycle: RESTING -> RAISING -> RAISED -> LOWERING -> RESTING
    switch (this.state) {
      case 'RESTING': {
        // Condition: arm elevation exceeds start threshold + hysteresis
        if (elevation >= (this.startThreshold + this.hysteresis)) {
          if (
            this.checkCandidateTransition(
              'RAISING',
              `Arm elevated above start threshold (${elevation.toFixed(1)}° >= ${(this.startThreshold + this.hysteresis).toFixed(1)}°)`,
              timestampSec
            )
          ) {
            this.currentRepStartTime = timestampSec;
            this.currentRepPeakElevation = elevation;
            this.currentRepMinElbow = elbow;
          }
        } else {
          this.pendingState = null;
          this.pendingFrames = 0;
        }
        break;
      }

      case 'RAISING': {
        // Track peak elevation and min elbow
        if (elevation > this.currentRepPeakElevation) {
          this.currentRepPeakElevation = elevation;
        }
        if (elbow < this.currentRepMinElbow) {
          this.currentRepMinElbow = elbow;
        }

        // Condition 1: Reached target apex
        if (elevation >= this.raisedThreshold) {
          this.checkCandidateTransition(
            'RAISED',
            `Reached target apex elevation (${elevation.toFixed(1)}° >= ${this.raisedThreshold.toFixed(1)}°)`,
            timestampSec
          );
        }
        // Condition 2: Movement aborted before reaching apex (arm dropped back below start threshold)
        else if (elevation <= (this.startThreshold - this.hysteresis)) {
          if (
            this.checkCandidateTransition(
              'RESTING',
              `Movement aborted: arm lowered before reaching apex (${elevation.toFixed(1)}° <= ${(this.startThreshold - this.hysteresis).toFixed(1)}°)`,
              timestampSec
            )
          ) {
            this.currentRepStartTime = null;
            this.currentRepPeakElevation = 0;
            this.currentRepMinElbow = 180;
          }
        } else {
          this.pendingState = null;
          this.pendingFrames = 0;
        }
        break;
      }

      case 'RAISED': {
        // Track peak if continuing slightly higher
        if (elevation > this.currentRepPeakElevation) {
          this.currentRepPeakElevation = elevation;
        }
        if (elbow < this.currentRepMinElbow) {
          this.currentRepMinElbow = elbow;
        }

        // Condition: descent initiated from peak
        // Arm dropped 8° below peak OR dropped below raisedThreshold - hysteresis
        const descentThreshold = Math.min(
          this.currentRepPeakElevation - 8.0,
          this.raisedThreshold - this.hysteresis
        );
        if (elevation <= descentThreshold) {
          this.checkCandidateTransition(
            'LOWERING',
            `Descent initiated from peak (${elevation.toFixed(1)}° <= ${descentThreshold.toFixed(1)}°)`,
            timestampSec
          );
        } else {
          this.pendingState = null;
          this.pendingFrames = 0;
        }
        break;
      }

      case 'LOWERING': {
        if (elbow < this.currentRepMinElbow) {
          this.currentRepMinElbow = elbow;
        }

        // Condition: returned to resting position
        if (elevation <= (this.returnThreshold + this.hysteresis)) {
          const duration = timestampSec - (this.currentRepStartTime ?? timestampSec);

          if (
            this.checkCandidateTransition(
              'RESTING',
              `Returned to resting position (${elevation.toFixed(1)}° <= ${(this.returnThreshold + this.hysteresis).toFixed(1)}°)`,
              timestampSec
            )
          ) {
            // Validate minimum repetition duration to filter out noise glitches
            if (duration >= this.minDurationSec && this.currentRepPeakElevation >= (this.raisedThreshold - this.hysteresis)) {
              this.repCount += 1;
              completedRep = {
                repNumber: this.repCount,
                duration: Math.round(duration * 10) / 10,
                peakElevation: Math.round(this.currentRepPeakElevation * 10) / 10,
                minElbow: Math.round(this.currentRepMinElbow * 10) / 10,
                rom: Math.round((this.currentRepPeakElevation - this.returnThreshold) * 10) / 10,
                completedAt: Math.round(timestampSec * 10) / 10,
              };
              this.repHistory.push(completedRep);
              this.lastTransitionReason = `Rep #${this.repCount} complete! Duration: ${completedRep.duration}s, Peak: ${completedRep.peakElevation}°`;
            } else {
              this.lastTransitionReason = `Movement finished but invalid (duration: ${duration.toFixed(1)}s, peak: ${this.currentRepPeakElevation.toFixed(1)}°)`;
            }

            // Reset cycle values
            this.currentRepStartTime = null;
            this.currentRepPeakElevation = 0;
            this.currentRepMinElbow = 180;
          }
        } else {
          this.pendingState = null;
          this.pendingFrames = 0;
        }
        break;
      }
    }

    this.updateDiagnostics(elevation, true, timestampSec, trackingDetails);
    this.rateLimitedLog(timestampSec);

    return {
      repCount: this.repCount,
      state: this.state,
      completedRep,
      instruction: this.getInstruction(),
      diagnostics: this.diagnostics,
    };
  }

  private updateDiagnostics(
    elevation: number,
    trackingValid: boolean,
    timestampSec: number,
    trackingDetails?: LandmarkTrackingDetails
  ) {
    this.diagnostics = {
      measuredAngle: Math.round(elevation * 10) / 10,
      trackingValid,
      thresholds: {
        start: this.startThreshold,
        raised: this.raisedThreshold,
        returnThresh: this.returnThreshold,
        hysteresis: this.hysteresis,
      },
      previousState: this.previousState,
      currentState: this.state,
      transitionReason: this.lastTransitionReason,
      trackingDetails,
      repCount: this.repCount,
      lastUpdated: timestampSec,
    };
  }

  private rateLimitedLog(timestampSec: number) {
    if (timestampSec - this.lastLogTimestamp >= 1.0) {
      this.lastLogTimestamp = timestampSec;
      this.logDiagnostic(timestampSec, false);
    }
  }

  private logDiagnostic(timestampSec: number, isTransition: boolean) {
    const prefix = isTransition ? '[FSM TRANSITION]' : '[FSM DIAGNOSTIC]';
    let trackInfo = this.diagnostics.trackingValid ? 'VALID' : 'LOST';
    if (this.diagnostics.trackingDetails) {
      const td = this.diagnostics.trackingDetails;
      trackInfo += ` (Sh:${td.shoulderScore.toFixed(2)}, Elb:${td.elbowScore.toFixed(2)}, Hip:${td.hipScore.toFixed(2)}, Wri:${td.wristScore.toFixed(2)} [thresh:${td.threshold.toFixed(2)}])`;
      if (!this.diagnostics.trackingValid && td.invalidReason) {
        trackInfo += ` [Reason: ${td.invalidReason}]`;
      }
    }
    console.log(
      `${prefix} State: ${this.previousState} -> ${this.state} | ` +
      `Angle: ${this.diagnostics.measuredAngle}° | ` +
      `Tracking: ${trackInfo} | ` +
      `Thresholds: [Start: ${this.startThreshold}°, Apex: ${this.raisedThreshold}°, Return: ${this.returnThreshold}° (±${this.hysteresis}°)] | ` +
      `Reps: ${this.repCount} | ` +
      `Reason: ${this.lastTransitionReason}`
    );
  }

  public reset() {
    this.state = 'RESTING';
    this.previousState = 'RESTING';
    this.repCount = 0;
    this.currentRepStartTime = null;
    this.currentRepPeakElevation = 0;
    this.currentRepMinElbow = 180;
    this.repHistory = [];
    this.pendingState = null;
    this.pendingFrames = 0;
    this.trackingLossStartTimestamp = null;
    this.lastTransitionReason = 'Reset to initial state';
  }
}
