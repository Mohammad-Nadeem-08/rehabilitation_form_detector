/**
 * Frontend Exercise Registry for RehabVision AI.
 * 
 * Centralizes exercise definitions, primary and secondary metric labels,
 * target range of motion, and FSM threshold configurations.
 * Allows adding new exercises without modifying core tracking or DTW logic.
 */

import { ExerciseStrategyConfig } from '../types';

export const EXERCISE_REGISTRY: Record<string, ExerciseStrategyConfig> = {
  'standing-right-arm-abduction': {
    id: 'standing-right-arm-abduction',
    name: 'Standing Right-Arm Shoulder Abduction',
    target_joint: 'Right Shoulder & Elbow',
    description: 'Coronal plane shoulder elevation from 0° resting arm up to 90°-100° with extended elbow (~170°-180°), controlled eccentric descent.',
    primary_metric_name: 'Arm Elevation',
    secondary_metric_name: 'Elbow Extension',
    target_rom_min: 80.0,
    target_rom_max: 105.0,
    fsm_thresholds: {
      start_threshold: 30.0,
      raised_threshold: 80.0,
      return_threshold: 25.0,
      hysteresis: 5.0,
      min_duration_sec: 0.5,
      confirmation_frames: 2,
    },
    required_landmarks: [
      { index: 12, name: 'Right Shoulder', role: 'Elevation vertex' },
      { index: 14, name: 'Right Elbow', role: 'Arm lever segment' },
      { index: 24, name: 'Right Hip', role: 'Vertical torso alignment' },
      { index: 23, name: 'Left Hip', role: 'Fallback vertical alignment' },
      { index: 16, name: 'Right Wrist', role: 'Elbow extension angle' },
    ],
  },
  'seated-right-bicep-curl': {
    id: 'seated-right-bicep-curl',
    name: 'Seated Right-Arm Bicep Curl',
    target_joint: 'Right Biceps Brachii & Elbow Flexor',
    description: 'Seated elbow flexion through full active range with humeral stabilization (elbow pinned near torso, avoiding excessive shoulder compensation).',
    primary_metric_name: 'Elbow Flexion',
    secondary_metric_name: 'Humeral Stability',
    target_rom_min: 95.0,
    target_rom_max: 135.0,
    fsm_thresholds: {
      start_threshold: 35.0,
      raised_threshold: 110.0,
      return_threshold: 25.0,
      hysteresis: 5.0,
      min_duration_sec: 0.6,
      confirmation_frames: 2,
    },
    required_landmarks: [
      { index: 12, name: 'Right Shoulder', role: 'Proximal shoulder anchor' },
      { index: 14, name: 'Right Elbow', role: 'Hinge joint vertex' },
      { index: 16, name: 'Right Wrist', role: 'Distal lever arm' },
      { index: 24, name: 'Right Hip', role: 'Torso alignment baseline' },
    ],
  },
};

export function getExerciseConfig(exerciseId: string): ExerciseStrategyConfig {
  return EXERCISE_REGISTRY[exerciseId] || EXERCISE_REGISTRY['standing-right-arm-abduction'];
}

export function listExerciseConfigs(): ExerciseStrategyConfig[] {
  return Object.values(EXERCISE_REGISTRY);
}
