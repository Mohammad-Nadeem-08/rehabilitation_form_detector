"""
Exercise Configuration and Strategy Registry for RehabVision AI.

Provides an extensible strategy pattern for multi-exercise rehabilitation.
Adding a new exercise requires registering a strategy subclass or configuration,
without modifying the core DTW comparison engine, repetition FSM, or database pipelines.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import math

from pose_features import (
    calculate_angle_2d,
    calculate_arm_elevation,
    calculate_elbow_angle,
    validate_right_arm_landmarks,
    get_landmark_score,
)

@dataclass
class ExerciseStrategy:
    id: str
    name: str
    target_joint: str
    description: str
    primary_metric_name: str
    secondary_metric_name: str
    target_rom_min: float
    target_rom_max: float
    fsm_thresholds: Dict[str, float]
    required_landmarks: List[Dict[str, Any]]
    
    def calculate_kinematics(self, landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute primary and secondary angles from landmark array."""
        raise NotImplementedError

    def validate_tracking(self, landmarks: List[Dict[str, float]], min_visibility: float = 0.30) -> Dict[str, Any]:
        """Validate required landmarks are tracked with sufficient confidence."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "target_joint": self.target_joint,
            "description": self.description,
            "primary_metric_name": self.primary_metric_name,
            "secondary_metric_name": self.secondary_metric_name,
            "target_rom_min": self.target_rom_min,
            "target_rom_max": self.target_rom_max,
            "fsm_thresholds": self.fsm_thresholds,
            "required_landmarks": self.required_landmarks,
        }


class StandingRightArmAbductionStrategy(ExerciseStrategy):
    """
    Exercise 1: Standing Right-Arm Shoulder Abduction (Primary Coronal Elevation).
    Target: 80° - 105° elevation with extended elbow (165° - 180°).
    """
    def __init__(self):
        super().__init__(
            id="standing-right-arm-abduction",
            name="Standing Right-Arm Shoulder Abduction",
            target_joint="Right Shoulder & Elbow",
            description="Coronal plane shoulder elevation from 0° resting arm up to 90°-100° with extended elbow (~170°-180°), controlled eccentric descent.",
            primary_metric_name="Arm Elevation",
            secondary_metric_name="Elbow Extension",
            target_rom_min=80.0,
            target_rom_max=105.0,
            fsm_thresholds={
                "start_threshold": 30.0,
                "raised_threshold": 80.0,
                "return_threshold": 25.0,
                "hysteresis": 5.0,
                "min_duration_sec": 0.5,
                "confirmation_frames": 2,
            },
            required_landmarks=[
                {"index": 12, "name": "Right Shoulder", "role": "Vertex for elevation"},
                {"index": 14, "name": "Right Elbow", "role": "Arm segment endpoint"},
                {"index": 24, "name": "Right Hip", "role": "Torso vertical baseline"},
                {"index": 23, "name": "Left Hip", "role": "Fallback torso baseline"},
                {"index": 16, "name": "Right Wrist", "role": "Elbow angle extension"},
            ]
        )

    def calculate_kinematics(self, landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        if not landmarks or len(landmarks) < 25:
            return {"primary_angle": 0.0, "secondary_angle": 180.0, "elevation": 0.0, "elbow": 180.0}
        
        shoulder = landmarks[12]
        elbow = landmarks[14]
        hip = landmarks[24] if landmarks[24].get("visibility", 0) > 0.3 else landmarks[23]
        wrist = landmarks[16] if len(landmarks) > 16 else {"x": elbow.get("x", 0.0), "y": elbow.get("y", 0.0) + 0.2}

        elevation = calculate_arm_elevation(shoulder, elbow, hip)
        elbow_ang = calculate_elbow_angle(shoulder, elbow, wrist)

        return {
            "primary_angle": round(float(elevation), 1),
            "secondary_angle": round(float(elbow_ang), 1),
            "elevation": round(float(elevation), 1),
            "elbow": round(float(elbow_ang), 1)
        }

    def validate_tracking(self, landmarks: List[Dict[str, float]], min_visibility: float = 0.30) -> Dict[str, Any]:
        return validate_right_arm_landmarks(landmarks, min_visibility=min_visibility)


class SeatedRightBicepCurlStrategy(ExerciseStrategy):
    """
    Exercise 2: Seated Right Bicep Curl (Sagittal Elbow Flexion).
    Demonstrates zero-duplication multi-exercise registration.
    Primary metric: Elbow Flexion (180° extended -> 45°-60° curled apex).
    Secondary metric: Upper Arm / Humeral Stability (Shoulder elevation <= 25°).
    """
    def __init__(self):
        super().__init__(
            id="seated-right-bicep-curl",
            name="Seated Right-Arm Bicep Curl",
            target_joint="Right Biceps Brachii & Elbow Flexor",
            description="Seated elbow flexion through full active range with humeral stabilization (elbow pinned near torso, avoiding excessive shoulder compensation).",
            primary_metric_name="Elbow Flexion",
            secondary_metric_name="Humeral Stability",
            target_rom_min=95.0,
            target_rom_max=135.0,
            fsm_thresholds={
                "start_threshold": 35.0,   # Flexion > 35° (i.e. elbow angle < 145°)
                "raised_threshold": 110.0, # Apex flexion > 110° (elbow angle < 70°)
                "return_threshold": 25.0,  # Return near extension (flexion < 25°)
                "hysteresis": 5.0,
                "min_duration_sec": 0.6,
                "confirmation_frames": 2,
            },
            required_landmarks=[
                {"index": 12, "name": "Right Shoulder", "role": "Proximal anchor"},
                {"index": 14, "name": "Right Elbow", "role": "Hinge joint vertex"},
                {"index": 16, "name": "Right Wrist", "role": "Distal lever arm"},
                {"index": 24, "name": "Right Hip", "role": "Torso alignment baseline"},
            ]
        )

    def calculate_kinematics(self, landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        if not landmarks or len(landmarks) < 25:
            return {"primary_angle": 0.0, "secondary_angle": 0.0, "elevation": 0.0, "elbow": 180.0}

        shoulder = landmarks[12]
        elbow = landmarks[14]
        wrist = landmarks[16]
        hip = landmarks[24] if landmarks[24].get("visibility", 0) > 0.3 else landmarks[23]

        raw_elbow = calculate_elbow_angle(shoulder, elbow, wrist)
        # Flexion angle = 180° - raw_elbow (0° at full extension, ~130° at full contraction)
        flexion_angle = max(0.0, min(160.0, 180.0 - raw_elbow))
        shoulder_sway = calculate_arm_elevation(shoulder, elbow, hip)

        return {
            "primary_angle": round(float(flexion_angle), 1),
            "secondary_angle": round(float(shoulder_sway), 1),
            "elevation": round(float(flexion_angle), 1), # compatible mapping for comparison
            "elbow": round(float(raw_elbow), 1),
        }

    def validate_tracking(self, landmarks: List[Dict[str, float]], min_visibility: float = 0.30) -> Dict[str, Any]:
        if not landmarks or len(landmarks) < 25:
            return {"has_required": False, "reason": "Insufficient landmarks", "scores": {}}

        sh_score = get_landmark_score(landmarks[12])
        elb_score = get_landmark_score(landmarks[14])
        wri_score = get_landmark_score(landmarks[16])
        hip_score = max(get_landmark_score(landmarks[24]), get_landmark_score(landmarks[23]))

        has_required = (sh_score >= min_visibility and elb_score >= min_visibility and wri_score >= min_visibility)
        reason = None
        if not has_required:
            missing = []
            if sh_score < min_visibility: missing.append(f"Right shoulder ({sh_score:.2f})")
            if elb_score < min_visibility: missing.append(f"Right elbow ({elb_score:.2f})")
            if wri_score < min_visibility: missing.append(f"Right wrist ({wri_score:.2f})")
            reason = f"Missing required bicep curl landmarks: {', '.join(missing)}"

        return {
            "has_required": has_required,
            "shoulder_score": sh_score,
            "elbow_score": elb_score,
            "wrist_score": wri_score,
            "hip_score": hip_score,
            "reason": reason
        }


# Global Registry Instance
class ExerciseRegistry:
    def __init__(self):
        self._strategies: Dict[str, ExerciseStrategy] = {}
        # Register default supported exercises
        self.register(StandingRightArmAbductionStrategy())
        self.register(SeatedRightBicepCurlStrategy())

    def register(self, strategy: ExerciseStrategy):
        self._strategies[strategy.id] = strategy

    def get(self, exercise_id: str) -> Optional[ExerciseStrategy]:
        return self._strategies.get(exercise_id)

    def list_all(self) -> List[ExerciseStrategy]:
        return list(self._strategies.values())

    def list_dicts(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._strategies.values()]


exercise_registry = ExerciseRegistry()
