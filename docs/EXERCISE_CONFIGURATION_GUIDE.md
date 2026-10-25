# RehabVision AI — Exercise Configuration & Strategy Guide

This guide describes how to register new exercises in **RehabVision AI** using the pluggable strategy pattern. 

The architecture guarantees **zero code duplication**: registering an exercise automatically wires it into:
- Landmark tracking & visibility validation
- Finite State Machine (FSM) repetition counting
- Canonical cycle segmentation
- Dynamic Time Warping (DTW) movement comparison
- SQLite session & reference database persistence
- Publication-quality ReportLab PDF generation

---

## 1. Architecture of an Exercise Strategy

An exercise strategy encapsulates:
1. **Metadata:** Unique ID, human-readable name, target joints, and clinical description.
2. **Biomechanical Metrics:** Primary metric label (e.g., `Arm Elevation`, `Elbow Flexion`), secondary metric label (e.g., `Elbow Extension`, `Humeral Stability`), and target active range-of-motion bounds.
3. **Tracking Prerequisites:** Required MediaPipe landmark indices, roles, and confidence thresholds.
4. **Kinematic Calculators:** Conversion of raw $(X, Y, Z)$ landmarks into primary and secondary joint angles.
5. **FSM Thresholds:** Angle thresholds governing repetition states.

---

## 2. Backend Registration (`backend/exercise_registry.py`)

### Step 1: Subclass `ExerciseStrategy`
Create a new subclass in `backend/exercise_registry.py` inheriting from `ExerciseStrategy`:

```python
from exercise_registry import ExerciseStrategy, exercise_registry
from pose_features import calculate_angle_2d, get_landmark_score

class StandingLeftArmAbductionStrategy(ExerciseStrategy):
    """
    Example: Standing Left-Arm Shoulder Abduction.
    Mirrors right-arm abduction using left shoulder (11), left elbow (13), and left hip (23).
    """
    def __init__(self):
        super().__init__(
            id="standing-left-arm-abduction",
            name="Standing Left-Arm Shoulder Abduction",
            target_joint="Left Shoulder & Elbow",
            description="Coronal plane left arm elevation from resting torso up to 90°-100° with extended elbow.",
            primary_metric_name="Left Arm Elevation",
            secondary_metric_name="Left Elbow Extension",
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
                {"index": 11, "name": "Left Shoulder", "role": "Elevation vertex"},
                {"index": 13, "name": "Left Elbow", "role": "Arm lever endpoint"},
                {"index": 23, "name": "Left Hip", "role": "Torso vertical baseline"},
                {"index": 24, "name": "Right Hip", "role": "Fallback torso baseline"},
                {"index": 15, "name": "Left Wrist", "role": "Elbow extension angle"},
            ]
        )

    def calculate_kinematics(self, landmarks):
        if not landmarks or len(landmarks) < 25:
            return {"primary_angle": 0.0, "secondary_angle": 180.0, "elevation": 0.0, "elbow": 180.0}

        shoulder = landmarks[11]
        elbow = landmarks[13]
        hip = landmarks[23] if landmarks[23].get("visibility", 0) > 0.3 else landmarks[24]
        wrist = landmarks[15]

        # Elevation angle at shoulder formed by hip -> shoulder -> elbow
        elevation = calculate_angle_2d(hip, shoulder, elbow)
        # Elbow angle at elbow formed by shoulder -> elbow -> wrist
        elbow_angle = calculate_angle_2d(shoulder, elbow, wrist)

        return {
            "primary_angle": round(float(elevation), 1),
            "secondary_angle": round(float(elbow_angle), 1),
            "elevation": round(float(elevation), 1),
            "elbow": round(float(elbow_angle), 1)
        }

    def validate_tracking(self, landmarks, min_visibility=0.30):
        if not landmarks or len(landmarks) < 25:
            return {"has_required": False, "reason": "Insufficient landmarks"}

        sh_score = get_landmark_score(landmarks[11])
        elb_score = get_landmark_score(landmarks[13])
        hip_score = max(get_landmark_score(landmarks[23]), get_landmark_score(landmarks[24]))

        has_required = (sh_score >= min_visibility and elb_score >= min_visibility and hip_score >= min_visibility)
        reason = None
        if not has_required:
            missing = []
            if sh_score < min_visibility: missing.append("Left Shoulder")
            if elb_score < min_visibility: missing.append("Left Elbow")
            if hip_score < min_visibility: missing.append("Torso Hip")
            reason = f"Low tracking confidence for {', '.join(missing)}"

        return {
            "has_required": has_required,
            "shoulder_score": sh_score,
            "elbow_score": elb_score,
            "hip_score": hip_score,
            "reason": reason
        }

# Step 2: Register with global registry
exercise_registry.register(StandingLeftArmAbductionStrategy())
```

Once registered, `GET /api/exercises` and `GET /api/health` immediately advertise the new exercise, and `database.py` will automatically seed it into the SQLite `exercises` table on boot.

---

## 3. Frontend Registration (`frontend/src/services/exerciseRegistry.ts`)

Mirror the configuration in the TypeScript registry:

```typescript
// frontend/src/services/exerciseRegistry.ts
export const EXERCISE_REGISTRY: Record<string, ExerciseStrategyConfig> = {
  // Existing exercises...
  'standing-left-arm-abduction': {
    id: 'standing-left-arm-abduction',
    name: 'Standing Left-Arm Shoulder Abduction',
    target_joint: 'Left Shoulder & Elbow',
    description: 'Coronal plane left arm elevation from resting torso up to 90°-100° with extended elbow.',
    primary_metric_name: 'Left Arm Elevation',
    secondary_metric_name: 'Left Elbow Extension',
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
      { index: 11, name: 'Left Shoulder', role: 'Elevation vertex' },
      { index: 13, name: 'Left Elbow', role: 'Arm lever endpoint' },
      { index: 23, name: 'Left Hip', role: 'Torso vertical baseline' },
      { index: 24, name: 'Right Hip', role: 'Fallback torso baseline' },
      { index: 15, name: 'Left Wrist', role: 'Elbow extension angle' },
    ],
  },
};
```

---

## 4. Tuning FSM Thresholds for New Exercises

When tuning FSM thresholds for a new movement:
- **`start_threshold`:** The angle that distinguishes an intentional movement onset from postural drift (typically $20\%\text{--}30\%$ of active ROM).
- **`raised_threshold`:** The target apex peak required to confirm full contraction/elevation (typically $80\%\text{--}90\%$ of desired ROM).
- **`return_threshold`:** The resting return angle required to complete a full repetition (typically within $10\%\text{--}15\%$ of baseline).
- **`hysteresis`:** Recommended $5.0^\circ$. Prevents false-state chatter when the user hovers near a boundary.
- **`confirmation_frames`:** Recommended 2 consecutive frames at 30 FPS to ensure optical landmark stability.
- **`min_duration_sec`:** Minimum duration (e.g. 0.5s–0.6s) to reject fast twitching or camera vibration.

---

## 5. MediaPipe Landmark Quick Reference Table

For reference when configuring new joint angles:

| Body Part | Left Index | Right Index |
|---|---|---|
| **Shoulder** | 11 | 12 |
| **Elbow** | 13 | 14 |
| **Wrist** | 15 | 16 |
| **Pinky** | 17 | 18 |
| **Index Finger** | 19 | 20 |
| **Thumb** | 21 | 22 |
| **Hip** | 23 | 24 |
| **Knee** | 25 | 26 |
| **Ankle** | 27 | 28 |
| **Heel** | 29 | 30 |
| **Foot Index** | 31 | 32 |
