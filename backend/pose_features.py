import math
from typing import Dict, List, Tuple, Optional, Any

def calculate_angle_2d(p1: Dict[str, float], p2: Dict[str, float], p3: Dict[str, float]) -> float:
    """
    Calculate the interior angle at vertex p2 formed by lines (p2 -> p1) and (p2 -> p3).
    Returns degrees in [0, 180].
    """
    v1_x = p1["x"] - p2["x"]
    v1_y = p1["y"] - p2["y"]
    v2_x = p3["x"] - p2["x"]
    v2_y = p3["y"] - p2["y"]

    dot = v1_x * v2_x + v1_y * v2_y
    mag1 = math.sqrt(v1_x * v1_x + v1_y * v1_y)
    mag2 = math.sqrt(v2_x * v2_x + v2_y * v2_y)

    if mag1 < 1e-7 or mag2 < 1e-7:
        return 0.0

    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_theta))

def calculate_arm_elevation(right_shoulder: Dict[str, float],
                            right_elbow: Dict[str, float],
                            right_hip: Dict[str, float]) -> float:
    """
    Calculate right-arm shoulder elevation angle relative to the ipsilateral torso.
    Torso vector: Right Shoulder -> Right Hip (pointing downward).
    Arm vector: Right Shoulder -> Right Elbow.
    When arm hangs down alongside torso, angle is near 0-15 degrees.
    When abducted horizontally, angle is near 90 degrees.
    When elevated overhead, angle approaches 160-180 degrees.
    """
    return calculate_angle_2d(right_hip, right_shoulder, right_elbow)

def calculate_elbow_angle(right_shoulder: Dict[str, float],
                          right_elbow: Dict[str, float],
                          right_wrist: Dict[str, float]) -> float:
    """
    Calculate right elbow angle (extension vs flexion).
    Vertex is right_elbow.
    Full extension: ~170-180 degrees.
    90-degree flexion: ~90 degrees.
    """
    return calculate_angle_2d(right_shoulder, right_elbow, right_wrist)

def calculate_trunk_lean(mid_shoulder: Dict[str, float], mid_hip: Dict[str, float]) -> float:
    """
    Calculate lateral trunk lean angle relative to vertical.
    Vertical vector in image coordinates points straight down (dx=0, dy>0).
    Torso vector: mid_shoulder -> mid_hip.
    """
    dx = mid_hip["x"] - mid_shoulder["x"]
    dy = mid_hip["y"] - mid_shoulder["y"]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-7:
        return 0.0
    
    cos_theta = max(-1.0, min(1.0, dy / length))
    return math.degrees(math.acos(cos_theta))

def get_landmark_score(lm: Optional[Dict[str, Any]]) -> float:
    """Extract confidence score from MediaPipe landmark (evaluating visibility and presence)."""
    if lm is None:
        return 0.0
    vis = lm.get("visibility")
    pres = lm.get("presence")
    if vis is not None and pres is not None:
        return min(float(vis), float(pres))
    if vis is not None:
        return float(vis)
    if pres is not None:
        return float(pres)
    return 1.0

def validate_right_arm_landmarks(landmarks: List[Dict[str, Any]], min_visibility: float = 0.30) -> Dict[str, Any]:
    """
    Validate tracking suitability for right-arm shoulder abduction.
    Required landmarks:
      - Right Shoulder (index 12)
      - Right Elbow (index 14)
      - Torso reference: Right Hip (24) or Left Hip (23)
    Auxiliary landmarks:
      - Right Wrist (index 16): used for elbow flexion only; low score does NOT invalidate elevation tracking.
    """
    if not landmarks or len(landmarks) < 25:
        return {
            "has_required": False,
            "shoulder_score": 0.0,
            "elbow_score": 0.0,
            "hip_score": 0.0,
            "r_hip_score": 0.0,
            "l_hip_score": 0.0,
            "wrist_score": 0.0,
            "wrist_valid": False,
            "reason": "Insufficient pose landmarks (< 25 points detected)"
        }

    def get_score(idx: int) -> float:
        if idx >= len(landmarks):
            return 0.0
        return get_landmark_score(landmarks[idx])

    sh_score = get_score(12)
    elb_score = get_score(14)
    r_hip_score = get_score(24)
    l_hip_score = get_score(23)
    wrist_score = get_score(16)
    hip_score = max(r_hip_score, l_hip_score)

    reasons = []
    if sh_score < min_visibility:
        reasons.append(f"Right shoulder (12) score {sh_score:.2f} < {min_visibility:.2f}")
    if elb_score < min_visibility:
        reasons.append(f"Right elbow (14) score {elb_score:.2f} < {min_visibility:.2f}")
    if hip_score < min_visibility:
        reasons.append(f"Torso hip (24/23) score {hip_score:.2f} < {min_visibility:.2f}")

    has_required = len(reasons) == 0
    wrist_valid = wrist_score >= min_visibility

    return {
        "has_required": has_required,
        "shoulder_score": round(sh_score, 2),
        "elbow_score": round(elb_score, 2),
        "hip_score": round(hip_score, 2),
        "r_hip_score": round(r_hip_score, 2),
        "l_hip_score": round(l_hip_score, 2),
        "wrist_score": round(wrist_score, 2),
        "wrist_valid": wrist_valid,
        "reason": "; ".join(reasons) if reasons else None
    }

class AngleSmoother:
    """
    Exponential moving average smoother for landmark angles to reduce camera noise
    without dampening genuine movement trajectory.
    """
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self.last_value: Optional[float] = None

    def update(self, value: float) -> float:
        if self.last_value is None:
            self.last_value = value
            return value
        smoothed = self.alpha * value + (1.0 - self.alpha) * self.last_value
        self.last_value = smoothed
        return smoothed

    def reset(self):
        self.last_value = None

class RepetitionCounterFSM:
    """
    Finite State Machine for Standing Right-Arm Shoulder Abduction.
    
    State Progression:
      RESTING -> RAISING -> RAISED -> LOWERING -> RESTING
      
    Features:
      - Validates arm elevation (degrees from vertical trunk line)
      - Consecutive confirmation frames and hysteresis to reject jitter
      - Safe tracking-loss handling (aborts without false repetition count)
      - Real-time instructional feedback and diagnostic metadata
    """
    def __init__(self,
                 start_threshold: float = 30.0,
                 raised_threshold: float = 80.0,
                 return_threshold: float = 25.0,
                 hysteresis: float = 5.0,
                 min_duration_sec: float = 0.6,
                 confirmation_frames: int = 2,
                 max_tracking_loss_sec: float = 1.5,
                 peak_threshold: Optional[float] = None):
        # Support peak_threshold as alias for raised_threshold for backwards compatibility
        if peak_threshold is not None:
            raised_threshold = peak_threshold

        self.start_threshold = float(start_threshold)
        self.raised_threshold = float(raised_threshold)
        self.return_threshold = float(return_threshold)
        self.hysteresis = float(hysteresis)
        self.min_duration_sec = float(min_duration_sec)
        self.confirmation_frames = int(confirmation_frames)
        self.max_tracking_loss_sec = float(max_tracking_loss_sec)

        # FSM State
        self.state = "RESTING"
        self.previous_state = "RESTING"
        self.rep_count = 0
        self.current_rep_start_time: Optional[float] = None
        self.current_rep_peak_elevation = 0.0
        self.current_rep_min_elbow = 180.0
        self.rep_history: List[Dict[str, Any]] = []

        # Buffer for multi-frame transition confirmation
        self._pending_state: Optional[str] = None
        self._pending_frames = 0
        self._tracking_loss_start_time: Optional[float] = None
        self.last_transition_reason = "Initial resting position"

    def get_instruction(self) -> str:
        instructions = {
            "RESTING": "Stand upright with arm at your side. Begin by raising right arm outward.",
            "RAISING": "Raising: Continue lifting arm toward shoulder height (90°).",
            "RAISED": "Apex reached! Hold briefly, then lower arm slowly.",
            "LOWERING": "Lowering: Return arm smoothly back down to your side."
        }
        return instructions.get(self.state, "Perform right-arm shoulder abduction.")

    def _transition_to(self, new_state: str, reason: str):
        self.previous_state = self.state
        self.state = new_state
        self.last_transition_reason = reason
        self._pending_state = None
        self._pending_frames = 0

    def _check_candidate_transition(self, candidate: str, reason: str) -> bool:
        if self._pending_state == candidate:
            self._pending_frames += 1
        else:
            self._pending_state = candidate
            self._pending_frames = 1

        if self._pending_frames >= self.confirmation_frames:
            self._transition_to(candidate, reason)
            return True
        return False

    def update(self, elevation: float, elbow: float, timestamp: float, has_tracking: bool = True) -> Tuple[int, str, Optional[Dict[str, Any]]]:
        """
        Process a new frame and return (rep_count, state, completed_rep_info_if_any).
        """
        completed_rep = None

        # 1. Handle Lost Tracking Safely
        if not has_tracking:
            if self._tracking_loss_start_time is None:
                self._tracking_loss_start_time = timestamp
            lost_duration = timestamp - self._tracking_loss_start_time

            # If tracking lost for > max_tracking_loss_sec during active rep, reset to RESTING without counting
            if self.state != "RESTING" and lost_duration >= self.max_tracking_loss_sec:
                self._transition_to(
                    "RESTING",
                    f"Tracking lost for {lost_duration:.1f}s; safely reset to RESTING without counting."
                )
                self.current_rep_start_time = None
                self.current_rep_peak_elevation = 0.0
                self.current_rep_min_elbow = 180.0

            return self.rep_count, self.state, None

        # Tracking is valid
        self._tracking_loss_start_time = None

        # 2. State Machine Transitions
        if self.state == "RESTING":
            if elevation >= (self.start_threshold + self.hysteresis):
                if self._check_candidate_transition("RAISING", f"Arm raised above start threshold ({elevation:.1f}° >= {self.start_threshold + self.hysteresis:.1f}°)"):
                    self.current_rep_start_time = timestamp
                    self.current_rep_peak_elevation = elevation
                    self.current_rep_min_elbow = elbow
            else:
                self._pending_state = None
                self._pending_frames = 0

        elif self.state == "RAISING":
            if elevation > self.current_rep_peak_elevation:
                self.current_rep_peak_elevation = elevation
            if elbow < self.current_rep_min_elbow:
                self.current_rep_min_elbow = elbow

            # Target apex reached
            if elevation >= self.raised_threshold:
                self._check_candidate_transition("RAISED", f"Reached apex elevation ({elevation:.1f}° >= {self.raised_threshold:.1f}°)")
            # Incomplete movement aborted before reaching apex
            elif elevation <= (self.start_threshold - self.hysteresis):
                if self._check_candidate_transition("RESTING", f"Movement aborted: dropped before apex ({elevation:.1f}° <= {self.start_threshold - self.hysteresis:.1f}°)"):
                    self.current_rep_start_time = None
                    self.current_rep_peak_elevation = 0.0
                    self.current_rep_min_elbow = 180.0
            else:
                self._pending_state = None
                self._pending_frames = 0

        elif self.state == "RAISED":
            if elevation > self.current_rep_peak_elevation:
                self.current_rep_peak_elevation = elevation
            if elbow < self.current_rep_min_elbow:
                self.current_rep_min_elbow = elbow

            # Descent started from apex
            descent_thresh = min(self.current_rep_peak_elevation - 8.0, self.raised_threshold - self.hysteresis)
            if elevation <= descent_thresh:
                self._check_candidate_transition("LOWERING", f"Descent started from apex ({elevation:.1f}° <= {descent_thresh:.1f}°)")
            else:
                self._pending_state = None
                self._pending_frames = 0

        elif self.state == "LOWERING":
            if elbow < self.current_rep_min_elbow:
                self.current_rep_min_elbow = elbow

            # Return to starting resting angle
            if elevation <= (self.return_threshold + self.hysteresis):
                duration = timestamp - (self.current_rep_start_time or timestamp)
                if self._check_candidate_transition("RESTING", f"Returned to resting angle ({elevation:.1f}° <= {self.return_threshold + self.hysteresis:.1f}°)"):
                    if duration >= self.min_duration_sec and self.current_rep_peak_elevation >= (self.raised_threshold - self.hysteresis):
                        self.rep_count += 1
                        completed_rep = {
                            "rep_number": self.rep_count,
                            "duration": round(duration, 2),
                            "peak_elevation": round(self.current_rep_peak_elevation, 1),
                            "min_elbow": round(self.current_rep_min_elbow, 1),
                            "rom": round(self.current_rep_peak_elevation - self.return_threshold, 1),
                            "completed_at": round(timestamp, 2)
                        }
                        self.rep_history.append(completed_rep)
                        self.last_transition_reason = f"Rep #{self.rep_count} complete! Duration: {completed_rep['duration']}s, Peak: {completed_rep['peak_elevation']}°"
                    else:
                        self.last_transition_reason = f"Movement completed but discarded (duration: {duration:.1f}s, peak: {self.current_rep_peak_elevation:.1f}°)"

                    self.current_rep_start_time = None
                    self.current_rep_peak_elevation = 0.0
                    self.current_rep_min_elbow = 180.0
            else:
                self._pending_state = None
                self._pending_frames = 0

        return self.rep_count, self.state, completed_rep

    def reset(self):
        self.state = "RESTING"
        self.previous_state = "RESTING"
        self.rep_count = 0
        self.current_rep_start_time = None
        self.current_rep_peak_elevation = 0.0
        self.current_rep_min_elbow = 180.0
        self.rep_history = []
        self._pending_state = None
        self._pending_frames = 0
        self._tracking_loss_start_time = None
        self.last_transition_reason = "Reset to initial state"
