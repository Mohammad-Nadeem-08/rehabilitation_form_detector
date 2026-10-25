import os
import sys
import json
import pytest
import numpy as np

# Ensure backend directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from pose_features import (
    calculate_angle_2d, calculate_arm_elevation, calculate_elbow_angle,
    calculate_trunk_lean, validate_right_arm_landmarks, AngleSmoother, RepetitionCounterFSM
)
from movement_comparison import (
    compute_dtw_alignment, resample_sequence, compare_angle_trajectories
)
from inference import lstm_engine, RehabLSTM
from database import init_db, get_db_connection
from reporting import generate_session_pdf

def test_landmark_tracking_validity_all_required_present():
    # 33 landmarks with standard visibility
    landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.85, "presence": 0.90} for _ in range(33)]
    res = validate_right_arm_landmarks(landmarks, min_visibility=0.30)
    assert res["has_required"] is True
    assert res["wrist_valid"] is True
    assert res["shoulder_score"] >= 0.85
    assert res["elbow_score"] >= 0.85
    assert res["reason"] is None

def test_landmark_tracking_validity_missing_shoulder():
    landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.85} for _ in range(33)]
    landmarks[12] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.15}  # Shoulder below 0.30
    res = validate_right_arm_landmarks(landmarks, min_visibility=0.30)
    assert res["has_required"] is False
    assert "Right shoulder (12)" in res["reason"]

def test_landmark_tracking_validity_missing_elbow():
    landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.85} for _ in range(33)]
    landmarks[14] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.10}  # Elbow below 0.30
    res = validate_right_arm_landmarks(landmarks, min_visibility=0.30)
    assert res["has_required"] is False
    assert "Right elbow (14)" in res["reason"]

def test_landmark_tracking_validity_unrelated_missing_wrist():
    # Crucial test for the bug: wrist occluded or out of frame (score 0.10 < 0.30)
    # Right shoulder (12), elbow (14), hip (24) are all well tracked (> 0.80)
    landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.85} for _ in range(33)]
    landmarks[16] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.08}  # Wrist missing/blurry
    res = validate_right_arm_landmarks(landmarks, min_visibility=0.30)
    # Unrelated wrist missing must NOT mark elevation tracking as lost!
    assert res["has_required"] is True
    assert res["wrist_valid"] is False
    assert res["reason"] is None

def test_landmark_tracking_validity_hip_fallback():
    # Right hip (24) is occluded (score 0.12), but Left hip (23) is visible (score 0.85)
    landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.85} for _ in range(33)]
    landmarks[24] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.12}
    landmarks[23] = {"x": 0.5, "y": 0.7, "z": 0.0, "visibility": 0.88}
    res = validate_right_arm_landmarks(landmarks, min_visibility=0.30)
    assert res["has_required"] is True
    assert res["hip_score"] >= 0.85

def test_landmark_tracking_validity_both_hips_missing():
    landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.85} for _ in range(33)]
    landmarks[24] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.10}
    landmarks[23] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.15}
    res = validate_right_arm_landmarks(landmarks, min_visibility=0.30)
    assert res["has_required"] is False
    assert "Torso hip (24/23)" in res["reason"]

def test_fsm_with_high_elevation_angles_and_unrelated_wrist_loss():
    # Simulates user's reported symptom:
    # Right arm elevates to 94°, 106.5°, 103.4°, where wrist visibility drops,
    # but tracking validity remains True for elevation, completing the repetition cycle.
    fsm = RepetitionCounterFSM(
        start_threshold=30.0,
        raised_threshold=80.0,
        return_threshold=25.0,
        min_duration_sec=0.5,
        confirmation_frames=2,
        hysteresis=5.0
    )

    trajectory = [
        # (elevation, elbow, timestamp, has_tracking)
        (15.0, 175.0, 0.0, True),
        (20.0, 175.0, 0.1, True),
        (45.0, 172.0, 0.3, True),  # Crossing start 35° -> RAISING
        (65.0, 170.0, 0.5, True),
        (94.0, 168.0, 0.7, True),  # Crossing apex 80° -> RAISED (user reported angle: 94°)
        (106.5, 165.0, 0.8, True), # Peak apex (user reported angle: 106.5°)
        (103.4, 166.0, 0.9, True), # Peak apex (user reported angle: 103.4°)
        (75.0, 170.0, 1.1, True),  # Descent initiated -> LOWERING
        (50.0, 172.0, 1.3, True),
        (22.0, 175.0, 1.5, True),  # Return below 30° -> RESTING
        (18.0, 175.0, 1.6, True),
    ]

    reps = 0
    states = []
    for elev, elb, ts, track in trajectory:
        reps, state, comp = fsm.update(elev, elb, ts, has_tracking=track)
        if not states or states[-1] != state:
            states.append(state)

    assert states == ["RESTING", "RAISING", "RAISED", "LOWERING", "RESTING"]
    assert reps == 1, f"Expected 1 repetition, got {reps}"
    assert fsm.rep_history[0]["peak_elevation"] >= 106.0

def test_joint_angle_calculation():
    # Right-angle test: (0, 1) -> (0, 0) -> (1, 0) should be 90 degrees
    p1 = {"x": 0.0, "y": 1.0}
    p2 = {"x": 0.0, "y": 0.0}
    p3 = {"x": 1.0, "y": 0.0}
    angle = calculate_angle_2d(p1, p2, p3)
    assert abs(angle - 90.0) < 1e-4

    # Straight line: 180 degrees
    p4 = {"x": -1.0, "y": 0.0}
    angle_straight = calculate_angle_2d(p3, p2, p4)
    assert abs(angle_straight - 180.0) < 1e-4

def test_arm_elevation_and_elbow_angles():
    # Shoulder at (0.5, 0.3), Hip at (0.5, 0.7), Elbow at (0.8, 0.3) -> 90 deg elevation
    shoulder = {"x": 0.5, "y": 0.3}
    hip = {"x": 0.5, "y": 0.7}
    elbow_abducted = {"x": 0.8, "y": 0.3}
    elev = calculate_arm_elevation(shoulder, elbow_abducted, hip)
    assert abs(elev - 90.0) < 1e-3

    # Extended elbow with wrist along same line
    wrist = {"x": 1.1, "y": 0.3}
    elbow_angle = calculate_elbow_angle(shoulder, elbow_abducted, wrist)
    assert abs(elbow_angle - 180.0) < 1e-3

def test_angle_smoother_and_missing_landmarks():
    smoother = AngleSmoother(alpha=0.6)
    v1 = smoother.update(100.0)
    assert v1 == 100.0
    v2 = smoother.update(120.0)
    # 0.6 * 120 + 0.4 * 100 = 72 + 40 = 112.0
    assert abs(v2 - 112.0) < 1e-4

    # Missing landmarks handled safely (zero length vectors return 0.0)
    zero_p = {"x": 0.0, "y": 0.0}
    safe_angle = calculate_angle_2d(zero_p, zero_p, zero_p)
    assert safe_angle == 0.0

def test_repetition_counting_fsm_full_cycle():
    fsm = RepetitionCounterFSM(
        start_threshold=30.0,
        raised_threshold=80.0,
        return_threshold=25.0,
        min_duration_sec=0.5,
        confirmation_frames=2,
        hysteresis=5.0
    )

    timestamps = np.linspace(0.0, 2.0, 50)
    # Generate elevation curve: rises smoothly from 15 deg to 90 deg, descends to 15 deg
    elevations = 15.0 + 75.0 * np.sin(np.linspace(0, np.pi, 50))

    states_seen = []
    reps = 0
    for t, e in zip(timestamps, elevations):
        reps, state, comp = fsm.update(float(e), 175.0, float(t), has_tracking=True)
        if not states_seen or states_seen[-1] != state:
            states_seen.append(state)

    # Must follow RESTING -> RAISING -> RAISED -> LOWERING -> RESTING
    assert states_seen == ["RESTING", "RAISING", "RAISED", "LOWERING", "RESTING"], f"States seen: {states_seen}"
    assert reps == 1, f"Expected 1 repetition, counted {reps}"
    assert len(fsm.rep_history) == 1
    assert fsm.rep_history[0]["peak_elevation"] >= 85.0
    assert fsm.rep_history[0]["duration"] >= 0.5

def test_repetition_counting_fsm_incomplete_movement():
    fsm = RepetitionCounterFSM(
        start_threshold=30.0,
        raised_threshold=80.0,
        return_threshold=25.0,
        confirmation_frames=2
    )

    timestamps = np.linspace(0.0, 2.0, 40)
    # Rises only to 50 deg (never reaching raised_threshold 80 deg) and lowers back down
    elevations = 15.0 + 35.0 * np.sin(np.linspace(0, np.pi, 40))

    reps = 0
    for t, e in zip(timestamps, elevations):
        reps, state, comp = fsm.update(float(e), 175.0, float(t), has_tracking=True)

    # Incomplete rep must not increment count
    assert reps == 0, f"Expected 0 repetitions for incomplete movement, got {reps}"
    assert fsm.state == "RESTING"

def test_repetition_counting_fsm_noisy_angles():
    fsm = RepetitionCounterFSM(
        start_threshold=30.0,
        raised_threshold=80.0,
        return_threshold=25.0,
        confirmation_frames=2,
        hysteresis=5.0
    )

    # Resting arm with small noise jittering around 32-34 deg (near start threshold)
    np.random.seed(42)
    resting_noise = 25.0 + np.random.uniform(-4, 4, 30)
    for i, e in enumerate(resting_noise):
        reps, state, _ = fsm.update(float(e), 175.0, float(i * 0.05), has_tracking=True)
        # Should remain in RESTING due to hysteresis and confirmation
        assert state == "RESTING"
        assert reps == 0

def test_repetition_counting_fsm_repeated_frames():
    fsm = RepetitionCounterFSM(
        start_threshold=30.0,
        raised_threshold=80.0,
        return_threshold=25.0,
        confirmation_frames=2
    )

    # 50 repeated static frames at apex 88 deg
    for i in range(50):
        reps, state, _ = fsm.update(88.0, 175.0, float(i * 0.05), has_tracking=True)
        assert reps == 0
    assert fsm.state in ["RAISING", "RAISED"]

def test_repetition_counting_fsm_tracking_loss():
    fsm = RepetitionCounterFSM(
        start_threshold=30.0,
        raised_threshold=80.0,
        return_threshold=25.0,
        max_tracking_loss_sec=1.0,
        confirmation_frames=2
    )

    # Raise arm to 85 deg (reaches RAISED)
    for i in range(10):
        fsm.update(40.0 + i * 5.0, 175.0, float(i * 0.1), has_tracking=True)
    assert fsm.state in ["RAISING", "RAISED"]

    # Tracking lost for 2.0s (> max_tracking_loss_sec of 1.0s)
    for i in range(10, 30):
        reps, state, _ = fsm.update(0.0, 180.0, float(i * 0.1), has_tracking=False)

    # Must safely reset to RESTING without counting a repetition
    assert reps == 0, f"Tracking loss must NOT increment rep count! Got {reps}"
    assert fsm.state == "RESTING"

def test_sequence_resampling():
    orig = [10.0, 20.0, 30.0]
    resampled = resample_sequence(orig, 5)
    assert len(resampled) == 5
    assert resampled[0] == 10.0
    assert resampled[-1] == 30.0

def test_dtw_comparison_identical_sequences():
    seq = [15.0, 45.0, 90.0, 45.0, 15.0]
    dist, path = compute_dtw_alignment(np.array(seq), np.array(seq))
    assert dist == 0.0
    assert len(path) == len(seq)

def test_synthetic_known_joint_positions():
    """
    Test known synthetic joint positions:
    1. Arm-down (resting vertical): elevation 0.0°, elbow 180.0° (extended)
    2. Horizontal abduction: elevation 90.0°, elbow 180.0°
    3. Elevated overhead arm: elevation 180.0°, elbow 180.0°
    4. Bent elbow (90° flexion): elbow 90.0°
    5. Bent elbow (30° flexion): elbow 30.0°
    """
    shoulder = {"x": 0.5, "y": 0.3}
    hip = {"x": 0.5, "y": 0.7}

    # 1. Arm-down: hanging straight down alongside torso
    elbow_down = {"x": 0.5, "y": 0.6}
    wrist_down = {"x": 0.5, "y": 0.9}
    elev_down = calculate_arm_elevation(shoulder, elbow_down, hip)
    elbow_down_ang = calculate_elbow_angle(shoulder, elbow_down, wrist_down)
    assert abs(elev_down - 0.0) < 1e-4, f"Expected 0.0°, got {elev_down}°"
    assert abs(elbow_down_ang - 180.0) < 1e-4, f"Expected 180.0°, got {elbow_down_ang}°"

    # 2. Horizontal abduction: arm extended to side at shoulder level (90°)
    elbow_horiz = {"x": 0.8, "y": 0.3}
    wrist_horiz = {"x": 1.1, "y": 0.3}
    elev_horiz = calculate_arm_elevation(shoulder, elbow_horiz, hip)
    elbow_horiz_ang = calculate_elbow_angle(shoulder, elbow_horiz, wrist_horiz)
    assert abs(elev_horiz - 90.0) < 1e-4, f"Expected 90.0°, got {elev_horiz}°"
    assert abs(elbow_horiz_ang - 180.0) < 1e-4, f"Expected 180.0°, got {elbow_horiz_ang}°"

    # 3. Elevated overhead arm: pointing straight up (180°)
    elbow_overhead = {"x": 0.5, "y": 0.0}
    wrist_overhead = {"x": 0.5, "y": -0.3}
    elev_overhead = calculate_arm_elevation(shoulder, elbow_overhead, hip)
    elbow_overhead_ang = calculate_elbow_angle(shoulder, elbow_overhead, wrist_overhead)
    assert abs(elev_overhead - 180.0) < 1e-4, f"Expected 180.0°, got {elev_overhead}°"
    assert abs(elbow_overhead_ang - 180.0) < 1e-4, f"Expected 180.0°, got {elbow_overhead_ang}°"

    # 4. Bent elbow at 90° flexion: humerus horizontal, forearm vertical
    wrist_bent_90 = {"x": 0.8, "y": 0.0}
    elbow_bent_90_ang = calculate_elbow_angle(shoulder, elbow_horiz, wrist_bent_90)
    assert abs(elbow_bent_90_ang - 90.0) < 1e-4, f"Expected 90.0°, got {elbow_bent_90_ang}°"

    # 5. Acutely bent elbow at 30° flexion
    # Humerus along negative x (-0.3, 0); forearm at 30° relative to humerus
    wrist_bent_30 = {
        "x": 0.8 - 0.3 * np.cos(np.radians(30.0)),
        "y": 0.3 - 0.3 * np.sin(np.radians(30.0))
    }
    elbow_bent_30_ang = calculate_elbow_angle(shoulder, elbow_horiz, wrist_bent_30)
    assert abs(elbow_bent_30_ang - 30.0) < 1e-3, f"Expected 30.0°, got {elbow_bent_30_ang}°"

def test_reference_versus_identical_trajectory():
    """
    Comparing a trajectory against itself must produce:
    100% similarity score, 0.0° mean deviation, 0.0° RMSE.
    """
    t = np.linspace(0, np.pi, 45)
    traj = [
        {"elevation": float(15.0 + 80.0 * np.sin(x)), "elbow": 172.0, "timestamp": float(i * 0.05), "confidence": 0.95}
        for i, x in enumerate(t)
    ]
    res = compare_angle_trajectories(traj, traj)
    assert res["status"] == "success"
    assert res["comparison_score"] == 100.0, f"Expected 100.0, got {res['comparison_score']}"
    assert res["mean_elevation_deviation"] == 0.0
    assert res["max_elevation_deviation"] == 0.0
    assert res["mean_elbow_deviation"] == 0.0
    assert res["rmse"] == 0.0
    assert len(res["warnings"]) == 0

def test_multi_rep_session_against_single_rep_reference():
    """
    Verify that comparing a multi-repetition workout session against a single-repetition
    reference template extracts the canonical cycle cleanly, eliminating phase-shift errors.
    """
    # 1. Single-rep reference template (peak 95°, extended elbow 172°)
    t_ref = np.linspace(0, np.pi, 64)
    ref = [
        {"elevation": float(15.0 + 80.0 * np.sin(x)), "elbow": 172.0, "timestamp": float(i * 0.05), "confidence": 0.95}
        for i, x in enumerate(t_ref)
    ]

    # 2. 4-repetition workout session with setup idle period and clean repetitions
    sess = []
    # 20 idle frames at start (~15°)
    for i in range(20):
        sess.append({"elevation": 15.0, "elbow": 175.0, "timestamp": round(i * 0.05, 2), "confidence": 0.95})
    
    # 4 repetitions of shoulder abduction (peaks ~93°–97°)
    cur_t = 1.0
    for rep in range(4):
        peak_var = 78.0 + (rep % 2) * 4.0
        for x in np.linspace(0, np.pi, 30):
            sess.append({
                "elevation": float(15.0 + peak_var * np.sin(x)),
                "elbow": float(170.0 - 5.0 * np.sin(x)),
                "timestamp": round(cur_t, 2),
                "confidence": 0.95
            })
            cur_t += 0.05
        # Inter-rep rest
        for _ in range(5):
            sess.append({"elevation": 15.0, "elbow": 175.0, "timestamp": round(cur_t, 2), "confidence": 0.95})
            cur_t += 0.05

    res = compare_angle_trajectories(ref, sess)
    assert res["status"] == "success"
    assert res["comparison_score"] >= 88.0, f"Expected similarity >= 88%, got {res['comparison_score']}%"
    assert res["mean_elevation_deviation"] < 8.0, f"Expected mean dev < 8.0°, got {res['mean_elevation_deviation']}°"
    assert res["rmse"] < 10.0, f"Expected RMSE < 10.0°, got {res['rmse']}°"
    assert len(res["warnings"]) == 0

def test_reach_artifact_rejection_in_trajectory_comparison():
    """
    Verify that an end-of-session reach-to-stop movement (150° elevation, 15° collapsed elbow)
    is properly rejected from the exercise repetition cycle and does not trigger false warnings.
    """
    t_ref = np.linspace(0, np.pi, 64)
    ref = [
        {"elevation": float(15.0 + 80.0 * np.sin(x)), "elbow": 172.0, "timestamp": float(i * 0.05), "confidence": 0.95}
        for i, x in enumerate(t_ref)
    ]

    sess = []
    cur_t = 0.0
    # 3 clean exercise repetitions
    for rep in range(3):
        for x in np.linspace(0, np.pi, 30):
            sess.append({
                "elevation": float(16.0 + 80.0 * np.sin(x)),
                "elbow": 170.0,
                "timestamp": round(cur_t, 2),
                "confidence": 0.95
            })
            cur_t += 0.05
        for _ in range(4):
            sess.append({"elevation": 16.0, "elbow": 172.0, "timestamp": round(cur_t, 2), "confidence": 0.95})
            cur_t += 0.05

    # Trailing reach-for-mouse artifact: elevation 150°, elbow collapsed to 15°
    for x in np.linspace(0, np.pi, 20):
        sess.append({
            "elevation": float(16.0 + 134.0 * np.sin(x)),
            "elbow": float(170.0 - 155.0 * np.sin(x)), # drops to 15°
            "timestamp": round(cur_t, 2),
            "confidence": 0.95
        })
        cur_t += 0.05

    res = compare_angle_trajectories(ref, sess)
    assert res["status"] == "success"
    # Reach artifact must NOT degrade score to ~60% or cause 55° RMSE
    assert res["comparison_score"] >= 85.0, f"Expected score >= 85%, got {res['comparison_score']}%"
    assert res["mean_elevation_deviation"] < 8.0, f"Expected mean dev < 8.0°, got {res['mean_elevation_deviation']}°"
    assert res["rmse"] < 10.0, f"Expected RMSE < 10.0°, got {res['rmse']}°"
    # Must NOT produce false elbow bending warning for the reach
    elbow_warnings = [w for w in res["warnings"] if "Significant elbow bending" in w]
    assert len(elbow_warnings) == 0, f"Unexpected elbow warning: {elbow_warnings}"

def test_movement_comparison_metrics():
    t = np.linspace(0, np.pi, 30)
    ref = [{"elevation": float(15.0 + 80.0 * np.sin(x)), "elbow": 175.0, "confidence": 0.95} for x in t]
    home = [{"elevation": float(15.0 + 78.0 * np.sin(x)), "elbow": 170.0, "confidence": 0.95} for x in t]

    res = compare_angle_trajectories(ref, home)
    assert res["status"] == "success"
    assert res["comparison_score"] >= 85.0
    assert res["mean_elevation_deviation"] < 5.0
    assert len(res["aligned_reference_elevation"]) == 64

def test_movement_comparison_insufficient_data():
    ref = [{"elevation": 50.0, "elbow": 170.0}] * 30
    too_short_home = [{"elevation": 50.0, "elbow": 170.0}] * 5
    res = compare_angle_trajectories(ref, too_short_home)
    assert res["status"] == "insufficient_data"

def test_lstm_model_loading_and_inference():
    assert lstm_engine.is_loaded, f"Model failed to load: {lstm_engine.load_error}"
    assert lstm_engine.model is not None

    # Test valid sequence shape (64, 80)
    sample_seq = lstm_engine.generate_sample_mocap_repetition(is_correct=True)
    res = lstm_engine.predict_compatible_sequence(sample_seq)

    assert res["status"] == "success"
    assert res["predicted_class"] in [0, 1]
    assert "probabilities" in res
    assert abs(sum(res["probabilities"].values()) - 1.0) < 1e-3
    assert res["input_shape"] == [1, 64, 80]

def test_lstm_invalid_shape_handling():
    invalid_seq = [[0.0] * 80 for _ in range(30)] # 30 instead of 64
    res = lstm_engine.predict_compatible_sequence(invalid_seq)
    assert res["status"] == "invalid_shape"

def test_database_persistence():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM exercises")
    ex_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM references_table")
    ref_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM sessions")
    sess_count = cursor.fetchone()[0]
    conn.close()

    assert ex_count >= 1
    assert ref_count >= 1
    assert sess_count >= 1

def test_pdf_report_generation():
    dummy_session = {
        "id": "test-sess-123",
        "timestamp": "2026-10-25T14:00:00",
        "exercise_name": "Standing Right-Arm Shoulder Abduction",
        "duration_seconds": 12.5,
        "repetitions_completed": 3,
        "reference_id": "ref-gold-standard-demo",
        "comparison_score": 87.2,
        "model_status": "input_requires_validation",
        "angle_stats": {
            "peak_elevation": 91.5,
            "mean_elevation": 48.0,
            "min_elevation": 14.0,
            "mean_elbow": 172.0,
            "rom": 77.5
        },
        "joint_deviations": {
            "mean_elevation_deviation": 6.1,
            "max_elevation_deviation": 12.0,
            "mean_elbow_deviation": 4.5,
            "rmse": 7.2
        },
        "warnings": ["Slight lateral trunk lean observed."],
        "notes": "Test report generation output verification.",
        "raw_trajectory": [
            {"timestamp": i * 0.1, "elevation": 15 + 75 * np.sin(i * 0.2), "elbow": 170.0}
            for i in range(25)
        ]
    }

    pdf_bytes = generate_session_pdf(dummy_session)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")

def test_exercise_registry_registration_and_lookup():
    from exercise_registry import exercise_registry, StandingRightArmAbductionStrategy, SeatedRightBicepCurlStrategy
    
    assert exercise_registry.get("standing-right-arm-abduction") is not None
    assert exercise_registry.get("seated-right-bicep-curl") is not None
    
    sh_ab = exercise_registry.get("standing-right-arm-abduction")
    assert sh_ab.primary_metric_name == "Arm Elevation"
    assert sh_ab.target_rom_min == 80.0
    assert "start_threshold" in sh_ab.fsm_thresholds
    
    curl = exercise_registry.get("seated-right-bicep-curl")
    assert curl.primary_metric_name == "Elbow Flexion"
    assert curl.target_rom_min == 95.0
    assert curl.fsm_thresholds["start_threshold"] == 35.0

def test_multi_exercise_bicep_curl_kinematics():
    from exercise_registry import SeatedRightBicepCurlStrategy
    strategy = SeatedRightBicepCurlStrategy()
    
    # Synthetic landmarks: shoulder at (0.5, 0.3), elbow at (0.5, 0.6), wrist at (0.5, 0.9) -> extended arm (180 deg)
    # Flexion = 180 - 180 = 0 deg
    landmarks_ext = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9} for _ in range(33)]
    landmarks_ext[12] = {"x": 0.5, "y": 0.3, "z": 0.0, "visibility": 0.9}
    landmarks_ext[14] = {"x": 0.5, "y": 0.6, "z": 0.0, "visibility": 0.9}
    landmarks_ext[16] = {"x": 0.5, "y": 0.9, "z": 0.0, "visibility": 0.9}
    landmarks_ext[24] = {"x": 0.5, "y": 0.8, "z": 0.0, "visibility": 0.9}
    
    k_ext = strategy.calculate_kinematics(landmarks_ext)
    assert abs(k_ext["primary_angle"] - 0.0) < 1.0  # 0 deg flexion (full extension)
    
    # Curled arm: wrist at (0.5, 0.3) -> 0 deg elbow angle -> 180 deg flexion (or 90 deg when at right angle)
    landmarks_curl = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9} for _ in range(33)]
    landmarks_curl[12] = {"x": 0.5, "y": 0.3, "z": 0.0, "visibility": 0.9}
    landmarks_curl[14] = {"x": 0.5, "y": 0.6, "z": 0.0, "visibility": 0.9}
    landmarks_curl[16] = {"x": 0.8, "y": 0.6, "z": 0.0, "visibility": 0.9} # 90 deg bend
    landmarks_curl[24] = {"x": 0.5, "y": 0.8, "z": 0.0, "visibility": 0.9}
    
    k_curl = strategy.calculate_kinematics(landmarks_curl)
    # Raw elbow is 90 deg, so flexion = 180 - 90 = 90 deg
    assert abs(k_curl["primary_angle"] - 90.0) < 1.0

def test_two_reference_types_persistence_and_provenance():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Standard Reference exists
    cursor.execute("SELECT * FROM references_table WHERE id = 'ref-gold-standard-demo'")
    std_row = cursor.fetchone()
    assert std_row is not None
    assert std_row["reference_type"] == "standard"
    assert std_row["version"] == "1.0"
    assert "Protocol" in std_row["provenance"]
    
    # 2. Patient-Specific Reference exists
    cursor.execute("SELECT * FROM references_table WHERE id = 'ref-patient-101-approved'")
    pat_row = cursor.fetchone()
    assert pat_row is not None
    assert pat_row["reference_type"] == "patient_specific"
    assert pat_row["version"] == "1.1"
    assert pat_row["patient_id"] == "patient-101"
    assert "Dr. Sarah Jenkins" in pat_row["provenance"]
    conn.close()

def test_reference_isolation_by_patient():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create another patient-specific reference for patient-999
    cursor.execute("DELETE FROM references_table WHERE id = 'ref-patient-999-test'")
    cursor.execute("""
    INSERT INTO references_table (
        id, exercise_id, exercise_name, notes, duration_seconds, frame_count,
        joint_angles, landmarks_summary, created_at, is_demo,
        reference_type, version, provenance, patient_id, is_active
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "ref-patient-999-test",
        "standing-right-arm-abduction",
        "Standing Right-Arm Shoulder Abduction",
        "Patient 999 reference",
        3.0, 60, "[]", "{}", "2026-10-25T15:00:00", 0,
        "patient_specific", "1.0", "Dr. Test PT", "patient-999", 1
    ))
    conn.commit()
    
    # Query for patient-101: should get patient-101's reference + standard references, but NOT patient-999
    cursor.execute("SELECT id, patient_id, reference_type FROM references_table WHERE patient_id = ? OR reference_type = 'standard'", ("patient-101",))
    rows = cursor.fetchall()
    row_ids = [r["id"] for r in rows]
    
    assert "ref-gold-standard-demo" in row_ids
    assert "ref-patient-101-approved" in row_ids
    assert "ref-patient-999-test" not in row_ids, "Patient-specific reference for patient-999 leaked into patient-101 scope!"
    
    # Clean up test row
    cursor.execute("DELETE FROM references_table WHERE id = 'ref-patient-999-test'")
    conn.commit()
    conn.close()

def test_patient_specific_pdf_report_provenance():
    patient_session = {
        "id": "sess-patient-101-test",
        "timestamp": "2026-10-25T15:30:00",
        "exercise_name": "Standing Right-Arm Shoulder Abduction",
        "duration_seconds": 15.0,
        "repetitions_completed": 4,
        "reference_id": "ref-patient-101-approved",
        "reference_type": "patient_specific",
        "reference_version": "1.1",
        "reference_provenance": "Approved by Dr. Sarah Jenkins, PT on 2026-10-24",
        "patient_id": "patient-101",
        "comparison_score": 93.4,
        "model_status": "input_requires_validation",
        "angle_stats": {
            "peak_elevation": 86.2,
            "mean_elevation": 46.5,
            "min_elevation": 14.0,
            "mean_elbow": 171.0,
            "rom": 72.2
        },
        "joint_deviations": {
            "mean_elevation_deviation": 5.2,
            "max_elevation_deviation": 9.8,
            "mean_elbow_deviation": 3.8,
            "rmse": 5.9
        },
        "warnings": [],
        "notes": "Patient-101 session adhering closely to customized 85 deg recovery arc.",
        "raw_trajectory": [
            {"timestamp": i * 0.1, "elevation": 15 + 71 * np.sin(i * 0.2), "elbow": 171.0}
            for i in range(25)
        ]
    }
    
    pdf_bytes = generate_session_pdf(patient_session)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")


# ==============================================================================
# PHASE 9 — HYBRID ENSEMBLE ASSESSMENT PIPELINE TESTS
# ==============================================================================

from statistical_assessment import StatisticalAssessmentEngine, StatisticalAssessmentResult
from model_assessment import ModelAssessmentEngine, ModelAssessmentResult
from ensemble import EnsembleDecisionEngine, EnsembleDecisionResult
from evaluation import evaluate_cohort
from fastapi.testclient import TestClient
from main import app

def test_model_input_schema_and_feature_dimensions():
    """Verify exact (64, 80) mocap schema enforcement and shape validation."""
    engine = ModelAssessmentEngine()
    
    # 1. Correct (64, 80) shape
    valid_seq = np.zeros((64, 80), dtype=np.float32)
    res_valid = engine.evaluate_sequence(valid_seq, input_source="verified_mocap")
    assert res_valid.input_quality_status == "VALID"
    assert res_valid.input_schema == "mocap_3d_64x80"
    assert res_valid.predicted_class in [0, 1]
    assert res_valid.inference_latency_ms > 0.0

    # 2. Incompatible temporal length (30, 80)
    invalid_time = np.zeros((30, 80), dtype=np.float32)
    res_time = engine.evaluate_sequence(invalid_time, input_source="verified_mocap")
    assert res_time.input_quality_status == "INVALID_SHAPE"
    assert res_time.predicted_class is None

    # 3. Incompatible feature dimension (64, 33)
    invalid_feat = np.zeros((64, 33), dtype=np.float32)
    res_feat = engine.evaluate_sequence(invalid_feat, input_source="verified_mocap")
    assert res_feat.input_quality_status == "INVALID_SHAPE"
    assert res_feat.predicted_class is None

def test_model_numerical_integrity_nan_inf_rejection():
    """Verify non-finite and NaN numerical anomalies are rejected safely."""
    engine = ModelAssessmentEngine()
    
    # NaN in tensor
    nan_seq = np.zeros((64, 80), dtype=np.float32)
    nan_seq[10, 5] = np.nan
    res_nan = engine.evaluate_sequence(nan_seq, input_source="verified_mocap")
    assert res_nan.input_quality_status == "NUMERICAL_ANOMALY"
    assert res_nan.predicted_class is None

    # Inf in tensor
    inf_seq = np.zeros((64, 80), dtype=np.float32)
    inf_seq[20, 10] = np.inf
    res_inf = engine.evaluate_sequence(inf_seq, input_source="verified_mocap")
    assert res_inf.input_quality_status == "NUMERICAL_ANOMALY"
    assert res_inf.predicted_class is None

def test_model_domain_shift_safeguard_for_webcam():
    """Verify webcam MediaPipe landmarks are rejected from 3D mocap LSTM."""
    engine = ModelAssessmentEngine()
    
    # Even if shaped (64, 80), live webcam input source is guarded
    webcam_seq = np.zeros((64, 80), dtype=np.float32)
    res = engine.evaluate_sequence(webcam_seq, input_source="webcam_mediapipe")
    assert res.input_quality_status == "INCOMPATIBLE_REPRESENTATION"
    assert res.predicted_class is None
    assert "domain shift" in res.rejection_reason.lower()

def test_statistical_scoring_and_metric_directionality():
    """Verify biomechanical sub-scoring, directionality documentation, and range boundaries."""
    stat_engine = StatisticalAssessmentEngine()
    
    # Excellent repetition (peaks at 90°, elbow 175°, confidence 0.95)
    t = np.linspace(0, np.pi, 40)
    traj_good = [
        {
            "timestamp": round(i * 0.05, 3),
            "elevation": float(15.0 + 75.0 * np.sin(t[i])),
            "elbow": 175.0,
            "confidence": 0.95
        }
        for i in range(40)
    ]
    res_good = stat_engine.evaluate_repetition(traj_good, exercise_id="standing-right-arm-abduction", duration_sec=2.0)
    assert res_good.score >= 80.0
    assert res_good.classification_estimate == "CORRECT"
    assert res_good.quality_status == "VALID"
    assert "peak_primary_angle" in res_good.metric_directions
    assert "rmse" in res_good.metric_directions

    # Sub-score directionalities: higher RMSE is worse (closer to 0 sub-score)
    assert res_good.sub_scores["rom_criterion"] > 0.8
    assert res_good.sub_scores["secondary_posture"] > 0.8

def test_statistical_low_landmark_confidence_warning():
    """Verify low landmark confidence triggers DEGRADED quality warning."""
    stat_engine = StatisticalAssessmentEngine()
    t = np.linspace(0, np.pi, 30)
    traj_low_conf = [
        {
            "timestamp": round(i * 0.05, 3),
            "elevation": float(15.0 + 75.0 * np.sin(t[i])),
            "elbow": 170.0,
            "confidence": 0.25 # below 0.40 threshold
        }
        for i in range(30)
    ]
    res = stat_engine.evaluate_repetition(traj_low_conf, exercise_id="standing-right-arm-abduction")
    assert res.quality_status == "DEGRADED"
    assert any("tracking confidence" in w.lower() for w in res.warnings)

def test_statistical_incomplete_repetition_rejection():
    """Verify trajectory with fewer than 8 frames is rejected."""
    stat_engine = StatisticalAssessmentEngine()
    short_traj = [{"timestamp": 0.1, "elevation": 20.0, "elbow": 170.0}]
    res = stat_engine.evaluate_repetition(short_traj, exercise_id="standing-right-arm-abduction")
    assert res.quality_status == "INSUFFICIENT_FRAMES"
    assert res.score == 0.0

def test_ensemble_decision_branch_agreement():
    """Verify ensemble reaches CORRECT_ESTIMATE or INCORRECT_ESTIMATE when branches agree."""
    ens_engine = EnsembleDecisionEngine()
    
    # 1. Both branches correct
    stat_corr = StatisticalAssessmentResult(
        score=92.0,
        classification_estimate="CORRECT",
        quality_status="VALID",
        primary_metric_name="Arm Elevation",
        secondary_metric_name="Elbow Extension",
        raw_measurements={"peak_elevation": 92.0},
        sub_scores={"rom": 0.95, "dtw_similarity": 0.90},
        metric_directions={},
        warnings=[],
        exercise_id="standing-right-arm-abduction"
    )
    model_corr = ModelAssessmentResult(
        model_id="rehab-lstm-v1",
        model_version="1.0.0",
        architecture="2-Layer LSTM",
        input_schema="mocap_3d_64x80",
        input_quality_status="VALID",
        predicted_class=1,
        class_label="correct",
        probabilities={"incorrect": 0.10, "correct": 0.90},
        inference_latency_ms=3.5
    )
    res_agree = ens_engine.evaluate(stat_corr, model_corr)
    assert res_agree.decision == "CORRECT_ESTIMATE"
    assert res_agree.branch_agreement is True
    assert res_agree.combined_score is not None
    assert res_agree.combined_score >= 80.0

    # 2. Both branches incorrect
    stat_inc = StatisticalAssessmentResult(
        score=35.0,
        classification_estimate="INCORRECT",
        quality_status="VALID",
        primary_metric_name="Arm Elevation",
        secondary_metric_name="Elbow Extension",
        raw_measurements={"peak_elevation": 45.0},
        sub_scores={"rom": 0.35, "dtw_similarity": 0.40},
        metric_directions={},
        warnings=["Restricted elevation"],
        exercise_id="standing-right-arm-abduction"
    )
    model_inc = ModelAssessmentResult(
        model_id="rehab-lstm-v1",
        model_version="1.0.0",
        architecture="2-Layer LSTM",
        input_schema="mocap_3d_64x80",
        input_quality_status="VALID",
        predicted_class=0,
        class_label="incorrect",
        probabilities={"incorrect": 0.85, "correct": 0.15},
        inference_latency_ms=3.2
    )
    res_inc = ens_engine.evaluate(stat_inc, model_inc)
    assert res_inc.decision == "INCORRECT_ESTIMATE"
    assert res_inc.branch_agreement is True

def test_ensemble_decision_branch_disagreement_abstention():
    """Verify ensemble abstains with NEEDS_REVIEW when branches conflict."""
    ens_engine = EnsembleDecisionEngine()
    
    # Statistical says correct (score 90.0), Model says incorrect (P(correct)=0.10)
    stat_corr = StatisticalAssessmentResult(
        score=90.0,
        classification_estimate="CORRECT",
        quality_status="VALID",
        primary_metric_name="Arm Elevation",
        secondary_metric_name="Elbow Extension",
        raw_measurements={"peak_elevation": 90.0},
        sub_scores={"rom": 0.90},
        metric_directions={},
        warnings=[],
        exercise_id="standing-right-arm-abduction"
    )
    model_inc = ModelAssessmentResult(
        model_id="rehab-lstm-v1",
        model_version="1.0.0",
        architecture="2-Layer LSTM",
        input_schema="mocap_3d_64x80",
        input_quality_status="VALID",
        predicted_class=0,
        class_label="incorrect",
        probabilities={"incorrect": 0.90, "correct": 0.10},
        inference_latency_ms=3.0
    )
    res = ens_engine.evaluate(stat_corr, model_inc)
    assert res.decision == "NEEDS_REVIEW"
    assert res.branch_agreement is False
    assert any("disagreement" in r.lower() for r in res.reasons)

def test_ensemble_abstention_on_missing_or_incompatible_branch():
    """Verify ensemble refuses single-branch decision if one branch is missing or incompatible."""
    ens_engine = EnsembleDecisionEngine()
    stat_valid = StatisticalAssessmentResult(
        score=88.0,
        classification_estimate="CORRECT",
        quality_status="VALID",
        primary_metric_name="Arm Elevation",
        secondary_metric_name="Elbow Extension",
        raw_measurements={},
        sub_scores={},
        metric_directions={},
        warnings=[],
        exercise_id="standing-right-arm-abduction"
    )

    # 1. Missing model branch
    res_no_model = ens_engine.evaluate(stat_valid, None)
    assert res_no_model.decision == "NEEDS_REVIEW"
    assert res_no_model.branch_agreement is None
    assert "not executed" in res_no_model.explanation.lower()

    # 2. Incompatible representation model branch (webcam)
    model_incomp = ModelAssessmentResult(
        model_id="rehab-lstm-v1",
        model_version="1.0.0",
        architecture="2-Layer LSTM",
        input_schema="mocap_3d_64x80",
        input_quality_status="INCOMPATIBLE_REPRESENTATION",
        rejection_reason="Webcam MediaPipe representation requires validation"
    )
    res_incomp = ens_engine.evaluate(stat_valid, model_incomp)
    assert res_incomp.decision == "NEEDS_REVIEW"
    assert "input requires validation" in res_incomp.explanation.lower()

    # 3. Missing statistical branch
    model_valid = ModelAssessmentResult(
        model_id="rehab-lstm-v1",
        model_version="1.0.0",
        architecture="2-Layer LSTM",
        input_schema="mocap_3d_64x80",
        input_quality_status="VALID",
        predicted_class=1,
        class_label="correct",
        probabilities={"incorrect": 0.15, "correct": 0.85}
    )
    res_no_stat = ens_engine.evaluate(None, model_valid)
    assert res_no_stat.decision == "NEEDS_REVIEW"

def test_ensemble_determinism():
    """Verify identical inputs produce identical decisions and scores."""
    ens_engine = EnsembleDecisionEngine()
    stat = StatisticalAssessmentResult(
        score=85.0,
        classification_estimate="CORRECT",
        quality_status="VALID",
        primary_metric_name="Arm Elevation",
        secondary_metric_name="Elbow Extension",
        raw_measurements={},
        sub_scores={"rom": 0.85},
        metric_directions={},
        warnings=[],
        exercise_id="standing-right-arm-abduction"
    )
    model = ModelAssessmentResult(
        model_id="rehab-lstm-v1",
        model_version="1.0.0",
        architecture="2-Layer LSTM",
        input_schema="mocap_3d_64x80",
        input_quality_status="VALID",
        predicted_class=1,
        class_label="correct",
        probabilities={"incorrect": 0.20, "correct": 0.80}
    )
    res1 = ens_engine.evaluate(stat, model)
    res2 = ens_engine.evaluate(stat, model)
    assert res1.decision == res2.decision
    assert res1.combined_score == res2.combined_score
    assert res1.confidence == res2.confidence

def test_api_hybrid_assessment_evaluate_endpoint():
    """Verify POST /api/assessment/evaluate works via FastAPI testclient."""
    client = TestClient(app)
    payload = {
        "exercise_id": "standing-right-arm-abduction",
        "trajectory": [
            {"timestamp": round(i * 0.05, 3), "elevation": 15.0 + 75.0 * np.sin(i * 0.1), "elbow": 172.0, "confidence": 0.95}
            for i in range(32)
        ],
        "input_source": "webcam_mediapipe",
        "duration_sec": 1.6
    }
    resp = client.post("/api/assessment/evaluate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "statistical_assessment" in data
    assert "model_assessment" in data
    assert "ensemble_decision" in data
    # Because webcam input was passed without mocap, ensemble must abstain with NEEDS_REVIEW
    assert data["ensemble_decision"]["decision"] == "NEEDS_REVIEW"

def test_api_cohort_evaluation_endpoint():
    """Verify GET /api/assessment/evaluate-cohort returns independent metrics."""
    client = TestClient(app)
    resp = client.get("/api/assessment/evaluate-cohort?split=test")
    assert resp.status_code == 200
    data = resp.json()
    assert data["split"] == "test"
    assert data["sample_count"] == 40
    assert "statistical_only" in data
    assert "lstm_only" in data
    assert "ensemble" in data
    assert data["statistical_only"]["accuracy"] > 0.0
    assert data["ensemble"]["accuracy"] > 0.0

def test_pdf_report_with_hybrid_ensemble_data():
    """Verify PDF report renders with 3-component hybrid ensemble assessment cards."""
    session_with_ensemble = {
        "id": "sess-ens-test-001",
        "timestamp": "2026-10-25T16:00:00",
        "exercise_name": "Standing Right-Arm Shoulder Abduction",
        "duration_seconds": 12.0,
        "repetitions_completed": 3,
        "reference_id": "ref-gold-standard-demo",
        "reference_type": "standard",
        "reference_version": "1.0",
        "reference_provenance": "Physiotherapy Clinical Protocol v1.0",
        "comparison_score": 88.5,
        "angle_stats": {
            "peak_elevation": 91.5,
            "mean_elevation": 50.2,
            "min_elevation": 14.0,
            "mean_elbow": 172.0,
            "rom": 77.5
        },
        "joint_deviations": {
            "mean_elevation_deviation": 5.1,
            "max_elevation_deviation": 10.2,
            "mean_elbow_deviation": 4.0,
            "rmse": 6.1
        },
        "statistical_assessment": {
            "score": 88.5,
            "classification_estimate": "CORRECT",
            "quality_status": "VALID",
            "sub_scores": {"rom": 0.90, "dtw_similarity": 0.88, "posture_compliance": 0.92, "cadence": 0.85}
        },
        "model_assessment": {
            "model_id": "rehab-lstm-rehab24-6-v1",
            "input_schema": "mocap_3d_64x80",
            "input_quality_status": "INCOMPATIBLE_REPRESENTATION",
            "rejection_reason": "Monocular webcam features require calibrated representation."
        },
        "ensemble_decision": {
            "decision": "NEEDS_REVIEW",
            "combined_score": None,
            "confidence": 0.50,
            "branch_agreement": None,
            "reasons": ["Webcam MediaPipe input safely isolated from 3D mocap model."]
        },
        "warnings": [],
        "raw_trajectory": [
            {"timestamp": i * 0.1, "elevation": 15 + 75 * np.sin(i * 0.2), "elbow": 172.0}
            for i in range(25)
        ]
    }
    pdf_bytes = generate_session_pdf(session_with_ensemble)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")


# =====================================================================
# Scientific Audit Regression Tests
# =====================================================================

def test_class_index_and_ground_truth_mapping():
    """Verify exact mapping of class indices to 'correct' and 'incorrect' across all components."""
    import torch
    from evaluation import independent_evaluator
    from model_assessment import model_assessment_engine

    # 1. Metadata mapping
    meta = lstm_engine.metadata
    assert meta["class_mapping"]["0"] == "incorrect"
    assert meta["class_mapping"]["1"] == "correct"

    # 2. Checkpoint mapping
    ckpt = torch.load(lstm_engine.model_path, map_location="cpu", weights_only=False)
    assert ckpt["class_names"]["0"] == "incorrect"
    assert ckpt["class_names"]["1"] == "correct"

    # 3. Model assessment engine output labels
    # If predicted_class is 0 -> class_label must be "incorrect"
    # If predicted_class is 1 -> class_label must be "correct"
    res_fake = model_assessment_engine.evaluate_sequence(
        sequence=[[0.0] * 80 for _ in range(64)],
        input_source="verified_mocap"
    )
    if res_fake.predicted_class is not None:
        expected = "correct" if res_fake.predicted_class == 1 else "incorrect"
        assert res_fake.class_label == expected

    # 4. Metric calculation ground-truth alignment
    metrics = independent_evaluator.compute_metrics(
        system_name="AlignmentTest",
        true_labels=[1, 0, 1, 0],
        predicted_classes=[1, 0, 0, 1],
        predicted_probs_correct=[0.9, 0.1, 0.2, 0.8],
        latencies_ms=[1.0, 1.0, 1.0, 1.0]
    )
    assert metrics.confusion_matrix["tp"] == 1  # true=1, pred=1
    assert metrics.confusion_matrix["tn"] == 1  # true=0, pred=0
    assert metrics.confusion_matrix["fp"] == 1  # true=0, pred=1
    assert metrics.confusion_matrix["fn"] == 1  # true=1, pred=0
    assert metrics.accuracy == 0.5


def test_feature_order_and_preprocessing_invariants():
    """Verify 80-feature shape, ordering, stationary joint clamping, and z-score invariants."""
    prep = np.load(lstm_engine.prep_path)
    mean = prep["feature_mean"]
    std = prep["feature_std"]

    assert mean.shape == (1, 1, 80)
    assert std.shape == (1, 1, 80)
    assert (std > 0).all(), "All feature standard deviations must be strictly positive"

    # Stationary root/pelvis joint (features 0, 1, 2) clamped to epsilon 1e-6
    assert std[0, 0, 0] <= 1.001e-6
    assert std[0, 0, 1] <= 1.001e-6
    assert std[0, 0, 2] <= 1.001e-6

    # Features 78 (arm elevation) and 79 (elbow angle)
    assert 0.04 < mean[0, 0, 78] < 0.06
    assert 0.04 < std[0, 0, 78] < 0.06
    assert 0.70 < mean[0, 0, 79] < 0.80
    assert 0.10 < std[0, 0, 79] < 0.20

    # Z-score normalization invariant: (mean - mean) / std == 0
    norm_mean = (mean - mean) / (std + 1e-8)
    assert np.allclose(norm_mean, 0.0)


def test_validation_test_separation_for_ensemble_calibration():
    """Verify participant-level disjointness between train, validation, and test cohorts."""
    from evaluation import independent_evaluator

    train_p = set(independent_evaluator.train_participants)
    val_p = set(independent_evaluator.validation_participants)
    test_p = set(independent_evaluator.test_participants)

    assert train_p == {1, 3, 4, 6, 9}
    assert val_p == {5, 7}
    assert test_p == {2, 8}

    # Strict disjointness - zero cross-contamination
    assert len(train_p.intersection(val_p)) == 0
    assert len(train_p.intersection(test_p)) == 0
    assert len(val_p.intersection(test_p)) == 0
    assert len(train_p.union(val_p).union(test_p)) == 9

    # Verify generated test cohort only contains test participants
    test_cohort = independent_evaluator.generate_partitioned_test_cohort()
    for rep in test_cohort:
        assert rep["participant_id"] in test_p


def test_probability_and_metric_calculations():
    """Verify Brier score, ROC-AUC, Macro-F1, and probability metrics."""
    from evaluation import independent_evaluator

    # Perfect predictions: Brier score must be 0.0
    perfect_metrics = independent_evaluator.compute_metrics(
        system_name="Perfect",
        true_labels=[1, 1, 0, 0],
        predicted_classes=[1, 1, 0, 0],
        predicted_probs_correct=[1.0, 1.0, 0.0, 0.0],
        latencies_ms=[1.0] * 4
    )
    assert perfect_metrics.brier_score_decided == 0.0
    assert perfect_metrics.accuracy == 1.0
    assert perfect_metrics.macro_f1 == 1.0

    # Complete uninformative prior (0.5 for all): Brier score must be (0.5 - 1)^2 = 0.25
    prior_metrics = independent_evaluator.compute_metrics(
        system_name="Prior",
        true_labels=[1, 1, 0, 0],
        predicted_classes=[1, 1, 0, 0],
        predicted_probs_correct=[0.5, 0.5, 0.5, 0.5],
        latencies_ms=[1.0] * 4
    )
    assert abs(prior_metrics.brier_score_decided - 0.25) < 1e-4


def test_abstention_and_partial_coverage_reporting():
    """Verify coverage, abstention, cohort accuracy vs decided accuracy reporting."""
    from evaluation import independent_evaluator

    # 4 samples: 2 decided (both correct), 2 abstained
    metrics = independent_evaluator.compute_metrics(
        system_name="AbstentionTest",
        true_labels=[1, 0, 1, 0],
        predicted_classes=[1, 0, None, None],
        predicted_probs_correct=[0.9, 0.1, None, None],
        latencies_ms=[1.0] * 4
    )
    assert metrics.total_samples == 4
    assert metrics.decided_samples == 2
    assert metrics.abstained_samples == 2
    assert metrics.decision_coverage_pct == 50.0
    assert metrics.abstention_rate_pct == 50.0

    # Decided accuracy is 2/2 = 1.0
    assert metrics.decided_accuracy == 1.0
    assert metrics.accuracy == 1.0

    # Cohort accuracy across all 4 samples is 2/4 = 0.5
    assert metrics.cohort_accuracy == 0.5
    assert metrics.cohort_accuracy < metrics.decided_accuracy

    # Cohort Brier score assigns 0.5 to abstained samples:
    # (0.9-1)^2 = 0.01; (0.1-0)^2 = 0.01; (0.5-1)^2 = 0.25; (0.5-0)^2 = 0.25 -> mean = 0.52 / 4 = 0.13
    assert abs(metrics.brier_score_cohort - 0.13) < 1e-3


def test_incompatible_live_inputs_safeguard():
    """Verify that monocular webcam MediaPipe inputs safely trigger NEEDS_REVIEW and cannot enter LSTM."""
    from model_assessment import model_assessment_engine
    from ensemble import ensemble_engine

    # 1. Model assessment engine directly rejects webcam inputs
    webcam_mocap = [[0.5] * 80 for _ in range(64)]
    m_res = model_assessment_engine.evaluate_sequence(
        sequence=webcam_mocap,
        input_source="webcam_mediapipe"
    )
    assert m_res.input_quality_status == "INCOMPATIBLE_REPRESENTATION"
    assert m_res.predicted_class is None
    assert m_res.class_label is None
    assert "Domain shift" in m_res.rejection_reason

    # 2. Ensemble decision engine with incompatible model branch safely abstains
    dummy_stat = {
        "score": 92.0,
        "classification_estimate": "CORRECT",
        "quality_status": "VALID",
        "sub_scores": {"rom": 0.95, "dtw_similarity": 0.90, "posture_compliance": 0.90, "cadence": 0.90}
    }
    e_res = ensemble_engine.evaluate(
        statistical_result=dummy_stat,
        model_result=m_res,
        exercise_id="standing-right-arm-abduction"
    )
    assert e_res.decision == "NEEDS_REVIEW"
    assert e_res.combined_score is None
    assert any("Domain shift" in r or "abstained" in r for r in e_res.reasons)
    assert any("uncalibrated webcam" in w for w in e_res.warnings)



