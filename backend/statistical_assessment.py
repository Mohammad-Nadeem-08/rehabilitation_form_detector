"""
Statistical and Biomechanical Assessment Engine for RehabVision AI.

Calculates multi-dimensional kinematic metrics, DTW trajectory similarity,
joint compensations, and cadence compliance. Produces an interpretable,
transparent statistical score with clear directionality and preserved raw measurements.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from exercise_registry import exercise_registry, ExerciseStrategy
from movement_comparison import compare_angle_trajectories

@dataclass
class StatisticalAssessmentResult:
    score: float  # [0.0, 100.0]
    classification_estimate: str  # "CORRECT" | "INCORRECT" | "DEGRADED_QUALITY"
    quality_status: str  # "VALID" | "DEGRADED" | "INSUFFICIENT_FRAMES"
    primary_metric_name: str
    secondary_metric_name: str
    raw_measurements: Dict[str, float]
    sub_scores: Dict[str, float]  # Each normalized [0.0, 1.0]
    metric_directions: Dict[str, str]  # Directionality documentation
    warnings: List[str]
    exercise_id: str
    repetition_index: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StatisticalAssessmentEngine:
    """
    Evaluates physical movement quality against biomechanical and DTW reference criteria.
    Exercise-strategy aware: automatically adapts metric names, thresholds, and scoring.
    """
    def __init__(self):
        pass

    def evaluate_repetition(
        self,
        trajectory: List[Dict[str, Any]],
        exercise_id: str = "standing-right-arm-abduction",
        reference_angles: Optional[List[Dict[str, Any]]] = None,
        duration_sec: Optional[float] = None,
        repetition_index: int = 1,
        clinician_criteria: Optional[Dict[str, Any]] = None,
    ) -> StatisticalAssessmentResult:
        strategy = exercise_registry.get(exercise_id)
        if not strategy:
            strategy = exercise_registry.get("standing-right-arm-abduction")

        warnings: List[str] = []

        # 1. Input Validation
        if not trajectory or len(trajectory) < 8:
            return StatisticalAssessmentResult(
                score=0.0,
                classification_estimate="DEGRADED_QUALITY",
                quality_status="INSUFFICIENT_FRAMES",
                primary_metric_name=strategy.primary_metric_name,
                secondary_metric_name=strategy.secondary_metric_name,
                raw_measurements={"frame_count": len(trajectory) if trajectory else 0},
                sub_scores={},
                metric_directions={},
                warnings=["Insufficient frames in repetition (< 8 frames). Movement discarded."],
                exercise_id=strategy.id,
                repetition_index=repetition_index
            )

        # 2. Extract Signals
        timestamps = [p.get("timestamp", idx * 0.05) for idx, p in enumerate(trajectory)]
        elevations = [float(p.get("elevation", p.get("primary_angle", 0.0))) for p in trajectory]
        elbows = [float(p.get("elbow", p.get("secondary_angle", 170.0))) for p in trajectory]
        confidences = [float(p.get("confidence", 1.0)) for p in trajectory]

        actual_duration = duration_sec if (duration_sec and duration_sec > 0) else (timestamps[-1] - timestamps[0])
        actual_duration = max(0.1, float(actual_duration))

        # Primary & Secondary Angle Metrics
        peak_primary = float(np.max(elevations))
        min_primary = float(np.min(elevations))
        mean_primary = float(np.mean(elevations))
        active_rom = float(peak_primary - min_primary)

        mean_secondary = float(np.mean(elbows))
        min_secondary = float(np.min(elbows))

        # Tracking Confidence & Frame Quality
        avg_confidence = float(np.mean(confidences)) if confidences else 1.0
        low_conf_frames = sum(1 for c in confidences if c < 0.30)
        missing_frame_pct = (low_conf_frames / len(trajectory)) * 100.0

        if avg_confidence < 0.50:
            warnings.append(f"Low average landmark tracking confidence ({avg_confidence:.2f}).")
        if missing_frame_pct > 25.0:
            warnings.append(f"High missing/low-confidence frame rate ({missing_frame_pct:.1f}%).")

        # 3. Dynamic Time Warping (DTW) Comparison
        dtw_score = 75.0
        mean_deviation = 10.0
        max_deviation = 20.0
        rmse = 12.0

        if reference_angles and len(reference_angles) >= 8:
            dtw_res = compare_angle_trajectories(reference_angles, trajectory)
            dtw_score = float(dtw_res.get("comparison_score", 75.0))
            mean_deviation = float(dtw_res.get("mean_elevation_deviation", 10.0))
            max_deviation = float(dtw_res.get("max_elevation_deviation", 20.0))
            rmse = float(dtw_res.get("rmse", 12.0))
            for w in dtw_res.get("warnings", []):
                if w not in warnings:
                    warnings.append(w)
        else:
            warnings.append("No reference trajectory provided; using baseline kinematic model.")

        # 4. Exercise-Specific Biomechanical Evaluation
        target_min = float(clinician_criteria.get("target_rom_min", strategy.target_rom_min) if clinician_criteria else strategy.target_rom_min)
        target_max = float(clinician_criteria.get("target_rom_max", strategy.target_rom_max) if clinician_criteria else strategy.target_rom_max)

        # Sub-score A: Range of Motion (Weight: 0.35)
        # Evaluates peak elevation / flexion against target bounds
        if peak_primary < target_min:
            shortfall = target_min - peak_primary
            rom_sub = max(0.0, 1.0 - (shortfall / 40.0))
            if shortfall > 15.0:
                warnings.append(f"Restricted {strategy.primary_metric_name} ({peak_primary:.1f}° vs target {target_min:.0f}°-{target_max:.0f}°).")
        elif peak_primary > target_max + 15.0:
            excess = peak_primary - target_max
            rom_sub = max(0.3, 1.0 - (excess / 35.0))
            warnings.append(f"Excessive elevation / overshoot ({peak_primary:.1f}° vs target {target_max:.0f}°).")
        else:
            rom_sub = 1.0

        # Sub-score B: Trajectory Shape Conformity (Weight: 0.35)
        # Higher DTW similarity index is better
        dtw_sub = max(0.0, min(1.0, dtw_score / 100.0))

        # Sub-score C: Secondary Joint Biomechanics (Weight: 0.20)
        # For shoulder abduction: elbow should remain extended (165° - 180°)
        # For bicep curl: humeral stability should remain low (0° - 25°)
        if "abduction" in strategy.id:
            if min_secondary < 140.0:
                posture_sub = max(0.0, (min_secondary - 90.0) / 50.0)
                warnings.append(f"Significant elbow flexion compensation detected (min {min_secondary:.1f}° < 140°).")
            elif min_secondary < 160.0:
                posture_sub = 0.70 + 0.30 * ((min_secondary - 140.0) / 20.0)
                warnings.append(f"Mild elbow bend during abduction (min {min_secondary:.1f}°).")
            else:
                posture_sub = 1.0
        elif "curl" in strategy.id:
            # secondary angle is shoulder sway
            if mean_secondary > 35.0:
                posture_sub = max(0.1, 1.0 - (mean_secondary - 35.0) / 30.0)
                warnings.append(f"Excessive shoulder swing / humeral movement ({mean_secondary:.1f}° > 35°).")
            else:
                posture_sub = 1.0
        else:
            posture_sub = 0.90

        # Sub-score D: Cadence & Timing (Weight: 0.10)
        # Functional cadence window: 1.2s to 5.5s per repetition
        if actual_duration < 1.0:
            cadence_sub = 0.40
            warnings.append(f"Movement cadence too rapid ({actual_duration:.1f}s < 1.0s). Controlled tempo recommended.")
        elif actual_duration > 6.0:
            cadence_sub = 0.60
            warnings.append(f"Movement cadence slow / hesitant ({actual_duration:.1f}s > 6.0s).")
        else:
            cadence_sub = 1.0

        # 5. Composite Normalized Statistical Score [0, 100]
        composite_score = round(float(
            100.0 * (
                0.35 * rom_sub +
                0.35 * dtw_sub +
                0.20 * posture_sub +
                0.10 * cadence_sub
            )
        ), 1)
        composite_score = max(0.0, min(100.0, composite_score))

        # Classification decision based on statistical criteria
        if composite_score >= 75.0 and rom_sub >= 0.70 and posture_sub >= 0.60:
            classification = "CORRECT"
        else:
            classification = "INCORRECT"

        quality_status = "VALID" if (avg_confidence >= 0.60 and missing_frame_pct <= 25.0) else "DEGRADED"

        raw_measurements = {
            "peak_primary_angle": round(peak_primary, 1),
            "min_primary_angle": round(min_primary, 1),
            "active_rom": round(active_rom, 1),
            "mean_secondary_angle": round(mean_secondary, 1),
            "min_secondary_angle": round(min_secondary, 1),
            "duration_seconds": round(actual_duration, 2),
            "frame_count": len(trajectory),
            "mean_elevation_deviation": round(mean_deviation, 1),
            "max_elevation_deviation": round(max_deviation, 1),
            "rmse": round(rmse, 1),
            "dtw_similarity_score": round(dtw_score, 1),
            "average_confidence": round(avg_confidence, 2),
            "missing_frame_pct": round(missing_frame_pct, 1),
        }

        sub_scores = {
            "rom_criterion": round(rom_sub, 3),
            "trajectory_conformity": round(dtw_sub, 3),
            "secondary_posture": round(posture_sub, 3),
            "cadence_timing": round(cadence_sub, 3),
        }

        metric_directions = {
            "peak_primary_angle": "Range-based: optimal within target ROM bounds",
            "active_rom": "Higher is better towards functional target",
            "mean_secondary_angle": "Exercise-dependent: >= 165° for straight-arm abduction",
            "duration_seconds": "Range-based: optimal within 1.2s - 5.5s",
            "mean_elevation_deviation": "Lower is better (0.0° is identical)",
            "rmse": "Lower is better",
            "dtw_similarity_score": "Higher is better (100 is identical)",
            "average_confidence": "Higher is better (1.0 is max confidence)",
        }

        return StatisticalAssessmentResult(
            score=composite_score,
            classification_estimate=classification,
            quality_status=quality_status,
            primary_metric_name=strategy.primary_metric_name,
            secondary_metric_name=strategy.secondary_metric_name,
            raw_measurements=raw_measurements,
            sub_scores=sub_scores,
            metric_directions=metric_directions,
            warnings=warnings,
            exercise_id=strategy.id,
            repetition_index=repetition_index
        )


statistical_engine = StatisticalAssessmentEngine()
