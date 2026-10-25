"""
Hybrid Ensemble Decision Engine for RehabVision AI.

Combines statistical/biomechanical assessment and deep-learning model assessment
into a deterministic, explainable clinical decision estimate.
Enforces strict abstention policies (NEEDS_REVIEW) when either branch is unavailable,
incompatible, low-confidence, or when branches significantly disagree.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Union
import math

from statistical_assessment import StatisticalAssessmentResult
from model_assessment import ModelAssessmentResult

@dataclass
class EnsembleDecisionResult:
    decision: str  # "CORRECT_ESTIMATE" | "INCORRECT_ESTIMATE" | "NEEDS_REVIEW"
    combined_score: Optional[float]  # [0.0, 100.0] or None if abstained
    confidence: float  # [0.0, 1.0]
    branch_agreement: Optional[bool]  # True, False, or None
    explanation: str
    reasons: List[str]
    warnings: List[str]
    statistical_branch: Optional[Dict[str, Any]] = None
    model_branch: Optional[Dict[str, Any]] = None
    ensemble_policy: str = "experimental_weighted_agreement_v1"
    weights_applied: Dict[str, float] = field(default_factory=lambda: {"statistical": 0.50, "model": 0.50})
    exercise_id: str = "standing-right-arm-abduction"
    reference_id: Optional[str] = None
    repetition_index: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnsembleDecisionEngine:
    """
    Evaluates both statistical and deep-learning branches under a conservative clinical ensemble policy.
    Never manufactures an ensemble decision if either branch is missing or incompatible.
    """
    def __init__(
        self,
        default_stat_weight: float = 0.50,
        default_model_weight: float = 0.50,
        correct_threshold: float = 0.70,
        incorrect_threshold: float = 0.40,
        conflict_margin: float = 0.25,
    ):
        self.default_stat_weight = default_stat_weight
        self.default_model_weight = default_model_weight
        self.correct_threshold = correct_threshold
        self.incorrect_threshold = incorrect_threshold
        self.conflict_margin = conflict_margin

    def evaluate(
        self,
        statistical_result: Optional[Union[StatisticalAssessmentResult, Dict[str, Any]]],
        model_result: Optional[Union[ModelAssessmentResult, Dict[str, Any]]],
        input_quality_metadata: Optional[Dict[str, Any]] = None,
        exercise_id: str = "standing-right-arm-abduction",
        reference_id: Optional[str] = None,
        repetition_index: int = 1,
        weights: Optional[Dict[str, float]] = None,
    ) -> EnsembleDecisionResult:
        reasons: List[str] = []
        warnings: List[str] = []

        w_stat = weights.get("statistical", self.default_stat_weight) if weights else self.default_stat_weight
        w_model = weights.get("model", self.default_model_weight) if weights else self.default_model_weight
        total_w = w_stat + w_model
        if total_w <= 0:
            w_stat, w_model = 0.5, 0.5
        else:
            w_stat, w_model = w_stat / total_w, w_model / total_w

        weights_applied = {"statistical": round(w_stat, 2), "model": round(w_model, 2)}

        stat_dict = statistical_result.to_dict() if isinstance(statistical_result, StatisticalAssessmentResult) else (statistical_result or {})
        model_dict = model_result.to_dict() if isinstance(model_result, ModelAssessmentResult) else (model_result or {})

        # =====================================================================
        # 1. Check Statistical Branch Completeness & Quality
        # =====================================================================
        if not stat_dict or "score" not in stat_dict:
            reasons.append("Missing statistical biomechanical assessment branch.")
            return EnsembleDecisionResult(
                decision="NEEDS_REVIEW",
                combined_score=None,
                confidence=0.0,
                branch_agreement=None,
                explanation="Ensemble abstains: Statistical biomechanical assessment is unavailable for this repetition.",
                reasons=reasons,
                warnings=warnings,
                statistical_branch=None,
                model_branch=model_dict or None,
                exercise_id=exercise_id,
                reference_id=reference_id,
                repetition_index=repetition_index
            )

        stat_score = float(stat_dict.get("score", 0.0))
        stat_status = stat_dict.get("quality_status", "VALID")
        stat_class = stat_dict.get("classification_estimate", "INCORRECT")

        # Check for NaN / non-finite statistical scores
        if not math.isfinite(stat_score):
            reasons.append("Statistical branch returned a non-finite score.")
            return EnsembleDecisionResult(
                decision="NEEDS_REVIEW",
                combined_score=None,
                confidence=0.0,
                branch_agreement=None,
                explanation="Ensemble abstains: Statistical score is non-finite or corrupted.",
                reasons=reasons,
                warnings=warnings,
                statistical_branch=stat_dict,
                model_branch=model_dict or None,
                exercise_id=exercise_id,
                reference_id=reference_id,
                repetition_index=repetition_index
            )

        if stat_status == "INSUFFICIENT_FRAMES":
            reasons.append("Repetition has insufficient frames for reliable biomechanical assessment.")
            return EnsembleDecisionResult(
                decision="NEEDS_REVIEW",
                combined_score=None,
                confidence=0.0,
                branch_agreement=None,
                explanation="Ensemble abstains: Incomplete or truncated movement repetition.",
                reasons=reasons,
                warnings=warnings,
                statistical_branch=stat_dict,
                model_branch=model_dict or None,
                exercise_id=exercise_id,
                reference_id=reference_id,
                repetition_index=repetition_index
            )

        # =====================================================================
        # 2. Check Deep-Learning Model Branch Completeness & Quality
        # =====================================================================
        if not model_dict or "input_quality_status" not in model_dict:
            reasons.append("Missing deep-learning model assessment branch.")
            return EnsembleDecisionResult(
                decision="NEEDS_REVIEW",
                combined_score=None,
                confidence=round(stat_score / 100.0, 2),
                branch_agreement=None,
                explanation=(
                    f"Ensemble abstains: Deep-learning model branch was not executed. "
                    f"Statistical score is {stat_score:.1f}/100 ({stat_class}), but ensemble policy requires both validated branches."
                ),
                reasons=reasons,
                warnings=warnings,
                statistical_branch=stat_dict,
                model_branch=None,
                exercise_id=exercise_id,
                reference_id=reference_id,
                repetition_index=repetition_index
            )

        model_status = model_dict.get("input_quality_status", "INCOMPATIBLE_REPRESENTATION")

        # Incompatible representation (e.g. monocular MediaPipe webcam vs 3D mocap LSTM)
        if model_status == "INCOMPATIBLE_REPRESENTATION":
            rejection_reason = model_dict.get("rejection_reason", "Domain shift: input requires validation.")
            reasons.append(f"Model branch abstained: {rejection_reason}")
            warnings.append("Live deep-learning classification unavailable for uncalibrated webcam landmarks.")
            return EnsembleDecisionResult(
                decision="NEEDS_REVIEW",
                combined_score=None,
                confidence=0.50,
                branch_agreement=None,
                explanation=(
                    f"Ensemble abstains (NEEDS_REVIEW): Deep-learning branch input requires validation (MediaPipe monocular vs mocap domain shift). "
                    f"Statistical biomechanical branch scored {stat_score:.1f}/100 ({stat_class}). "
                    f"Under the ensemble clinical safety policy, an autonomous decision requires both branches to be validated."
                ),
                reasons=reasons,
                warnings=warnings,
                statistical_branch=stat_dict,
                model_branch=model_dict,
                exercise_id=exercise_id,
                reference_id=reference_id,
                repetition_index=repetition_index
            )

        # Other model errors (invalid shape, numerical anomaly, model unavailable)
        if model_status != "VALID":
            err_msg = model_dict.get("rejection_reason", f"Model input quality status is {model_status}")
            reasons.append(f"Model branch invalid: {err_msg}")
            return EnsembleDecisionResult(
                decision="NEEDS_REVIEW",
                combined_score=None,
                confidence=0.0,
                branch_agreement=None,
                explanation=f"Ensemble abstains: Deep-learning model branch failed validation ({model_status}).",
                reasons=reasons,
                warnings=warnings,
                statistical_branch=stat_dict,
                model_branch=model_dict,
                exercise_id=exercise_id,
                reference_id=reference_id,
                repetition_index=repetition_index
            )

        # Check model probabilities
        probs = model_dict.get("probabilities")
        if not probs or "correct" not in probs or "incorrect" not in probs:
            reasons.append("Model output is missing class probabilities.")
            return EnsembleDecisionResult(
                decision="NEEDS_REVIEW",
                combined_score=None,
                confidence=0.0,
                branch_agreement=None,
                explanation="Ensemble abstains: Model probabilities are missing or malformed.",
                reasons=reasons,
                warnings=warnings,
                statistical_branch=stat_dict,
                model_branch=model_dict,
                exercise_id=exercise_id,
                reference_id=reference_id,
                repetition_index=repetition_index
            )

        prob_cor = float(probs["correct"])
        prob_inc = float(probs["incorrect"])

        if not (math.isfinite(prob_cor) and math.isfinite(prob_inc)):
            reasons.append("Model class probabilities contain NaN or infinite values.")
            return EnsembleDecisionResult(
                decision="NEEDS_REVIEW",
                combined_score=None,
                confidence=0.0,
                branch_agreement=None,
                explanation="Ensemble abstains: Non-finite model probabilities detected.",
                reasons=reasons,
                warnings=warnings,
                statistical_branch=stat_dict,
                model_branch=model_dict,
                exercise_id=exercise_id,
                reference_id=reference_id,
                repetition_index=repetition_index
            )

        # =====================================================================
        # 3. Scale Alignment & Weighted Combination
        # =====================================================================
        # Statistical branch score: [0, 100] -> P_stat(correct) in [0.0, 1.0]
        p_stat_cor = max(0.0, min(1.0, stat_score / 100.0))
        p_model_cor = max(0.0, min(1.0, prob_cor))

        # Check binary branch agreement
        # Statistical classification: stat_class in {"CORRECT", "INCORRECT"}
        # Model classification: class_label in {"correct", "incorrect"}
        stat_binary_cor = (stat_class == "CORRECT")
        model_binary_cor = (model_dict.get("class_label") == "correct" or model_dict.get("predicted_class") == 1)
        branches_agree = (stat_binary_cor == model_binary_cor)

        # Combined Probability
        p_combined = (w_stat * p_stat_cor) + (w_model * p_model_cor)
        combined_score = round(p_combined * 100.0, 1)

        # Confidence: distance from decision boundary (0.50)
        confidence = round(float(2.0 * abs(p_combined - 0.50)), 2)

        # =====================================================================
        # 4. Conflict & Disagreement Policy
        # =====================================================================
        # If branches fundamentally disagree (one says correct, one says incorrect):
        if not branches_agree:
            reasons.append(
                f"Branch Disagreement: Statistical branch assessed {stat_class} ({stat_score:.1f}/100) "
                f"while Deep-Learning branch assessed {model_dict.get('class_label', 'unknown').upper()} (P_cor={prob_cor:.2f})."
            )
            warnings.append("Movement form criteria show discrepancy between kinematic rules and neural temporal pattern.")

            # If combined probability falls in conflict margin around boundary
            if abs(p_stat_cor - p_model_cor) >= self.conflict_margin and (self.incorrect_threshold < p_combined < self.correct_threshold):
                return EnsembleDecisionResult(
                    decision="NEEDS_REVIEW",
                    combined_score=combined_score,
                    confidence=confidence,
                    branch_agreement=False,
                    explanation=(
                        f"Ensemble flags NEEDS_REVIEW due to branch disagreement: "
                        f"Statistical assessment indicates {stat_class} ({stat_score:.1f}/100) "
                        f"whereas Deep-Learning model predicts {model_dict.get('class_label', 'unknown').upper()} (P_cor={prob_cor:.2f}). "
                        f"Combined score is {combined_score:.1f}/100. Manual clinician review is recommended."
                    ),
                    reasons=reasons,
                    warnings=warnings,
                    statistical_branch=stat_dict,
                    model_branch=model_dict,
                    weights_applied=weights_applied,
                    exercise_id=exercise_id,
                    reference_id=reference_id,
                    repetition_index=repetition_index
                )

        # =====================================================================
        # 5. Final Ensemble Decision Formulation
        # =====================================================================
        if p_combined >= self.correct_threshold:
            decision = "CORRECT_ESTIMATE"
            explanation = (
                f"Consensus Correct: Both branches concordantly estimate correct movement form. "
                f"Statistical score: {stat_score:.1f}/100 | DL model probability: {prob_cor:.2f} | "
                f"Combined score: {combined_score:.1f}/100 (Confidence: {confidence:.2f})."
            )
            reasons.append(f"Combined score ({combined_score:.1f}) exceeds correct threshold ({self.correct_threshold * 100:.0f}).")
        elif p_combined <= self.incorrect_threshold:
            decision = "INCORRECT_ESTIMATE"
            explanation = (
                f"Consensus Incorrect: Biomechanical deviations and model classification indicate incorrect movement form. "
                f"Statistical score: {stat_score:.1f}/100 | DL model probability: {prob_cor:.2f} | "
                f"Combined score: {combined_score:.1f}/100."
            )
            reasons.append(f"Combined score ({combined_score:.1f}) is below incorrect threshold ({self.incorrect_threshold * 100:.0f}).")
        else:
            decision = "NEEDS_REVIEW"
            explanation = (
                f"Ambiguous movement quality (Combined score: {combined_score:.1f}/100 in intermediate zone {self.incorrect_threshold*100:.0f}-{self.correct_threshold*100:.0f}). "
                f"Flagged for physiotherapist inspection."
            )
            reasons.append(f"Combined score ({combined_score:.1f}) falls in intermediate ambiguous range.")

        return EnsembleDecisionResult(
            decision=decision,
            combined_score=combined_score,
            confidence=confidence,
            branch_agreement=branches_agree,
            explanation=explanation,
            reasons=reasons,
            warnings=warnings,
            statistical_branch=stat_dict,
            model_branch=model_dict,
            weights_applied=weights_applied,
            exercise_id=exercise_id,
            reference_id=reference_id,
            repetition_index=repetition_index
        )


ensemble_engine = EnsembleDecisionEngine()
