# RehabVision AI — Model Evaluation & Benchmarking Guide

## Overview

RehabVision AI provides a reusable, leak-free evaluation framework in [`backend/evaluation.py`](../backend/evaluation.py) to independently evaluate machine-learning models, statistical biomechanical rules, and hybrid ensemble decisions on participant-partitioned cohorts.

This guide explains how to evaluate your trained model, establish baseline metrics, and prevent common scientific evaluation errors (such as data leakage, unrepresented abstention, and conflating decided accuracy with cohort accuracy).

---

## 1. The Reusable Evaluation Interface

The `IndependentEvaluator` class provides `evaluate_custom_model_cohort(...)` to benchmark any registered model adapter:

```python
from evaluation import independent_evaluator

report = independent_evaluator.evaluate_custom_model_cohort(
    cohort=my_labelled_test_cohort,
    model_id="bicep-curl-transformer-v1",
    exercise_id="seated-right-bicep-curl",
    input_key="sequence_input"  # Key containing the input feature array
)
```

### Expected Cohort Format
The cohort should be a list of dictionaries with ground-truth labels and participant identifiers:
```python
my_labelled_test_cohort = [
    {
        "repetition_id": "rep-p10-01",
        "participant_id": 10,
        "ground_truth_label": 1,  # 1 = correct form, 0 = movement fault
        "sequence_input": [[...], [...], ...],  # Features matching adapter's declared schema
    },
    ...
]
```

---

## 2. Standardized Metrics Reported

The evaluation output reports the following standardized metrics:

| Metric | Field in Report | Interpretation |
| :--- | :--- | :--- |
| **Dataset Sample Count** | `dataset_sample_count` | Total repetitions in the evaluation set. |
| **Participant Split** | `participant_distribution` | Count of repetitions per subject (checks balance). |
| **Class Distribution** | `class_distribution` | `{"0": incorrect_count, "1": correct_count}`. |
| **Decided Accuracy** | `metrics.decided_accuracy` | Accuracy strictly on repetitions where the system reached a decision: $\frac{\text{TP} + \text{TN}}{\text{Decided Samples}}$. |
| **Cohort Accuracy** | `metrics.cohort_accuracy` | Accuracy across the **entire evaluation cohort**: $\frac{\text{TP} + \text{TN}}{\text{Total Samples}}$. |
| **Decision Coverage** | `metrics.decision_coverage_pct` | Percentage of repetitions decided ($[0.0\%, 100.0\%]$). |
| **Abstention Rate** | `metrics.abstention_rate_pct` | Percentage of repetitions flagged `NEEDS_REVIEW` ($[0.0\%, 100.0\%]$). |
| **Precision (Class 1)** | `metrics.precision_correct` | Positive predictive value for correct exercise form: $\frac{\text{TP}}{\text{TP} + \text{FP}}$. |
| **Recall (Class 1)** | `metrics.recall_correct` | Sensitivity for identifying correct form: $\frac{\text{TP}}{\text{TP} + \text{FN}}$. |
| **F1-Score (Class 1)** | `metrics.f1_correct` | Harmonic mean of precision and recall for correct form. |
| **Precision (Class 0)** | `metrics.precision_incorrect` | Predictive value for movement faults: $\frac{\text{TN}}{\text{TN} + \text{FN}}$. |
| **Recall (Class 0)** | `metrics.recall_incorrect` | Sensitivity for detecting movement faults: $\frac{\text{TN}}{\text{TN} + \text{FP}}$. |
| **F1-Score (Class 0)** | `metrics.f1_incorrect` | Harmonic mean of precision and recall for faults. |
| **Macro F1** | `metrics.macro_f1` | Unweighted average of Class 0 F1 and Class 1 F1: $\frac{\text{F1}_0 + \text{F1}_1}{2}$. |
| **Confusion Matrix** | `metrics.confusion_matrix` | Counts: `{"tp": int, "tn": int, "fp": int, "fn": int}`. |
| **ROC-AUC** | `metrics.roc_auc` | Area under ROC curve (via Mann-Whitney U rank statistic). `None` if only one class is decided. |
| **Brier Score (Decided)** | `metrics.brier_score_decided` | Mean squared probability error on decided samples: $\frac{1}{N_{\text{dec}}} \sum (P_i - y_i)^2$. |
| **Brier Score (Cohort)** | `metrics.brier_score_cohort` | Mean squared probability error across full cohort (assigns $0.50$ prior to abstentions). |
| **Average Latency** | `metrics.average_latency_ms` | Mean inference runtime per repetition in milliseconds. |

---

## 3. Scientific Evaluation Rules

### Rule 1: Never Present Decided Accuracy Without Coverage
If an ensemble or model abstains (`NEEDS_REVIEW`) on $40\%$ of difficult repetitions, reporting only its decided accuracy ($100\%$) creates a false illusion of perfection. Always present:
$$\text{Decided Accuracy} = 100.0\% \quad \text{with} \quad \text{Coverage} = 60.0\% \implies \text{Cohort Accuracy} = 60.0\%$$

### Rule 2: Strictly Preserve Participant-Level Partitioning
- In rehabilitation and motion capture, multiple repetitions from the same participant are highly correlated (same body morphology, height, movement habits).
- Random frame-level or repetition-level shuffling across train and test sets leads to **catastrophic identity leakage**.
- **Always hold out entire participants** (e.g. Participants 2 & 8 strictly for test).

### Rule 3: Do Not Tune Thresholds on the Held-Out Test Set
- Ensemble weights, decision thresholds ($0.70 / 0.40$), and probability calibration (temperature scaling / Platt scaling) must be fitted on **validation participants only** (e.g. Participants 5 & 7).
- The test cohort must remain completely untouched until final evaluation.

---

## 4. Status of the Existing REHAB24-6 LSTM Evaluation

An exhaustive repository audit established the following verified status:
1. **Model Artifacts Present:** The trained PyTorch checkpoint [`backend/models/rehab_lstm.pth`](../backend/models/rehab_lstm.pth) and z-score parameters [`backend/models/preprocessing.npz`](../backend/models/preprocessing.npz) are present and functional.
2. **Raw Training Data Absent:** The raw 3D optical motion-capture files (C3D / BVH coordinate logs) from the 9 REHAB24-6 subjects are **not included** in this repository.
3. **Colab Test Result Context:** The $82.5\%$ test accuracy and $81.95\%$ macro F1 documented in `metadata.json` represent the original research findings reported in Google Colab.
4. **Synthetic Test Generator Finding:** Synthetic dev repetitions in `evaluation.py` filled joints 0–77 with static noise, which caused the frozen LSTM to predict near its classifier bias ($10\%$ accuracy). This demonstrates why synthetic noise tests are only for code verification, not scientific validation.

---

## 5. Running Automated Cohort Evaluation

To evaluate all three systems (Statistical-Only, LSTM-Only, Hybrid Ensemble) on the held-out partition:

### Via Pytest Suite:
```powershell
python -m pytest backend/tests/test_model_adapters.py -v
```

### Via REST API:
```powershell
curl -X GET "http://127.0.0.1:8000/api/assessment/evaluate-cohort?split=test"
```
Response contains complete JSON breakdown for `statistical_only`, `lstm_only`, and `ensemble`.
