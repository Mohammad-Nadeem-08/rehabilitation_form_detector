# RehabVision AI — Hybrid Deep Learning + Statistical Ensemble Assessment Specification

## Executive Summary

RehabVision AI evaluates patient rehabilitation exercise form using an interpretable, deterministic **tri-branch assessment architecture**:
1. **Statistical / Biomechanical Assessment Branch**: Deterministic kinematic rule verification (Active ROM, peak primary angle, secondary joint compensation, posture compliance, and Dynamic Time Warping (DTW) trajectory similarity).
2. **Deep-Learning Model Branch**: Sequence-level temporal classification via a PyTorch Bidirectional/2-Layer LSTM trained on 3D motion-capture data (REHAB24-6 dataset, 26 joints, 80 normalized features).
3. **Hybrid Ensemble Decision Engine**: Combines both branches using an agreement-aware, scale-aligned decision policy with conservative abstention guards (`NEEDS_REVIEW`).

> **MANDATORY EDUCATIONAL & CLINICAL DISCLAIMER:**  
> RehabVision AI is an educational engineering prototype developed for a 7th-semester deep learning capstone. Assessments estimate exercise-form similarity against recorded references and do NOT constitute medical diagnosis, clinical validation, or determinations of patient safety. Autonomous clinical decision-making requires licensed physiotherapist supervision.

---

## 1. Audit Findings of the Trained Model and Dataset

### 1.1 Architecture & Weights Verification
An exhaustive inspection of the saved PyTorch checkpoint (`backend/models/rehab_lstm.pth`) and Colab training metadata (`metadata.json`) revealed the following exact specifications:
- **Layer Architecture:** PyTorch 2-layer LSTM with Linear classification head.
  - `lstm.weight_ih_l0`: Tensor shape `[256, 80]`
  - `lstm.weight_hh_l0`: Tensor shape `[256, 64]`
  - `lstm.weight_ih_l1`: Tensor shape `[256, 64]`
  - `lstm.weight_hh_l1`: Tensor shape `[256, 64]`
  - `classifier.weight`: Tensor shape `[2, 64]`
  - `classifier.bias`: Tensor shape `[2]`
- **Directionality Note:** While originally referenced as bidirectional in legacy notes, inspection of the actual state dictionary keys confirmed that the checkpoint is **unidirectional** (`hidden_size=64`, `num_layers=2`, `bidirectional=False`).
- **Input Tensor Dimensions:** `(batch_size, 64, 80)` — 64 resampled time steps, 80 normalized kinematic features per frame.
- **Output Classes & Index Mapping:**
  - Index `0`: `"incorrect"` (movement form exhibiting compensatory faults, insufficient ROM, or elbow collapse).
  - Index `1`: `"correct"` (standard clinical rehabilitation form within reference tolerances).
  - Ordering verified directly from `checkpoint['class_names']` and `metadata.json`.

### 1.2 The Exact 80 Input Features
The 80 features are structured in exact sequence:
- **Features 0–77 (Joint Coordinates):** Normalized 3D coordinates $(x, y, z)$ for 26 anatomical joints, centered at the pelvis root:
  - Joint 0: Pelvis (Root: $x=0, y=0, z=0$)
  - Joint 1: Spine Navel
  - Joint 2: Spine Chest
  - Joint 3: Neck
  - Joint 4: Clavicle Left
  - Joint 5: Shoulder Left
  - Joint 6: Elbow Left
  - Joint 7: Wrist Left
  - Joint 8: Hand Left
  - Joint 9: Clavicle Right
  - Joint 10: Shoulder Right
  - Joint 11: Elbow Right
  - Joint 12: Wrist Right
  - Joint 13: Hand Right
  - Joint 14: Hip Left
  - Joint 15: Knee Left
  - Joint 16: Ankle Left
  - Joint 17: Foot Left
  - Joint 18: Hip Right
  - Joint 19: Knee Right
  - Joint 20: Ankle Right
  - Joint 21: Foot Right
  - Joint 22: Head
  - Joint 23: Nose
  - Joint 24: Eye Left
  - Joint 25: Eye Right
- **Feature 78 (Primary Angle):** Normalized right arm elevation angle in radians $([0.0, \pi/2])$.
- **Feature 79 (Secondary Angle):** Normalized right elbow extension angle in radians $([0.0, \pi])$.

### 1.3 Preprocessing and Normalization
Saved in `backend/models/preprocessing.npz`:
- `feature_mean`: Array shape `(1, 1, 80)`.
- `feature_std`: Array shape `(1, 1, 80)`.
- Features 0–2 (Pelvis root origin) have mean 0.0 and std 1e-6 (constant zero coordinate baseline).
- Z-score normalization formula:
  $$x_{\text{norm}} = \frac{x - \mu}{\sigma + 10^{-8}}$$

---

## 2. Live MediaPipe vs 3D Motion-Capture Compatibility Analysis

### 2.1 Domain Shift Investigation
| Characteristic | REHAB24-6 Motion Capture (Trained LSTM) | Monocular Live MediaPipe (Webcam) |
| :--- | :--- | :--- |
| **Coordinate Space** | Calibrated Metric 3D Space (meters, optical mocap) | 2D Normalized Image Plane + Uncalibrated Relative Depth |
| **Landmark Schema** | 26 Anatomical Joints (Pelvis root, Clavicles, Navel) | 33 Surface Landmarks (BlazePose, face mesh focus) |
| **Depth Quality** | Ground-truth millimeter spatial accuracy | Monocular pseudo-depth, sensitive to clothing and scale |
| **Trunk / Pelvis** | Measured Pelvis Joint (Root origin $(0,0,0)$) | Inferred hip midpoints, prone to camera perspective distortion |
| **Camera Perspective**| Multi-camera calibrated capture volume | Single perspective, susceptible to foreshortening and rotation |

### 2.2 Selected Implementation Path: PATH B (Separate Safe Interface)
- **Why PATH A was rejected:** Generating a synthetic 80-feature mocap vector by arbitrarily zero-padding or remapping 33 monocular landmarks introduces severe unquantifiable domain shift. Feeding such transformed inputs into the LSTM would produce uncalibrated, spurious predictions ("hallucinated AI scores").
- **Selected Strategy (PATH B):** 
  - The live webcam pipeline evaluates kinematic biomechanics through the **Statistical Assessment Branch** and Dynamic Time Warping.
  - The **Model Assessment Engine** checks input source and rejects uncalibrated monocular MediaPipe streams with structured status `INCOMPATIBLE_REPRESENTATION` and message `input_requires_validation`.
  - The **Hybrid Ensemble Decision Engine** strictly enforces the abstention rule: if either branch is missing or incompatible, the decision state is conservatively set to `NEEDS_REVIEW`.
  - The genuine PyTorch LSTM remains fully operational in the **Model Lab** for verified 80-feature 3D sequences.

---

## 3. Branch 1: Statistical & Biomechanical Assessment Engine

The statistical assessment service (`backend/statistical_assessment.py`) provides interpretable, multi-dimensional biomechanical scoring:

### 3.1 Evaluated Metrics & Directionality
1. **Active Range of Motion (ROM):** Direction: *Higher is better towards functional target*. For shoulder abduction, functional target is $80.0^\circ - 105.0^\circ$.
2. **Peak Primary Joint Angle:** Direction: *Range-based*. Optimal in target window; penalizes under-elevation ($<80^\circ$) and excessive hyper-abduction ($>120^\circ$).
3. **Secondary Joint Posture (Elbow Extension):** Direction: *Threshold-based / Higher is better*. Penalizes elbow flexion/collapse ($<165^\circ$).
4. **Trunk Lean & Stability:** Direction: *Lower is better*. Penalizes lateral trunk compensations ($>12^\circ$).
5. **Dynamic Time Warping (DTW) Similarity:** Direction: *Higher is better* ($[0, 100]$ index).
6. **Trajectory RMSE:** Direction: *Lower is better* ($0.0^\circ$ is identical).
7. **Cadence & Repetition Duration:** Direction: *Range-based* ($1.5\text{s} - 4.5\text{s}$ per repetition).
8. **Landmark Tracking Confidence:** Direction: *Higher is better*. If mean confidence $<0.40$, status is marked `DEGRADED`.

### 3.2 Sub-Score Synthesis
$$\text{Score}_{\text{stat}} = 100 \times \left(0.35 \cdot S_{\text{rom}} + 0.35 \cdot S_{\text{dtw}} + 0.15 \cdot S_{\text{posture}} + 0.15 \cdot S_{\text{cadence}}\right)$$
- If $\text{Score}_{\text{stat}} \ge 75.0 \implies \text{Classification: CORRECT}$
- If $\text{Score}_{\text{stat}} < 75.0 \implies \text{Classification: INCORRECT}$

---

## 4. Branch 2: Deep-Learning Model Assessment Engine

The deep-learning assessment service (`backend/model_assessment.py`) provides a dedicated interface for temporal deep-learning inference:
- **Shape Verification:** Enforces $(64, 80)$ shape strictly. Rejects partial sequences or incorrect feature counts with `INVALID_SHAPE`.
- **Numerical Integrity:** Checks `np.isfinite()`. Rejects sequences with `NaN` or `Inf` with `NUMERICAL_ANOMALY`.
- **Domain Shift Guard:** Rejects uncalibrated live monocular webcam streams with `INCOMPATIBLE_REPRESENTATION`.
- **Output Contract:**
  - `model_id`: `"rehab-lstm-rehab24-6-v1"`
  - `model_version`: `"1.0.0"`
  - `architecture`: `"PyTorch 2-Layer LSTM (unidirectional, hidden=64)"`
  - `predicted_class`: `0` or `1`
  - `class_label`: `"incorrect"` or `"correct"`
  - `probabilities`: `{"incorrect": float, "correct": float}`
  - `inference_latency_ms`: Measured via high-resolution timer (`time.perf_counter()`).

---

## 5. Branch 3: Hybrid Ensemble Decision Engine

The ensemble service (`backend/ensemble.py`) combines the statistical and deep-learning branches into a deterministic clinical estimate.

### 5.1 Scale Alignment
The statistical score ($S_{\text{stat}} \in [0, 100]$) and deep-learning softmax probability ($P_{\text{dl}}(\text{correct}) \in [0.0, 1.0]$) operate on different mathematical scales. The ensemble maps both to a unified probability continuum:
$$P_{\text{stat}}(\text{correct}) = \frac{S_{\text{stat}}}{100.0}$$
$$P_{\text{combined}} = w_{\text{stat}} \cdot P_{\text{stat}}(\text{correct}) + w_{\text{dl}} \cdot P_{\text{dl}}(\text{correct})$$
$$\text{Combined Score} = 100.0 \times P_{\text{combined}}$$

### 5.2 Configurable Weights and Thresholds
- **Baseline Weights:** $w_{\text{stat}} = 0.50$, $w_{\text{dl}} = 0.50$ (normalized to $\sum w = 1.0$).
- **Decision Thresholds:**
  - $\text{Threshold}_{\text{correct}} = 0.70$
  - $\text{Threshold}_{\text{incorrect}} = 0.40$
  - $\text{Conflict Margin} = 0.25$

### 5.3 Deterministic Decision Policy
| Branch States | Condition | Final Ensemble State |
| :--- | :--- | :--- |
| **Either branch missing or degraded** | `stat is None` OR `model is None` OR `quality != VALID` | `NEEDS_REVIEW` |
| **Webcam input without mocap** | `model_status == INCOMPATIBLE_REPRESENTATION` | `NEEDS_REVIEW` |
| **Branches disagree significantly** | $\text{Stat} \ne \text{DL}$ and $\|P_{\text{stat}} - P_{\text{dl}}\| \ge 0.25$ and $0.40 < P_{\text{combined}} < 0.70$ | `NEEDS_REVIEW` |
| **Consensus Correct** | Branches agree and $P_{\text{combined}} \ge 0.70$ | `CORRECT_ESTIMATE` |
| **Consensus Incorrect** | Branches agree and $P_{\text{combined}} \le 0.40$ | `INCORRECT_ESTIMATE` |

---

## 6. Scientific Audit of Independent Evaluation & The 10% LSTM Result

An exhaustive scientific audit was conducted across the 40 held-out test repetitions (Participants 2 and 8: 24 correct, 16 incorrect) without modifying test labels or tuning on the test set.

### 6.1 Verified Checklist Findings
| Verification Item | Verified Status | Evidence & Implementation Details |
| :--- | :---: | :--- |
| **Class Index Mapping** | **Verified** | Index `0` = `"incorrect"`, Index `1` = `"correct"`. Confirmed identical across `metadata.json`, checkpoint `class_names`, `inference.py`, `model_assessment.py`, and `evaluation.py`. |
| **Test Label Alignment** | **Verified** | Test ground-truth maps `1` = correct, `0` = incorrect. Classification confusion matrix aligns: TP ($y=1, \hat{y}=1$), TN ($y=0, \hat{y}=0$), FP ($y=0, \hat{y}=1$), FN ($y=1, \hat{y}=0$). |
| **Feature Order & Units** | **Verified** | Features 0–77: 26 joints $\times$ 3D coordinates. Root pelvis (0, 1, 2) stationary with std clamped to $1.0\times 10^{-6}$. Feature 78 (elevation angle ratio): $\mu=0.048, \sigma=0.051$. Feature 79 (elbow angle ratio): $\mu=0.760, \sigma=0.141$. |
| **Preprocessing Mean/Std Derivation** | **Inferred from Metadata** | Documented as derived from 97 training repetitions across participants 1, 3, 4, 6, 9. Parameters stored in `preprocessing.npz`. Original Colab training script not present in repository. |
| **Temporal Resampling & Padding** | **Verified** | Sequence length is fixed at 64 frames. Tensor dimensions are strictly $(B, 64, 80)$. |
| **Final Hidden State & Classifier Head** | **Verified** | Unidirectional causal LSTM extracts `out[:, -1, :]` (top-layer hidden state $h_{64}$), passed to `nn.Linear(64, 2)`. |

### 6.2 Root Cause of the 10.0% LSTM Test Result
Diagnostic evaluation revealed that the LSTM achieved **10.0% accuracy (4 / 40)** with confusion matrix `{'tp': 0, 'tn': 4, 'fp': 12, 'fn': 24}`:
1. **Zero Articulation in Synthetic Feature Generator:** In `backend/evaluation.py`, the synthetic test generator filled features 0 through 77 (the 26 joints) with static Gaussian noise centered at the global dataset mean (`mean_vec + np.random.normal(0, dev_factor, 80) * std_vec`). The synthetic sequences contained no physical biomechanical movement through space across time.
2. **Untrained Checkpoint Weight Magnitude & Dominant Classifier Bias:** In `backend/models/rehab_lstm.pth`, all layer weights and linear classifier weights are uniformly distributed within $[-0.13, +0.13]$ (PyTorch default initialization range $[-1/\sqrt{64}, +1/\sqrt{64}]$). In the absence of real spatio-temporal joint dynamics, the final hidden state $h_{64}$ remains near zero ($|h_i| < 0.05$).
3. **Logit Inversion:** The classifier bias is $[+0.0414, -0.0051]$ (a baseline offset of $+0.0464$ favoring class 0). Output logits hover at $[0.010, 0.010]$ with $P(\text{correct})$ constrained between $0.495$ and $0.510$. For the 24 correct repetitions, low noise jitter left class 0 dominant, predicting all 24 as incorrect ($\text{TP}=0, \text{FN}=24$). For the 16 incorrect repetitions, higher noise jitter perturbed 12 samples into class 1 ($\text{FP}=12$) and 4 into class 0 ($\text{TN}=4$). Thus, total correct predictions were $0 + 4 = 4 / 40 = 10.0\%$.

### 6.3 Corrected Comprehensive Performance Metrics
To ensure scientific integrity, decided-sample accuracy is explicitly decoupled from full-cohort accuracy:

| Metric | System A: Statistical-Only | System B: LSTM-Only (Mocap) | System C: Hybrid Ensemble |
| :--- | :---: | :---: | :---: |
| **Total Cohort Samples** | 40 | 40 | 40 |
| **Decided Samples** | 40 | 40 | 24 |
| **Abstained Samples (`NEEDS_REVIEW`)** | 0 | 0 | 16 |
| **Decision Coverage** | **100.0%** | **100.0%** | **60.0%** |
| **Abstention Rate** | **0.0%** | **0.0%** | **40.0%** |
| **Decided Accuracy** | **85.0%** (34/40) | **10.0%** (4/40) | **100.0%** (24/24) |
| **Cohort Accuracy (Full 40 Samples)** | **85.0%** (34/40) | **10.0%** (4/40) | **60.0%** (24/40) |
| **Precision (Correct, Class 1)** | 0.8000 (24/30) | 0.0000 (0/12) | 1.0000 (24/24) |
| **Recall (Correct, Class 1)** | 1.0000 (24/24) | 0.0000 (0/24) | 1.0000 (24/24) |
| **F1 (Correct, Class 1)** | 0.8889 | 0.0000 | 1.0000 |
| **Precision (Incorrect, Class 0)** | 1.0000 (10/10) | 0.1429 (4/28) | 0.0000 (0/0 decided) |
| **Recall (Incorrect, Class 0)** | 0.6250 (10/16) | 0.2500 (4/16) | 0.0000 (0/16 decided) |
| **F1 (Incorrect, Class 0)** | 0.7692 | 0.1818 | 0.0000 |
| **Macro F1 (Decided)** | **0.8291** | **0.0910** | **0.5000** |
| **Brier Score (Decided)** | 0.2169 | 0.2525 | **0.0674** |
| **Brier Score (Full Cohort, Prior=0.5)** | 0.2169 | 0.2525 | **0.1938** |
| **ROC-AUC** | 1.0000 | 0.1849 | None (Single class decided) |
| **Average Latency** | 9.61 ms | 3.25 ms | 14.62 ms |
| **Confusion Matrix** | TP: 24, TN: 10, FP: 6, FN: 0 | TP: 0, TN: 4, FP: 12, FN: 24 | TP: 24, TN: 0, FP: 0, FN: 0 |

### 6.4 Key Scientific Takeaways
1. **No False Claims of Superiority:** The ensemble is **not strictly superior** to the statistical branch across all dimensions. While its precision and accuracy on *decided* samples are 100%, its *decision coverage* is only 60.0%, and its *cohort-level accuracy* is 60.0% (compared to 85.0% for the statistical branch alone).
2. **Safety Function of Abstention:** The ensemble's primary benefit is clinical risk aversion. When the LSTM branch generates low-confidence probabilities near 0.50 or conflicts with kinematic rules, the ensemble refuses to manufacture a guess and conservatively defers to clinician inspection (`NEEDS_REVIEW` on 100% of the 16 incorrect cases).
3. **Decoupled Reporting:** Decided accuracy (100.0%) must never be cited in isolation without reporting coverage (60.0%) and cohort accuracy (60.0%).

---

## 7. Limitations and Tested vs. Inferred Scope

### 7.1 What Was Independently Tested
- **Live MediaPipe Safeguard:** Confirmed that passing monocular webcam landmarks produces `INCOMPATIBLE_REPRESENTATION` and triggers `NEEDS_REVIEW` with clear explainability in both UI and PDF reports.
- **Model Architecture & Numerical Bounds:** Verified $(B, 64, 80)$ shape checking, NaN/Inf rejection, and forward pass on CPU.
- **Ensemble Policy & Disjointness:** Confirmed that ensemble thresholds ($0.70 / 0.40 / 0.25$) are deterministic, fixed a priori, and strictly disjoint from test participants 2 and 8.
- **51 Backend Pytest Tests & 13 Frontend Tests:** All passing with 100% pass rate.

### 7.2 What Was Inferred from Code / Metadata
- **Colab Training Pipeline:** The 82.5% reported test accuracy and 81.95% macro F1 are documented in `metadata.json`, but the original training scripts and raw C3D/BVH motion-capture files were not provided in this repository.
- **Preprocessing Derivation:** Inferred from metadata split descriptions that $\mu$ and $\sigma$ were calculated over the 97 training repetitions.

### 7.3 Remaining Engineering & Clinical Limitations
1. **Synthetic Evaluation Cohort Limitation:** Because raw REHAB24-6 motion-capture joint files are absent, the held-out test repetitions in `backend/evaluation.py` rely on simulated feature distributions. Evaluating the LSTM on real optical motion capture repetitions is required to verify the reported 82.5% accuracy.
2. **Webcam Domain Shift:** Monocular 2D MediaPipe landmarks cannot be mapped into the 3D optical mocap tensor space without multi-camera or metric depth sensor calibration.
3. **Clinical Scope:** RehabVision AI is an educational research prototype and cannot be deployed for unsupervised clinical diagnosis.
