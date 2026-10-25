import os
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List

class RehabLSTM(nn.Module):
    def __init__(self, input_size: int = 80, hidden_size: int = 64, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        out, (h_n, c_n) = self.lstm(x)
        logits = self.classifier(out[:, -1, :])
        return logits

class LSTMInferenceEngine:
    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
        self.models_dir = models_dir
        
        self.model_path = os.path.join(models_dir, "rehab_lstm.pth")
        self.prep_path = os.path.join(models_dir, "preprocessing.npz")
        self.meta_path = os.path.join(models_dir, "metadata.json")

        self.model: Optional[RehabLSTM] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None

        self._load_artifacts()

    def _load_artifacts(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Missing model checkpoint: {self.model_path}")
            if not os.path.exists(self.prep_path):
                raise FileNotFoundError(f"Missing preprocessing parameters: {self.prep_path}")
            if not os.path.exists(self.meta_path):
                raise FileNotFoundError(f"Missing model metadata: {self.meta_path}")

            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)

            prep = np.load(self.prep_path)
            self.feature_mean = prep["feature_mean"].astype(np.float32)
            self.feature_std = prep["feature_std"].astype(np.float32)

            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)

            input_size = int(ckpt.get("input_size", 80))
            hidden_size = int(ckpt.get("hidden_size", 64))
            num_layers = int(ckpt.get("num_layers", 2))
            num_classes = 2

            self.model = RehabLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes
            )
            self.model.load_state_dict(state_dict)
            self.model.eval()

            # Startup validation check
            test_seq = np.zeros((1, 64, 80), dtype=np.float32)
            norm_test = (test_seq - self.feature_mean) / (self.feature_std + 1e-8)
            with torch.no_grad():
                test_out = self.model(torch.tensor(norm_test, dtype=torch.float32))
            assert test_out.shape == (1, 2), f"Startup verification failed: expected shape (1, 2), got {test_out.shape}"

            self.is_loaded = True
            print("LSTMInferenceEngine initialized successfully and startup test passed.")
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            print(f"LSTMInferenceEngine initialization warning/failure: {e}")

    def predict_compatible_sequence(self, sequence: List[List[float]]) -> Dict[str, Any]:
        """
        Run inference on a verified 80-feature sequence of length 64.
        sequence shape: (64, 80)
        """
        if not self.is_loaded or self.model is None:
            return {
                "status": "error",
                "message": f"Model not loaded: {self.load_error}"
            }

        seq_np = np.array(sequence, dtype=np.float32)
        if seq_np.ndim == 2:
            if seq_np.shape != (64, 80):
                return {
                    "status": "invalid_shape",
                    "message": f"Invalid sequence shape {seq_np.shape}. Expected (64, 80): 64 timesteps x 80 features."
                }
            seq_np = np.expand_dims(seq_np, axis=0) # shape (1, 64, 80)
        elif seq_np.ndim == 3:
            if seq_np.shape[1:] != (64, 80):
                return {
                    "status": "invalid_shape",
                    "message": f"Invalid batch sequence shape {seq_np.shape}. Expected (B, 64, 80)."
                }
        else:
            return {
                "status": "invalid_shape",
                "message": f"Invalid dimensions: {seq_np.ndim}. Expected 2D or 3D array."
            }

        # Apply exact preprocessing normalization
        norm_seq = (seq_np - self.feature_mean) / (self.feature_std + 1e-8)
        tensor_input = torch.tensor(norm_seq, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))

        class_map = self.metadata.get("class_mapping", {"0": "incorrect", "1": "correct"})
        class_label = class_map.get(str(pred_class), f"class_{pred_class}")

        return {
            "status": "success",
            "predicted_class": pred_class,
            "class_label": class_label,
            "probabilities": {
                "incorrect": round(float(probs[0]), 4),
                "correct": round(float(probs[1]), 4)
            },
            "input_shape": list(seq_np.shape),
            "warning": "Evaluated on compatible 80-feature motion-capture sequence format. Preliminary evaluation only (82.5% held-out test accuracy, 81.95% macro F1). Not validated for clinical decision-making.",
            "educational_note": "This prediction uses the real trained PyTorch LSTM weights from the REHAB24-6 dataset."
        }

    def generate_sample_mocap_repetition(self, is_correct: bool = True) -> List[List[float]]:
        """
        Synthesize a realistic 80-feature sequence based on feature mean and standard deviation
        for demonstration and testing of the compatible LSTM inference pipeline.
        """
        if self.feature_mean is None or self.feature_std is None:
            # fallback zeros
            return [[0.0] * 80 for _ in range(64)]

        mean_vec = self.feature_mean[0, 0, :]
        std_vec = self.feature_std[0, 0, :]

        seq = np.zeros((64, 80), dtype=np.float32)
        t = np.linspace(0, np.pi, 64)

        for step in range(64):
            # Base posture near feature mean
            seq[step, :] = mean_vec + np.random.normal(0, 0.05, 80) * std_vec
            # Arm elevation feature (feature index 78)
            if is_correct:
                seq[step, 78] = 15.0 + 80.0 * np.sin(t[step]) # smooth 95 deg arc
                seq[step, 79] = 172.0 - 5.0 * np.sin(t[step]) # extended elbow
            else:
                seq[step, 78] = 15.0 + 45.0 * np.sin(t[step]) # restricted 60 deg
                seq[step, 79] = 130.0 - 20.0 * np.sin(t[step]) # bent elbow compensation

        return seq.tolist()

# Singleton engine instance
lstm_engine = LSTMInferenceEngine()
