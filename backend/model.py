import torch
import torch.nn as nn
import numpy as np

class InterpretableCNN(nn.Module):
    """Cui et al., TNNLS 2022 implementation.

    The paper's BatchNorm uses per-batch statistics only (track_running_stats=False).
    That makes single-epoch inference degenerate: with a batch of one, every channel
    is normalized to zero mean, global average pooling then outputs the BN bias, and
    the prediction is a constant regardless of input. For live streaming we keep
    running statistics from training so each epoch can be scored independently.
    """
    def __init__(self, channels=30, n1=16, kernel_length=64, sample_length=384, classes=2):
        super().__init__()
        self.pointwise = nn.Conv2d(1, n1, (channels, 1))
        self.depthwise = nn.Conv2d(n1, n1 * 2, (1, kernel_length), groups=n1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(n1 * 2)
        self.gap = nn.AvgPool2d((1, sample_length - kernel_length + 1))
        self.fc = nn.Linear(n1 * 2, classes)

    def forward(self, x):
        x = self.pointwise(x)
        x = self.depthwise(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DrowsinessPredictor:
    def __init__(self, weights_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InterpretableCNN().to(self.device)
        state = torch.load(weights_path, map_location=self.device)
        if "bn.running_mean" not in state:
            raise RuntimeError(
                f"{weights_path} has no BatchNorm running statistics (trained with the old "
                "per-batch BN) and would give a constant prediction. Retrain with training/train.py."
            )
        self.model.load_state_dict(state)
        self.model.eval()
        self.softmax = nn.Softmax(dim=1)

    def predict_epoch(self, epoch_data: np.ndarray) -> float:
        """
        Input shape: (30, 384) or (1, 30, 384)
        Returns: float probability of drowsiness (class 1)
        """
        if epoch_data.ndim == 2:
            epoch_data = epoch_data[np.newaxis, np.newaxis, :, :]  # Shape: (1, 1, 30, 384)
        elif epoch_data.ndim == 3:
            epoch_data = epoch_data[np.newaxis, :, :, :]           # Shape: (1, 1, 30, 384)

        tensor_in = torch.from_numpy(epoch_data.astype(np.float32)).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = self.softmax(logits)
            drowsy_prob = probs[0, 1].item()

        return drowsy_prob
