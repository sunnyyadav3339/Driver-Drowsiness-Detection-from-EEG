"""Feature-based baselines: log band power and Riemannian tangent space, each with logistic regression."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FS

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)}


def band_power_features(x):
    """x (N, C, T) -> log band power (N, C * n_bands)."""
    freqs, psd = welch(x, fs=FS, nperseg=FS * 2, axis=-1)
    feats = [np.log(psd[..., (freqs >= lo) & (freqs < hi)].mean(axis=-1) + 1e-12) for lo, hi in BANDS.values()]
    return np.stack(feats, axis=-1).reshape(len(x), -1)


class BandPowerLR:
    def __init__(self, seed):
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=seed))

    def fit(self, x, y):
        self.clf.fit(band_power_features(x), y)
        return self

    def predict_proba(self, x):
        return self.clf.predict_proba(band_power_features(x))[:, 1]


class RiemannTangentLR:
    """OAS covariance -> tangent space at the training-set Riemannian mean -> logistic regression."""

    def __init__(self, seed):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace

        self.clf = make_pipeline(
            Covariances(estimator="oas"),
            TangentSpace(metric="riemann"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=5000, random_state=seed),
        )

    def fit(self, x, y):
        self.clf.fit(x.astype(np.float64), y)
        return self

    def predict_proba(self, x):
        return self.clf.predict_proba(x.astype(np.float64))[:, 1]
