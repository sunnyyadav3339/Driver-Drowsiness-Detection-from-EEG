"""Loading the SADT releases (Cui et al. 2022, extracted from Cao et al. 2019)."""
import glob
import os

import numpy as np
import scipy.io as sio

# Channel order of the SADT release (30 EEG channels, modified 10-20 system).
CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ", "FC4", "FT8", "T3", "C3", "CZ",
    "C4", "T4", "TP7", "CP3", "CPZ", "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2",
]
FS = 128


def find_dataset(path=None):
    """Return the .mat path: an explicit path, else search Kaggle inputs, else the repo's dataset/."""
    if path:
        return path
    candidates = glob.glob("/kaggle/input/**/dataset.mat", recursive=True)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [os.path.join(here, "..", "..", "dataset", "dataset.mat")]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError("dataset.mat not found; pass --data")


def load_sadt(path):
    """Returns x (N, 30, 384) float32, y (N,) int64 {0 alert, 1 drowsy}, subject (N,) int64."""
    mat = sio.loadmat(path)
    x = mat["EEGsample"].astype(np.float32)
    y = mat["substate"].flatten().astype(np.int64)
    subj = mat["subindex"].flatten().astype(np.int64)
    return x, y, subj


def find_cao(path=None):
    """Folder holding the converted Cao et al. sessions (*.npz from convert_cao.py)."""
    if path:
        return path
    for c in glob.glob("/kaggle/input/**/sessions", recursive=True):
        if glob.glob(os.path.join(c, "*.npz")):
            return c
    raise FileNotFoundError("converted Cao sessions not found; pass --data")


def load_cao(folder):
    """Returns x (N, 30, 384) float32 and a DataFrame of per-trial metadata, sorted by session then onset time."""
    import pandas as pd

    xs, metas = [], []
    for f in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        xs.append(d["x"])
        metas.append(pd.DataFrame({k: d[k] for k in d.files if k != "x"}))
    meta = pd.concat(metas, ignore_index=True)
    order = np.lexsort((meta.onset_s.values, meta.session.values))
    return np.concatenate(xs)[order], meta.iloc[order].reset_index(drop=True)
