"""Phase 3: full Cao et al. dataset (27 subjects), LOSO, with ONLINE label-free alignment of the held-out driver.

Training subjects are aligned per session with their whole session (available offline). The held-out driver's
sessions are aligned per session with one of these references (all label-free, computed from EEG windows only;
transitional and unlabeled trials count too, as in a real drive):
    none        no alignment (the model is trained on unaligned data)
    first30     fixed reference: the first 30 trials of the session (calibration at the start of the drive)
    cum         online: all trials of the session up to and including the current one
    slide100    online: the last 100 trials up to and including the current one (~15 min)
    all         whole session at once (transductive upper bound, not available live)
Only trials labelled alert (0) or drowsy (1) are scored, and only from the 31st trial of each session on, so every
variant is scored on exactly the same trials.

Alignment per model family (reference R = arithmetic mean of the windows' sample covariances X X^T / T):
    icnn_running   Euclidean alignment x <- R^{-1/2} x (one global rescale from the training subjects)
    riemann_lr     re-centring of OAS covariances C <- R^{-1/2} C R^{-1/2}, tangent space at the training mean, LR
    bandpower_lr   log band power minus the reference's mean log band power

    python -m research.src.run_phase3 --out research/results/phase3_local --data <folder of *.npz> --subjects 5,35 --seeds 0
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .classical import band_power_features
from .data import find_cao, load_cao
from .run_phase0 import RECIPES, append_csv, env_info, predict, train_deep
from .run_align import inv_sqrtm

MODELS = ["icnn_running", "riemann_lr", "bandpower_lr"]
MODES = ["none", "first30", "cum", "slide100", "all"]
SKIP_FIRST = 30
warnings.filterwarnings("ignore", category=UserWarning)  # single-class subjects in balanced accuracy


def reference_means(c, mode):
    """c: (n, ...) per-window statistics of ONE session in time order -> (n, ...) reference mean used for each window."""
    n = len(c)
    if mode == "all":
        return np.broadcast_to(c.mean(0), c.shape)
    if mode == "first30":
        return np.broadcast_to(c[:SKIP_FIRST].mean(0), c.shape)
    csum = np.cumsum(c, axis=0)
    if mode == "cum":
        return csum / np.arange(1, n + 1).reshape(-1, *[1] * (c.ndim - 1))
    if mode.startswith("slide"):
        w = int(mode[5:])
        lo = np.concatenate([np.zeros((1,) + c.shape[1:]), csum])[np.maximum(0, np.arange(n) - w + 1)]
        return (csum - lo) / np.minimum(np.arange(1, n + 1), w).reshape(-1, *[1] * (c.ndim - 1))
    raise ValueError(mode)


def whiteners(scov, sessions, mode):
    """Per-window R^{-1/2} for every window, each session aligned with `mode`."""
    out = np.empty_like(scov)
    for s in np.unique(sessions):
        m = np.where(sessions == s)[0]
        ref = reference_means(scov[m], mode)
        if mode in ("all", "first30"):
            out[m] = inv_sqrtm(ref[0])
        else:
            out[m] = np.stack([inv_sqrtm(r) for r in ref])
    return out


def scores(y, prob):
    pred = (prob >= 0.5).astype(int)
    two = len(np.unique(y)) == 2
    return dict(bacc=balanced_accuracy_score(y, pred), acc=accuracy_score(y, pred),
                f1_macro=f1_score(y, pred, average="macro", zero_division=0),
                auc=roc_auc_score(y, prob) if two else float("nan"), n_alert=int((y == 0).sum()), n_drowsy=int((y == 1).sum()))


def lr(seed):
    return LogisticRegression(C=1.0, max_iter=5000, random_state=seed)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data")
    p.add_argument("--out", required=True)
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--subjects", default="all")
    p.add_argument("--epoch-scale", type=float, default=1.0, help="multiply epoch counts (smoke tests only)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, modes, seeds = a.models.split(","), a.modes.split(","), [int(s) for s in a.seeds.split(",")]

    folder = find_cao(a.data)
    x, meta = load_cao(folder)
    subj, sess, y = meta.subject.values, meta.session.values, meta.label.values
    pos = meta.groupby("session").cumcount().values  # position of the trial within its session (time order)
    labelled = y >= 0
    scored = labelled & (pos >= SKIP_FIRST)
    subjects = sorted(np.unique(subj)) if a.subjects == "all" else [int(s) for s in a.subjects.split(",")]
    json.dump(dict(vars(a), data=folder, n_trials=len(y), n_labelled=int(labelled.sum()), skip_first=SKIP_FIRST,
                   icnn_recipe={k: v for k, v in RECIPES["icnn_running"].items() if k != "build"}),
              open(os.path.join(a.out, "config.json"), "w"), indent=2)
    json.dump(env_info(device), open(os.path.join(a.out, "env.json"), "w"), indent=2)
    print(f"device={device} trials={len(y)} labelled={labelled.sum()} subjects={subjects}", flush=True)

    first_aligned = next((m for m in modes if m != "none"), None)
    t0 = time.time()
    scov = np.einsum("nct,ndt->ncd", x.astype(np.float64), x.astype(np.float64)) / x.shape[-1]
    W = {m: whiteners(scov, sess, m) for m in set(modes) | {"all"} if m != "none"}
    print(f"[{time.time()-t0:5.0f}s] reference whiteners ready", flush=True)
    covs = feats = None
    if "riemann_lr" in models:
        from pyriemann.estimation import Covariances
        covs = Covariances(estimator="oas").fit_transform(x.astype(np.float64))
    if "bandpower_lr" in models:
        feats = band_power_features(x)
        fref = {m: np.concatenate([reference_means(feats[sess == s], m) for s in pd.unique(sess)]) for m in modes if m != "none"}
    print(f"[{time.time()-t0:5.0f}s] features ready", flush=True)

    ps_path, pred_path = os.path.join(a.out, "per_subject.csv"), os.path.join(a.out, "predictions.csv")

    def record(name, seed, s, mode, prob, test_idx, secs):
        ev = scored[test_idx]
        r = scores(y[test_idx][ev], prob[ev])
        append_csv(ps_path, [dict(model=name, mode=mode, seed=seed, subject=s, **r, seconds=round(secs, 1))])
        append_csv(pred_path, [dict(model=name, mode=mode, seed=seed, subject=s, session=meta.session.values[i],
                                    trial=int(meta.trial.values[i]), y=int(y[i]), prob=round(float(q), 5))
                               for i, q in zip(test_idx[ev], prob[ev])])
        print(f"[{time.time()-t0:6.0f}s] {name:12s} {mode:8s} seed {seed} subj {s:2d}  bacc {r['bacc']:.4f}  auc {r['auc']:.4f}  "
              f"(alert {r['n_alert']}, drowsy {r['n_drowsy']})", flush=True)

    for name in models:
        for seed in (seeds if name == "icnn_running" else seeds[:1]):
            for s in subjects:
                tf = time.time()
                test_all = np.where(subj == s)[0]
                train = np.where((subj != s) & labelled)[0]
                if name == "icnn_running":
                    yt = torch.from_numpy(y.clip(0)).to(device)
                    for trained_on in ["none", "aligned"]:
                        want = [m for m in modes if (m == "none") == (trained_on == "none")]
                        if not want:
                            continue
                        if trained_on == "none":
                            xtr, scale = x[train], 1.0
                        else:
                            xtr = np.einsum("ncd,ndt->nct", W["all"][train], x[train]).astype(np.float32)
                            scale = float(x[train].std() / xtr.std())
                        model, _ = train_deep("icnn_running", torch.from_numpy(xtr * scale).to(device), yt[train],
                                              None, None, seed, a.epoch_scale, device)
                        for m in want:
                            xte = x[test_all] if m == "none" else np.einsum("ncd,ndt->nct", W[m][test_all], x[test_all]).astype(np.float32)
                            prob = predict(model, torch.from_numpy(xte * scale).to(device))
                            record(name, seed, s, m, prob, test_all, time.time() - tf)
                elif name == "riemann_lr":
                    from pyriemann.tangentspace import TangentSpace
                    for m in modes:
                        if m == "none":
                            ctr, cte = covs[train], covs[test_all]
                        else:
                            ctr = W["all"][train] @ covs[train] @ W["all"][train]
                            cte = W[m][test_all] @ covs[test_all] @ W[m][test_all]
                        if m in ("none", first_aligned):  # one fit per training alignment
                            clf = make_pipeline(TangentSpace(metric="riemann"), StandardScaler(), lr(seed)).fit(ctr, y[train])
                        record(name, seed, s, m, clf.predict_proba(cte)[:, 1], test_all, time.time() - tf)
                else:
                    for m in modes:
                        if m == "none":
                            ftr, fte = feats[train], feats[test_all]
                        else:
                            ftr, fte = feats[train] - fref["all"][train], feats[test_all] - fref[m][test_all]
                        if m in ("none", first_aligned):
                            clf = make_pipeline(StandardScaler(), lr(seed)).fit(ftr, y[train])
                        record(name, seed, s, m, clf.predict_proba(fte)[:, 1], test_all, time.time() - tf)

    df = pd.read_csv(ps_path)
    summ = (df.groupby(["model", "mode", "seed"])[["bacc", "auc"]].mean().groupby(["model", "mode"])
            .agg(["mean", "std"]).round(4))
    summ.to_csv(os.path.join(a.out, "summary.csv"))
    print(summ.to_string())
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
