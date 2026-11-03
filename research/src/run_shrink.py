"""E1: Phase 3 with shrinkage-regularised references, on the full Cao et al. dataset (27 subjects, LOSO).

Phase 3's short references were near-singular (research/paper/LAB_NOTEBOOK.md, 2026-10-30), so their whiteners blew up the
signal. Here every reference, training included, is shrunk towards a scaled identity before whitening:
    R <- (1 - alpha) R + alpha tr(R) / C I
Everything else matches run_phase3.py: same modes, seeds, recipe, and scoring (labelled trials from the 31st of each session).

Results go to <out>/none and <out>/a<alpha>, each with per_subject.csv and predictions.csv in the run_phase3.py format.
<out>/conditioning.csv holds the condition number of each session's reference, before and after shrinkage.

    python -m research.src.run_shrink --out research/results/shrink_local --data <sessions folder> --subjects 5,35 --epoch-scale 0.1
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch

from .data import find_cao, load_cao
from .run_phase0 import RECIPES, append_csv, env_info, predict, train_deep
from .run_phase3 import SKIP_FIRST, reference_means, scores

ALIGNED_MODES = ["first30", "cum", "slide100", "all"]
warnings.filterwarnings("ignore", category=UserWarning)  # single-class subjects in balanced accuracy


def shrink(r, alpha):
    """r: (..., C, C). alpha = 0 returns r unchanged."""
    c = r.shape[-1]
    scale = np.trace(r, axis1=-2, axis2=-1)[..., None, None] / c
    return (1 - alpha) * r + alpha * scale * np.eye(c)


def inv_sqrtm_batch(r):
    """Inverse square root of a stack of symmetric matrices, floored as in run_align.inv_sqrtm. Also returns condition numbers."""
    w, v = np.linalg.eigh(r)
    cond = w[..., -1] / np.maximum(w[..., 0], 1e-300)
    w = np.maximum(w, 1e-10 * w[..., -1:])
    return (v / np.sqrt(w)[..., None, :]) @ np.swapaxes(v, -1, -2), cond


def whiteners(scov, sessions, mode, alpha, log):
    """Per-window R^{-1/2} with each session's reference shrunk; appends conditioning rows to log."""
    out = np.empty_like(scov)
    for s in np.unique(sessions):
        m = np.where(sessions == s)[0]
        ref = reference_means(scov[m], mode)
        fixed = mode in ("all", "first30")
        raw = ref[:1] if fixed else ref
        w, cond = inv_sqrtm_batch(shrink(raw, alpha))
        out[m] = w  # a fixed reference has one whitener, which broadcasts to every window of the session
        _, cond_raw = inv_sqrtm_batch(raw)
        at = min(SKIP_FIRST, len(cond) - 1)
        log.append(dict(alpha=alpha, mode=mode, session=s, cond_raw_at31=cond_raw[0 if fixed else at], cond_at31=cond[0 if fixed else at],
                        cond_median=float(np.median(cond)), cond_max=float(cond.max())))
    return out


def apply_whitener(w, x):
    return np.einsum("ncd,ndt->nct", w, x).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data")
    p.add_argument("--out", required=True)
    p.add_argument("--alphas", default="0.01,0.05,0.1,0.2")
    p.add_argument("--seeds", default="0")
    p.add_argument("--subjects", default="all")
    p.add_argument("--epoch-scale", type=float, default=1.0, help="multiply epoch counts (smoke tests only)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alphas = [float(v) for v in a.alphas.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]

    folder = find_cao(a.data)
    x, meta = load_cao(folder)
    subj, sess, y = meta.subject.values.astype(int), meta.session.values.astype(str), meta.label.values.astype(int)
    pos = meta.groupby("session").cumcount().values
    labelled = y >= 0
    scored = labelled & (pos >= SKIP_FIRST)
    subjects = [int(s) for s in (np.unique(subj) if a.subjects == "all" else a.subjects.split(","))]
    json.dump(dict(vars(a), data=folder, n_trials=len(y), n_labelled=int(labelled.sum()), skip_first=SKIP_FIRST,
                   icnn_recipe={k: v for k, v in RECIPES["icnn_running"].items() if k != "build"}),
              open(os.path.join(a.out, "config.json"), "w"), indent=2)
    json.dump(env_info(device), open(os.path.join(a.out, "env.json"), "w"), indent=2)
    print(f"device={device} trials={len(y)} labelled={labelled.sum()} subjects={subjects} alphas={alphas} seeds={seeds}", flush=True)

    t0 = time.time()
    scov = np.einsum("nct,ndt->ncd", x.astype(np.float64), x.astype(np.float64)) / x.shape[-1]
    yt = torch.from_numpy(y.clip(0)).to(device)

    def record(run, seed, s, mode, prob, test_idx):
        ev = scored[test_idx]
        r = scores(y[test_idx][ev], prob[ev])
        d = os.path.join(a.out, run)
        os.makedirs(d, exist_ok=True)
        append_csv(os.path.join(d, "per_subject.csv"), [dict(model="icnn_running", mode=mode, seed=seed, subject=s, **r)])
        append_csv(os.path.join(d, "predictions.csv"), [dict(model="icnn_running", mode=mode, seed=seed, subject=s, session=sess[i], trial=int(meta.trial.values[i]),
                                                             y=int(y[i]), prob=round(float(q), 5)) for i, q in zip(test_idx[ev], prob[ev])])
        print(f"[{time.time()-t0:6.0f}s] {run:6s} {mode:8s} seed {seed} subj {s:2d}  bacc {r['bacc']:.4f}  auc {r['auc']:.4f}", flush=True)

    for seed in seeds:  # no alignment: one training per fold, the reference for every alpha
        for s in subjects:
            test, train = np.where(subj == s)[0], np.where((subj != s) & labelled)[0]
            model, _ = train_deep("icnn_running", torch.from_numpy(x[train]).to(device), yt[train], None, None, seed, a.epoch_scale, device)
            record("none", seed, s, "none", predict(model, torch.from_numpy(x[test]).to(device)), test)

    for alpha in alphas:
        log = []
        W = {m: whiteners(scov, sess, m, alpha, log) for m in ALIGNED_MODES}
        pd.DataFrame(log).to_csv(os.path.join(a.out, "conditioning.csv"), mode="a", index=False,
                                 header=not os.path.exists(os.path.join(a.out, "conditioning.csv")))
        print(f"[{time.time()-t0:6.0f}s] alpha {alpha}: whiteners ready", flush=True)
        for seed in seeds:
            for s in subjects:
                test, train = np.where(subj == s)[0], np.where((subj != s) & labelled)[0]
                xtr = apply_whitener(W["all"][train], x[train])
                scale = float(x[train].std() / xtr.std())
                model, _ = train_deep("icnn_running", torch.from_numpy(xtr * scale).to(device), yt[train], None, None, seed, a.epoch_scale, device)
                for m in ALIGNED_MODES:
                    xte = apply_whitener(W[m][test], x[test])
                    record(f"a{alpha}", seed, s, m, predict(model, torch.from_numpy(xte * scale).to(device)), test)
        del W
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
