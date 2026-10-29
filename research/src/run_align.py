"""Phase 2 probe (direction B): label-free per-subject alignment under LOSO on SADT-2022.

Every subject (training and held-out) is re-centred using a *reference set* of its own epochs, without labels:
    none      no alignment (Phase 0 baseline)
    all       all epochs of the subject (transductive upper bound: needs the whole session)
    rand<k>   k random epochs of the subject (a short unlabeled calibration; class mix ~ the session's)
    alert<k>  the subject's first k alert epochs (simulates "first k*3 s of the drive, assumed alert").
              NOTE: SADT-2022 stores each subject's alert epochs first, so these are guaranteed alert;
              in a real drive the first minutes are only *probably* alert. Optimistic for that reason.
    mix<p>    30 random epochs of which p% are drowsy (labels used only to build the mix; the method never sees
              them). Measures how sensitive alignment is to the state mix of the calibration window.
A rule "A>B" aligns the training subjects with rule A and the held-out subject with rule B, e.g. "all>mix0":
training subjects are aligned with their whole sessions (available offline), the new driver with a short window.

Alignment per model family:
    icnn_running   Euclidean alignment (He & Wu 2020): x <- R^{-1/2} x, R = mean reference X X^T / T,
                   then a single global scale (from training subjects) to keep the µV-like range.
    riemann_lr     Riemannian re-centring (Zanini et al. 2018): C <- M^{-1/2} C M^{-1/2}, M = Riemannian mean
                   of the reference covariances, then tangent space at the training mean + LR.
    bandpower_lr   baseline normalisation: log band power minus the reference mean (= ratio to baseline power).

Run from the repository root:
    python -m research.src.run_align --out research/results/align_local --subjects 1,2 --seeds 0 --epoch-scale 0.1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .classical import band_power_features
from .data import find_dataset, load_sadt
from .run_phase0 import RECIPES, append_csv, env_info, metrics, predict, summarize, train_deep

MODELS = ["bandpower_lr", "riemann_lr", "icnn_running"]


def reference_indices(rule, y_s, rng):
    """Indices (within one subject) of the unlabeled reference set. Labels are used only to emulate 'alert<k>'."""
    n = len(y_s)
    if rule == "all":
        return np.arange(n)
    if rule.startswith("rand"):
        return np.sort(rng.choice(n, min(int(rule[4:]), n), replace=False))
    if rule.startswith("alert"):
        return np.where(y_s == 0)[0][:int(rule[5:])]
    if rule.startswith("mix"):
        k = 30
        n_d = int(round(k * float(rule[3:]) / 100))
        drowsy, alert = np.where(y_s == 1)[0], np.where(y_s == 0)[0]
        n_d, n_a = min(n_d, len(drowsy)), min(k - n_d, len(alert))
        return np.sort(np.concatenate([rng.choice(drowsy, n_d, replace=False), rng.choice(alert, n_a, replace=False)]))
    raise ValueError(rule)


def inv_sqrtm(m):
    w, v = np.linalg.eigh(m)
    w = np.maximum(w, 1e-10 * w.max())
    return (v / np.sqrt(w)) @ v.T


def subject_refs(rule, y, subj, seed):
    """{subject: absolute indices of its reference set} for every subject, or None for 'none'."""
    if rule == "none":
        return None
    out = {}
    for s in np.unique(subj):
        idx = np.where(subj == s)[0]
        rng = np.random.default_rng(10_000 * seed + int(s))
        out[s] = idx[reference_indices(rule, y[idx], rng)]
    return out


def align_raw(x, subj, refs):
    """Euclidean alignment, per subject."""
    xa = np.empty_like(x)
    for s, ref in refs.items():
        r = np.einsum("nct,ndt->cd", x[ref], x[ref]) / (len(ref) * x.shape[-1])
        m = np.where(subj == s)[0]
        xa[m] = np.einsum("cd,ndt->nct", inv_sqrtm(r), x[m]).astype(np.float32)
    return xa


def align_cov(covs, subj, refs):
    from pyriemann.utils.mean import mean_riemann

    ca = np.empty_like(covs)
    for s, ref in refs.items():
        p = inv_sqrtm(mean_riemann(covs[ref]))
        m = np.where(subj == s)[0]
        ca[m] = p @ covs[m] @ p
    return ca


def align_feat(f, subj, refs):
    fa = np.empty_like(f)
    for s, ref in refs.items():
        m = np.where(subj == s)[0]
        fa[m] = f[m] - f[ref].mean(axis=0)
    return fa


def lr(seed):
    return LogisticRegression(C=1.0, max_iter=5000, random_state=seed)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data")
    p.add_argument("--out", required=True)
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--refs", default="none,all,rand10,rand30,alert10,alert30")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--subjects", default="all")
    p.add_argument("--epoch-scale", type=float, default=1.0, help="multiply epoch counts (smoke tests only)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 1)
    models, rules, seeds = args.models.split(","), args.refs.split(","), [int(s) for s in args.seeds.split(",")]
    data_path = find_dataset(args.data)
    x, y, subj = load_sadt(data_path)
    subjects = sorted(np.unique(subj)) if args.subjects == "all" else [int(s) for s in args.subjects.split(",")]
    json.dump(dict(vars(args), data_path=data_path, n_samples=len(y), protocol="LOSO",
                   reference_rule_rng="numpy default_rng(10000*seed + subject)",
                   icnn_recipe={k: v for k, v in RECIPES["icnn_running"].items() if k != "build"}),
              open(os.path.join(args.out, "config.json"), "w"), indent=2)
    json.dump(env_info(device), open(os.path.join(args.out, "env.json"), "w"), indent=2)
    print(f"device={device} N={len(y)} models={models} refs={rules} seeds={seeds}", flush=True)

    # Label-free per-epoch features, computed once.
    feats = band_power_features(x) if "bandpower_lr" in models else None
    covs = None
    if "riemann_lr" in models:
        from pyriemann.estimation import Covariances
        covs = Covariances(estimator="oas").fit_transform(x.astype(np.float64))

    ps_path, pred_path = os.path.join(args.out, "per_subject.csv"), os.path.join(args.out, "predictions.csv")
    t_start = time.time()
    for name in models:
        for rule in rules:
            # Only the random reference depends on the seed for the deterministic classical models.
            run_seeds = seeds if (name == "icnn_running" or "rand" in rule or "mix" in rule) else [seeds[0]]
            for seed in run_seeds:
                train_rule, test_rule = rule.split(">") if ">" in rule else (rule, rule)
                aligned = {}
                for r in {train_rule, test_rule}:
                    refs = subject_refs(r, y, subj, seed)
                    if name == "bandpower_lr":
                        aligned[r] = (feats if refs is None else align_feat(feats, subj, refs), refs)
                    elif name == "riemann_lr":
                        aligned[r] = (covs if refs is None else align_cov(covs, subj, refs), refs)
                    else:
                        aligned[r] = (x if refs is None else align_raw(x, subj, refs), refs)
                data, _ = aligned[train_rule]
                data_test, refs = aligned[test_rule]
                for s in subjects:
                    t0 = time.time()
                    test, train = np.where(subj == s)[0], np.where(subj != s)[0]
                    if name == "bandpower_lr":
                        clf = make_pipeline(StandardScaler(), lr(seed)).fit(data[train], y[train])
                        prob = clf.predict_proba(data_test[test])[:, 1]
                    elif name == "riemann_lr":
                        from pyriemann.tangentspace import TangentSpace
                        clf = make_pipeline(TangentSpace(metric="riemann"), StandardScaler(), lr(seed)).fit(data[train], y[train])
                        prob = clf.predict_proba(data_test[test])[:, 1]
                    else:
                        # One global scale from training subjects only, so aligned inputs keep the raw data's range.
                        scale = float(x[train].std() / data[train].std())
                        yt = torch.from_numpy(y).to(device)
                        model, _ = train_deep("icnn_running", torch.from_numpy(data[train] * scale).to(device), yt[train],
                                              None, None, seed, args.epoch_scale, device)
                        prob = predict(model, torch.from_numpy(data_test[test] * scale).to(device))
                    m = metrics(y[test], prob)
                    secs = time.time() - t0
                    append_csv(ps_path, [dict(model=name, mode=rule, seed=seed, subject=s, n=len(test), **m,
                                              n_ref=0 if refs is None else len(refs[s]), seconds=round(secs, 1))])
                    append_csv(pred_path, [dict(model=name, mode=rule, seed=seed, subject=s, index=int(i), y=int(y[i]),
                                                prob=round(float(q), 6)) for i, q in zip(test, prob)])
                    print(f"[{time.time()-t_start:6.0f}s] {name:12s} {rule:8s} seed {seed} subj {s:2d}  acc {m['acc']:.4f}  "
                          f"auc {m['auc']:.4f}  ({secs:.1f}s)", flush=True)
        summarize(args.out)
    print(f"done in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
