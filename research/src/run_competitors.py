"""E6: competitors to prior-regularised alignment, on the full Cao et al. dataset (27 subjects, LOSO), same scoring as E2.

InterpretableCNN, trained unaligned with running-statistics BN (same seeds and recipe, so `none` must match E2):
    none        training statistics only
    bnadapt30   online BN adaptation: statistics blend training values with the session so far, weight 30, as prior30 does for covariances
    bnadapt10, bnadapt100   sensitivity to that weight
    bnema       exponential moving average of the statistics, momentum 0.1, starting from training values
Riemannian tangent-space LR (deterministic, seed 0):
    riemann_none; riemann_prior30 (online re-centring with the population prior)

Every update is causal: a window's statistics use only that window and earlier ones in its session.
Primary comparison, fixed before seeing results: E2's prior30 vs bnadapt30, balanced accuracy and AUC, from the 31st trial.
timing.csv: CPU milliseconds per window for each online update.

    python -m research.src.run_competitors --out research/results/comp_local --data <sessions folder> --subjects 5,35 --seeds 0 --epoch-scale 0.1
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import find_cao, load_cao
from .models import InterpretableCNN
from .run_phase0 import RECIPES, append_csv, env_info, predict, train_deep
from .run_phase3 import SKIP_FIRST, scores
from .run_prior import prior_references
from .run_shrink import apply_whitener, inv_sqrtm_batch

warnings.filterwarnings("ignore", category=UserWarning)  # single-class subjects in balanced accuracy
BN_KAPPAS = [30, 10, 100]


@torch.no_grad()
def session_features(model, x):
    """Pre-BN features: (n, 32, T) for one session, in time order."""
    return torch.cat([model.relu(model.depthwise(model.pointwise(x[i:i + 256].unsqueeze(1)))).squeeze(2) for i in range(0, len(x), 256)])


@torch.no_grad()
def logits_from(model, feat, mean, var):
    """Apply BN with per-window statistics (n, 32), then the pooling and classifier."""
    norm = (feat - mean.unsqueeze(2)) / torch.sqrt(var + model.bn.eps).unsqueeze(2)
    pooled = norm.mean(2) * model.bn.weight + model.bn.bias
    return model.fc(pooled)


@torch.no_grad()
def bn_stats_blend(model, feat, kappa):
    """Mean and variance per window from training statistics (weight kappa) plus the session so far."""
    m0, v0 = model.bn.running_mean, model.bn.running_var
    n = torch.arange(1, len(feat) + 1, device=feat.device, dtype=feat.dtype).unsqueeze(1)
    m1 = (torch.cumsum(feat.mean(2), 0) + kappa * m0) / (n + kappa)
    s2 = (torch.cumsum((feat ** 2).mean(2), 0) + kappa * (v0 + m0 ** 2)) / (n + kappa)
    return m1, (s2 - m1 ** 2).clamp(min=0)


@torch.no_grad()
def bn_stats_ema(model, feat, momentum=0.1):
    m, s = model.bn.running_mean.clone(), model.bn.running_var + model.bn.running_mean ** 2
    means, seconds = [], []
    wm, ws = feat.mean(2), (feat ** 2).mean(2)
    for t in range(len(feat)):
        m, s = (1 - momentum) * m + momentum * wm[t], (1 - momentum) * s + momentum * ws[t]
        means.append(m)
        seconds.append(s)
    means, seconds = torch.stack(means), torch.stack(seconds)
    return means, (seconds - means ** 2).clamp(min=0)


def self_test():
    torch.manual_seed(0)
    model = InterpretableCNN(bn_mode="running")
    x = torch.randn(40, 30, 384)
    model.train()
    model(x)
    model.eval()
    feat = session_features(model, x)
    plain = model(x)
    m, v = bn_stats_blend(model, feat, 1e9)
    assert (logits_from(model, feat, m, v) - plain).abs().max() < 1e-2, "huge kappa must equal plain inference"
    y = x.clone()
    y[20] += 5.0
    m2, v2 = bn_stats_blend(model, session_features(model, y), 30)
    m1, v1 = bn_stats_blend(model, feat, 30)
    assert (m1[:20] - m2[:20]).abs().max() < 1e-5 and (v1[:20] - v2[:20]).abs().max() < 1e-5, "statistics use future windows"
    me, _ = bn_stats_ema(model, feat)
    mf, _ = bn_stats_ema(model, session_features(model, y))
    assert (me[:20] - mf[:20]).abs().max() < 1e-5, "EMA uses future windows"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--subjects", default="all")
    p.add_argument("--epoch-scale", type=float, default=1.0, help="multiply epoch counts (smoke tests only)")
    a = p.parse_args()
    self_test()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in a.seeds.split(",")]

    folder = find_cao(a.data)
    x, meta = load_cao(folder)
    subj, sess, y = meta.subject.values.astype(int), meta.session.values.astype(str), meta.label.values.astype(int)
    pos = meta.groupby("session").cumcount().values
    labelled = y >= 0
    subjects = [int(s) for s in (np.unique(subj) if a.subjects == "all" else a.subjects.split(","))]
    json.dump(dict(vars(a), data=folder, n_trials=len(y), primary="E2 prior30 vs bnadapt30, from the 31st trial",
                   icnn_recipe={k: v for k, v in RECIPES["icnn_running"].items() if k != "build"}), open(os.path.join(a.out, "config.json"), "w"), indent=2)
    json.dump(env_info(device), open(os.path.join(a.out, "env.json"), "w"), indent=2)
    print(f"device={device} trials={len(y)} subjects={subjects} seeds={seeds}", flush=True)

    t0 = time.time()
    x64 = x.astype(np.float64)
    scov = np.einsum("nct,ndt->ncd", x64, x64) / x.shape[-1]
    w_all = np.empty_like(scov)
    for s in np.unique(sess):
        m = np.where(sess == s)[0]
        w_all[m] = inv_sqrtm_batch(scov[m].mean(0, keepdims=True))[0]
    yt = torch.from_numpy(y.clip(0)).to(device)

    def record(model_name, seed, s, mode, prob, test):
        ok = labelled[test]
        late = ok & (pos[test] >= SKIP_FIRST)
        r = scores(y[test][late], prob[late])
        append_csv(os.path.join(a.out, "per_subject.csv"), [dict(model=model_name, mode=mode, seed=seed, subject=s, **r)])
        append_csv(os.path.join(a.out, "per_subject_all.csv"), [dict(model=model_name, mode=mode, seed=seed, subject=s, **scores(y[test][ok], prob[ok]))])
        append_csv(os.path.join(a.out, "predictions.csv"), [dict(model=model_name, mode=mode, seed=seed, subject=s, session=sess[i], trial=int(meta.trial.values[i]),
                                                                 y=int(y[i]), prob=round(float(q), 5)) for i, q in zip(test[ok], prob[ok])])
        print(f"[{time.time()-t0:6.0f}s] {mode:15s} seed {seed} subj {s:2d}  bacc {r['bacc']:.4f}  auc {r['auc']:.4f}", flush=True)

    xt = torch.from_numpy(x).to(device)
    for seed in seeds:
        for s in subjects:
            test, train = np.where(subj == s)[0], np.where((subj != s) & labelled)[0]
            model, _ = train_deep("icnn_running", xt[train], yt[train], None, None, seed, a.epoch_scale, device)
            model.eval()
            record("icnn_running", seed, s, "none", predict(model, xt[test]), test)
            probs = {f"bnadapt{k}": np.zeros(len(test)) for k in BN_KAPPAS}
            probs["bnema"] = np.zeros(len(test))
            for ses in np.unique(sess[test]):
                m = np.where(sess[test] == ses)[0]
                feat = session_features(model, xt[test[m]])
                for k in BN_KAPPAS:
                    probs[f"bnadapt{k}"][m] = F.softmax(logits_from(model, feat, *bn_stats_blend(model, feat, k)), 1)[:, 1].cpu().numpy()
                probs["bnema"][m] = F.softmax(logits_from(model, feat, *bn_stats_ema(model, feat)), 1)[:, 1].cpu().numpy()
            for mode, pr in probs.items():
                record("icnn_running", seed, s, mode, pr, test)

    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    covs = Covariances(estimator="oas").fit_transform(x64)
    for s in subjects:
        test, train = np.where(subj == s)[0], np.where((subj != s) & labelled)[0]
        clf = make_pipeline(TangentSpace(metric="riemann"), StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=0)).fit(covs[train], y[train])
        record("riemann_lr", 0, s, "riemann_none", clf.predict_proba(covs[test])[:, 1], test)
        ctr = w_all[train] @ covs[train] @ w_all[train]
        clf = make_pipeline(TangentSpace(metric="riemann"), StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=0)).fit(ctr, y[train])
        r_pop = scov[subj != s].mean(0)
        w = np.empty_like(scov[test])
        for ses in np.unique(sess[test]):
            m = np.where(sess[test] == ses)[0]
            w[m] = inv_sqrtm_batch(prior_references(scov[test][m], r_pop, 30))[0]
        record("riemann_lr", 0, s, "riemann_prior30", clf.predict_proba(w @ covs[test] @ w)[:, 1], test)

    # CPU cost of one online update, one window at a time, as a vehicle would run it.
    cpu_model = InterpretableCNN(bn_mode="running")
    cpu_model.eval()
    sample, r_pop = x[:200], scov[:200].mean(0)
    rows, n, total, feat_m, feat_s = [], 0, np.zeros_like(r_pop), 0.0, 0.0
    for i in range(len(sample)):
        one = sample[i:i + 1]
        t = time.perf_counter()
        c = (one[0].astype(np.float64) @ one[0].astype(np.float64).T) / one.shape[-1]
        total, n = total + c, n + 1
        w, _ = inv_sqrtm_batch(((total + 30 * r_pop) / (n + 30))[None])
        aligned = apply_whitener(w, one)
        t1 = time.perf_counter()
        with torch.no_grad():
            f = session_features(cpu_model, torch.from_numpy(one))
            feat_m, feat_s = feat_m + f.mean(2), feat_s + (f ** 2).mean(2)
        t2 = time.perf_counter()
        with torch.no_grad():
            cpu_model(torch.from_numpy(aligned))
        t3 = time.perf_counter()
        if i >= 20:
            rows.append(dict(window=i, whiten_ms=1e3 * (t1 - t), bn_update_ms=1e3 * (t2 - t1), forward_ms=1e3 * (t3 - t2)))
    append_csv(os.path.join(a.out, "timing.csv"), rows)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
