"""E2: online alignment with a population prior, on the full Cao et al. dataset (27 subjects, LOSO).

The held-out driver's reference blends their running covariance with the training drivers' mean covariance:
    R_t = (n R_hat_t + kappa R_pop) / (n + kappa),   n = windows seen so far in the session, current one included
R_pop uses training subjects only and is recomputed in each fold; all their windows count, labelled or not, and no labels are used.
Alignment therefore exists from the first window. Training subjects are whitened with their whole sessions, as in Phase 3.

Modes:  none; prior<kappa>; pop (R_pop alone, a control with no data from the driver).
Primary comparison, fixed before seeing results: prior30 vs none, on labelled trials from the 31st of each session.
Files:  per_subject.csv (from the 31st trial, as Phase 3), per_subject_all.csv (from the first trial), predictions.csv (all labelled trials).

    python -m research.src.run_prior --out research/results/prior_local --data <sessions folder> --subjects 5,35 --seeds 0 --epoch-scale 0.1
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import torch

from .data import find_cao, load_cao
from .run_phase0 import RECIPES, append_csv, env_info, predict, train_deep
from .run_phase3 import SKIP_FIRST, reference_means, scores
from .run_shrink import apply_whitener, inv_sqrtm_batch

warnings.filterwarnings("ignore", category=UserWarning)  # single-class subjects in balanced accuracy


def prior_references(c, r_pop, kappa):
    """c: (n, C, C) sample covariances of one session in time order -> (n, C, C) references. kappa=None means R_pop alone."""
    if kappa is None:
        return np.broadcast_to(r_pop, c.shape).copy()
    n = np.arange(1, len(c) + 1).reshape(-1, 1, 1)
    return (n * reference_means(c, "cum") + kappa * r_pop) / (n + kappa)


def self_test():
    rng = np.random.default_rng(0)
    c = np.stack([np.cov(rng.normal(size=(30, 200))) for _ in range(400)])
    pop = np.eye(30) * 7.0
    assert np.allclose(prior_references(c[:1], pop, 0)[0], c[0]), "kappa 0 must be the driver's own reference"
    first = prior_references(c, pop, 30)[0]
    assert np.allclose(first, (c[0] + 30 * pop) / 31), "window 1 must be dominated by the prior"
    late = prior_references(c, pop, 1e-6)[-1]
    assert np.allclose(late, c.mean(0)), "small kappa must approach the running mean"
    assert np.allclose(prior_references(c, pop, None)[5], pop), "pop mode ignores the driver"
    # Causality: changing a later window must not change an earlier reference.
    d = c.copy()
    d[100] += np.eye(30)
    assert np.allclose(prior_references(c, pop, 30)[:100], prior_references(d, pop, 30)[:100]), "reference uses future windows"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data")
    p.add_argument("--out", required=True)
    p.add_argument("--kappas", default="30,10,100")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--subjects", default="all")
    p.add_argument("--epoch-scale", type=float, default=1.0, help="multiply epoch counts (smoke tests only)")
    a = p.parse_args()
    self_test()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kappas = [float(k) for k in a.kappas.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]

    folder = find_cao(a.data)
    x, meta = load_cao(folder)
    subj, sess, y = meta.subject.values.astype(int), meta.session.values.astype(str), meta.label.values.astype(int)
    pos = meta.groupby("session").cumcount().values
    labelled = y >= 0
    subjects = [int(s) for s in (np.unique(subj) if a.subjects == "all" else a.subjects.split(","))]
    json.dump(dict(vars(a), data=folder, n_trials=len(y), n_labelled=int(labelled.sum()), primary="prior30 vs none, from the 31st trial",
                   icnn_recipe={k: v for k, v in RECIPES["icnn_running"].items() if k != "build"}), open(os.path.join(a.out, "config.json"), "w"), indent=2)
    json.dump(env_info(device), open(os.path.join(a.out, "env.json"), "w"), indent=2)
    print(f"device={device} trials={len(y)} subjects={subjects} kappas={kappas} seeds={seeds}", flush=True)

    t0 = time.time()
    scov = np.einsum("nct,ndt->ncd", x.astype(np.float64), x.astype(np.float64)) / x.shape[-1]
    w_all = np.empty_like(scov)
    for s in np.unique(sess):
        m = np.where(sess == s)[0]
        w_all[m] = inv_sqrtm_batch(scov[m].mean(0, keepdims=True))[0]
    yt = torch.from_numpy(y.clip(0)).to(device)

    def record(seed, s, mode, prob, test):
        ok = labelled[test]
        late = ok & (pos[test] >= SKIP_FIRST)
        r = scores(y[test][late], prob[late])
        append_csv(os.path.join(a.out, "per_subject.csv"), [dict(model="icnn_running", mode=mode, seed=seed, subject=s, **r)])
        append_csv(os.path.join(a.out, "per_subject_all.csv"), [dict(model="icnn_running", mode=mode, seed=seed, subject=s, **scores(y[test][ok], prob[ok]))])
        append_csv(os.path.join(a.out, "predictions.csv"), [dict(model="icnn_running", mode=mode, seed=seed, subject=s, session=sess[i], trial=int(meta.trial.values[i]),
                                                                 y=int(y[i]), prob=round(float(q), 5)) for i, q in zip(test[ok], prob[ok])])
        print(f"[{time.time()-t0:6.0f}s] {mode:9s} seed {seed} subj {s:2d}  bacc {r['bacc']:.4f}  auc {r['auc']:.4f}", flush=True)

    for seed in seeds:
        for s in subjects:
            test, train = np.where(subj == s)[0], np.where((subj != s) & labelled)[0]
            r_pop = scov[subj != s].mean(0)
            cond_pop = inv_sqrtm_batch(r_pop[None])[1][0]

            model, _ = train_deep("icnn_running", torch.from_numpy(x[train]).to(device), yt[train], None, None, seed, a.epoch_scale, device)
            record(seed, s, "none", predict(model, torch.from_numpy(x[test]).to(device)), test)

            xtr = apply_whitener(w_all[train], x[train])
            scale = float(x[train].std() / xtr.std())
            model, _ = train_deep("icnn_running", torch.from_numpy(xtr * scale).to(device), yt[train], None, None, seed, a.epoch_scale, device)
            rows = []
            for kappa in [*kappas, None]:
                mode = "pop" if kappa is None else f"prior{kappa:g}"
                w = np.empty_like(scov[test])
                for ses in np.unique(sess[test]):
                    m = np.where(sess[test] == ses)[0]
                    w[m], cond = inv_sqrtm_batch(prior_references(scov[test][m], r_pop, kappa))
                    rows.append(dict(seed=seed, subject=s, mode=mode, session=ses, cond_first=cond[0], cond_max=cond.max(), cond_pop=cond_pop))
                record(seed, s, mode, predict(model, torch.from_numpy(apply_whitener(w, x[test]) * scale).to(device)), test)
            append_csv(os.path.join(a.out, "conditioning.csv"), rows)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
