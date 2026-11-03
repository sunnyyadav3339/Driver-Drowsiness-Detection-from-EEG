"""E7: does a batch-norm statistics update on top of online alignment help drowsiness detection? 27 subjects, LOSO, InterpretableCNN.

Same aligned model as E2 (same seeds and recipe, so prior30 must match E2). Windows are whitened online with kappa = 30, then:
    prior30            no BN update
    prior30_bnema07    EMA of BN statistics, weight 0.7 on the current trial: the setting of Duan et al. 2025 (arXiv:2509.19403, Eq. 11)
    prior30_bnema01    EMA, weight 0.1
    prior30_bn30       statistics blended with training values, weight 30, as in E6
This reproduces only the two alignment stages of Duan et al., not their self-supervised loss.
Primary comparison, fixed before seeing results: prior30_bnema07 vs prior30, balanced accuracy and AUC, from the 31st trial.

    python -m research.src.run_prior_bn --out research/results/e7_local --data <sessions folder> --subjects 5,35 --seeds 0 --epoch-scale 0.1
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from .data import find_cao, load_cao
from .run_competitors import bn_stats_blend, bn_stats_ema, logits_from, session_features
from .run_phase0 import RECIPES, append_csv, env_info, predict, train_deep
from .run_phase3 import SKIP_FIRST, scores
from .run_prior import prior_references
from .run_shrink import apply_whitener, inv_sqrtm_batch

warnings.filterwarnings("ignore", category=UserWarning)  # single-class subjects in balanced accuracy


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--subjects", default="all")
    p.add_argument("--epoch-scale", type=float, default=1.0, help="multiply epoch counts (smoke tests only)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in a.seeds.split(",")]
    folder = find_cao(a.data)
    x, meta = load_cao(folder)
    subj, sess, y = meta.subject.values.astype(int), meta.session.values.astype(str), meta.label.values.astype(int)
    pos = meta.groupby("session").cumcount().values
    labelled = y >= 0
    subjects = [int(s) for s in (np.unique(subj) if a.subjects == "all" else a.subjects.split(","))]
    json.dump(dict(vars(a), data=folder, n_trials=len(y), primary="prior30_bnema07 vs prior30, from the 31st trial",
                   icnn_recipe={k: v for k, v in RECIPES["icnn_running"].items() if k != "build"}), open(os.path.join(a.out, "config.json"), "w"), indent=2)
    json.dump(env_info(device), open(os.path.join(a.out, "env.json"), "w"), indent=2)
    print(f"device={device} trials={len(y)} subjects={subjects} seeds={seeds}", flush=True)

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
        append_csv(os.path.join(a.out, "predictions.csv"), [dict(model="icnn_running", mode=mode, seed=seed, subject=s, session=sess[i], trial=int(meta.trial.values[i]),
                                                                 y=int(y[i]), prob=round(float(q), 5)) for i, q in zip(test[ok], prob[ok])])
        print(f"[{time.time()-t0:6.0f}s] {mode:16s} seed {seed} subj {s:2d}  bacc {r['bacc']:.4f}  auc {r['auc']:.4f}", flush=True)

    for seed in seeds:
        for s in subjects:
            test, train = np.where(subj == s)[0], np.where((subj != s) & labelled)[0]
            r_pop = scov[subj != s].mean(0)
            xtr = apply_whitener(w_all[train], x[train])
            scale = float(x[train].std() / xtr.std())
            model, _ = train_deep("icnn_running", torch.from_numpy(xtr * scale).to(device), yt[train], None, None, seed, a.epoch_scale, device)
            model.eval()
            w = np.empty_like(scov[test])
            for ses in np.unique(sess[test]):
                m = np.where(sess[test] == ses)[0]
                w[m] = inv_sqrtm_batch(prior_references(scov[test][m], r_pop, 30))[0]
            zt = torch.from_numpy(apply_whitener(w, x[test]) * scale).to(device)
            record(seed, s, "prior30", predict(model, zt), test)
            probs = {k: np.zeros(len(test)) for k in ("prior30_bnema07", "prior30_bnema01", "prior30_bn30")}
            for ses in np.unique(sess[test]):
                m = np.where(sess[test] == ses)[0]
                feat = session_features(model, zt[m])
                for k, stats in (("prior30_bnema07", bn_stats_ema(model, feat, 0.7)), ("prior30_bnema01", bn_stats_ema(model, feat, 0.1)), ("prior30_bn30", bn_stats_blend(model, feat, 30))):
                    probs[k][m] = F.softmax(logits_from(model, feat, *stats), 1)[:, 1].cpu().numpy()
            for k, pr in probs.items():
                record(seed, s, k, pr, test)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
