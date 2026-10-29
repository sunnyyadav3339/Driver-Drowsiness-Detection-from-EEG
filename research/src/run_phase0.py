"""Phase 0: every baseline under one leave-one-subject-out (LOSO) protocol on SADT-2022.

Run from the repository root:
    python -m research.src.run_phase0 --out research/results/phase0_local
    python -m research.src.run_phase0 --out /tmp/smoke --subjects 1,2 --seeds 0 --epoch-scale 0.02   # smoke test

For each held-out subject and seed, a model is trained on the other subjects and scored on the held-out one.
Modes recorded per prediction:
    streaming     each epoch scored on its own (what a live system can do)  <- primary
    transductive  the whole held-out subject scored as one batch (only differs for per-batch BN)

Outputs in --out (written incrementally, so a timeout keeps finished folds):
    config.json, env.json, per_subject.csv, predictions.csv, summary.json, summary.md
"""
import argparse
import csv
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .classical import BandPowerLR, RiemannTangentLR
from .data import find_dataset, load_sadt
from .models import DeltaGateNet, EEGNet, InterpretableCNN

# Training recipes as published (or as used in the comparison paper for EEGNet).
# n_val: number of training subjects held out for checkpoint selection (0 = fixed epoch count, no validation).
RECIPES = {
    "icnn_batch": dict(build=lambda: InterpretableCNN(bn_mode="batch"), opt="adam", lr=1e-3, wd=0.0, batch=50, epochs=11, n_val=0,
                       source="Cui et al. 2022: Adam 1e-3, batch 50, 11 epochs, per-batch BN"),
    "icnn_running": dict(build=lambda: InterpretableCNN(bn_mode="running"), opt="adam", lr=1e-3, wd=0.0, batch=50, epochs=11, n_val=0,
                         source="Cui et al. 2022 recipe with running-stats BN (streaming-valid)"),
    "eegnet": dict(build=lambda: EEGNet(), opt="adam", lr=1e-3, wd=0.0, batch=32, epochs=100, n_val=2,
                   source="EEGNet-8,2; Adam 1e-3, batch 32, 100 epochs as in Jang et al. 2025; best validation-subject loss"),
    "deltagatenet": dict(build=lambda: DeltaGateNet(), opt="adamw", lr=1e-4, wd=1e-2, batch=32, epochs=200, n_val=2,
                         source="Po et al. 2026 official code: AdamW 1e-4, batch 32, 200 epochs, best validation-subject loss"),
}
CLASSICAL = {"bandpower_lr": BandPowerLR, "riemann_lr": RiemannTangentLR}
ALL_MODELS = list(RECIPES) + list(CLASSICAL)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else float("nan")
    return dict(acc=accuracy_score(y, pred), f1_macro=f1_score(y, pred, average="macro"),
                f1_drowsy=f1_score(y, pred, zero_division=0), auc=auc)


@torch.no_grad()
def predict(model, x, batch=256):
    model.eval()
    out = [F.softmax(model(x[i:i + batch]), dim=1)[:, 1] for i in range(0, len(x), batch)]
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def predict_streaming(model, x):
    model.eval()
    return np.array([F.softmax(model(x[i:i + 1]), dim=1)[0, 1].item() for i in range(len(x))])


def train_deep(name, xtr, ytr, xval, yval, seed, epoch_scale, device):
    r = RECIPES[name]
    set_seed(seed)
    model = r["build"]().to(device)
    opt_cls = torch.optim.AdamW if r["opt"] == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=r["lr"], weight_decay=r["wd"])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    epochs = max(1, round(r["epochs"] * epoch_scale))
    best_loss, best_state, best_epoch = float("inf"), None, epochs
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=gen).to(device)
        for i in range(0, len(perm), r["batch"]):
            idx = perm[i:i + r["batch"]]
            if len(idx) < 2:  # BatchNorm needs more than one sample
                continue
            opt.zero_grad()
            F.cross_entropy(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
        if xval is not None:
            model.eval()
            with torch.no_grad():
                val_loss = sum(F.cross_entropy(model(xval[i:i + 256]), yval[i:i + 256], reduction="sum").item()
                               for i in range(0, len(xval), 256)) / len(xval)
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, ep + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def append_csv(path, rows):
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def summarize(out):
    import pandas as pd

    df = pd.read_csv(os.path.join(out, "per_subject.csv"))
    rows = []
    for (model, mode), g in df.groupby(["model", "mode"], sort=False):
        per_seed = g.groupby("seed")[["acc", "f1_macro", "f1_drowsy", "auc"]].mean()
        subj_mean = g.groupby("subject")[["acc"]].mean()
        rows.append(dict(model=model, mode=mode, n_seeds=len(per_seed), n_subjects=g.subject.nunique(),
                         **{f"{m}_mean": per_seed[m].mean() for m in per_seed},
                         **{f"{m}_std_seeds": per_seed[m].std(ddof=0) for m in per_seed},
                         acc_std_subjects=subj_mean.acc.std(ddof=0)))
    summary = pd.DataFrame(rows)
    summary.to_json(os.path.join(out, "summary.json"), orient="records", indent=2)
    lines = ["| model | mode | seeds | acc | macro-F1 | drowsy-F1 | AUC | acc std (subjects) |", "|---|---|---|---|---|---|---|---|"]
    for r in summary.itertuples():
        lines.append(f"| {r.model} | {r.mode} | {r.n_seeds} | {100*r.acc_mean:.2f} ± {100*r.acc_std_seeds:.2f} | "
                     f"{100*r.f1_macro_mean:.2f} | {100*r.f1_drowsy_mean:.2f} | {r.auc_mean:.3f} | {100*r.acc_std_subjects:.2f} |")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("Mean over seeds of the subject-averaged LOSO metric; ± is the std across seeds.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


def env_info(device):
    info = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
                device=str(device), git_commit=os.environ.get("GIT_COMMIT", "unknown"),
                code_source=os.environ.get("CODE_SOURCE", "local"))
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    for mod in ("sklearn", "scipy", "pyriemann", "pandas"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            pass
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data")
    p.add_argument("--out", required=True)
    p.add_argument("--models", default=",".join(ALL_MODELS))
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--subjects", default="all")
    p.add_argument("--epoch-scale", type=float, default=1.0, help="multiply epoch counts (smoke tests only)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = args.models.split(",")
    unknown = set(models) - set(ALL_MODELS)
    if unknown:
        raise SystemExit(f"unknown models {unknown}; choose from {ALL_MODELS}")
    seeds = [int(s) for s in args.seeds.split(",")]

    data_path = find_dataset(args.data)
    x, y, subj = load_sadt(data_path)
    subjects = sorted(np.unique(subj)) if args.subjects == "all" else [int(s) for s in args.subjects.split(",")]

    config = dict(vars(args), data_path=data_path, n_samples=len(y), protocol="LOSO",
                  recipes={k: {kk: vv for kk, vv in v.items() if kk != "build"} for k, v in RECIPES.items() if k in models},
                  val_subject_rule="n_val training subjects drawn with numpy default_rng(1000*seed + test_subject)")
    json.dump(config, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    json.dump(env_info(device), open(os.path.join(args.out, "env.json"), "w"), indent=2)
    print(f"device={device} data={data_path} N={len(y)} models={models} seeds={seeds} subjects={subjects}", flush=True)

    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    ps_path, pred_path = os.path.join(args.out, "per_subject.csv"), os.path.join(args.out, "predictions.csv")
    t_start = time.time()

    for name in models:
        run_seeds = [0] if name in CLASSICAL else seeds  # classical pipelines are deterministic
        for seed in run_seeds:
            for s in subjects:
                t0 = time.time()
                test = np.where(subj == s)[0]
                train_subjects = [t for t in np.unique(subj) if t != s]
                n_val = RECIPES[name]["n_val"] if name in RECIPES else 0
                val_subjects = list(np.random.default_rng(1000 * seed + s).choice(train_subjects, n_val, replace=False)) if n_val else []
                train = np.where(np.isin(subj, train_subjects) & ~np.isin(subj, val_subjects))[0]
                val = np.where(np.isin(subj, val_subjects))[0]
                best_epoch = ""

                if name in CLASSICAL:
                    clf = CLASSICAL[name](seed).fit(x[train], y[train])
                    probs = {"streaming": clf.predict_proba(x[test])}
                else:
                    model, best_epoch = train_deep(
                        name, xt[train], yt[train], xt[val] if len(val) else None, yt[val] if len(val) else None,
                        seed, args.epoch_scale, device)
                    if name == "icnn_batch":
                        probs = {"streaming": predict_streaming(model, xt[test]),
                                 "transductive": predict(model, xt[test], batch=len(test))}
                    else:  # running-stats BN: batched scoring equals per-epoch scoring
                        probs = {"streaming": predict(model, xt[test])}

                secs = time.time() - t0
                rows, pred_rows = [], []
                for mode, pr in probs.items():
                    m = metrics(y[test], pr)
                    rows.append(dict(model=name, mode=mode, seed=seed, subject=s, n=len(test), **m,
                                     best_epoch=best_epoch, val_subjects=" ".join(map(str, val_subjects)), seconds=round(secs, 1)))
                    pred_rows += [dict(model=name, mode=mode, seed=seed, subject=s, index=int(i), y=int(y[i]), prob=round(float(q), 6))
                                  for i, q in zip(test, pr)]
                    print(f"[{time.time()-t_start:7.0f}s] {name:13s} {mode:12s} seed {seed} subj {s:2d}  "
                          f"acc {m['acc']:.4f}  f1 {m['f1_macro']:.4f}  auc {m['auc']:.4f}  ({secs:.0f}s, best ep {best_epoch})", flush=True)
                append_csv(ps_path, rows)
                append_csv(pred_path, pred_rows)
        summarize(args.out)

    print(f"done in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
