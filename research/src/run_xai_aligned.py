"""E4: does online alignment change what the model relies on? Full Cao et al. dataset, 27 subjects, LOSO, InterpretableCNN.

Two models per fold, same seeds and recipe as E2: `none` (unaligned) and `prior30` (whitened online with kappa = 30).
Attributions are taken with respect to the RAW input, through the fixed per-window whitener, so channels mean the same in both models.
Scored on labelled trials from the 31st of each session. A window's whitener stays fixed when that window is perturbed.

Tests, as in run_xai.py: point and channel deletion against a random baseline (saliency, gradient x input); sanity check against a
re-initialised model; removal of each band and of peripheral or central channels, change in P(true class), per true class; channel maps.
Primary questions, fixed before seeing results: for prior30 vs none, (1) does removing peripheral channels, beta or gamma change P(alert)
differently, and (2) does removing theta, alpha or central channels change P(drowsy) differently?

    python -m research.src.run_xai_aligned --out research/results/xai_al_local --data <sessions folder> --subjects 5,35 --seeds 0 --epoch-scale 0.1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .data import CHANNELS, find_cao, load_cao
from .run_phase0 import RECIPES, append_csv, env_info, metrics, train_deep
from .run_prior import prior_references
from .run_shrink import apply_whitener, inv_sqrtm_batch
from .run_xai import BANDS, CENTRAL_IDX, CHANNEL_LEVELS, PERIPHERAL_IDX, POINT_LEVELS, SANITY_SAMPLES, band_stop, frozen_model, margin

METHODS = ["saliency", "gradinput"]


class Bound:
    """A trained model plus per-window whiteners and scale: maps raw windows to logits."""

    def __init__(self, model, w, scale):
        self.model, self.w, self.scale = model, w, scale

    def logits(self, x, idx):
        z = x if self.w is None else torch.einsum("ncd,ndt->nct", self.w[idx], x)
        return self.model(z * self.scale)

    @torch.no_grad()
    def prob(self, x, target, size=256):
        out = [F.softmax(self.logits(x[i:i + size], torch.arange(i, min(i + size, len(x)), device=x.device)), 1) for i in range(0, len(x), size)]
        return torch.cat(out)[torch.arange(len(target)), target]

    def attribution(self, x, target, method, size=128):
        maps = []
        for i in range(0, len(x), size):
            a = x[i:i + size].clone().requires_grad_(True)
            idx = torch.arange(i, min(i + size, len(x)), device=x.device)
            g = torch.autograd.grad(margin(self.logits(a, idx), target[i:i + size]).sum(), a)[0]
            maps.append(g.abs() if method == "saliency" else g * a.detach())
        return torch.cat(maps)


def deletion(bound, x, target, rank):
    rows = {("point", 0): bound.prob(x, target).mean().item(), ("channel", 0): None}
    rows[("channel", 0)] = rows[("point", 0)]
    order = rank.flatten(1).argsort(dim=1, descending=True)
    for level in POINT_LEVELS:
        mask = torch.ones_like(x).flatten(1)
        mask.scatter_(1, order[:, :int(round(level * mask.shape[1]))], 0.0)
        rows[("point", level)] = bound.prob(x * mask.view_as(x), target).mean().item()
    order = rank.clamp(min=0).sum(2).argsort(dim=1, descending=True)
    for level in CHANNEL_LEVELS:
        mask = torch.ones(x.shape[:2], device=x.device)
        mask.scatter_(1, order[:, :level], 0.0)
        rows[("channel", level)] = bound.prob(x * mask.unsqueeze(2), target).mean().item()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--subjects", default="all")
    p.add_argument("--kappa", type=float, default=30)
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
    json.dump(dict(vars(a), data=folder, n_trials=len(y), methods=METHODS, channels=CHANNELS, bands=BANDS), open(os.path.join(a.out, "config.json"), "w"), indent=2)
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
    out = lambda n: os.path.join(a.out, n + ".csv")

    for seed in seeds:
        for s in subjects:
            test_all, train = np.where(subj == s)[0], np.where((subj != s) & labelled)[0]
            r_pop = scov[subj != s].mean(0)
            w = np.empty_like(scov[test_all])
            for ses in np.unique(sess[test_all]):
                m = np.where(sess[test_all] == ses)[0]
                w[m] = inv_sqrtm_batch(prior_references(scov[test_all][m], r_pop, a.kappa))[0]
            keep = labelled[test_all] & (pos[test_all] >= 30)
            xs, ys = torch.from_numpy(x[test_all][keep]).to(device), yt[test_all[keep]]
            wt = torch.from_numpy(w[keep].astype(np.float32)).to(device)
            xtr = apply_whitener(w_all[train], x[train])
            al_scale = float(x[train].std() / xtr.std())

            for name, tr, scale, wb in (("none", x[train], 1.0, None), ("prior30", xtr * al_scale, al_scale, wt)):
                model, _ = train_deep("icnn_running", torch.from_numpy(tr).to(device), yt[train], None, None, seed, a.epoch_scale, device)
                model.eval()
                for q in model.parameters():
                    q.requires_grad_(False)
                b = Bound(model, wb, scale)
                probs = torch.cat([F.softmax(b.logits(xs[i:i + 256], torch.arange(i, min(i + 256, len(xs)), device=device)).detach(), 1) for i in range(0, len(xs), 256)])
                pred = probs.argmax(1)
                common = dict(variant=name, seed=seed, subject=s)
                append_csv(out("per_subject"), [dict(common, n=len(xs), **metrics(ys.cpu().numpy(), probs[:, 1].cpu().numpy()))])
                maps = {k: b.attribution(xs, pred, k) for k in METHODS}
                gen = torch.Generator(device="cpu").manual_seed(1000 * seed + s)
                rows = []
                for k, r in {**maps, "random": torch.rand(xs.shape, generator=gen).to(device)}.items():
                    rows += [dict(common, method=k, unit=u, level=l, p=v) for (u, l), v in deletion(b, xs, pred, r).items()]
                append_csv(out("deletion"), rows)

                torch.manual_seed(9_000 + 1000 * seed + s)
                rnd = RECIPES["icnn_running"]["build"]().to(device)
                ctx = xs if wb is None else torch.einsum("ncd,ndt->nct", wb, xs)
                rb = Bound(frozen_model(rnd, ctx * scale, device), wb, scale)
                pick = torch.randperm(len(xs))[:SANITY_SAMPLES].to(device)
                rows = []
                # Whiteners follow window position, so the sanity maps are taken from the full set and then indexed.
                for k in METHODS:
                    rm = rb.attribution(xs, pred, k)[pick].abs().flatten(1).cpu().numpy()
                    tm = maps[k][pick].abs().flatten(1).cpu().numpy()
                    rows.append(dict(common, method=k, spearman=float(np.nanmean([spearmanr(u, v)[0] for u, v in zip(tm, rm)]))))
                append_csv(out("sanity"), rows)

                rows = []
                true_p = b.prob(xs, ys)
                for cls in (0, 1):
                    idx = (ys == cls).nonzero().flatten()
                    if len(idx) == 0:
                        continue
                    for band in BANDS:
                        v = b.prob(band_stop(xs, band), ys) - true_p
                        rows.append(dict(common, cls=cls, kind="band", part=band, dp=v[idx].mean().item(), n=len(idx)))
                    for group, chs in (("peripheral", PERIPHERAL_IDX), ("central", CENTRAL_IDX)):
                        z = xs.clone()
                        z[:, chs] = 0
                        rows.append(dict(common, cls=cls, kind="group", part=group, dp=(b.prob(z, ys) - true_p)[idx].mean().item(), n=len(idx)))
                append_csv(out("perturb"), rows)

                rows = []
                for k in METHODS:
                    c = maps[k].clamp(min=0).sum(2)
                    c = (c / c.sum(1, keepdim=True).clamp(min=1e-12)).cpu().numpy()
                    for cls in (0, 1):
                        mask = (ys == cls).cpu().numpy()
                        if mask.any():
                            rows.append(dict(common, method=k, cls=cls, peripheral=float(c[mask][:, PERIPHERAL_IDX].mean(0).sum()), **dict(zip(CHANNELS, c[mask].mean(0).round(5).tolist()))))
                append_csv(out("channel_maps"), rows)
                print(f"[{time.time()-t0:6.0f}s] {name:8s} seed {seed} subj {s:2d}  n {len(xs)}  acc {(pred == ys).float().mean().item():.4f}", flush=True)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
