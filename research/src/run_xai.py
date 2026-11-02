"""Case study for the review: how far can explanations of InterpretableCNN be trusted, on every held-out window?

Same LOSO protocol, seeds and training recipe as run_phase0.py, on the balanced 11-subject subset. Two variants:
    icnn_batch     normalised with statistics of the whole held-out subject (the published protocol)
    icnn_running   running statistics, so a window is scored on its own (what a live system can do)

Attribution methods (target: logit margin of the predicted class; statistics frozen, so a map explains one window):
    saliency       |gradient|
    gradinput      gradient x input (equals LRP-epsilon for this ReLU network, Cui et al. 2023)
    ig             integrated gradients, zero baseline, 50 steps
    cam_fix        CAM + CNN-Fixation tracing, the method published with this model (Cui et al. 2022): the top 100 CAM
                   entries are traced to one input channel and a 64-sample segment, then smoothed with a Gaussian (sigma 32)
    random         random ranking (deletion baseline only)

Tests:
    deletion       zero the top-ranked points or channels; track P(originally predicted class). Lower is more faithful.
    sanity         Spearman correlation between maps of the trained model and of a randomly re-initialised one
                   (Adebayo et al. 2018); a faithful map should change, so lower is better.
    band           remove one frequency band; change in P(true class), per true class
    group          zero peripheral or central channels; change in P(true class), per true class
    context        icnn_batch only: score the same windows with statistics from all, 30 random, only alert or only drowsy
                   windows of that subject. Labels are used only to build the contexts.

    python -m research.src.run_xai --out research/results/xai_local --subjects 1,2 --seeds 0 --epoch-scale 0.2 --max-per-subject 20
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .data import CHANNELS, FS, find_dataset, load_sadt
from .models import InterpretableCNN
from .run_phase0 import RECIPES, append_csv, env_info, metrics, set_seed, train_deep

VARIANTS = ["icnn_batch", "icnn_running"]
METHODS = ["saliency", "gradinput", "ig", "cam_fix"]
POINT_LEVELS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
CHANNEL_LEVELS = [1, 2, 3, 5, 10]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)}
# Outer ring of the montage, fixed before looking at any result; the other 15 channels are "central".
PERIPHERAL = ["FP1", "FP2", "F7", "F8", "FT7", "FT8", "T3", "T4", "TP7", "TP8", "T5", "T6", "O1", "OZ", "O2"]
PERIPHERAL_IDX = [CHANNELS.index(c) for c in PERIPHERAL]
CENTRAL_IDX = [i for i in range(len(CHANNELS)) if i not in PERIPHERAL_IDX]
TOP_N, SEGMENT, SIGMA, IG_STEPS, SANITY_SAMPLES = 100, 64, 32, 50, 30


def frozen_model(model, context, device):
    """Copy of a trained model whose BN statistics are fixed. context=None keeps the running statistics."""
    m = InterpretableCNN(bn_mode="running").to(device)
    missing = m.load_state_dict(model.state_dict(), strict=False)
    if context is None:
        assert not missing.missing_keys, missing
    else:
        with torch.no_grad():
            feat = m.relu(m.depthwise(m.pointwise(context.unsqueeze(1))))
            m.bn.running_mean.copy_(feat.mean(dim=(0, 2, 3)))
            m.bn.running_var.copy_(feat.var(dim=(0, 2, 3), unbiased=False))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def margin(logits, target):
    rows = torch.arange(len(target))
    return logits[rows, target] - logits[rows, 1 - target]


def batched(fn, x, target, size=64):
    return torch.cat([fn(x[i:i + size], target[i:i + size]) for i in range(0, len(x), size)])


def grad_map(model, x, target):
    x = x.clone().requires_grad_(True)
    return torch.autograd.grad(margin(model(x), target).sum(), x)[0]


def saliency(model, x, target):
    return batched(lambda a, t: grad_map(model, a, t).abs(), x, target)


def gradinput(model, x, target):
    return batched(lambda a, t: grad_map(model, a, t) * a, x, target)


def integrated_gradients(model, x, target):
    def one(a, t):
        total = sum(grad_map(model, a * (k + 0.5) / IG_STEPS, t) for k in range(IG_STEPS))
        return a * total / IG_STEPS
    return batched(one, x, target, size=32)


def gaussian_kernel(device):
    r = 3 * SIGMA
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=device, dtype=torch.float32) / SIGMA) ** 2)
    return (k / k.sum()).view(1, 1, -1)


def cam_fix(model, x, target):
    """CAM traced back to the input: each top CAM entry goes to the channel with the largest weighted contribution."""
    with torch.no_grad():
        act = model.bn(model.relu(model.depthwise(model.pointwise(x.unsqueeze(1))))).squeeze(2)  # (B, 32, 321)
        weights = model.fc.weight[target] - model.fc.weight[1 - target]  # (B, 32): margin, as for the other methods
        cam = (weights.unsqueeze(2) * act).flatten(1)
        spatial = model.pointwise.weight.squeeze(1).squeeze(-1)  # (16, 30)
        temporal = model.depthwise.weight.squeeze(1).squeeze(1)  # (32, 64)
        n_t = act.shape[2]
        out = torch.zeros_like(x)
        windows = x.unfold(2, SEGMENT, 1)  # (B, 30, 321, 64)
        values, top = cam.topk(TOP_N, dim=1)
        for b in range(len(x)):
            keep = values[b] > 0
            ks, ts = top[b][keep] // n_t, top[b][keep] % n_t
            if len(ks) == 0:
                continue
            contrib = (windows[b][:, ts] * temporal[ks]).sum(-1) * spatial[ks // 2].T  # (30, n)
            channels = contrib.argmax(0)
            for ch, t, v in zip(channels.tolist(), ts.tolist(), values[b][keep].tolist()):
                out[b, ch, t:t + SEGMENT] += v
        kernel = gaussian_kernel(x.device)
        smooth = F.conv1d(out.flatten(0, 1).unsqueeze(1), kernel, padding=kernel.shape[-1] // 2)
        return smooth.view_as(out)


ATTRIBUTION = {"saliency": saliency, "gradinput": gradinput, "ig": integrated_gradients, "cam_fix": cam_fix}


@torch.no_grad()
def prob_of(model, x, target, size=256):
    out = [F.softmax(model(x[i:i + size]), dim=1) for i in range(0, len(x), size)]
    return torch.cat(out)[torch.arange(len(target)), target]


def point_deletion(model, x, target, rank):
    """Mean P(target) after zeroing the top fraction of (channel, time) points of each window."""
    order = rank.flatten(1).argsort(dim=1, descending=True)
    out = {}
    for level in POINT_LEVELS:
        mask = torch.ones_like(x).flatten(1)
        mask.scatter_(1, order[:, :int(round(level * mask.shape[1]))], 0.0)
        out[level] = prob_of(model, x * mask.view_as(x), target).mean().item()
    return out


def channel_deletion(model, x, target, rank):
    order = rank.clamp(min=0).sum(2).argsort(dim=1, descending=True)
    out = {}
    for level in CHANNEL_LEVELS:
        mask = torch.ones(x.shape[:2], device=x.device)
        mask.scatter_(1, order[:, :level], 0.0)
        out[level] = prob_of(model, x * mask.unsqueeze(2), target).mean().item()
    return out


def band_stop(x, band):
    lo, hi = BANDS[band]
    spec = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / FS).to(x.device)
    spec[..., (freqs >= lo) & (freqs < hi)] = 0
    return torch.fft.irfft(spec, n=x.shape[-1], dim=-1)


def per_sample_channel_map(a):
    """Share of positive relevance per channel, per window."""
    c = a.clamp(min=0).sum(2)
    return c / c.sum(1, keepdim=True).clamp(min=1e-12)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--subjects", default="all")
    p.add_argument("--epoch-scale", type=float, default=1.0, help="multiply epoch counts (smoke tests only)")
    p.add_argument("--max-per-subject", type=int, default=0, help="limit windows explained per subject (smoke tests only)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in a.seeds.split(",")]
    x, y, subj = load_sadt(find_dataset(a.data))
    subjects = [int(s) for s in (np.unique(subj) if a.subjects == "all" else a.subjects.split(","))]
    json.dump(dict(vars(a), n_samples=len(y), methods=METHODS, point_levels=POINT_LEVELS, channel_levels=CHANNEL_LEVELS, bands=BANDS,
                   peripheral=PERIPHERAL, top_n=TOP_N, ig_steps=IG_STEPS, channels=CHANNELS,
                   recipes={k: {kk: vv for kk, vv in RECIPES[k].items() if kk != "build"} for k in VARIANTS}),
              open(os.path.join(a.out, "config.json"), "w"), indent=2)
    json.dump(env_info(device), open(os.path.join(a.out, "env.json"), "w"), indent=2)
    print(f"device={device} N={len(y)} seeds={seeds} subjects={subjects}", flush=True)

    xt, yt = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    path = lambda n: os.path.join(a.out, n + ".csv")
    t0 = time.time()
    store = {}

    for name in VARIANTS:
        for seed in seeds:
            for s in subjects:
                test_all = np.where(subj == s)[0]
                test = test_all
                if a.max_per_subject:
                    test = np.random.default_rng(s).choice(test_all, min(a.max_per_subject, len(test_all)), replace=False)
                train = np.where(subj != s)[0]
                model, _ = train_deep(name, xt[train], yt[train], None, None, seed, a.epoch_scale, device)
                ctx_all = xt[test_all] if name == "icnn_batch" else None
                m = frozen_model(model, ctx_all, device)
                xs, ys = xt[test], yt[test]
                with torch.no_grad():
                    probs = F.softmax(m(xs), dim=1)
                pred = probs.argmax(1)
                # Frozen statistics must reproduce the protocol exactly; otherwise the maps would explain another model.
                if name == "icnn_batch":
                    model.eval()
                    with torch.no_grad():
                        direct = F.softmax(model(xt[test_all]), dim=1)
                    frozen = F.softmax(m(xt[test_all]), dim=1)
                    assert (direct - frozen).abs().max().item() < 1e-3, "frozen BN differs from whole-subject BN"
                score = metrics(ys.cpu().numpy(), probs[:, 1].cpu().numpy())
                base_common = dict(model=name, seed=seed, subject=s)
                append_csv(path("per_subject"), [dict(base_common, n=len(test), **score)])

                maps = {k: ATTRIBUTION[k](m, xs, pred) for k in METHODS}
                ranks = {k: v for k, v in maps.items()}
                gen = torch.Generator(device="cpu").manual_seed(1000 * seed + s)
                ranks["random"] = torch.rand(xs.shape, generator=gen).to(device)
                base_p = prob_of(m, xs, pred).mean().item()
                rows = []
                for k, r in ranks.items():
                    rows.append(dict(base_common, method=k, unit="point", level=0, p=base_p))
                    rows += [dict(base_common, method=k, unit="point", level=l, p=v) for l, v in point_deletion(m, xs, pred, r).items()]
                    rows.append(dict(base_common, method=k, unit="channel", level=0, p=base_p))
                    rows += [dict(base_common, method=k, unit="channel", level=l, p=v) for l, v in channel_deletion(m, xs, pred, r).items()]
                append_csv(path("deletion"), rows)

                # Model-randomisation sanity check: same inputs and targets, freshly initialised weights.
                torch.manual_seed(9_000 + 1000 * seed + s)
                rnd = RECIPES[name]["build"]().to(device)
                rnd_m = frozen_model(rnd, xt[test_all], device)
                pick = torch.randperm(len(xs))[:SANITY_SAMPLES].to(device)
                rows = []
                for k in METHODS:
                    trained = maps[k][pick].abs().flatten(1).cpu().numpy()
                    randomised = ATTRIBUTION[k](rnd_m, xs[pick], pred[pick]).abs().flatten(1).cpu().numpy()
                    rhos = [spearmanr(u, v)[0] for u, v in zip(trained, randomised)]
                    rows.append(dict(base_common, method=k, spearman=float(np.nanmean(rhos))))
                append_csv(path("sanity"), rows)

                # What the model relies on, on every window: bands and channel groups, per true class.
                rows = []
                true_p = prob_of(m, xs, ys)
                for cls in (0, 1):
                    idx = (ys == cls).nonzero().flatten()
                    if len(idx) == 0:
                        continue
                    for band in BANDS:
                        dp = prob_of(m, band_stop(xs[idx], band), ys[idx]) - true_p[idx]
                        rows.append(dict(base_common, cls=cls, kind="band", part=band, dp=dp.mean().item(), n=len(idx)))
                    for group, chs in (("peripheral", PERIPHERAL_IDX), ("central", CENTRAL_IDX)):
                        z = xs[idx].clone()
                        z[:, chs] = 0
                        dp = prob_of(m, z, ys[idx]) - true_p[idx]
                        rows.append(dict(base_common, cls=cls, kind="group", part=group, dp=dp.mean().item(), n=len(idx)))
                append_csv(path("perturb"), rows)

                # Same windows, different statistics: does a decision depend on the rest of the session?
                if name == "icnn_batch":
                    rows = []
                    rng = np.random.default_rng(10_000 * seed + s)
                    y_all = y[test_all]
                    contexts = {"all": test_all, "random30": rng.choice(test_all, min(30, len(test_all)), replace=False),
                                "alert_only": test_all[y_all == 0], "drowsy_only": test_all[y_all == 1]}
                    base_map = maps["gradinput"].flatten(1)
                    for cname, idx in contexts.items():
                        if len(idx) < 2:
                            continue
                        c_m = frozen_model(model, xt[idx], device)
                        with torch.no_grad():
                            c_prob = F.softmax(c_m(xs), dim=1)
                        c_pred = c_prob.argmax(1)
                        c_map = gradinput(c_m, xs, c_pred).flatten(1)
                        r = F.cosine_similarity(base_map - base_map.mean(1, keepdim=True), c_map - c_map.mean(1, keepdim=True), dim=1)
                        rows.append(dict(base_common, context=cname, acc=(c_pred == ys).float().mean().item(), flips=(c_pred != pred).float().mean().item(),
                                         pred_drowsy=c_pred.float().mean().item(), map_corr=r.mean().item()))
                    append_csv(path("context"), rows)

                # Per-window channel maps and class-averaged time courses, for stability analysis and scalp maps.
                d = store.setdefault((name, seed), {})
                d.setdefault("index", []).append(test)
                d.setdefault("pred", []).append(pred.cpu().numpy())
                d.setdefault("prob", []).append(probs[:, 1].cpu().numpy())
                for k in METHODS:
                    d.setdefault("channel_" + k, []).append(per_sample_channel_map(maps[k]).cpu().numpy().astype(np.float32))
                    d.setdefault("time_" + k, []).append(maps[k].clamp(min=0).sum(1).cpu().numpy().astype(np.float32))
                print(f"[{time.time()-t0:6.0f}s] {name:12s} seed {seed} subj {s:2d}  acc {score['acc']:.4f}  auc {score['auc']:.4f}", flush=True)
            np.savez_compressed(os.path.join(a.out, f"maps_{name}_seed{seed}.npz"), **{k: np.concatenate(v) for k, v in store.pop((name, seed)).items()})
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
