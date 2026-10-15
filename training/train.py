"""Train the InterpretableCNN and export the files the live backend needs.

Run from the repository root:

    python training/train.py                     # LOSO evaluation + export
    python training/train.py --skip-loso         # export only

Steps:
1. Leave-one-subject-out (LOSO) cross-validation. Every held-out epoch is scored on
   its own, exactly as the streaming backend scores it.
2. Train the production model on every subject except --demo-subject.
3. Write modelfiles/interpretable_cnn_weights.pt, demo_subject_x.npy,
   demo_subject_y.npy and loso_metrics.json.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.model import InterpretableCNN  # noqa: E402


def load_dataset(path):
    mat = sio.loadmat(path)
    x = mat["EEGsample"].astype(np.float32)[:, np.newaxis]  # (N, 1, 30, 384)
    y = mat["substate"].flatten().astype(np.int64)          # 0 = alert, 1 = drowsy
    subj = mat["subindex"].flatten().astype(np.int64)       # 1..11
    return x, y, subj


def train_model(x, y, n_epoch, batch_size, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    net = InterpretableCNN().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    net.train()
    for _ in range(n_epoch):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(net(xb), yb).backward()
            optimizer.step()
    return net.eval()


def predict_proba(net, x, device):
    # Eval-mode BN uses running statistics, so batched scoring equals per-epoch scoring.
    with torch.no_grad():
        return net(torch.from_numpy(x).to(device)).softmax(dim=1)[:, 1].cpu().numpy()


def run_loso(x, y, subj, args, device):
    per_subject = {}
    for s in np.unique(subj):
        train, test = subj != s, subj == s
        net = train_model(x[train], y[train], args.epochs, args.batch_size, args.seed, device)
        prob = predict_proba(net, x[test], device)
        pred = (prob >= 0.5).astype(np.int64)
        per_subject[int(s)] = {
            "accuracy": float(accuracy_score(y[test], pred)),
            "f1": float(f1_score(y[test], pred)),
            "auc": float(roc_auc_score(y[test], prob)),
            "n_epochs": int(test.sum()),
        }
        m = per_subject[int(s)]
        print(f"Subject {s:2d} held out - acc {m['accuracy']:.4f}  f1 {m['f1']:.4f}  auc {m['auc']:.4f}", flush=True)

    mean = {k: float(np.mean([m[k] for m in per_subject.values()])) for k in ("accuracy", "f1", "auc")}
    print(f"\nMean LOSO - acc {mean['accuracy']:.4f}  f1 {mean['f1']:.4f}  auc {mean['auc']:.4f}")
    return {"mean": mean, "per_subject": per_subject}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=ROOT / "dataset" / "dataset.mat", type=pathlib.Path)
    parser.add_argument("--out", default=ROOT / "modelfiles", type=pathlib.Path)
    parser.add_argument("--demo-subject", default=1, type=int, help="subject held out for the live demo")
    parser.add_argument("--epochs", default=11, type=int)
    parser.add_argument("--batch-size", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--skip-loso", action="store_true", help="skip cross-validation, only export")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    x, y, subj = load_dataset(args.data)
    print(f"Loaded {x.shape[0]} epochs, {len(np.unique(subj))} subjects")
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.skip_loso:
        metrics = run_loso(x, y, subj, args, device)
        metrics["config"] = {"epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed}
        (args.out / "loso_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nTraining production model (subject {args.demo_subject} held out for the demo)...")
    train, demo = subj != args.demo_subject, subj == args.demo_subject
    net = train_model(x[train], y[train], args.epochs, args.batch_size, args.seed, device)
    torch.save(net.state_dict(), args.out / "interpretable_cnn_weights.pt")
    np.save(args.out / "demo_subject_x.npy", x[demo])
    np.save(args.out / "demo_subject_y.npy", y[demo])

    demo_acc = accuracy_score(y[demo], predict_proba(net, x[demo], device) >= 0.5)
    print(f"Demo subject {args.demo_subject} accuracy: {demo_acc:.4f}")
    print(f"Wrote weights and demo data to {args.out}")


if __name__ == "__main__":
    main()
