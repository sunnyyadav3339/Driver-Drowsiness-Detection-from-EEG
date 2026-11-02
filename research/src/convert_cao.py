"""Phase 1: convert the Cao et al. 2019 pre-processed dataset (figshare 7666055, CC BY 4.0) into trial windows.

For every lane-departure trial (event 251/252 = deviation onset, 253 = response onset):
    x            3 s of EEG before deviation onset, 30 channels, resampled 500 -> 128 Hz  (30, 384), as in Cui et al.
    rt           reaction time = response onset - deviation onset (s); NaN if no response before the next trial
    valid        False if rt < 0.3 s or missing: behavioural artefact per the dataset authors' tutorial
                 ("Reject trials with extremely short RT (RT < 0.3 sec)"); such trials get label -1 and are
                 excluded from global_rt and alert_rt
    long_rt      True if rt > 10 s (no correction for a long time: deep drowsiness or a recording gap); kept, flagged
                 so RT-based models can cap them
    global_rt    mean RT of the valid trials with onset in the 90 s up to and including this trial (Wei et al. 2018)
    alert_rt     5th percentile of the session's valid RTs
    label        1 drowsy (rt and global_rt > 2.5 * alert_rt), 0 alert (both < 1.5 * alert_rt), -1 in between or invalid
    onset_s      deviation onset time in the session (s), so time order is kept
Sessions are downloaded, converted and deleted one at a time to keep disk use low.

    python -m research.src.convert_cao --out /tmp/cao --sessions s01_051017m      # one session
    python -m research.src.convert_cao --out /tmp/cao --sessions all

Writes <out>/sessions/<name>.npz, <out>/trials.csv (one row per trial, without x) and <out>/sessions.csv.
If the SADT-2022 file is found, also checks how many of its epochs match a converted window (<out>/sadt_match.csv).
"""
import argparse
import glob
import json
import os
import shutil
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import resample_poly

ARTICLE = 7666055
FS_IN, FS_OUT, WIN_S = 500, 128, 3
MIN_RT = 0.3


def file_list():
    with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{ARTICLE}/files?page_size=100") as r:
        return [f for f in json.load(r) if f["name"].endswith(".set.zip")]


def download(url, path, tries=5):
    for i in range(tries):
        try:
            urllib.request.urlretrieve(url, path)
            return
        except Exception as e:  # flaky network: retry with backoff
            print(f"  download failed ({e}); retry {i + 1}", flush=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"could not download {url}")


def load_set(set_path):
    eeg = sio.loadmat(set_path, squeeze_me=True, struct_as_record=False)["EEG"]
    assert int(eeg.srate) == FS_IN, eeg.srate
    fdt = os.path.join(os.path.dirname(set_path), eeg.data)
    data = np.fromfile(fdt, dtype=np.float32).reshape(int(eeg.pnts), int(eeg.nbchan)).T  # EEGLAB: channel-fastest
    channels = [c.labels for c in np.atleast_1d(eeg.chanlocs)]
    events = [(str(e.type), int(round(float(e.latency))) - 1) for e in np.atleast_1d(eeg.event)]  # 0-based samples
    return data, channels, sorted(events, key=lambda e: e[1])


def trials_from_events(events):
    """[(onset_sample, response_sample or None, direction)] in time order."""
    out = []
    for i, (typ, lat) in enumerate(events):
        if typ not in ("251", "252"):
            continue
        resp = None
        for typ2, lat2 in events[i + 1:]:
            if typ2 in ("251", "252"):
                break
            if typ2 == "253":
                resp = lat2
                break
        out.append((lat, resp, "left" if typ == "251" else "right"))
    return out


def convert_session(name, data, events):
    n_win = WIN_S * FS_IN
    rows, xs = [], []
    trials = trials_from_events(events)
    onset = np.array([t[0] for t in trials]) / FS_IN
    rt = np.array([(t[1] - t[0]) / FS_IN if t[1] is not None else np.nan for t in trials])
    valid = np.nan_to_num(rt, nan=0.0) >= MIN_RT
    alert_rt = np.percentile(rt[valid], 5)
    for k, (lat, _, direction) in enumerate(trials):
        recent = (onset > onset[k] - 90) & (onset <= onset[k]) & valid
        g = rt[recent].mean() if recent.any() else np.nan
        if not valid[k]:
            label = -1
        else:
            label = 1 if (rt[k] > 2.5 * alert_rt and g > 2.5 * alert_rt) else 0 if (rt[k] < 1.5 * alert_rt and g < 1.5 * alert_rt) else -1
        if lat - n_win < 0:
            continue  # not enough signal before the first trial
        xs.append(resample_poly(data[:, lat - n_win:lat].astype(np.float64), FS_OUT, FS_IN, axis=1).astype(np.float32))
        rows.append(dict(session=name, subject=int(name[1:3]), trial=k, onset_s=round(onset[k], 3), direction=direction,
                         rt=rt[k], valid=bool(valid[k]), long_rt=bool(rt[k] > 10), global_rt=g, alert_rt=alert_rt, label=label))
    return np.stack(xs), pd.DataFrame(rows)


def check_against_sadt(out, x_all, meta):
    """For each SADT-2022 epoch, the best Pearson correlation with any converted window."""
    from .data import find_dataset, load_sadt

    try:
        xs, ys, ss = load_sadt(find_dataset(None))
    except FileNotFoundError:
        print("SADT-2022 not found; skipping match check")
        return
    def z(a):
        a = a.reshape(len(a), -1).astype(np.float64)
        a = a - a.mean(1, keepdims=True)
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    zc, best, arg = z(x_all), np.full(len(xs), -1.0), np.zeros(len(xs), int)
    zs = z(xs)
    for i in range(0, len(zs), 256):
        c = zs[i:i + 256] @ zc.T
        best[i:i + 256], arg[i:i + 256] = c.max(1), c.argmax(1)
    m = meta.iloc[arg].reset_index(drop=True)
    res = pd.DataFrame(dict(sadt_index=np.arange(len(xs)), sadt_subject=ss, sadt_label=ys, corr=best.round(4),
                            session=m.session, trial=m.trial, label=m.label, rt=m.rt))
    res.to_csv(os.path.join(out, "sadt_match.csv"), index=False)
    good = res[res["corr"] > 0.99]
    print(f"SADT epochs matched (corr > 0.99): {len(good)}/{len(res)}; label agreement on matches: "
          f"{(good.sadt_label == good.label).mean() if len(good) else float('nan'):.3f}")
    print(good.groupby(["sadt_subject", "session"]).size().to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--sessions", default="all", help="'all' or comma-separated session names, e.g. s01_051017m")
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "sessions"), exist_ok=True)
    tmp = os.path.join(a.out, "_tmp")
    files = file_list()
    want = None if a.sessions == "all" else set(a.sessions.split(","))
    files = [f for f in files if want is None or f["name"].replace(".set.zip", "") in want]
    print(f"{len(files)} sessions to convert", flush=True)

    sess_rows, t0 = [], time.time()
    for f in files:
        name = f["name"].replace(".set.zip", "")
        os.makedirs(tmp, exist_ok=True)
        zpath = os.path.join(tmp, f["name"])
        download(f["download_url"], zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        os.remove(zpath)
        # The zips contain a folder named "<name>.set/" holding the .set and .fdt files.
        set_path = [p for p in glob.glob(os.path.join(tmp, "**", "*.set"), recursive=True) if os.path.isfile(p)][0]
        data, channels, events = load_set(set_path)
        x, meta = convert_session(name, data, events)
        np.savez(os.path.join(a.out, "sessions", name + ".npz"), x=x, **{c: meta[c].values for c in meta})
        meta.to_csv(os.path.join(a.out, "trials.csv"), mode="a", index=False,
                    header=not os.path.exists(os.path.join(a.out, "trials.csv")))
        sess_rows.append(dict(session=name, subject=int(name[1:3]), duration_s=data.shape[1] / FS_IN, n_trials=len(meta),
                              n_alert=int((meta.label == 0).sum()), n_drowsy=int((meta.label == 1).sum()),
                              n_invalid=int((~meta.valid).sum()), n_long_rt=int(meta.long_rt.sum()), alert_rt=meta.alert_rt.iloc[0],
                              median_rt=meta.rt.median(), channels=" ".join(channels)))
        print(f"[{time.time()-t0:6.0f}s] {name}: {data.shape[1]/FS_IN/60:.1f} min, {len(meta)} trials, "
              f"alert {sess_rows[-1]['n_alert']}, drowsy {sess_rows[-1]['n_drowsy']}, alert_rt {meta.alert_rt.iloc[0]:.3f}s",
              flush=True)
        shutil.rmtree(tmp)
    pd.DataFrame(sess_rows).to_csv(os.path.join(a.out, "sessions.csv"), index=False)

    meta = pd.read_csv(os.path.join(a.out, "trials.csv"))
    x_all = np.concatenate([np.load(os.path.join(a.out, "sessions", s + ".npz"))["x"] for s in meta.session.unique()])
    check_against_sadt(a.out, x_all, meta)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
