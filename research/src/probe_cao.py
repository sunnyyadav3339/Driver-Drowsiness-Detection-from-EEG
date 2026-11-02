"""Phase 1 probe: download one session of the Cao et al. 2019 pre-processed dataset (figshare 7666055) and print its format.

    python -m research.src.probe_cao --out /tmp/probe
"""
import argparse
import collections
import json
import os
import urllib.request
import zipfile

import numpy as np
import scipy.io as sio

ARTICLE = 7666055


def file_list():
    with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{ARTICLE}/files?page_size=100") as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--index", type=int, default=0, help="which file of the listing to probe")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = file_list()
    json.dump([dict(name=f["name"], size=f["size"], url=f["download_url"]) for f in files],
              open(os.path.join(a.out, "file_list.json"), "w"), indent=1)
    print(f"{len(files)} files, {sum(f['size'] for f in files)/1e9:.2f} GB")
    for f in files:
        print(f"  {f['name']}  {f['size']/1e6:.0f} MB")

    f = files[a.index]
    zpath = os.path.join(a.out, f["name"])
    urllib.request.urlretrieve(f["download_url"], zpath)
    with zipfile.ZipFile(zpath) as z:
        print("zip contents:", [(i.filename, i.file_size) for i in z.infolist()])
        z.extractall(a.out)
    os.remove(zpath)
    set_path = next(os.path.join(dp, n) for dp, _, ns in os.walk(a.out) for n in ns if n.endswith(".set"))
    print("loading", set_path)

    m = sio.loadmat(set_path, squeeze_me=True, struct_as_record=False)
    eeg = m.get("EEG", None)
    if eeg is None:  # newer EEGLAB files store the fields at top level
        print("top-level keys:", [k for k in m if not k.startswith("__")])
        eeg = type("E", (), {k: m[k] for k in m if not k.startswith("__")})
    fields = [k for k in dir(eeg) if not k.startswith("_")]
    print("EEG fields:", fields)
    for k in ("nbchan", "pnts", "srate", "trials", "xmin", "xmax", "ref"):
        if hasattr(eeg, k):
            print(f"  {k} = {getattr(eeg, k)}")
    data = getattr(eeg, "data", None)
    print("data:", type(data), getattr(data, "shape", data))
    if hasattr(eeg, "chanlocs"):
        print("channels:", [c.labels for c in np.atleast_1d(eeg.chanlocs)])
    ev = np.atleast_1d(getattr(eeg, "event", []))
    print("n events:", len(ev))
    if len(ev):
        print("event fields:", [k for k in dir(ev[0]) if not k.startswith("_")])
        types = [str(e.type) for e in ev]
        print("event type counts:", collections.Counter(types).most_common(20))
        for e in ev[:25]:
            print("  ", {k: getattr(e, k) for k in dir(e) if not k.startswith("_")})
    for k in fields:  # anything RT-like stored elsewhere
        if k.lower() in ("etc", "epoch", "urevent", "comments", "history"):
            v = getattr(eeg, k)
            print(f"{k}: {str(v)[:600]}")


if __name__ == "__main__":
    main()
