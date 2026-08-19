# SPDX-License-Identifier: Apache-2.0
"""P6 — DVS-Gesture as the second task (generality check).

IBM DVS128 Gesture: 11 hand gestures, 128x128x2-polarity event camera.
Preprocessing: spatial 8x downsample to 16x16x2 = 512 channels, events
binned to T count bins (same D7 integer-injection encoding as SHD).
Dataset via Tonic. Parts mirror p5_shd.py: myc-gated-<H> and gru-<h>.

Usage: python experiments/p6_dvs.py --part myc-gated-512 [--data-dir DIR]
Writes experiments/results/p6dvs_<part>.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from celiumsnn.model import Mycelium  # noqa: E402
from experiments import p5_shd  # noqa: E402
from experiments.p5_shd import GRUBaseline, train  # noqa: E402

N_IN, N_CLASSES = 512, 11  # 16 x 16 x 2 polarity
SPATIAL_DOWN = 8


def load_dvs(data_dir: str, T: int = 32):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / f"dvsg_T{T}.pt"
    if cache.exists():
        return torch.load(cache)
    import numpy as np
    import tonic
    out = {}
    for split, train_flag in (("train", True), ("test", False)):
        ds = tonic.datasets.DVSGesture(save_to=str(data_dir), train=train_flag)
        xs = np.zeros((len(ds), T, N_IN), dtype=np.uint8)
        ys = np.zeros(len(ds), dtype=np.int64)
        side = 128 // SPATIAL_DOWN
        for i in range(len(ds)):
            events, label = ds[i]
            t = events["t"].astype(np.float64)
            tb = np.minimum(((t - t.min()) / max(t.ptp(), 1) * T).astype(int), T - 1)
            cx = (events["x"] // SPATIAL_DOWN).astype(int)
            cy = (events["y"] // SPATIAL_DOWN).astype(int)
            cp = events["p"].astype(int)
            chan = (cp * side + cy) * side + cx
            np.add.at(xs[i], (tb, chan), 1)
            ys[i] = label
        out[split] = (torch.from_numpy(np.minimum(xs, 7)).to(torch.uint8),
                      torch.from_numpy(ys))
        print(f"{split}: {len(ys)} samples")
    torch.save(out, cache)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--data-dir", default="/root/dvsg")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    p5_shd.N_CLASSES = N_CLASSES  # GRUBaseline reads the module global
    p5_shd.N_IN = N_IN
    data = load_dvs(args.data_dir, T=args.T)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    path = out / f"p6dvs_{args.part}-s{args.seed}.json"

    if args.part.startswith("gru-"):
        model = GRUBaseline(int(args.part[4:]))
        best, history = train(model, data, args.epochs, cosine=True,
                              lr=1e-3, seed=args.seed)
        payload = {"acc_best": best, "history": history,
                   "bytes": model.param_bytes(),
                   "macs_per_step": model.macs_per_step()}
    elif args.part.startswith("myc-gated-"):
        hidden = int(args.part.rsplit("-", 1)[1])
        model = Mycelium(N_IN, hidden, N_CLASSES, gated=True, theta0=48.0,
                         dropout=0.25, seed=args.seed)
        ep_a = args.epochs * 2 // 3
        _, hist_a = train(model, data, ep_a, lam=args.lam, lr=args.lr,
                          seed=args.seed)
        model = model.freeze()
        best, hist_b = train(model, data, args.epochs - ep_a, cosine=True,
                             lr=args.lr, seed=args.seed)
        f_in, f_rec = model._fracs()
        payload = {"acc_best": best,
                   "history": hist_a + [{"freeze": True}] + hist_b,
                   "bytes": model.param_bytes(),
                   "macs_per_step": model.macs_per_step(),
                   "flop_fraction": {"in": f_in, "rec": f_rec},
                   "topology_sha256": model.hash()}
    else:
        raise SystemExit(f"unknown part {args.part}")

    payload["part"] = args.part
    payload["config"] = {"T": args.T, "seed": args.seed, "epochs": args.epochs}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
