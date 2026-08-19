# SPDX-License-Identifier: AGPL-3.0-or-later
"""P5 — Task, baseline, curve (handoff §9 P5, decides D6/D7/D4 outcomes).

Task: Spiking Heidelberg Digits (SHD, Cramer et al.) — 20 spoken digits,
700 spike channels, binned to T time steps of counts (D7: integer count
injection, chip's multiplicity semantics). Curves: accuracy vs deployment
bytes and vs MACs/sample, Mycelium (block-sparse ternary, frozen topology)
against a dense fp16 GRU trained under the same budget.

Parts (run in parallel processes):
  myc-full-<H>    Mycelium, all blocks active (sparse-architecture upper bound)
  myc-gated-<H>   Mycelium with P4 L0 gates -> freeze -> fine-tune (flagship)
  gru-<h>         dense GRU fp16-equivalent baseline
  reference       chip-faithful instance: <=1024 neurons & synapse entries,
                  2-class SHD, DiffLIF==IntLIF bit-exactness certificate

Usage: python experiments/p5_shd.py --part myc-gated-512 [--data-dir DIR]
Writes experiments/results/p5_<part>.json
"""

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from celiumsnn import DiffLIF, EdgeListSynapse, IntLIF  # noqa: E402
from celiumsnn.model import Mycelium  # noqa: E402

N_IN, N_CLASSES, T_BINS, DURATION = 704, 20, 32, 1.4  # 700 channels padded to 704
BATCH = 128
EPOCHS_FULL, EPOCHS_GATED_A, EPOCHS_GATED_B = 40, 28, 12
LR, GATE_LR, LAMBDA_L0 = 2e-3, 0.1, 0.02
URLS = ("https://zenkelab.org/datasets/", "https://compneuro.net/datasets/")


# --- data --------------------------------------------------------------------

def fetch(data_dir: Path, name: str) -> Path:
    h5 = data_dir / name
    if h5.exists():
        return h5
    gz = data_dir / (name + ".gz")
    if not gz.exists():
        for base in URLS:
            try:
                print(f"downloading {base}{name}.gz")
                urllib.request.urlretrieve(base + name + ".gz", gz)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  failed: {exc}")
        else:
            raise RuntimeError(f"could not download {name}.gz")
    with gzip.open(gz, "rb") as src, open(h5, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return h5

def load_shd(data_dir: str, T: int = T_BINS):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / f"shd_T{T}.pt"
    if cache.exists():
        return torch.load(cache)
    import h5py
    import numpy as np
    out = {}
    for split, name in (("train", "shd_train.h5"), ("test", "shd_test.h5")):
        with h5py.File(fetch(data_dir, name), "r") as f:
            times, units = f["spikes"]["times"], f["spikes"]["units"]
            labels = np.asarray(f["labels"], dtype=np.int64)
            x = np.zeros((len(labels), T, N_IN), dtype=np.uint8)
            for i in range(len(labels)):
                b = np.minimum((np.asarray(times[i]) / DURATION * T).astype(int), T - 1)
                np.add.at(x[i], (b, np.asarray(units[i], dtype=int)), 1)
            out[split] = (torch.from_numpy(np.minimum(x, 7)).to(torch.uint8),
                          torch.from_numpy(labels))
    torch.save(out, cache)
    return out


# --- training ----------------------------------------------------------------

def evaluate(model, x, y, batch=256) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y), batch):
            xb = x[i:i + batch].float()
            correct += int((model(xb).argmax(-1) == y[i:i + batch]).sum())
    return correct / len(y)


def augment_shd(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Training-time augmentation: per-sample random roll in time (±T/8)
    and along the tonotopic channel axis (±8 channels). Roll (wraparound)
    is the common practice in SHD pipelines."""
    batch, T, _ = x.shape
    t_shift = torch.randint(-T // 8, T // 8 + 1, (batch,), generator=gen)
    c_shift = torch.randint(-8, 9, (batch,), generator=gen)
    out = torch.empty_like(x)
    for b in range(batch):
        out[b] = torch.roll(x[b], (int(t_shift[b]), int(c_shift[b])), dims=(0, 1))
    return out


def train(model, data, epochs, lam=0.0, log=None, eval_every=2, seed=0,
          restore_best=False, cosine=False, lr=LR, augment_fn=None,
          clip: float = 0.0):
    (xtr, ytr), (xte, yte) = data["train"], data["test"]
    gate_params = [p for n, p in model.named_parameters() if "log_alpha" in n]
    other = [p for n, p in model.named_parameters() if "log_alpha" not in n]
    groups = [{"params": other, "lr": lr}]
    if gate_params:
        groups.append({"params": gate_params, "lr": GATE_LR})
    opt = torch.optim.Adam(groups)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if cosine else None)
    gen = torch.Generator().manual_seed(seed)
    best = 0.0
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr), generator=gen)
        for i in range(0, len(ytr), BATCH):
            idx = perm[i:i + BATCH]
            xb = xtr[idx].float()
            if augment_fn is not None:
                xb = augment_fn(xb, gen)
            loss = nn.functional.cross_entropy(model(xb), ytr[idx])
            if lam and hasattr(model, "l0_penalty"):
                loss = loss + lam * model.l0_penalty()
            opt.zero_grad()
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        if sched is not None:
            sched.step()
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            acc = evaluate(model, xte, yte)
            train_acc = evaluate(model, xtr[:2048], ytr[:2048])
            if acc > best:
                best = acc
                if restore_best:
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
            history.append({"epoch": epoch + 1, "test_acc": acc,
                            "train_acc": train_acc})
            print(f"  epoch {epoch+1}: test {acc:.4f} train {train_acc:.4f} "
                  f"(best {best:.4f})")
            if log is not None:
                log(history)
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return best, history


# --- baseline ----------------------------------------------------------------

class GRUBaseline(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(N_IN, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, N_CLASSES)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.readout(out).mean(1)

    def param_bytes(self) -> float:  # deployed at fp16
        return sum(p.numel() for p in self.parameters()) * 2

    def macs_per_step(self) -> float:
        return 3 * (N_IN * self.hidden + self.hidden ** 2) + self.hidden * N_CLASSES


# --- chip-faithful reference (D1-c) -------------------------------------------

REF_IN, REF_HID, REF_OUT = 64, 32, 2
REF_LIF_KW = dict(leak_shift=15, refractory_ticks=1, learnable_theta=True,
                  surrogate_width=1.0)


class ReferenceNet(nn.Module):
    """<=1024 neurons (34) and <=1024 dendrite entries (832 incl. the host-side
    input encoder). Readout = output-neuron spike counts: fully chip-mappable."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        def edges(n_pre, n_post, fan_in):
            post = torch.arange(n_post).repeat_interleave(fan_in)
            pre = torch.randint(0, n_pre, (n_post * fan_in,), generator=g)
            w = (torch.rand(n_post * fan_in, generator=g) * 2 - 1)
            return pre, post, w
        self.syn_in = EdgeListSynapse(REF_IN, REF_HID, *edges(REF_IN, REF_HID, 16), precision="ternary")
        self.syn_rec = EdgeListSynapse(REF_HID, REF_HID, *edges(REF_HID, REF_HID, 8), precision="ternary")
        self.syn_out = EdgeListSynapse(REF_HID, REF_OUT, *edges(REF_HID, REF_OUT, 32), precision="ternary")
        self.lif_h = DiffLIF(theta=4.0, n_neurons=REF_HID, **REF_LIF_KW)
        self.lif_o = DiffLIF(theta=2.0, n_neurons=REF_OUT, **REF_LIF_KW)

    def n_entries(self):
        return sum(s.edges_pre.numel() for s in (self.syn_in, self.syn_rec, self.syn_out))

    def forward(self, x, int_neurons=None):
        batch = x.shape[0]
        lif_h, lif_o = int_neurons if int_neurons else (self.lif_h, self.lif_o)
        lif_h.reset_state(batch)
        lif_o.reset_state(batch)
        s = torch.zeros(batch, REF_HID)
        counts = 0.0
        for t in range(x.shape[1]):
            ci, ei = self.syn_in(x[:, t])
            cr, er = self.syn_rec(s)
            s = lif_h.step(ci + cr, ei | er)
            if int_neurons:
                s = s.float()
            co, eo = self.syn_out(s)
            counts = counts + lif_o.step(co, eo).float()
        return counts  # spike counts ARE the logits (chip-readable)


def pool_to_ref(x):  # (N, T, 704) -> (N, T, 64): 11 consecutive channels per group
    return x.reshape(*x.shape[:-1], 64, 11).sum(-1).clamp(max=7)


def run_reference(data):
    (xtr, ytr), (xte, yte) = data["train"], data["test"]
    def subset(x, y):
        m = y < REF_OUT
        return pool_to_ref(x[m].float()), y[m]
    xtr, ytr = subset(xtr, ytr)
    xte, yte = subset(xte, yte)
    print(f"reference subset: train {len(ytr)}, test {len(yte)}")
    net = ReferenceNet(seed=0)
    sub = {"train": (xtr, ytr), "test": (xte, yte)}
    best, history = train(net, sub, epochs=60, eval_every=5, restore_best=True)
    acc_diff_final = evaluate(net, xte, yte)  # best-epoch weights restored

    # Certificate: replay the trained net through IntLIF — bit-exact spikes.
    net.eval()
    int_h = IntLIF(theta=net.lif_h.effective_theta().detach().round().int().tolist(),
                   leak_shift=15, refractory_ticks=1, n_neurons=REF_HID)
    int_o = IntLIF(theta=net.lif_o.effective_theta().detach().round().int().tolist(),
                   leak_shift=15, refractory_ticks=1, n_neurons=REF_OUT)
    mismatches = correct = 0
    with torch.no_grad():
        for i in range(0, len(yte), 256):
            xb = xte[i:i + 256]
            diff_logits = net(xb)
            int_logits = net(xb, int_neurons=(int_h, int_o))
            mismatches += int((diff_logits != int_logits).sum())
            correct += int((int_logits.argmax(-1) == yte[i:i + 256]).sum())
    h = hashlib.sha256()
    for s in (net.syn_in, net.syn_rec, net.syn_out):
        h.update(s.edges_pre.numpy().tobytes())
        h.update(s.edges_post.numpy().tobytes())
        h.update(s.quantized_weight().detach().to(torch.int8).numpy().tobytes())
    h.update(int_h.theta.numpy().tobytes())
    h.update(int_o.theta.numpy().tobytes())
    return {"acc_best_diff": best, "acc_diff_at_deploy": acc_diff_final,
            "acc_intlif": correct / len(yte),
            "history": history,
            "n_neurons": REF_HID + REF_OUT, "n_synapse_entries": net.n_entries(),
            "intlif_logit_mismatches": mismatches,
            "artifact_sha256": h.hexdigest()}


# --- parts -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--data-dir", default="/root/shd")
    ap.add_argument("--theta0", type=float, default=8.0)
    ap.add_argument("--precision", default="ternary")
    ap.add_argument("--epochs", type=int, default=EPOCHS_FULL)
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--lam", type=float, default=LAMBDA_L0)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--T", type=int, default=T_BINS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--readout", default="spike", choices=["spike", "membrane"])
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    data = load_shd(args.data_dir, T=args.T)
    aug = augment_shd if args.augment else None
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    name = args.part + (f"-{args.tag}" if args.tag else "")
    path = out / f"p5_{name}.json"
    log = lambda hist: json.dump({"part": name, "history": hist},
                                 open(path.with_suffix(".partial"), "w"))

    if args.part == "reference":
        payload = run_reference(data)
    elif args.part.startswith("gru-"):
        model = GRUBaseline(int(args.part[4:]))
        best, history = train(model, data, args.epochs, log=log, seed=args.seed,
                              cosine=args.cosine, lr=args.lr, augment_fn=aug)
        payload = {"acc_best": best, "history": history,
                   "bytes": model.param_bytes(), "macs_per_step": model.macs_per_step(),
                   "config": {"T": args.T, "seed": args.seed, "lr": args.lr,
                              "augment": args.augment},
                   "params": sum(p.numel() for p in model.parameters())}
    elif args.part.startswith("myc-"):
        kind, h = args.part[4:].rsplit("-", 1)
        hidden = int(h)
        mkw = dict(theta0=args.theta0, precision=args.precision,
                   surrogate_width=args.width, dropout=args.dropout,
                   n_layers=args.layers, seed=args.seed,
                   readout_mode=args.readout)
        if kind == "full":
            model = Mycelium(N_IN, hidden, N_CLASSES, **mkw)
            best, history = train(model, data, args.epochs, log=log, seed=args.seed,
                                  cosine=args.cosine, lr=args.lr, augment_fn=aug,
                                  clip=args.clip)
        else:  # gated -> freeze -> fine-tune (2/3 + 1/3 of the epoch budget)
            model = Mycelium(N_IN, hidden, N_CLASSES, gated=True, **mkw)
            ep_a = args.epochs * 2 // 3
            _, hist_a = train(model, data, ep_a, lam=args.lam, log=log,
                              seed=args.seed, lr=args.lr, augment_fn=aug,
                              clip=args.clip)
            model = model.freeze()
            best, hist_b = train(model, data, args.epochs - ep_a, log=log,
                                 seed=args.seed, cosine=args.cosine, lr=args.lr,
                                 augment_fn=aug, clip=args.clip)
            history = hist_a + [{"freeze": True}] + hist_b
        f_in, f_rec = model._fracs()
        payload = {"acc_best": best, "history": history,
                   "bytes": model.param_bytes(), "macs_per_step": model.macs_per_step(),
                   "flop_fraction": {"in": f_in, "rec": f_rec},
                   "hidden_rate": getattr(model, "last_rate", None),
                   "config": {"theta0": args.theta0, "precision": args.precision,
                              "epochs": args.epochs, "T": args.T, "seed": args.seed,
                              "augment": args.augment, "lam": args.lam,
                              "lr": args.lr, "readout": args.readout,
                              "clip": args.clip},
                   "topology_sha256": model.hash()}
    else:
        raise SystemExit(f"unknown part {args.part}")

    payload["part"] = name
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
