"""Frozen action-chunk representation Phi for the perceptual action distance
    d_Phi(a, b) = alpha * ||a - b||^2 + ||Phi(a) - Phi(b)||^2        (alpha > 0 keeps physical differences)
Phi = encoder of a small autoencoder on normalized 16x7 action chunks of the 80 LIBERO-90 training
tasks' demos (reconstruction grounding: the latent must retain what distinguishes chunks).

    python -m horsea.phi train      -> experiments/phi/phi.pt
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

import horsea  # noqa: F401
from horsea.bank import FeatureBank
from horsea.paths import EXP, FEAT_DIR, TRAIN_90

CHUNK, ADIM, LATENT = 16, 7, 64
PATH = os.path.join(EXP, "phi", "phi.pt")


class ActionAE(nn.Module):
    def __init__(self, hid=512, latent=LATENT):
        super().__init__()
        d = CHUNK * ADIM
        self.enc = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, d))

    def forward(self, a):
        z = self.enc(a.flatten(1))
        return z, self.dec(z).view(-1, CHUNK, ADIM)


class Phi(nn.Module):
    """Frozen encoder + the perceptual distance."""

    def __init__(self, path=PATH, alpha=1.0, device="cuda:0"):
        super().__init__()
        ae = ActionAE().to(device)
        sd = torch.load(path, map_location=device, weights_only=False)
        ae.load_state_dict(sd["ae"])
        self.enc = ae.enc.eval().requires_grad_(False)
        self.scale = sd["latent_std"]  # normalise latent so both terms are O(1)
        self.alpha = alpha

    def forward(self, a):
        return self.enc(a.flatten(1)) / self.scale

    def dist(self, a, b):
        """Per-row d_Phi(a, b)."""
        return self.alpha * ((a - b) ** 2).flatten(1).mean(1) + ((self(a) - self(b)) ** 2).mean(1)


def train(args):
    dev = args.device
    torch.manual_seed(0)
    bank = FeatureBank(os.path.join(FEAT_DIR, "libero_90.pt"), dev)
    idx = torch.cat([bank.task_frames(t, range(bank.n_demos(t))) for t in TRAIN_90])
    perm = torch.randperm(len(idx), device=idx.device)
    val, tr = idx[perm[:4096]], idx[perm[4096:]]
    ae = ActionAE().to(dev)
    opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    t0, log = time.time(), []
    for s in range(1, args.steps + 1):
        _, a = bank.gather(tr[torch.randint(len(tr), (args.bs,), device=dev)])
        a = a.clamp(-1, 1)
        _, rec = ae(a)
        loss = ((rec - a) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if s % 1000 == 0 or s == args.steps:
            with torch.no_grad():
                _, av = bank.gather(val)
                av = av.clamp(-1, 1)
                zv, rv = ae(av)
                vl = ((rv - av) ** 2).mean().item()
                rel = vl / av.var().item()
            rec_ = {"step": s, "train": round(loss.item(), 5), "val_mse": round(vl, 5), "val_rel": round(rel, 4),
                    "min": round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec_), flush=True)
            log.append(rec_)
    with torch.no_grad():
        z, _ = ae(av)
    torch.save({"ae": ae.state_dict(), "latent_std": z.std().item(), "log": log}, PATH)
    print("saved", PATH, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
