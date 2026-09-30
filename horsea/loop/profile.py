"""Inference cost of one action-chunk generation (spec sec. 11): synchronized GPU timing after warm-up, same
precision and batch size, no simulator; observation encoding included once per chunk (native sampler).

    python -m horsea.loop.profile --configs base_K10 K12 K13 loop_l1_all ... --out experiments/loop/profile.json
"""
import argparse
import json
import time

import numpy as np
import torch

import horsea  # noqa: F401
from horsea.base import load_policy
from horsea.loop.core import CallCounter, LoopConfig, install
from horsea.paths import BASE_CKPT


def chunk_latency(policy, K, cfg, B=1, reps=200, warm=30):
    vnet = policy.velocity_net
    counter = CallCounter()
    install(vnet, cfg, counter)
    dev = next(vnet.parameters()).device
    cond = torch.randn(B, 3, 256, device=dev)
    times = []
    with torch.no_grad():
        for i in range(warm + reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            enc = vnet.forward_enc(cond)
            z = torch.randn(B, 16, 7, device=dev)
            t = torch.zeros(B, device=dev)
            counter.reset()
            for _ in range(K):
                z = z + vnet.forward_dec(z, t, enc) / K
                t = t + 1.0 / K
            torch.cuda.synchronize()
            if i >= warm:
                times.append(time.perf_counter() - t0)
    return {"median_ms": 1e3 * float(np.median(times)), "p95_ms": 1e3 * float(np.percentile(times, 95)),
            "block_calls": counter.logical // B}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="JSON dict name -> {K, loop}")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    policy, _ = load_policy(BASE_CKPT, "cuda:0")
    policy.eval()
    res = {}
    for name, c in json.loads(open(args.configs).read() if args.configs.endswith(".json") else args.configs).items():
        res[name] = chunk_latency(policy, c["K"], LoopConfig(**c.get("loop", {"enabled": False})), B=args.B)
        print(name, res[name], flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
