"""Precompute per-frame encoder summaries + normalized action chunks for LIBERO-90 and LIBERO-10.

Because the frozen encoders are never updated (memories, fine-tuning and consolidation all
train only the decoder or memory modules), every frame's conditioning can be computed once:
encm (L=4, D=256) plus its normalized 16-step action chunk. ~2.5 GB for both suites in fp16,
which fits on the GPU, so meta-training and distillation never touch images.

    python -m horsea.features --suite libero_90   (and libero_10)
"""
import argparse
import os

import h5py
import numpy as np
import torch
from libero.libero.benchmark import get_benchmark
from tqdm import tqdm

import horsea  # noqa: F401
from horsea.base import Flow, load_policy
from horsea.paths import BASE_CKPT, DATA_PREFIX, FEAT_DIR

IMG_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
LOWDIM_KEYS = ("robot0_eef_pos", "robot0_gripper_qpos")


def task_embeddings(benchmark):
    from imitation.envs.libero.utils import get_task_embs

    descs = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    return torch.stack([e["task_emb"] for e in get_task_embs("clip", descs)]), descs


def action_chunks(actions, chunk):
    """actions (T, A) -> (T, chunk, A), padded by repeating the last action (as the dataset does)."""
    T = actions.shape[0]
    idx = np.minimum(np.arange(T)[:, None] + np.arange(chunk)[None], T - 1)
    return actions[idx]


@torch.no_grad()
def encode_frames(flow, f_demo, task_emb, device, bs=256):
    obs = f_demo["obs"]
    T = obs[IMG_KEYS[0]].shape[0]
    imgs = {k: obs[k][()] for k in IMG_KEYS}
    low = {k: obs[k][()].astype(np.float32) for k in LOWDIM_KEYS}
    out = []
    for s in range(0, T, bs):
        sl = slice(s, min(T, s + bs))
        B = sl.stop - sl.start
        o = {k: torch.from_numpy(imgs[k][sl]).to(device).permute(0, 3, 1, 2)[:, None].float() for k in IMG_KEYS}
        o.update({k: torch.from_numpy(low[k][sl]).to(device)[:, None] for k in LOWDIM_KEYS})
        out.append(flow.encode({"obs": o, "task_emb": task_emb.to(device).expand(B, -1)}).half().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=["libero_90", "libero_10"])
    ap.add_argument("--ckpt", default=BASE_CKPT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_demos", type=int, default=50)
    ap.add_argument("--max_tasks", type=int, default=None, help="smoke tests only")
    ap.add_argument("--instruction", default=None,
                    help="encode every task with this one instruction (RoboTTT one-shot protocol)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    policy, _ = load_policy(args.ckpt, args.device)
    flow = Flow(policy)
    bench = get_benchmark(args.suite)()
    task_emb, descs = task_embeddings(bench)
    if args.instruction is not None:
        from imitation.envs.libero.utils import get_task_embs
        generic = get_task_embs("clip", [args.instruction])[0]["task_emb"]
        task_emb = generic.unsqueeze(0).expand(len(descs), -1).clone()
    n_tasks = args.max_tasks or bench.n_tasks
    norm = policy.normalizer

    encm, act, ptr = [], [], torch.zeros(n_tasks, args.max_demos, 2, dtype=torch.long)
    n = 0
    for ti in tqdm(range(n_tasks), desc=args.suite):
        path = os.path.join(DATA_PREFIX, "libero", bench.get_task_demonstration(ti))
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda s: int(s[5:]))[: args.max_demos]
            for di, dname in enumerate(demos):
                d = f["data"][dname]
                e = encode_frames(flow, d, task_emb[ti : ti + 1], args.device)
                raw = torch.from_numpy(action_chunks(d["actions"][()].astype(np.float32), flow.chunk))
                a = norm.normalize({"actions": raw})["actions"].clamp(-1, 1).half()
                encm.append(e)
                act.append(a)
                ptr[ti, di] = torch.tensor([n, e.shape[0]])
                n += e.shape[0]

    out = args.out or os.path.join(FEAT_DIR, f"{args.suite}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(
        {"encm": torch.cat(encm), "act": torch.cat(act), "ptr": ptr, "task_emb": task_emb,
         "descs": descs[:n_tasks], "ckpt": args.ckpt, "instruction": args.instruction},
        out,
    )
    print(f"saved {n} frames from {n_tasks} tasks to {out}")


if __name__ == "__main__":
    main()
