"""Train a flow-matching policy on RoboTwin with validation loss and early stopping.

The imitation repo's train.py has no validation split and no early-stopping criterion -- it runs a
fixed epoch count. This driver adds both, holding out whole EPISODES (frames inside an episode are
near-duplicates, so a frame-level split leaks and the validation curve becomes uninformative).

    python scripts/train_robotwin.py --algo fm_policy_M --out DIR
    python scripts/train_robotwin.py --algo fm_policy_M --out DIR --smoke   # 1 short epoch

Checkpoints are written in the trainer's format (model / config / norm_stats), so
horsea.base.load_policy can read them unchanged.
"""
import argparse
import json
import os
import sys
import time

HORSEA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Python puts this file's directory (scripts/) on sys.path, where scripts/queue.py shadows the
# stdlib `queue` that torch.fx imports. Drop it before importing torch.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if not p or os.path.abspath(p) != _HERE]
sys.path.insert(0, HORSEA)

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import horsea  # noqa: E402,F401
import horsea.base  # noqa: E402,F401  registers OmegaConf's "eval" resolver, which
#                                      imitation's rollout config interpolations use
from horsea.paths import IMITATION  # noqa: E402

DATA_ROOT = os.environ.get(
    "ROBOTWIN_DATA_ROOT",
    "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/RoboTwin-data/demo_clean")


def build(args):
    """Compose the imitation config for this policy, then instantiate policy + datasets."""
    from imitation.envs.robotwin.hdf5_dataset import build_robotwin_hdf5_dataset

    with initialize_config_dir(config_dir=os.path.join(IMITATION, "config"), version_base=None):
        cfg = compose(config_name="train", overrides=[
            "task=robotwin", f"algo={args.algo}", "task.zarr_glob=unused",
            "rollout.enabled=false", "logging.mode=disabled",
        ])
    policy = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta)
    common = dict(data_root=DATA_ROOT, tasks=args.tasks, seq_len=cfg.algo.dataset.seq_len,
                  obs_seq_len=cfg.algo.dataset.obs_seq_len,
                  img_height=cfg.task.img_height, img_width=cfg.task.img_width,
                  task_embedding_format=cfg.task.task_embedding_format,
                  val_fraction=args.val_fraction, max_episodes=args.max_episodes)
    train = build_robotwin_hdf5_dataset(split="train", **common)
    val = build_robotwin_hdf5_dataset(split="val", **common)
    return cfg, policy, train, val


def norm_stats(policy, dataset, loader_kw, n_batches=40):
    """Action/lowdim ranges for the policy normalizer, from the TRAIN split only."""
    dl = torch.utils.data.DataLoader(dataset, shuffle=True, **loader_kw)
    keys = list(policy.shape_meta["observation"]["lowdim"].keys())
    acc = {"actions": []}
    acc.update({k: [] for k in keys})
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        acc["actions"].append(batch["actions"].reshape(-1, batch["actions"].shape[-1]))
        for k in keys:
            acc[k].append(batch["obs"][k].reshape(-1, batch["obs"][k].shape[-1]))
    return {k: {"min": torch.cat(v).min(0).values.numpy(),
                "max": torch.cat(v).max(0).values.numpy()} for k, v in acc.items()}


@torch.no_grad()
def validate(policy, loader, device, max_batches=None):
    policy.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = move(batch, device)
        loss, _ = policy.compute_loss(batch)
        total += float(loss)
        n += 1
    policy.train()
    return total / max(n, 1)


def move(batch, device):
    batch["obs"] = {k: v.to(device, non_blocking=True) for k, v in batch["obs"].items()}
    batch["actions"] = batch["actions"].to(device, non_blocking=True)
    batch["task_emb"] = batch["task_emb"].to(device, non_blocking=True)
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="fm_policy_M",
                    help="fm_policy_S (18.3M) | fm_policy_M (embed 512, ff 1024) | fm_policy_L")
    ap.add_argument("--tasks", nargs="*", default=None, help="default: every task under the data root")
    ap.add_argument("--epochs", type=int, default=400, help="upper bound; early stopping decides")
    ap.add_argument("--patience", type=int, default=15,
                    help="stop after this many epochs without a new best validation loss")
    ap.add_argument("--min-delta", type=float, default=1e-4,
                    help="improvement smaller than this does not count as progress")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=.15)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--max-steps-per-epoch", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true", help="1 tiny epoch; verifies the whole path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.max_steps_per_epoch, args.max_episodes, args.workers = 1, 6, 3, 2

    os.makedirs(args.out, exist_ok=True)
    device = args.device
    cfg, policy, train_ds, val_ds = build(args)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"algo={args.algo}  parameters={n_params:,}", flush=True)

    # fork is fine here: this process holds no simulator/EGL state, unlike scripts/train_base.py
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                     pin_memory=True, persistent_workers=args.workers > 0,
                     drop_last=False)
    stats = norm_stats(policy, train_ds, dict(batch_size=args.batch_size, num_workers=args.workers))
    policy.normalizer.fit(stats)
    policy.to(device).train()

    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kw)
    val_dl = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kw)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    best, best_epoch, history = float("inf"), -1, []
    t0 = time.time()
    for epoch in range(args.epochs):
        run, n = 0.0, 0
        for i, batch in enumerate(train_dl):
            if args.max_steps_per_epoch and i >= args.max_steps_per_epoch:
                break
            loss, _ = policy.compute_loss(move(batch, device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            run += float(loss)
            n += 1
        tr = run / max(n, 1)
        va = validate(policy, val_dl, device, max_batches=args.max_steps_per_epoch)
        improved = va < best - args.min_delta
        rec = {"epoch": epoch, "train": round(tr, 6), "val": round(va, 6),
               "best": round(min(best, va), 6), "improved": bool(improved),
               "min": round((time.time() - t0) / 60, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)

        if improved:
            best, best_epoch = va, epoch
            torch.save({"model": policy.state_dict(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "norm_stats": stats, "epoch": epoch, "val": va},
                       os.path.join(args.out, "multitask_model.pth"))
        # written after `best` is updated, so the recorded best matches the saved checkpoint
        json.dump({"algo": args.algo, "parameters": n_params, "history": history,
                   "best_val": best if best < float("inf") else None, "best_epoch": best_epoch,
                   "patience": args.patience, "early_stopped": False},
                  open(os.path.join(args.out, "history.json"), "w"), indent=1)

        if not improved and epoch - best_epoch >= args.patience:
            print(f"EARLY STOP: no validation improvement for {args.patience} epochs "
                  f"(best {best:.6f} at epoch {best_epoch})", flush=True)
            break
    print(f"DONE best_val={best:.6f} at epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
