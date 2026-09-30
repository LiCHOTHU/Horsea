"""Immutable run manifest for the internal-looping study (spec sec. 3, 6, 13).

    python -m horsea.loop.manifest   -> experiments/loop/manifest.json (refuses to overwrite)
"""
import hashlib
import json
import os
import subprocess
import sys

import torch

OUT = "experiments/loop/manifest.json"
IMIT = os.path.expanduser("~/workspace/imitation")


def sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists (immutable)")
    import horsea  # noqa: F401
    from horsea.paths import BASE_CKPT
    import libero, mujoco, numpy, robosuite
    proto = json.load(open("experiments/manifest/protocol_v2.json"))
    sd = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    pol = sd["config"]["algo"]["policy"]
    names = {t["id"]: t["name"] for t in proto["tasks"]}
    dev_tasks = [23, 32, 81]
    confirm = [72, 3, 6, 33, 69, 5, 7, 11, 15, 16]
    m = {
        "study": "internal looping of DiT decoder blocks in the LIBERO FM policy (spec 2026-09-30)",
        "checkpoint": {"path": BASE_CKPT, "sha256": sha(BASE_CKPT), "epoch": sd["epoch"], "steps": sd["steps"]},
        "code": {"horsea_commit": git(".", "rev-parse", "HEAD"), "horsea_dirty": git(".", "status", "--porcelain"),
                 "imitation_commit": git(IMIT, "rev-parse", "HEAD"),
                 "imitation_dirty_files": git(IMIT, "status", "--porcelain").splitlines(),
                 "imitation_diff_sha256": hashlib.sha256(git(IMIT, "diff").encode()).hexdigest(),
                 "native_entry_points": {"fm_loss_and_sampler": "imitation/algos/fm_policy.py FlowMatchingPolicy",
                                         "block": "dit_modules.py _DiTDecoder.forward",
                                         "decoder_stack": "dit_modules.py _TransformerDecoder.forward (replaced "
                                                          "per-instance by horsea.loop.core.install -> forward_dec)",
                                         "horsea_wrapper": "horsea/base.py Flow.decode -> horsea.loop.core.decoder_forward"}},
        "suite": "LIBERO-90, base trained on the 80-task subset libero_90_train80 (NOT the named LIBERO-10 suite)",
        "train80_ids": proto["train_80"],
        "demonstrations": {"per_task": 50, "continuation_train": "demo_0..demo_44", "diagnostic": "demo_45..demo_49",
                           "note": "all 50 demos per task were in the base's historical pretraining; diagnostic "
                                   "episodes are held out from continuation training only"},
        "development_tasks": {str(t): names[t] for t in dev_tasks},
        "development_rule": "chosen before any loop result from the existing base audit (experiments/rdm/audit, "
                            "receding-horizon regime, starts 0-9): success away from 0 and 1 (23: 0.5, 32: 0.6, 81: 0.4)",
        "confirmation_tasks": {str(t): names[t] for t in confirm},
        "confirmation_rule": "declared before any loop result: the 10 train-80 tasks outside the development set with "
                             "the lowest nonzero audit success, ties by task id (audit covered 57/80 tasks; task 51 "
                             "at 0.0 excluded)",
        "smoke_task": {"23": names[23], "starts": "40-49"},
        "starts": {"development": "0-19", "confirmation": "20-39", "smoke": "40-49",
                   "note": "LIBERO fixed init states; starts 0-49 were used by earlier Horsea studies but never for "
                           "tuning in this study"},
        "control_protocol": {"temporal_agg": pol["temporal_agg"], "replanning": "every env step (temporal aggregation "
                             "averages overlapping predictions)", "chunk": pol["chunk_size"],
                             "action_horizon_config": pol["action_horizon"], "horizon_env_steps": 300,
                             "episode_end": "first success or 300 steps"},
        "fm": {"sig_min": pol["flow_sig_min"], "time": "t in [0,1): noise at 0, actions at 1",
               "train_time_sampler": f"t = (1 - sig_min)(1 - b), b ~ Beta({pol['flow_alpha']}, {pol['flow_beta']})",
               "sampler": f"uniform Euler, K = {pol['num_inference_steps']} evaluations at t = 0, 0.1, ..., 0.9",
               "path": "z_t = [1 - (1 - sig_min) t] eps + t a; u_t = a - (1 - sig_min) eps"},
        "actions": {"dim": 7, "representation": "delta (abs_action false), axis-angle rotation, gripper",
                    "normalization": "checkpoint norm_stats via policy.normalizer", "clipping": "[-1, 1] after sampling"},
        "observations": {"rgb": list(sd["config"]["task"]["shape_meta"]["observation"]["rgb"].keys()),
                         "lowdim": list(sd["config"]["task"]["shape_meta"]["observation"]["lowdim"].keys()),
                         "image": "128x128", "language": "CLIP task embeddings (train split embedding at training, "
                                                         "test embedding set on the benchmark at evaluation)"},
        "precision": "fp32 (tf32 matmul as torch default; parity tests with tf32 disabled)",
        "seeds": {"eval_noise": "sha256(eval seed | task | start | replanning index) -> per-row generator",
                  "train": "data order per (seed, epoch); FM noise/time per (seed, update); augmentation per (seed, update)"},
        "versions": {"torch": torch.__version__, "numpy": numpy.__version__, "mujoco": mujoco.__version__,
                     "robosuite": getattr(robosuite, "__version__", "?"), "libero": getattr(libero, "__version__", "?")},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(m, open(OUT, "w"), indent=1)
    print(json.dumps({k: m[k] for k in ("development_tasks", "confirmation_tasks")}, indent=1))


if __name__ == "__main__":
    main()
