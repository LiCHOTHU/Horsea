"""Plumbing smoke test for the rollout harness (random policy, a few env steps).

    MUJOCO_GL=osmesa python tests/smoke_rollout.py cpu     (no GPU)
    python tests/smoke_rollout.py cuda:0
"""
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import horsea  # noqa: E402,F401
from horsea.base import Flow  # noqa: E402
from horsea.memory import build_memory  # noqa: E402
from horsea.rollout import install_sampler, make_runner, run_task  # noqa: E402
from test_cpu import CFG, make_policy  # noqa: E402

def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    policy = make_policy().to(dev)
    policy.device = dev
    flow = Flow(policy)
    shape_meta = OmegaConf.to_container(OmegaConf.load(CFG).task.shape_meta, resolve=True)

    for arm in ["fmw", "ttt"]:
        mem = build_memory(arm).to(dev).requires_grad_(False)
        state = mem.init_state(1, requires_grad=True)
        state = mem.write(flow, state, torch.randn(30, 4, 256, device=dev), torch.rand(30, 16, 7, device=dev) * 2 - 1, False)
        rec = []
        runner = make_runner(shape_meta, "libero_90", n_rollouts=2, n_par=int(os.environ.get("NPAR", 2)),
                             init_offset=25, device=dev)
        runner.max_episode_length = 12
        res = run_task(runner, install_sampler(policy, flow, mem, state, rec), task_id=2)
        print(arm, {k: res[k] for k in ("env_name", "success", "length", "init", "rollout_sec")},
              "recorded contexts:", sum(r.shape[0] for r in rec), tuple(rec[0].shape))
        assert res["init"] == [25, 26] and len(rec) > 0
    print("ROLLOUT SMOKE OK")


if __name__ == "__main__":
    main()
