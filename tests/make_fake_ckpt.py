"""Write a randomly initialised fm_policy_S checkpoint in the trainer's format (CPU smoke tests),
plus a fake 90-task feature bank."""
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_cpu import CFG, make_policy  # noqa: E402

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "smoke")
os.makedirs(os.path.join(out, "features"), exist_ok=True)
p = make_policy()
cfg = OmegaConf.to_container(OmegaConf.load(CFG), resolve=True)
cfg["algo"]["policy"]["encoder"]["image_encoder_factory"]["pretrained"] = False
torch.save({"config": cfg, "model": p.state_dict(), "norm_stats": p.normalizer.stats}, os.path.join(out, "fake_base.pth"))

g = torch.Generator().manual_seed(0)
n_tasks, n_demos, T = 90, 14, 24
ptr = torch.zeros(n_tasks, 50, 2, dtype=torch.long)
n = 0
for t in range(n_tasks):
    for d in range(n_demos):
        ptr[t, d] = torch.tensor([n, T])
        n += T
torch.save({"encm": torch.randn(n, 4, 256, generator=g).half(), "act": (torch.rand(n, 16, 7, generator=g) * 2 - 1).half(),
            "ptr": ptr, "task_emb": torch.randn(n_tasks, 512), "descs": [f"task {i}" for i in range(n_tasks)]},
           os.path.join(out, "features", "libero_90.pt"))
print("wrote", out)
