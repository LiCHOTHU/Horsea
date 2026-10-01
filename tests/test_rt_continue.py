"""Checks for the Stage-B continuation trainer (horsea/rt/continue_train.py).

    PYTHONPATH=. /home/licho/anaconda3/envs/robotwin/bin/python tests/test_rt_continue.py
"""
import collections
import copy

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from horsea.rt.continue_train import BASE, LOOPS, fm_per_example, sample_graph, set_modes, trainable  # noqa: E402
from horsea.rt.graph import install, library  # noqa: E402
from horsea.rt.policy import RTFlowPolicy  # noqa: E402
from horsea.rt.train import augment  # noqa: E402

dev = "cuda:0"
s = torch.load(BASE, map_location=dev, weights_only=False)
lib = library("primary")
g = torch.Generator(device=dev).manual_seed(0)
B = 6
imgs = torch.rand(B, 3, 3, 120, 160, device=dev, generator=g)
state = torch.rand(B, 14, device=dev, generator=g)
lang = torch.randn(B, 512, device=dev, generator=g)
actions = torch.rand(B, 16, 14, device=dev, generator=g)


def fresh(graph=None):
    m = RTFlowPolicy(s["stats"]).to(dev)
    m.load_state_dict(s["model"])
    install(m.velocity_net, lib[graph] if graph else None)
    m.velocity_net._graph_step = None
    m.eval()                      # deterministic for equality checks (dropout off)
    return m


def gen(k):
    return torch.Generator(device=dev).manual_seed(k)


def test_per_example_equals_native_loss():
    native = RTFlowPolicy(s["stats"]).to(dev).eval()
    native.load_state_dict(s["model"])
    l_native = native.loss(imgs, state, lang, actions, gen(7))
    per, t = fm_per_example(fresh("G0"), imgs, state, lang, actions, gen(7))
    assert torch.allclose(per.mean(), l_native, atol=1e-7), (per.mean(), l_native)
    return f"fm_per_example(G0 via executor) == native RTFlowPolicy.loss ({float(l_native):.6f})"


def test_streams_identical_across_arms():
    # same update index -> same augmentation, same FM noise/time, regardless of the graph drawn for M
    out = []
    for gname in ("G0", "b1_late"):
        m = fresh(gname)
        x = augment(imgs.clone(), gen(11))
        _, t = fm_per_example(m, x, state, lang, actions, gen(13))
        out.append((x, t))
    assert torch.equal(out[0][0], out[1][0]) and torch.equal(out[0][1], out[1][1])
    return "augmentation and FM time/noise identical across graphs at the same update"


def test_shared_gradients_and_frozen_params():
    grads = {}
    for gname in ("G0", "b1_middle"):
        m = fresh(gname)
        names, params = trainable(m)
        n_before = sum(p.numel() for p in m.parameters())
        set_modes(m)
        for mod in m.modules():                 # dropout off so the two graphs are comparable
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
        t_mid = torch.full((B,), 0.45, device=dev)   # inside the middle window [0.2, 0.7)
        cond = m.obs_tokens(imgs, state, lang)
        x1 = m.norm_a(actions).clamp(-1, 1)
        x0 = torch.randn(x1.shape, device=dev, generator=gen(3))
        psi = (1 - (1 - m.sig_min) * t_mid[:, None, None]) * x0 + t_mid[:, None, None] * x1
        _, v = m.velocity_net(psi, t_mid, cond)
        ((v - x1) ** 2).mean().backward()
        blk = m.velocity_net.decoder.layers[1].linear1.weight
        assert blk.grad is not None and torch.isfinite(blk.grad).all() and blk.grad.abs().sum() > 0
        assert all(p.grad is None for n, p in m.named_parameters() if not p.requires_grad)
        assert sum(p.numel() for p in m.parameters()) == n_before
        grads[gname] = blk.grad.clone()
        # optimizer step changes only trainable tensors
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        torch.optim.AdamW(params, lr=1e-3).step()
        changed = {n for n, p in m.named_parameters() if not torch.equal(p, before[n])}
        assert changed <= set(names), sorted(changed - set(names))[:3]
        assert not any(n.startswith(("vision.", "lang.", "state.", "cam_emb", "velocity_net.encoder.")) for n in changed)
        assert "velocity_net.time_net.w" not in changed
    assert not torch.allclose(grads["G0"], grads["b1_middle"]), "repeat visit contributed no gradient"
    return f"both visits accumulate in the single shared block; {len(names)} trainable tensors; encoders frozen"


def test_graph_sampler_distribution():
    c = collections.Counter(sample_graph("M", 0, i) for i in range(24000))
    p0 = c["G0"] / 24000
    loops = [c[l] / 24000 for l in LOOPS]
    assert abs(p0 - 0.5) < 0.015 and all(abs(x - 1 / 24) < 0.006 for x in loops), (p0, loops)
    assert all(sample_graph("C", 0, i) == "G0" for i in range(100))
    assert sample_graph("M", 0, 123) == sample_graph("M", 0, 123)
    return f"P(G0)={p0:.3f}, loop shares {min(loops):.4f}-{max(loops):.4f} (target 0.0417); deterministic per update"


if __name__ == "__main__":
    for f in (test_per_example_equals_native_loss, test_streams_identical_across_arms,
              test_shared_gradients_and_frozen_params, test_graph_sampler_distribution):
        print("PASS", f.__name__, "--", f(), flush=True)
