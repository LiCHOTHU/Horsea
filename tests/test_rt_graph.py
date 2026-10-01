"""Acceptance checks for the RoboTwin execution-graph study (spec 2026-10-01, sec. 7). FP32, TF32 off.

    PYTHONPATH=. /home/licho/anaconda3/envs/robotwin/bin/python tests/test_rt_graph.py
"""
import copy
import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from horsea.rt.graph import ORIGINAL, Tracer, WINDOWS, block_calls, decoder_forward, install, library, run_visits  # noqa
from horsea.rt.policy import ADIM, CHUNK, RTFlowPolicy  # noqa: E402

dev = "cuda:0"
CK = "experiments/rt/base/step300000.pt"
s = torch.load(CK, map_location=dev, weights_only=False)
base = RTFlowPolicy(s["stats"]).to(dev).eval()
base.load_state_dict(s["model"])
LIB = library("all")
g = torch.Generator(device=dev).manual_seed(0)
imgs = torch.rand(3, 3, 3, 120, 160, device=dev, generator=g)
state = (torch.rand(3, 14, device=dev, generator=g) * (base.s_max - base.s_min) + base.s_min)
lang = torch.randn(3, 512, device=dev, generator=g)
noise = torch.randn(3, CHUNK, ADIM, device=dev, generator=g)


def model_with(graph, tracer=None):
    m = copy.deepcopy(base)
    install(m.velocity_net, LIB[graph] if graph else None, tracer)
    return m


def test_original_parity():
    ref = base.sample(imgs, state, lang, noise=noise.clone())                 # native, nothing installed
    out = model_with("G0").sample(imgs, state, lang, noise=noise.clone())
    assert torch.equal(ref, out), (ref - out).abs().max()
    # executed commands = first 8 actions of the chunk (receding horizon): identical as well
    return "G0 through the executor == native sampler (action chunks bit-identical, so executed commands too)"


def test_trace_counts_and_identities():
    out = {}
    for name in ("b1_middle", "b3_early", "pair_alternating_late", "depth3_middle"):
        tr = Tracer()
        model_with(name, tr).sample(imgs[:1], state[:1], lang[:1], noise=noise[:1].clone())
        steps = set(WINDOWS[LIB[name]["window"]][0])
        for k in range(10):
            seq = [(b, w) for (st, b, w, _) in tr.events if st == k]
            exp = list(LIB[name]["visits"]) if k in steps else list(ORIGINAL)
            assert seq == exp, (name, k, seq, exp)
        assert tr.calls() == block_calls(LIB[name]), (name, tr.calls())
        out[name] = tr.calls()
    return f"per-step visit sequences match the manifest; block calls {out}"


def test_second_visit_receives_h1():
    blk = base.velocity_net.decoder.layers[1]
    seen = []
    h0 = blk.register_forward_pre_hook(lambda mod, inp: seen.append(inp[0].detach().clone()))
    x = torch.randn(16, 2, 256, device=dev, generator=g)
    te = base.velocity_net.time_net(torch.rand(2, device=dev, generator=g))
    cond = [torch.randn(5, 2, 256, device=dev, generator=g) for _ in range(4)]
    run_visits(base.velocity_net.decoder.layers, x, te, cond, ((1, 0.5), (1, 0.5)))
    h0.remove()
    h1 = x + 0.5 * (blk(x, te, cond[1]) - x)
    assert torch.equal(seen[0], x) and torch.allclose(seen[1], h1, atol=1e-6)
    # cached-residual placebo: reusing the first residual for both half steps reproduces the original block map
    d = blk(x, te, cond[1]) - x
    placebo = (x + 0.5 * d) + 0.5 * d
    assert torch.allclose(placebo, blk(x, te, cond[1]), atol=1e-5), (placebo - blk(x, te, cond[1])).abs().max()
    return "visit 2 sees h1 = h + 0.5(B(h) - h); cached-residual placebo == B(h) (max err %.1e)" % \
        (placebo - blk(x, te, cond[1])).abs().max()


def test_no_duplicated_projection_or_head():
    m = model_with("b2_late")
    calls = {"ac_proj": 0, "eps_out": 0, "time_net": 0}
    hooks = [getattr(m.velocity_net, k).register_forward_hook(lambda *a, k=k: calls.__setitem__(k, calls[k] + 1))
             for k in calls]
    m.sample(imgs[:1], state[:1], lang[:1], noise=noise[:1].clone())
    for h in hooks:
        h.remove()
    assert calls == {"ac_proj": 10, "eps_out": 10, "time_net": 10}, calls
    return f"input projection / time embedding / velocity head once per evaluation: {calls}"


def test_inactive_windows_exact_and_training_grouping():
    vb, vg = base.velocity_net, model_with("b1_late").velocity_net
    cond = base.obs_tokens(imgs, state, lang)
    enc = vb.forward_enc(cond)
    z = noise.clone()
    for k in (0, 3, 4):                                 # outside the late window (steps 5-9): exact original path
        t = torch.full((3,), k / 10, device=dev)
        vg._graph_step = k
        assert torch.equal(vg.forward_dec(z, t, enc), vb.forward_dec(z, t, enc)), k
    vg._graph_step = None                               # training mode: per-row continuous FM time gating
    t = torch.tensor([0.1, 0.55, 0.95], device=dev)
    mixed = vg.forward_dec(z, t, enc)
    ref = vb.forward_dec(z, t, enc)
    assert torch.allclose(mixed[0], ref[0], atol=1e-6), "row outside the window changed"
    assert (mixed[1:] - ref[1:]).abs().max() > 1e-4, "rows inside the window unchanged"
    return "steps outside the window bit-identical to G0; training rows gated by FM time (grouped)"


def test_eval_deterministic_and_changes_actions():
    m = model_with("b2_middle")
    a1 = m.sample(imgs, state, lang, noise=noise.clone())
    a2 = m.sample(imgs, state, lang, noise=noise.clone())
    assert torch.equal(a1, a2) and torch.isfinite(a1).all()
    a0 = base.sample(imgs, state, lang, noise=noise.clone())
    diffs = {}
    for name in [n for n in LIB if n != "G0"]:
        a = model_with(name).sample(imgs, state, lang, noise=noise.clone())
        diffs[name] = round(float((a - a0)[:, :8].abs().max()), 4)
    assert all(v > 0 for v in diffs.values()), diffs
    return f"eval deterministic (no dropout); executed-prefix change vs G0 (max |rad|): {diffs}"


def test_actor_noise_isolated():
    sys.path.insert(0, "/home/licho/workspace/RoboTwin/policy")
    from HorseaFM.deploy_policy import _Model
    m = _Model.__new__(_Model)
    m.dev, m.seed, m.task, m.noise_rep, m.decision = dev, 10, "lift_pot", 0, 3
    n1 = m.noise(episode=2)
    torch.manual_seed(123)
    torch.randn(1000, device=dev)                       # advancing the global RNG (env seeding, router, warm-up)
    n2 = m.noise(episode=2)
    m.noise_rep = 1
    n3 = m.noise(episode=2)
    assert torch.equal(n1, n2) and not torch.equal(n1, n3)
    return "actor noise keyed by (seed, task, episode, decision, replicate); unaffected by the global RNG"


if __name__ == "__main__":
    for f in (test_original_parity, test_trace_counts_and_identities, test_second_visit_receives_h1,
              test_no_duplicated_projection_or_head, test_inactive_windows_exact_and_training_grouping,
              test_eval_deterministic_and_changes_actions, test_actor_noise_isolated):
        print("PASS", f.__name__, "--", f(), flush=True)
