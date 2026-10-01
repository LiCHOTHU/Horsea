"""Checks for per-evaluation schedule graphs (loop-consistency study 2026-10-01; horsea.rt.graph.schedule). FP32, TF32 off.

    PYTHONPATH=. /home/licho/anaconda3/envs/robotwin/bin/python tests/test_rt_sched.py
"""
import copy

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from horsea.rt.graph import ORIGINAL, Tracer, block_calls, install, library, r2, resolve, schedule  # noqa: E402
from horsea.rt.policy import ADIM, CHUNK, RTFlowPolicy  # noqa: E402

dev = "cuda:0"
s = torch.load("experiments/rt/base/step300000.pt", map_location=dev, weights_only=False)
base = RTFlowPolicy(s["stats"]).to(dev).eval()
base.load_state_dict(s["model"])
g = torch.Generator(device=dev).manual_seed(0)
imgs = torch.rand(3, 3, 3, 120, 160, device=dev, generator=g)
state = torch.rand(3, 14, device=dev, generator=g) * (base.s_max - base.s_min) + base.s_min
lang = torch.randn(3, 512, device=dev, generator=g)
noise = torch.randn(3, CHUNK, ADIM, device=dev, generator=g)
NAMED = {"N": "s:----------", "S0": "s:0000000000", "S1": "s:1111111111", "S2": "s:2222222222", "S3": "s:3333333333"}


def model_with(graph, tracer=None):
    m = copy.deepcopy(base)
    install(m.velocity_net, graph if isinstance(graph, dict) or graph is None else resolve(graph), tracer)
    return m


def test_N_parity():
    ref = base.sample(imgs, state, lang, noise=noise.clone())
    out = model_with(NAMED["N"]).sample(imgs, state, lang, noise=noise.clone())
    assert torch.equal(ref, out), (ref - out).abs().max()
    return "N (s:----------) through the executor == native sampler, bit-identical"


def test_traces_and_block_calls():
    out = {}
    for name, sch in list(NAMED.items()) + [("T1-like", "s:0000011111"), ("Tfree-like", "s:3210-01233"),
                                            ("U5-like", "s:1-1-1-1-1-")]:
        tr = Tracer()
        model_with(sch, tr).sample(imgs[:1], state[:1], lang[:1], noise=noise[:1].clone())
        for k in range(10):
            seq = [(b, w) for (st, b, w, _) in tr.events if st == k]
            c = sch[2 + k]
            assert seq == list(ORIGINAL if c == "-" else r2(int(c))), (name, k, seq)
        assert tr.calls() == block_calls(resolve(sch)), (name, tr.calls())
        out[name] = tr.calls()
    assert out["N"] == 40 and all(out[f"S{l}"] == 50 for l in range(4)) and out["U5-like"] == 45, out
    return f"per-evaluation visit sequences follow the schedule; block calls per chunk {out}"


def test_schedule_equals_window_graph():
    # the validated window graphs are special schedules: early = evaluations 0-4, middle = 2-6, late = 5-9
    lib = library("primary")
    eq = {"b1_early": "s:11111-----", "b2_middle": "s:--22222---", "b0_late": "s:-----00000"}
    cond = base.obs_tokens(imgs, state, lang)
    enc = base.velocity_net.forward_enc(cond)
    t = torch.rand(64, device=dev, generator=g)
    z = torch.randn(64, CHUNK, ADIM, device=dev, generator=g)
    encb = [e.repeat(1, 22, 1)[:, :64] for e in enc]
    worst = 0.0
    for wname, sch in eq.items():
        a = model_with(lib[wname]).sample(imgs, state, lang, noise=noise.clone())
        b = model_with(sch).sample(imgs, state, lang, noise=noise.clone())
        assert torch.equal(a, b), (wname, (a - b).abs().max())
        va, vb = model_with(lib[wname]).velocity_net, model_with(sch).velocity_net
        da, db = va.forward_dec(z, t, encb), vb.forward_dec(z, t, encb)     # training mode: per-row FM-time gating
        worst = max(worst, float((da - db).abs().max()))
        assert torch.allclose(da, db, atol=1e-6), (wname, (da - db).abs().max())
    return f"schedules reproduce the window graphs: sampling bit-identical; training-mode rows max diff {worst:.1e}"


def test_training_rows_follow_their_interval():
    sch = "s:0123-32100"
    vs = model_with(sch).velocity_net
    cond = base.obs_tokens(imgs, state, lang)
    enc = base.velocity_net.forward_enc(cond)
    t = torch.tensor([0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.999], device=dev)
    n = len(t)
    z = torch.randn(n, CHUNK, ADIM, device=dev, generator=g)
    encn = [e[:, :1].expand(-1, n, -1).contiguous() for e in enc]
    mixed = vs.forward_dec(z, t, encn)
    for i in range(n):
        k = min(int(float(t[i]) * 10), 9)
        c = sch[2 + k]
        single = model_with("s:" + c * 10).velocity_net.forward_dec(z[i:i + 1], t[i:i + 1], [e[:, i:i + 1] for e in encn])
        assert torch.allclose(mixed[i], single[0], atol=1e-5), (i, k, (mixed[i] - single[0]).abs().max())
    return "each training row runs the entry of its own FM-time interval (12 rows over all 10 intervals)"


def test_backprop_through_both_calls():
    m = model_with(NAMED["S2"])
    m.eval()
    outs = []
    blk = m.velocity_net.decoder.layers[2]

    def keep(mod, inp, out):                    # returns None: the module output is not replaced
        out.retain_grad()
        outs.append(out)
    h = blk.register_forward_hook(keep)
    cond = m.obs_tokens(imgs, state, lang)
    t = torch.tensor([0.1, 0.5, 0.9], device=dev)
    _, v = m.velocity_net(noise.clone(), t, cond)
    v.pow(2).mean().backward()
    h.remove()
    assert len(outs) == 2 and all(o.grad is not None and o.grad.abs().sum() > 0 for o in outs), len(outs)
    gS = blk.linear1.weight.grad.clone()
    mN = model_with(NAMED["N"])
    _, vN = mN.velocity_net(noise.clone(), t, mN.obs_tokens(imgs, state, lang))
    vN.pow(2).mean().backward()
    assert not torch.allclose(gS, mN.velocity_net.decoder.layers[2].linear1.weight.grad)
    return "repeated block: both calls on the autograd path with nonzero output gradients; shared weight grad differs from N"


def test_rowcode_override():
    vs = model_with(NAMED["S1"]).velocity_net
    cond = base.obs_tokens(imgs, state, lang)
    enc = base.velocity_net.forward_enc(cond)
    t = torch.tensor([0.1, 0.5, 0.9], device=dev)
    vs._rowcode = torch.tensor([-1, 1, -1], device=dev)
    mixed = vs.forward_dec(noise, t, enc)
    vs._rowcode = None
    ref_n = base.velocity_net.forward_dec(noise, t, enc)
    full = vs.forward_dec(noise, t, enc)
    assert torch.allclose(mixed[[0, 2]], ref_n[[0, 2]], atol=1e-6) and torch.allclose(mixed[1], full[1], atol=1e-6)
    return "explicit per-row codes (random-subset rule) override the interval gate"


def test_resolve_and_deploy_resolution():
    assert resolve("b1_middle") == library("all")["b1_middle"] and resolve("G0") == library("all")["G0"]
    assert resolve("s:1111111111")["sched"] == (1,) * 10 and resolve("s:----------")["sched"] == (None,) * 10
    for bad in ("s:111", "s:11111111x1"):
        try:
            schedule(bad)
            raise RuntimeError(bad)
        except AssertionError:
            pass
    import ast
    src = open("/home/licho/workspace/RoboTwin/policy/HorseaFM/deploy_policy.py").read()
    assert "resolve(graph)" in src and "library(\"all\")[graph]" not in src
    ast.parse(src)
    return "resolve: library names and s:<10 chars>; malformed schedules rejected; deploy resolves graphs by name"


if __name__ == "__main__":
    for f in (test_N_parity, test_traces_and_block_calls, test_schedule_equals_window_graph,
              test_training_rows_follow_their_interval, test_backprop_through_both_calls, test_rowcode_override,
              test_resolve_and_deploy_resolution):
        print("PASS", f.__name__, "--", f(), flush=True)
