"""Acceptance checks for the recurrent-denoising memory (spec sec. 9), offline (no simulator).

    PYTHONPATH=. python tests/test_rdm.py
"""
import numpy as np
import torch

from horsea.base import Flow, load_policy
from horsea.paths import BASE_CKPT
from horsea.rdm import model as M
from horsea.rdm.model import EXEC, RDM, VARIANTS, make_model, policy_logp
from horsea.rdm.rollout import Controller
from horsea.rdm.train import build_dataset, recompute, returns

dev = "cuda:0"
policy, _ = load_policy(BASE_CKPT, dev)
policy.eval()
flow = Flow(policy)
torch.manual_seed(0)


def open_cond(model, scale=0.05):
    """Make the zero-initialised conditioning projection non-zero (as after some training)."""
    with torch.no_grad():
        for p in model.reader.cond[-1].parameters():
            p.normal_(0, scale)


def fake_events(B, N):
    g = torch.Generator().manual_seed(1)
    r = lambda *s: torch.randn(*s, generator=g)
    return {"pre": r(B, N, 4, 256).half(), "post": r(B, N, 4, 256).half(), "act": r(B, N, EXEC, 7).clamp(-1, 1),
            "amask": torch.ones(B, N, EXEC), "p_pre": r(B, N, 5), "p_post": r(B, N, 5),
            "attempt": torch.randint(0, 5, (B, N), generator=g), "dstep": torch.randint(0, 38, (B, N), generator=g),
            "reset": torch.randint(0, 2, (B, N), generator=g)}


def test_zero_init_matches_base():
    B = 3
    encm = torch.randn(B, 4, 256, device=dev)
    eps = torch.randn(B, 16, 7, device=dev)
    ev = {k: v.to(dev) for k, v in fake_events(B, 12).items()}
    base = RDM(flow, "plain").to(dev).mean(encm, eps)
    for v in VARIANTS[1:]:
        m = make_model(flow, v).to(dev)
        if v == "ttt_info":
            st = m.init_state(B, requires_grad=True)
            for i in range(12):
                st = m.write(st, {k: x[:, i] for k, x in ev.items()}, create_graph=False)
            z = m.mean(encm, eps, st)
        else:
            z = m.mean(encm, eps, m.event_tokens(ev), torch.full((B,), 12, device=dev))
        assert torch.allclose(z, base, atol=1e-5), (v, (z - base).abs().max())
    return "zero-init == base for " + ", ".join(VARIANTS[1:])


def test_causality_and_workspace_reset():
    B, N = 2, 20
    encm = torch.randn(B, 4, 256, device=dev)
    eps = torch.randn(B, 16, 7, device=dev)
    out = []
    for v in ("readonce", "reread"):
        m = RDM(flow, v).to(dev)
        open_cond(m)
        ev = {k: x.to(dev) for k, x in fake_events(B, N).items()}
        nv = torch.full((B,), 10, device=dev)
        z1 = m.mean(encm, eps, m.event_tokens(ev), nv)
        ev2 = {k: x.clone() for k, x in ev.items()}
        ev2["post"][:, 10:] = torch.randn_like(ev2["post"][:, 10:])  # alter FUTURE events (index >= n_valid)
        ev2["act"][:, 10:] = -ev2["act"][:, 10:]
        z2 = m.mean(encm, eps, m.event_tokens(ev2), nv)
        assert torch.allclose(z1, z2, atol=1e-6), (v, "future events changed the output")
        ev3 = {k: x.clone() for k, x in ev.items()}
        ev3["post"][:, 3] = torch.randn_like(ev3["post"][:, 3])  # altering a PAST event must matter
        z3 = m.mean(encm, eps, m.event_tokens(ev3), nv)
        assert (z1 - z3).abs().max() > 1e-4, (v, "past events have no effect")
        z4 = m.mean(encm, eps, m.event_tokens(ev), nv)   # workspace does not leak across calls
        assert torch.allclose(z1, z4, atol=1e-6), (v, "workspace leaked across calls")
        out.append(v)
    return "causal prefix, past events matter, workspace reset: " + ", ".join(out)


def test_same_K():
    calls = {"n": 0}
    orig = flow.decode

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    flow.decode = counting
    try:
        B = 2
        encm, eps = torch.randn(B, 4, 256, device=dev), torch.randn(B, 16, 7, device=dev)
        ev = {k: v.to(dev) for k, v in fake_events(B, 5).items()}
        counts = {}
        for v in VARIANTS:
            calls["n"] = 0
            m = make_model(flow, v).to(dev)
            if v == "ttt_info":
                st = m.init_state(B, requires_grad=True)
                calls["n"] = 0
                m.mean(encm, eps, st)
            else:
                m.mean(encm, eps, m.event_tokens(ev), torch.full((B,), 5, device=dev))
            counts[v] = calls["n"]
    finally:
        flow.decode = orig
    assert len(set(counts.values())) == 1 and counts["plain"] == flow.n_steps, counts
    return f"denoiser evaluations per call: {counts}"


def test_logged_vs_recomputed_likelihood():
    """Run the rollout controller on synthetic observations (two attempts), then recompute every logged
    decision's likelihood with freshly recomputed tokens and the full graph: ratios must be 1."""
    out = []
    for v in ("adapter", "looped", "readonce", "reread", "ttt_info"):
        torch.manual_seed(2)
        m = make_model(flow, v).to(dev)
        if v == "ttt_info":  # open the output gate and the information adapters
            with torch.no_grad():
                m.mem.alpha.fill_(0.3)
                m.A.weight.normal_(0, 0.01)
                m.P.weight.normal_(0, 0.01)
        else:
            open_cond(m)
        ctrl = Controller(m, flow, 0.1, dev)
        B = 3
        ctrl.start(B)
        for a in range(2):
            ctrl.attempt, ctrl.first_of_attempt, ctrl.dstep = a, True, 0
            for d in range(6):
                enc = torch.randn(B, 4, 256, device=dev).half().float()  # as the rollout encoder emits
                ctrl.prop = np.random.randn(B, 5).astype(np.float32)
                if ctrl.pending is not None:
                    ctrl.finalize(enc, ctrl.prop, EXEC)
                ctrl.decide(enc)
            ctrl.finalize(torch.randn(B, 4, 256, device=dev).half().float(), np.random.randn(B, 5).astype(np.float32), EXEC)
        for b in range(B):
            ctrl.records[b][4]["reward"] = 1.0 if b == 0 else 0.0
        data, events = build_dataset([(ctrl.records, ctrl.bank.cpu_events())])
        idx = torch.arange(len(data["u"]))
        with torch.no_grad():
            mu, _ = recompute(m, data, events, idx, dev)
        lp = policy_logp(data["u"].to(dev), mu, 0.1)
        dev_ = float((lp - data["logp"].to(dev)).abs().max())
        assert dev_ < 1e-3, (v, dev_)
        # causal history size is the number of completed decisions before each decision
        assert data["n_hist"].tolist()[:7] == [0, 1, 2, 3, 4, 5, 6], data["n_hist"].tolist()[:7]
        out.append(f"{v} max|dlogp|={dev_:.1e}")
    return "; ".join(out)


def test_returns_cross_attempts():
    recs = [{"reward": 0.0} for _ in range(10)]
    recs[8]["reward"] = 1.0          # success in a later attempt
    G = returns(recs)
    assert G[0] == 1.0 and G[8] == 1.0 and G[9] == 0.0, G
    return "later-attempt reward reaches earlier-attempt decisions"


def test_grads_finite_and_theta_frozen():
    B = 4
    m = RDM(flow, "reread").to(dev)
    open_cond(m, 0.01)
    encm, eps = torch.randn(B, 4, 256, device=dev), torch.randn(B, 16, 7, device=dev)
    ev = {k: v.to(dev) for k, v in fake_events(B, 15).items()}
    z = m.mean(encm, eps, m.event_tokens(ev), torch.full((B,), 15, device=dev))
    z.pow(2).sum().backward()
    missing = [n for n, p in m.reader.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()
               or p.grad.abs().sum() == 0]
    assert not missing, missing
    assert all(not p.requires_grad for p in policy.parameters())
    assert not policy.training
    return "all reader params get finite non-zero grads; theta frozen, eval mode"


if __name__ == "__main__":
    for f in (test_zero_init_matches_base, test_causality_and_workspace_reset, test_same_K,
              test_logged_vs_recomputed_likelihood, test_returns_cross_attempts, test_grads_finite_and_theta_frozen):
        print("PASS", f.__name__, "--", f())
