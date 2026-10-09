"""One meta-training step for every arm of the Horsea-vs-TTT comparison, on fake data.

Arms (fair-comparison plan section 2):
  A1 ttt2           context + own action,        FM objective,    TTT mechanism
  A2 ttt_info       + observed consequence,      FM objective,    TTT mechanism
  A3 ttt_info_dphi  + observed consequence,      d_Phi objective, TTT mechanism
  A4 horsea         + observed consequence,      d_Phi objective, energy mechanism
  A5 horsea-mse     + observed consequence,      regression,      energy mechanism

Checks per arm: finite loss, meta-gradients reach the slow parameters, the frozen base gets none,
and an optimizer step moves the model. Also reports the fast/slow parameter counts the plan asks to
be matched (item 3), and asserts A2 sees information A1 cannot.

Random policy and random data, so losses are meaningless -- the graph and the plumbing are the point.
Needs a Phi at $HORSEA_EXP/phi/phi.pt (scripts/smoke.sh trains a throwaway one).

    python tests/test_fair_arms.py [cuda:0|cpu]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import horsea  # noqa: E402,F401
from horsea import energy as en  # noqa: E402
from horsea.base import Flow  # noqa: E402
from horsea.phi import Phi  # noqa: E402
from horsea.writer import EXEC, Meta  # noqa: E402
from test_cpu import make_policy  # noqa: E402

K_PROBE = 4


def fake_episode(n, seed, dev):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).to(dev)
    return {"encm": r(n, 4, 256), "label": r(n, 16, 7).clamp(-1, 1), "chunk": r(n, 16, 7).clamp(-1, 1),
            "probes": r(n, K_PROBE, 16, 7), "cmd": r(n, EXEC, 7).clamp(-1, 1), "dprop": r(n, EXEC, 5),
            "mask": torch.ones(n, EXEC, device=dev), "success": False, "n": n}


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    assert cond, name


def grad_sum(ps):
    return sum(p.grad.abs().sum().item() for p in ps if p.grad is not None)


def run_ttt_arm(arm, objective, policy, flow, phi, dev):
    meta = Meta(arm, dev, flow, flow, n_inner=2, lam_old=0.0, pw=6, po=4,
                objective=objective, phi=phi if objective == "dphi" else None)
    hist = [fake_episode(5, 10 + i, dev) for i in range(2)]
    fut = [fake_episode(5, 20 + i, dev) for i in range(2)]
    state = meta.adapt(hist, create_graph=True)
    loss = meta.outer(state, fut)["future"]
    for p in meta.params():
        p.grad = None
    loss.backward()
    g = grad_sum(meta.params())
    fast = meta.memory.fast_numel()
    slow = sum(p.numel() for p in meta.params())
    return loss, g, fast, slow, meta


def run_energy_arm(objective, policy, flow, phi, dev):
    en.WRITER["kind"] = "batch"
    en.EnergyMemory.nohist = False
    en.OBJECTIVE["kind"] = objective
    model = en.Model(K=10, n_iter=2, structured=False, max_step=0.05).to(dev)  # generic candidates
    hist = [fake_episode(5, 10 + i, dev) for i in range(2)]
    fut = [fake_episode(5, 20 + i, dev) for i in range(2)]
    state = en.adapt(model, phi, hist, create_graph=True)
    loss = en.outer(model, phi, flow, state, fut, "final", create_graph=True, po=4)
    model.zero_grad(set_to_none=True)
    loss.backward()
    g = grad_sum(model.parameters())
    fast = model.mem.fast_numel()
    slow = sum(p.numel() for p in model.parameters())
    return loss, g, fast, slow, model


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    policy = make_policy().to(dev)
    policy.device = dev
    flow = Flow(policy)
    phi = Phi(device=dev)
    rows = []

    for label, arm, objective in [("A1 ttt2", "ttt2", "fm"),
                                  ("A2 ttt_info", "ttt_info", "fm"),
                                  ("A3 ttt_info_dphi", "ttt_info_dphi", "dphi")]:
        torch.manual_seed(0)
        loss, g, fast, slow, meta = run_ttt_arm(arm, objective, policy, flow, phi, dev)
        check(f"{label}: finite loss", torch.isfinite(loss).item(), f"{loss.item():.4f}")
        check(f"{label}: meta-grad reaches slow params", g > 0, f"{g:.3e}")
        check(f"{label}: frozen base gets no grad", all(p.grad is None for p in policy.parameters()))
        opt = torch.optim.Adam(meta.params(), lr=1e-3)
        before = next(iter(meta.memory.W0.values())).detach().clone()
        opt.step()
        check(f"{label}: optimizer step moves W0",
              not torch.equal(before, next(iter(meta.memory.W0.values()))))
        rows.append((label, fast, slow))

    for label, objective in [("A4 horsea", "dphi"), ("A5 horsea-mse", "mse")]:
        torch.manual_seed(0)
        loss, g, fast, slow, model = run_energy_arm(objective, policy, flow, phi, dev)
        check(f"{label}: finite loss", torch.isfinite(loss).item(), f"{loss.item():.4f}")
        check(f"{label}: meta-grad reaches slow params", g > 0, f"{g:.3e}")
        check(f"{label}: frozen base gets no grad", all(p.grad is None for p in policy.parameters()))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.writer.head[-1].weight.detach().clone()
        opt.step()
        check(f"{label}: optimizer step moves the writer",
              not torch.equal(before, model.writer.head[-1].weight))
        rows.append((label, fast, slow))

    # A2 must actually USE the consequence: zeroing dprop has to change its write.
    torch.manual_seed(0)
    meta = Meta("ttt_info", dev, flow, flow, n_inner=1, lam_old=0.0, pw=6, po=4)
    with torch.no_grad():  # exp_proj starts at zero (A2 == A1 before training); make it live
        for pr in meta.memory.exp_proj:
            pr.weight.normal_(0, 0.05)
    ep = fake_episode(5, 10, dev)
    encm, cmd, dprop, mask, chunk = ep["encm"], ep["cmd"], ep["dprop"], ep["mask"], ep["chunk"]
    s0 = meta.memory.init_state(1, requires_grad=True)
    torch.manual_seed(1)
    a = meta.memory.write(flow, s0, encm, chunk, False, exp=(cmd, dprop, mask))
    s0 = meta.memory.init_state(1, requires_grad=True)
    torch.manual_seed(1)
    b = meta.memory.write(flow, s0, encm, chunk, False, exp=(cmd, torch.zeros_like(dprop), mask))
    d = max((a[k] - b[k]).abs().max().item() for k in a)
    check("A2 write depends on the observed consequence", d > 1e-7, f"max delta {d:.2e}")

    print("\nparameter counts (plan item 3: these should be matched across arms)")
    print(f"  {'arm':18} {'fast':>10} {'slow':>12}")
    for label, fast, slow in rows:
        print(f"  {label:18} {fast:>10,} {slow:>12,}")
    print("\nFAIR ARMS OK")


if __name__ == "__main__":
    main()
