"""One full meta-training step of energy Horsea on fake data: adapt -> solve -> outer loss -> backward.

This is the part of the current method the other tests miss. It exercises the double backward
through the frozen decoder (the reason Solver.solve forces SDPBackend.MATH: the fused attention
kernels have no double-backward), and checks that meta-gradients actually reach the writer, the
memory's W0, the inner learning rates and the solver's own lambda/beta.

Random policy and random data, so the loss value is meaningless -- the graph is what is tested.
Needs a Phi checkpoint at $HORSEA_EXP/phi/phi.pt (scripts/smoke.sh trains a throwaway one).

    python tests/test_energy_step.py [cuda:0|cpu]
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
from test_cpu import make_policy  # noqa: E402

K_PROBE = 4  # horsea.selfplay.K_PROBE has 4 entries
EXEC = 8


def fake_episode(n, seed, dev):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).to(dev)
    return {"encm": r(n, 4, 256), "label": r(n, 16, 7).clamp(-1, 1), "chunk": r(n, 16, 7).clamp(-1, 1),
            "probes": r(n, K_PROBE, 16, 7), "cmd": r(n, EXEC, 7).clamp(-1, 1), "dprop": r(n, EXEC, 5),
            "mask": torch.ones(n, EXEC, device=dev), "success": False, "n": n}


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    assert cond, name


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    policy = make_policy().to(dev)
    policy.device = dev
    flow = Flow(policy)
    phi = Phi(device=dev)

    for solve in ("final", "joint"):
        en.WRITER["kind"] = "batch"
        en.EnergyMemory.nohist = False
        model = en.Model(K=10, n_iter=2, structured=True, max_step=0.05).to(dev)
        E = 2
        hist = [fake_episode(5, 10 + i, dev) for i in range(E)]
        fut = [fake_episode(5, 20 + i, dev) for i in range(E)]

        state = en.adapt(model, phi, hist, create_graph=True)
        loss = en.outer(model, phi, flow, state, fut, solve, create_graph=True, po=4)
        check(f"{solve}: finite outer loss", torch.isfinite(loss).item(), f"{loss.item():.4f}")

        model.zero_grad(set_to_none=True)
        loss.backward()
        g = lambda ps: sum(p.grad.abs().sum().item() for p in ps if p.grad is not None)
        check(f"{solve}: grad reaches the writer", g(model.writer.parameters()) > 0,
              f"{g(model.writer.parameters()):.3e}")
        check(f"{solve}: grad reaches memory W0", g(model.mem.W0.values()) > 0,
              f"{g(model.mem.W0.values()):.3e}")
        check(f"{solve}: grad reaches inner lr (2nd order)", g(model.mem.log_lr.values()) > 0,
              f"{g(model.mem.log_lr.values()):.3e}")
        check(f"{solve}: grad reaches solver lambda/beta",
              g([model.solver.log_lam, model.solver.log_beta]) > 0,
              f"{g([model.solver.log_lam, model.solver.log_beta]):.3e}")

        # the frozen base must stay frozen
        check(f"{solve}: base policy received no grad",
              all(p.grad is None for p in policy.parameters()))

        # an optimizer step must be finite and must move the parameters
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.writer.head[-1].weight.detach().clone()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        check(f"{solve}: finite grad norm", torch.isfinite(gn).item(), f"{gn:.3e}")
        check(f"{solve}: optimizer step changes the writer",
              not torch.equal(before, model.writer.head[-1].weight))

    # the empty-memory path must reproduce the base policy exactly (reset == base)
    model = en.Model(K=10, n_iter=2, structured=True, max_step=0.05).to(dev)
    c = torch.randn(3, 4, 256, device=dev)
    eps = torch.randn(3, 16, 7, device=dev)
    with torch.no_grad():
        a_none = model.solver.solve(flow, None, c, eps, "none", False)
        a_base = flow.sample(c, eps)
    check("solver 'none' == base policy", torch.allclose(a_none, a_base, atol=1e-6),
          f"{(a_none - a_base).abs().max():.2e}")
    print("\nENERGY META-STEP OK")


if __name__ == "__main__":
    main()
