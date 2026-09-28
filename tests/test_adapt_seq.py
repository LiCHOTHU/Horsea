"""Regression test for energy.adapt_seq (audit P2): with ragged histories, each episode's memory must be
the same as if its history had been written alone (no truncation to the shortest history, and no
decay/write for episodes whose history has ended).

Needs a GPU and the trained action encoder (experiments/phi/phi.pt).
    python tests/test_adapt_seq.py
"""
import torch

from horsea import energy as en
from horsea.phi import Phi


def fake_history(n, seed, dev):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).to(dev)
    return {"encm": r(n, 4, 256), "cmd": r(n, 8, 7).clamp(-1, 1), "dprop": r(n, 8, 5), "mask": torch.ones(n, 8, device=dev),
            "chunk": r(n, 16, 7).clamp(-1, 1), "n": n}


def check(kind, dev):
    en.WRITER["kind"] = kind
    torch.manual_seed(0)
    model = en.Model(structured=True).to(dev)
    with torch.no_grad():  # make the gate non-trivial
        model.gate.weight.normal_(0, 0.5)
    phi = Phi(device=dev)
    ha, hb = fake_history(3, 1, dev), fake_history(7, 2, dev)
    both = en.adapt(model, phi, [ha, hb], create_graph=False)
    counts = list(en.WRITE_COUNT["last"])
    alone_b = en.adapt(model, phi, [hb], create_graph=False)
    alone_a = en.adapt(model, phi, [ha], create_graph=False)
    for k in both:
        assert torch.allclose(both[k][1], alone_b[k][0], atol=1e-5), (kind, k, "long history differs when batched")
        assert torch.allclose(both[k][0], alone_a[k][0], atol=1e-5), (kind, k, "short history differs when batched")
    assert counts == [1, 2], counts  # chunk 4: 3 rows -> 1 write, 7 rows -> 2 writes
    return counts


if __name__ == "__main__":
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    for kind in ("seq", "gated"):
        print(kind, "writes per episode", check(kind, dev), "ok")
