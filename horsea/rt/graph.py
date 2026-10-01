"""Execution graphs over the decoder blocks of the RoboTwin FM policy (spec 2026-10-01, sec. 5-6, 14).

A graph assigns to every denoising evaluation a sequence of block VISITS (block index, residual weight):
    visit (b, w):  h <- B_b(h)                  if w == 1   (exact original call)
                   h <- h + w * (B_b(h) - h)    otherwise   (normalized whole-block residual, recomputed)
Original graph G0: [(0,1), (1,1), (2,1), (3,1)] at every step. A loop graph applies its alternative visit list inside a
window of solver steps (inference) / FM-time interval (training) and G0 elsewhere. Input projection, positional
embedding and the velocity head run once per evaluation; time embedding and each block's own conditioning are shared
by all visits of that block.

Windows (K = 10 Euler evaluations, t_k = k/10): early = steps 0-4 = t in [0, 0.5); middle = steps 2-6 = [0.2, 0.7);
late = steps 5-9 = [0.5, 1]. Inference gates by the exact solver index (no floating-point boundary issues);
training gates by the continuous FM time of each example (grouped execution, rows outside the window are exact).
"""
import json

import torch

L = 4
K = 10
WINDOWS = {"early": (range(0, 5), (0.0, 0.5)), "middle": (range(2, 7), (0.2, 0.7)), "late": (range(5, 10), (0.5, 1.0))}
ORIGINAL = tuple((b, 1.0) for b in range(L))


def r2(block):
    """Block `block` visited twice with half of its whole-block residual each time; other blocks unchanged."""
    out = []
    for b in range(L):
        out += [(b, 0.5), (b, 0.5)] if b == block else [(b, 1.0)]
    return tuple(out)


def secondary(kind):
    if kind == "pair_grouped":       # 0, 1, 1, 2, 2, 3
        return ((0, 1.0), (1, 0.5), (1, 0.5), (2, 0.5), (2, 0.5), (3, 1.0))
    if kind == "pair_alternating":   # 0, 1, 2, 1, 2, 3
        return ((0, 1.0), (1, 0.5), (2, 0.5), (1, 0.5), (2, 0.5), (3, 1.0))
    if kind == "depth3":             # 0, 1, 1, 1, 2, 3
        return ((0, 1.0), (1, 1 / 3), (1, 1 / 3), (1, 1 / 3), (2, 1.0), (3, 1.0))
    raise ValueError(kind)


def library(family="primary"):
    """name -> {"window": name or None, "visits": tuple}"""
    lib = {"G0": {"window": None, "visits": ORIGINAL}}
    if family in ("primary", "all"):
        for b in range(L):
            for w in WINDOWS:
                lib[f"b{b}_{w}"] = {"window": w, "visits": r2(b)}
    if family in ("secondary", "all"):
        for kind in ("pair_grouped", "pair_alternating", "depth3"):
            for w in WINDOWS:
                lib[f"{kind}_{w}"] = {"window": w, "visits": secondary(kind)}
    return lib


SCHED_CHARS = "-0123"


def schedule(s):
    """Per-evaluation execution graph (loop-consistency study, 2026-10-01). `s` has one character per outer FM evaluation
    k = 0..K-1 (solver time t_k = k/K; in training, rows with FM time in [k/K, (k+1)/K)): '-' runs the original
    0->1->2->3, a digit l runs r2(l) = block l visited twice with half of its whole-block residual each time."""
    s = s[2:] if s.startswith("s:") else s
    assert len(s) == K and set(s) <= set(SCHED_CHARS), s
    return {"window": "sched", "sched": tuple(None if c == "-" else int(c) for c in s), "visits": None, "name": "s:" + s}


def resolve(name):
    """Graph by name: a library('all') entry (window graphs, G0) or a schedule 's:<K chars>'."""
    return schedule(name) if name.startswith("s:") else library("all")[name]


def block_calls(graph, K_=K):
    """Decoder block visits per generated chunk."""
    if graph.get("sched") is not None:
        return sum(L if b is None else len(r2(b)) for b in graph["sched"])
    act = set(WINDOWS[graph["window"]][0]) if graph["window"] else set()
    return sum(len(graph["visits"]) if k in act else L for k in range(K_))


def manifest():
    out = {"L": L, "K": K, "t_grid": [k / K for k in range(K)], "windows": {w: {"steps": list(s), "t_interval": list(i)}
                                                                            for w, (s, i) in WINDOWS.items()}, "graphs": {}}
    for fam in ("primary", "secondary"):
        for n, g in library(fam).items():
            out["graphs"][n] = {"family": "original" if n == "G0" else fam, "window": g["window"],
                                "visits_in_window": [list(v) for v in g["visits"]], "block_calls_per_chunk": block_calls(g)}
    return out


class Tracer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.events = []   # (step or None, block, weight, rows)

    def calls(self):
        return sum(e[3] for e in self.events)


def _visit(block, idx, h, te, cond, w, tracer, step):
    if tracer is not None:
        tracer.events.append((step, idx, w, h.shape[1]))
    out = block(h, te, cond)
    if w == 1.0:
        return out
    return h + w * (out - h)


def run_visits(layers, x, te, conds, visits, tracer=None, step=None):
    for b, w in visits:
        x = _visit(layers[b], b, x, te, conds[b], w, tracer, step)
    return x


def decoder_forward(vnet, x, te, enc, t, graph, step=None, tracer=None):
    """x (C, B, D) projected tokens; te (B, D); enc list of per-layer conditioning (Lc, B, D); t (B,).
    step: solver index (inference) or None (training: gate each row by its continuous FM time)."""
    layers = vnet.decoder.layers
    if graph is not None and graph.get("sched") is not None:
        return sched_forward(layers, x, te, enc, t, graph["sched"], step, tracer, getattr(vnet, "_rowcode", None))
    if graph is None or graph["window"] is None:
        return run_visits(layers, x, te, enc, ORIGINAL, tracer, step)
    steps, (lo, hi) = WINDOWS[graph["window"]]
    if step is not None:
        return run_visits(layers, x, te, enc, graph["visits"] if step in steps else ORIGINAL, tracer, step)
    act = (t >= lo) & ((t <= hi) if hi >= 1.0 else (t < hi))
    n = int(act.sum())
    if n == 0:
        return run_visits(layers, x, te, enc, ORIGINAL, tracer)
    if n == x.shape[1]:
        return run_visits(layers, x, te, enc, graph["visits"], tracer)
    ia, ib = act.nonzero().squeeze(1), (~act).nonzero().squeeze(1)
    sub = lambda v, i, d=1: v.index_select(d, i)
    xa = run_visits(layers, sub(x, ia), te.index_select(0, ia), [sub(c, ia) for c in enc], graph["visits"], tracer)
    xb = run_visits(layers, sub(x, ib), te.index_select(0, ib), [sub(c, ib) for c in enc], ORIGINAL, tracer)
    return torch.empty_like(x).index_copy(1, ia, xa).index_copy(1, ib, xb)


def visits_of(b):
    return ORIGINAL if b is None or b < 0 else r2(b)


def row_codes(sched, t):
    """Training gate: the schedule entry of each row's FM-time interval k = floor(K t) (-1 = no repeat)."""
    k = torch.clamp((t * K).floor().long(), 0, K - 1)
    return torch.tensor([-1 if b is None else b for b in sched], device=t.device)[k]


def sched_forward(layers, x, te, enc, t, sched, step, tracer, rowcode=None):
    """Inference (step given): the solver index picks the entry. Training (step None): each row runs the entry of its own
    FM-time interval (or an explicit per-row code, e.g. the random-subset rule); rows are grouped by code."""
    if step is not None:
        return run_visits(layers, x, te, enc, visits_of(sched[step]), tracer, step)
    code = row_codes(sched, t) if rowcode is None else rowcode
    groups = code.unique().tolist()
    if len(groups) == 1:
        return run_visits(layers, x, te, enc, visits_of(groups[0]), tracer)
    out = torch.empty_like(x)
    for c in groups:
        i = (code == c).nonzero().squeeze(1)
        xi = run_visits(layers, x.index_select(1, i), te.index_select(0, i), [e.index_select(1, i) for e in enc],
                        visits_of(c), tracer)
        out = out.index_copy(1, i, xi)
    return out


def install(vnet, graph=None, tracer=None):
    """Route the native DiTNoiseNet.forward_dec of this INSTANCE (used by RTFlowPolicy.loss and .sample) through the
    graph executor. vnet._graph_step is set by the sampler to the current solver index (None during training)."""
    import types

    def forward_dec(self, noise_actions, time, enc_cache):
        te = self.time_net(time)
        x = self.ac_proj(noise_actions).transpose(0, 1) + self.dec_pos
        x = decoder_forward(self, x, te, enc_cache, time, self._graph, self._graph_step, self._tracer)
        return self.eps_out(x, te, enc_cache[-1])

    vnet._graph, vnet._graph_step, vnet._tracer, vnet._rowcode = graph, None, tracer, None
    vnet.forward_dec = types.MethodType(forward_dec, vnet)
    return vnet


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=1))
