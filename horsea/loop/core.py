"""Internal looping of selected DiT decoder blocks (spec 2026-09-30, sec. 5).

One shared block-execution path used by BOTH the native policy (DiTNoiseNet.forward_dec, patched on the loaded
instance) and Horsea's Flow.decode, so training and every evaluation path run the same schedule.

Normalized refinement of block l (F_l(h) = B_l(h) - h):
    h <- h + (lambda / R) * F_l(h)        R times, same block object (no parameter copies)
Raw repetition (separate ablation):      h <- B_l(h)  R times (lambda must be 1)
Cached-residual control:                 d = F_l(h_in) once; h <- h + (lambda/R) d, R times (== B_l at lambda = 1)

Per FM evaluation: action projection + positional embedding once, then layers in order, each selected layer
repeated before moving on (2,2,3,3), fixed time embedding and that layer's conditioning, output head once.
A layer loops only for examples whose FM time t is inside the configured interval; other examples take the
exact original path. Mixed batches are executed GROUPED (active rows only are recomputed), and both logical and
physical block calls are counted.
"""
import dataclasses
import json

import torch

MODES = ("normalized", "raw", "cached")


@dataclasses.dataclass
class LoopConfig:
    enabled: bool = False
    mode: str = "normalized"
    layer_ids: tuple = (1,)
    repeats: int = 2
    strength: float = 1.0
    interval: tuple = (0.0, 1.0)     # [lo, hi); hi == 1.0 is inclusive (covers the action-side end)

    def __post_init__(self):
        assert self.mode in MODES, self.mode
        assert self.repeats >= 1
        if self.mode == "raw":
            assert self.strength == 1.0, "raw repetition has no strength parameter"
        self.layer_ids = tuple(int(x) for x in self.layer_ids)
        self.interval = (float(self.interval[0]), float(self.interval[1]))

    def to_json(self):
        return json.dumps(dataclasses.asdict(self))

    def trivial(self):
        return (not self.enabled) or (self.repeats == 1 and self.strength == 1.0 and self.mode != "raw")

    def active_rows(self, t):
        lo, hi = self.interval
        return (t >= lo) & ((t <= hi) if hi >= 1.0 else (t < hi))


class CallCounter:
    """Logical block calls (the schedule's cost per example) and physical calls (rows actually processed)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.logical = 0      # sum over examples of block calls
        self.physical = 0     # sum over block invocations of rows processed
        self.invocations = 0  # number of block forward calls
        self.order = []       # layer index of each invocation (for order checks; cleared by reset)


def run_block(block, h, te, cond, repeats=1, strength=1.0, mode="normalized", counter=None, layer=None):
    """h: (C, B, D) tokens; te: (B, D) time embedding; cond: (Lc, B, D) conditioning tokens of this layer."""
    def call(x):
        if counter is not None:
            counter.invocations += 1
            counter.physical += x.shape[1]
            counter.order.append(layer)
        return block(x, te, cond)

    if repeats == 1 and strength == 1.0 and mode != "cached":
        return call(h)                                    # exact original code path
    if mode == "normalized":
        step = strength / repeats
        for _ in range(repeats):
            h = h + step * (call(h) - h)                 # no in-place updates (training graph)
        return h
    if mode == "raw":
        for _ in range(repeats):
            h = call(h)
        return h
    if mode == "cached":                                  # control: residual computed once and reused
        d = call(h) - h
        step = strength / repeats
        for _ in range(repeats):
            h = h + step * d
        return h
    raise ValueError(mode)


def decoder_forward(layers, x, te, conds, t, cfg, counter=None, layer_hook=None):
    """Run the decoder stack. x (C, B, D); te (B, D); conds[l] (Lc, B, D); t (B,) raw FM time.

    Every layer is first called once on the FULL batch (the exact original computation, so rows outside the loop
    interval are bit-identical to the unmodified policy). For a looped layer that call is also the first proposal
    of the active rows; only the active rows then run the remaining R - 1 proposals (grouped execution)."""
    B = x.shape[1]
    for l, (layer, cond) in enumerate(zip(layers, conds)):
        loop = cfg is not None and cfg.enabled and l in cfg.layer_ids and not cfg.trivial()
        p0 = run_block(layer, x, te, cond, counter=counter, layer=l)
        n_act = 0
        if loop:
            act = cfg.active_rows(t)
            n_act = int(act.sum())
        if n_act == 0:
            x = p0
        else:
            full = n_act == B
            ia = None if full else act.nonzero().squeeze(1)
            sub = (lambda v, d=1: v) if full else (lambda v, d=1: v.index_select(d, ia))
            h_in, p, te_a, c_a = sub(x), sub(p0), sub(te, 0), sub(cond)
            R, mode = cfg.repeats, cfg.mode
            if mode == "normalized":
                step = cfg.strength / R
                h = h_in + step * (p - h_in)
                for _ in range(R - 1):
                    h = h + step * (run_block(layer, h, te_a, c_a, counter=counter, layer=l) - h)
            elif mode == "raw":
                h = p
                for _ in range(R - 1):
                    h = run_block(layer, h, te_a, c_a, counter=counter, layer=l)
            else:  # cached residual control: d computed once, reused for all R normalized substeps
                d = p - h_in
                h = h_in
                for _ in range(R):
                    h = h + (cfg.strength / R) * d
            x = h if full else p0.index_copy(1, ia, h)
        if counter is not None:
            per = cfg.repeats if (loop and cfg.mode != "cached") else 1
            counter.logical += n_act * per + (B - n_act)
        if layer_hook is not None:
            x = layer_hook(l, x)
    return x


def install(vnet, cfg, counter=None):
    """Patch a DiTNoiseNet INSTANCE: forward_dec routes through decoder_forward with cfg (native path, used by
    FlowMatchingPolicy.sample_actions and compute_loss). Flow.decode reads the same vnet._loop_cfg."""
    import types

    def forward_dec(self, noise_actions, time, enc_cache):
        time_enc = self.time_net(time)
        dec_in = self.ac_proj(noise_actions).transpose(0, 1) + self.dec_pos
        dec_out = decoder_forward(self.decoder.layers, dec_in, time_enc, enc_cache, time, self._loop_cfg,
                                  self._loop_counter)
        return self.eps_out(dec_out, time_enc, enc_cache[-1])

    vnet._loop_cfg = cfg
    vnet._loop_counter = counter if counter is not None else CallCounter()
    vnet.forward_dec = types.MethodType(forward_dec, vnet)
    return vnet
