"""Stage-0 checks for internal looping (spec 2026-09-30, sec. 7). FP32.

    PYTHONPATH=. python tests/test_loop.py
"""
import types

import numpy as np
import torch

from horsea.base import Flow, load_policy
from horsea.loop.core import CallCounter, LoopConfig, install
from horsea.paths import BASE_CKPT

dev = "cuda:0"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
policy, sd = load_policy(BASE_CKPT, dev)   # load_state_dict is strict by default: no missing/unexpected keys
policy.eval()
vnet = policy.velocity_net
ORIG_FWD_DEC = type(vnet).forward_dec      # the unpatched class method
L = len(vnet.decoder.layers)
counter = CallCounter()
install(vnet, LoopConfig(enabled=False), counter)
flow = Flow(policy)
g = torch.Generator(device=dev).manual_seed(0)


def rand_inputs(B=6, t=None):
    cond = torch.randn(B, 3, 256, device=dev, generator=g)     # obs encoder tokens as (B, T, D) before transpose
    enc_cache = vnet.forward_enc(cond)
    z = torch.randn(B, 16, 7, device=dev, generator=g)
    t = torch.rand(B, device=dev, generator=g) if t is None else t
    return z, t, enc_cache


def set_cfg(**kw):
    vnet._loop_cfg = LoopConfig(**kw)


def old_flow_decode(z, t, encm):
    te = vnet.time_net(t)
    x = vnet.ac_proj(z).transpose(0, 1) + vnet.dec_pos
    for l, layer in enumerate(vnet.decoder.layers):
        x = layer(x, te, encm[:, l][None])
    return vnet.eps_out(x, te, encm[:, -1][None])


def test_architecture_audit():
    blk = vnet.decoder.layers[0]
    info = {"decoder_blocks": L, "width": blk.linear1.in_features, "heads": blk.self_attn.num_heads,
            "mlp": blk.linear1.out_features, "K": policy.num_inference_steps, "sig_min": policy.flow_sig_min,
            "chunk": policy.chunk_size, "temporal_agg": policy.temporal_agg, "action_horizon": policy.action_horizon}
    assert info["decoder_blocks"] == 4 and info["width"] == 256 and info["heads"] == 4 and info["mlp"] == 512, info
    return info


def test_R1_parity_native_and_wrapper():
    z, t, enc = rand_inputs()
    ref = ORIG_FWD_DEC(vnet, z, t, enc)
    out = {}
    for name, kw in {"disabled": dict(enabled=False), "R1": dict(enabled=True, repeats=1, strength=1.0),
                     "R1_other_layers": dict(enabled=True, repeats=1, layer_ids=(0, 1, 2, 3))}.items():
        set_cfg(**kw)
        v = vnet.forward_dec(z, t, enc)
        assert torch.equal(v, ref), (name, (v - ref).abs().max())
        out[name] = "exact"
    encm = torch.stack([e.mean(0) for e in enc], 1)
    set_cfg(enabled=False)
    assert torch.equal(flow.decode(z, t, encm), old_flow_decode(z, t, encm)), "Horsea wrapper changed at R=1"
    # sampled chunks with identical initial noise through the real policy sampler
    obs_cond = torch.randn(4, 3, 256, device=dev, generator=g)
    a = []
    for patched in (False, True):
        torch.manual_seed(7)
        enc_c = vnet.forward_enc(obs_cond)
        zz = torch.randn(4, 16, 7, device=dev)
        tt = torch.zeros(4, device=dev)
        for _ in range(policy.num_inference_steps):
            v = vnet.forward_dec(zz, tt, enc_c) if patched else ORIG_FWD_DEC(vnet, zz, tt, enc_c)
            zz = zz + v / policy.num_inference_steps
            tt = tt + 1.0 / policy.num_inference_steps
        a.append(zz)
    assert torch.equal(a[0], a[1])
    return "native forward_dec, Flow.decode wrapper and sampled chunks bit-identical at R=1 / disabled"


def test_call_counts_and_order():
    z, t, enc = rand_inputs(B=5, t=torch.full((5,), 0.5, device=dev))
    res = {}
    for layers, exp_order in (((1,), [0, 1, 1, 2, 3]), ((2, 3), [0, 1, 2, 2, 3, 3])):
        set_cfg(enabled=True, layer_ids=layers, repeats=2)
        counter.reset()
        vnet.forward_dec(z, t, enc)
        assert counter.order == exp_order, (layers, counter.order)
        assert counter.logical == 5 * len(exp_order) and counter.physical == 5 * len(exp_order), vars(counter)
        res[str(layers)] = counter.order
    # one full K=10 generation with layer 1 looped at all t: 50 block calls vs 40
    set_cfg(enabled=True, layer_ids=(1,), repeats=2)
    counter.reset()
    zz = z.clone()
    tt = torch.zeros(5, device=dev)
    for _ in range(10):
        zz = zz + vnet.forward_dec(zz, tt, enc) / 10
        tt = tt + 0.1
    assert counter.logical == 5 * 50, counter.logical
    return f"orders {res}; K=10 generation: {counter.logical // 5} block calls per chunk"


def test_cached_control_equals_block():
    z, t, enc = rand_inputs()
    set_cfg(enabled=False)
    ref = vnet.forward_dec(z, t, enc)
    set_cfg(enabled=True, mode="cached", layer_ids=(1,), repeats=2, strength=1.0)
    v = vnet.forward_dec(z, t, enc)
    assert torch.allclose(v, ref, atol=1e-5), (v - ref).abs().max()
    set_cfg(enabled=True, mode="normalized", layer_ids=(1,), repeats=2)
    vn = vnet.forward_dec(z, t, enc)
    assert (vn - ref).abs().max() > 1e-4, "normalized loop identical to the block: recomputation had no effect"
    return f"cached-residual control == ordinary block (max {(v - ref).abs().max():.1e}); normalized loop differs " \
           f"(max {(vn - ref).abs().max():.3f})"


def test_inactive_interval_and_mixed_batch():
    t = torch.tensor([0.1, 0.5, 0.8, 0.95, 0.2, 0.7], device=dev)
    z, _, enc = rand_inputs(B=6, t=t)
    set_cfg(enabled=False)
    ref = vnet.forward_dec(z, t, enc)
    set_cfg(enabled=True, layer_ids=(1,), repeats=2, interval=(2 / 3, 1.0))
    counter.reset()
    v = vnet.forward_dec(z, t, enc)
    inact = t < 2 / 3
    assert torch.equal(v[inact], ref[inact]), "inactive rows not on the exact original path"
    # active rows equal a batch where every row is active (grouping does not change the math)
    set_cfg(enabled=True, layer_ids=(1,), repeats=2, interval=(0.0, 1.0))
    full = vnet.forward_dec(z, t, enc)
    assert torch.allclose(v[~inact], full[~inact], atol=1e-5), (v[~inact] - full[~inact]).abs().max()
    n_act = int((~inact).sum())
    return f"inactive rows exact; grouped active rows == full loop; logical calls {counter.logical} " \
           f"(= 6*4 + {n_act})"


def test_gradients_through_repeats_shared_params():
    policy.train()
    for m in policy.modules():  # dropout off (continuation arms disable decoder dropout)
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    n_params = sum(p.numel() for p in vnet.parameters())
    blk = vnet.decoder.layers[1]
    for p in vnet.parameters():
        p.requires_grad_(True)
    z, t, enc = rand_inputs()
    z.requires_grad_(True)
    grads = {}
    for R in (1, 2):
        set_cfg(enabled=True, layer_ids=(1,), repeats=R)
        vnet.zero_grad(set_to_none=True)
        z.grad = None
        vnet.forward_dec(z, t, [e.detach() for e in enc]).pow(2).mean().backward()
        grads[R] = blk.linear1.weight.grad.clone()
        assert torch.isfinite(grads[R]).all() and grads[R].abs().sum() > 0
        assert z.grad is not None and z.grad.abs().sum() > 0
    assert sum(p.numel() for p in vnet.parameters()) == n_params, "looping created parameter copies"
    assert not torch.allclose(grads[1], grads[2]), "R=2 gradient identical to R=1"
    # the loop's hidden state is not detached: gradient through the FIRST repeat reaches the block via the second
    set_cfg(enabled=True, layer_ids=(1,), repeats=2)
    h = torch.randn(16, 3, 256, device=dev, requires_grad=True)
    te = vnet.time_net(torch.rand(3, device=dev))
    cond = torch.randn(1, 3, 256, device=dev)
    from horsea.loop.core import run_block
    out = run_block(blk, h, te, cond, repeats=2, strength=1.0)
    jac_via_loop = torch.autograd.grad(out.sum(), h)[0]
    one_half_step = torch.autograd.grad((h + 0.5 * (blk(h, te, cond) - h)).sum(), h)[0]
    assert not torch.allclose(jac_via_loop, one_half_step), "second repeat contributes no gradient to the input"
    policy.eval()
    for p in vnet.parameters():
        p.requires_grad_(False)
    return "finite grads through both repeats accumulate in the single shared block; no parameter copies"


def test_frozen_params_unchanged_after_step():
    from horsea.loop.train import trainable_params
    policy.train()
    names, params = trainable_params(policy)
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=1e-3)
    set_cfg(enabled=True, layer_ids=(1,), repeats=2)
    z, t, enc = rand_inputs()
    vnet.forward_dec(z, t, [e.detach() for e in enc]).pow(2).mean().backward()
    opt.step()
    changed = {n for n, p in policy.named_parameters() if not torch.equal(p, before[n])}
    train_set = set(names)
    assert changed <= train_set, sorted(changed - train_set)[:5]
    assert not any(n.startswith("velocity_net.encoder") or "encoder." in n.split("velocity_net.")[0] for n in changed)
    assert "velocity_net.time_net.w" not in changed and "velocity_net.time_net.w" not in train_set
    for n, p in policy.named_parameters():  # restore
        p.data.copy_(before[n])
        p.requires_grad_(False)
    policy.eval()
    return f"{len(changed)} of {len(train_set)} trainable tensors changed; encoders and time_net.w untouched"


def test_eval_deterministic_finite_and_reaches_commands():
    policy.eval()
    set_cfg(enabled=True, layer_ids=(1,), repeats=2)
    z, t, enc = rand_inputs()
    a, b = vnet.forward_dec(z, t, enc), vnet.forward_dec(z, t, enc)
    assert torch.equal(a, b) and torch.isfinite(a).all(), "eval not deterministic (dropout?) or non-finite"
    # executed commands: normalized chunk -> unnormalize -> postprocess -> final_postprocess (the env action path)
    obs_cond = torch.randn(3, 3, 256, device=dev, generator=g)
    enc_c = vnet.forward_enc(obs_cond)
    outs = []
    for cfg in (dict(enabled=False), dict(enabled=True, layer_ids=(1,), repeats=2)):
        set_cfg(**cfg)
        torch.manual_seed(3)
        zz = torch.randn(3, 16, 7, device=dev)
        tt = torch.zeros(3, device=dev)
        for _ in range(10):
            zz = zz + vnet.forward_dec(zz, tt, enc_c) / 10
            tt = tt + 0.1
        act = torch.clamp(zz, -1, 1)
        un = policy.normalizer.unnormalize({"actions": act})["actions"]
        fin = policy.final_postprocess_actions(un[:, 0])
        outs.append(fin)
    diff = (outs[0] - outs[1]).abs().max()
    assert diff > 1e-4, "loop does not reach the executed command"
    set_cfg(enabled=False)
    return f"eval deterministic and finite; executed command changes by up to {diff:.3f}"


if __name__ == "__main__":
    for f in (test_architecture_audit, test_R1_parity_native_and_wrapper, test_call_counts_and_order,
              test_cached_control_equals_block, test_inactive_interval_and_mixed_batch,
              test_gradients_through_repeats_shared_params, test_frozen_params_unchanged_after_step,
              test_eval_deterministic_finite_and_reaches_commands):
        print("PASS", f.__name__, "--", f())
