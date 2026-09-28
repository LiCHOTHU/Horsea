"""CPU invariants with a randomly initialised fm_policy_S (no GPU, no trained checkpoint).

    python tests/test_cpu.py
"""
import copy
import os
import sys

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import horsea  # noqa: E402,F401
from horsea.base import Flow, decoder_parameters  # noqa: E402
from horsea.memory import build_memory  # noqa: E402

torch.manual_seed(0)
CFG = os.path.join(os.path.dirname(__file__), "fixtures", "fm_policy_S_train80_config.yaml")


def make_policy():
    cfg = OmegaConf.load(CFG)
    pcfg = OmegaConf.to_container(cfg.algo.policy, resolve=True)
    pcfg["device"] = "cpu"
    pcfg["encoder"]["image_encoder_factory"]["pretrained"] = False  # no download in tests
    p = instantiate(pcfg)
    # the base's final layer is zero-initialised; randomise so velocities are non-trivial
    for prm in p.velocity_net.eps_out.parameters():
        torch.nn.init.normal_(prm, std=0.05)
    stats = {"actions": {"min": -torch.ones(7).numpy(), "max": torch.ones(7).numpy()},
             "robot0_eef_pos": {"min": -torch.ones(3).numpy(), "max": torch.ones(3).numpy()},
             "robot0_gripper_qpos": {"min": -torch.ones(2).numpy(), "max": torch.ones(2).numpy()}}
    p.normalizer.fit(stats)
    p.eval()
    for prm in p.parameters():
        prm.requires_grad_(False)
    return p


def fake_batch(B):
    obs = {"agentview_image": torch.rand(B, 1, 3, 128, 128) * 255,
           "robot0_eye_in_hand_image": torch.rand(B, 1, 3, 128, 128) * 255,
           "robot0_eef_pos": torch.randn(B, 1, 3), "robot0_gripper_qpos": torch.randn(B, 1, 2)}
    return {"obs": obs, "task_emb": torch.randn(B, 512)}


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    assert cond, name


def main():
    p = make_policy()
    flow = Flow(p)
    B = 6
    data = fake_batch(B)

    # 1. encoder-summary shortcut == original decoder
    d2 = copy.deepcopy(data)
    d2 = p.preprocess_input(d2, train_mode=False)
    cond = p.get_cond(d2)
    enc_cache = p.velocity_net.forward_enc(cond)
    z = torch.randn(B, 16, 7)
    t = torch.rand(B)
    v_ref = p.velocity_net.forward_dec(z, t, enc_cache)
    encm = flow.encode(copy.deepcopy(data))
    v_new = flow.decode(z, t, encm)
    err = (v_ref - v_new).abs().max().item()
    check("encoder-summary decoder == forward_dec", err < 1e-5, f"max abs err {err:.2e}")

    # 2. base sampler equivalence (same noise)
    torch.manual_seed(1)
    ref = p.sample_actions(copy.deepcopy(data))
    torch.manual_seed(1)
    new = flow.sample(flow.encode(copy.deepcopy(data))).numpy()
    check("Flow.sample == policy.sample_actions", abs(ref - new).max() < 1e-5, f"{abs(ref - new).max():.2e}")

    # 3. action preprocessing is the identity for axis-angle (features.py relies on it)
    a = torch.randn(4, 16, 7)
    dd = {"actions": a.clone(), "obs": {"hand_mat_inv": torch.eye(4).expand(4, 1, 4, 4)}}
    p.preprocess_actions(dd)
    check("preprocess_actions is identity", torch.allclose(dd["actions"], a))

    # 4. per-arm: meta-gradients, centring, deployment writes
    E, F, Q, K = 3, 4, 5, 2
    sup = [(torch.randn(E * F, 4, 256), torch.rand(E * F, 16, 7) * 2 - 1) for _ in range(K)]
    qe, qa = torch.randn(E * Q, 4, 256), torch.rand(E * Q, 16, 7) * 2 - 1
    for arm in ["ttt", "kv", "fmw", "res"]:
        m = build_memory(arm)
        print(f"--- {arm}: fast params {m.fast_numel():,}")
        # meta-gradient through K writes
        state = m.init_state(E)
        for e, a_ in sup:
            state = m.write(flow, state, e, a_, create_graph=True)
        loss = m.outer_loss(flow, state, qe, qa)
        loss.backward()
        g_w0 = sum(prm.grad.abs().sum().item() for prm in m.W0.values() if prm.grad is not None)
        g_lr = sum(prm.grad.abs().sum().item() for prm in m.log_lr.values() if prm.grad is not None)
        check(f"{arm}: finite outer loss", torch.isfinite(loss).item(), f"{loss.item():.4f}")
        check(f"{arm}: grad reaches W0", g_w0 > 0, f"{g_w0:.3e}")
        check(f"{arm}: grad reaches inner lr (2nd order)", g_lr > 0, f"{g_lr:.3e}")
        m.zero_grad()

        # centring: empty memory == base, exactly
        s0 = m.init_state(1)
        e1 = torch.randn(7, 4, 256)
        noise = torch.randn(7, 16, 7)
        a_mem = m.sample(flow, s0, e1, noise)
        a_base = flow.sample(e1, noise)
        if m.centred:
            check(f"{arm}: empty memory == base policy", torch.allclose(a_mem, a_base, atol=1e-6),
                  f"{(a_mem - a_base).abs().max():.2e}")
        else:
            print(f"INFO {arm}: not centred; empty-memory deviation {(a_mem - a_base).abs().max():.2e}")

        # deployment writes (no meta-graph) change the policy and lower the write loss
        s = m.init_state(1, requires_grad=True)
        e_d, a_d = torch.randn(20, 4, 256), torch.rand(20, 16, 7) * 2 - 1
        for _ in range(3):
            s = m.write(flow, s, e_d, a_d, create_graph=False)
        a_after = m.sample(flow, s, e1, noise)
        check(f"{arm}: writes change the output", (a_after - a_mem).abs().max() > 1e-6,
              f"{(a_after - a_mem).abs().max():.2e}")
        check(f"{arm}: deployment state is detached", all(v.grad_fn is None for v in s.values()))

    # 5. decoder_parameters excludes both encoders
    names = {id(q) for q in decoder_parameters(p)}
    enc = [q for n, q in p.velocity_net.named_parameters() if n.startswith("encoder.")]
    check("decoder params exclude obs encoder", all(id(q) not in names for q in enc))
    print("\nALL CPU TESTS PASSED")


if __name__ == "__main__":
    main()
