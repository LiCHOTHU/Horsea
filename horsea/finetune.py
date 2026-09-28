"""Slow-weight updates of the base decoder: the direct fine-tuning baseline and consolidation.

Both train only decoder_parameters() (the denoising network); both encoders stay frozen so the
precomputed encoder summaries stay valid. Loss = the base's own FM loss (same time sampling,
interpolant and target) on (encm, x1) pairs, where x1 is either demonstrated actions (fine-tune)
or teacher samples drawn fresh at every step (consolidation = endpoint FM distillation, eq. 22
of the proposal, with fresh source noise independent of the teacher's).
"""
import copy

import torch

from horsea.base import Flow, decoder_parameters


def clone_trainable(policy):
    student = copy.deepcopy(policy)
    student.__dict__.pop("sample_actions", None)  # never inherit another policy's sampler hook
    for p in student.parameters():
        p.requires_grad_(False)
    for p in decoder_parameters(student):
        p.requires_grad_(True)
    return student


def train_decoder(policy, new_batch, steps, lr=5e-5, replay_batch=None, clip=1.0, log_every=500, tag="",
                  extra_loss=None):
    """new_batch() -> (encm, x1); replay_batch() -> (encm, x1) or None; extra_loss(flow) -> scalar
    (e.g. an anchor that keeps another input channel unchanged). Returns a frozen copy."""
    student = clone_trainable(policy)
    flow = Flow(student)
    opt = torch.optim.AdamW(decoder_parameters(student), lr=lr, weight_decay=1e-6)
    student.train()  # dropout on, as in base training; encoders are unused (summaries precomputed)
    hist = []
    for s in range(steps):
        e, x1 = new_batch()
        loss_new = flow.fm_loss(e, x1)
        loss = loss_new
        if replay_batch is not None:
            er, xr = replay_batch()
            loss_rep = flow.fm_loss(er, xr)
            loss = loss + loss_rep
        if extra_loss is not None:
            loss_extra = extra_loss(flow)
            loss = loss + loss_extra
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder_parameters(student), clip)
        opt.step()
        if (s + 1) % log_every == 0 or s == 0:
            rec = {"step": s + 1, "loss_new": round(loss_new.item(), 5)}
            if replay_batch is not None:
                rec["loss_replay"] = round(loss_rep.item(), 5)
            if extra_loss is not None:
                rec["loss_extra"] = round(loss_extra.item(), 5)
            hist.append(rec)
            print(tag, rec, flush=True)
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    return student, hist


def demo_batcher(encm, act, bs, g=None):
    n = encm.shape[0]

    def f():
        i = torch.randint(n, (bs,), device=encm.device)
        return encm[i], act[i]

    return f
