"""Frozen fm_policy_S base: loading, and its velocity field evaluated from encoder summaries.

The DiT decoder in imitation/algos/utils/diffusion_policy_utils/dit_modules.py only sees the
observation through `torch.mean(enc_cache[l], axis=0)` (adaLN conditioning per decoder layer,
and the last encoder layer for the final layer). So a frame's whole conditioning is
encm = stack_l mean(enc_cache[l]) with shape (L=4, D=256). Passing encm[:, l][None] as the
layer's "cond" reproduces the decoder exactly; tests/test_base_equivalence.py checks this.
"""
import copy

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

DECODER_PREFIXES = ("time_net.", "ac_proj.", "dec_pos", "decoder.", "eps_out.")


def load_policy(ckpt_path, device):
    """Instantiate a trained FlowMatchingPolicy exactly like imitation/evaluate.py does."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy_cfg = copy.deepcopy(sd["config"]["algo"]["policy"])
    policy_cfg["device"] = str(device)
    policy = instantiate(policy_cfg)
    policy.load_state_dict(sd["model"])
    policy.normalizer.fit(sd["norm_stats"])
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy, sd


def decoder_parameters(policy):
    """The slow weights that consolidation and fine-tuning update (the denoising decoder).

    The ResNet encoder and the DiT observation encoder stay frozen everywhere, which keeps the
    precomputed encoder summaries valid for every student.
    """
    return [p for n, p in policy.velocity_net.named_parameters() if n.startswith(DECODER_PREFIXES)]


class Flow:
    """Velocity field of a (possibly fine-tuned) fm_policy_S from encoder summaries."""

    def __init__(self, policy):
        self.policy = policy
        self.vnet = policy.velocity_net
        self.sig_min = policy.flow_sig_min
        self.n_steps = policy.num_inference_steps
        self.chunk = policy.chunk_size
        self.adim = policy.network_action_dim

    # ----- observation side ------------------------------------------------------------
    @torch.no_grad()
    def encode(self, data):
        """Raw batch (as built by the dataset or ChunkPolicy._make_batch) -> encm (B, L, D)."""
        data = self.policy.preprocess_input(data, train_mode=False)
        cond = self.policy.get_cond(data)
        enc_cache = self.vnet.forward_enc(cond)
        return torch.stack([e.mean(0) for e in enc_cache], dim=1)

    # ----- action side -----------------------------------------------------------------
    def decode(self, z, t, encm, layer_hook=None, return_hidden=False):
        """v(z, t | encm). layer_hook(l, x) may modify the (chunk, B, D) tokens after layer l."""
        te = self.vnet.time_net(t)
        x = self.vnet.ac_proj(z).transpose(0, 1) + self.vnet.dec_pos
        for l, layer in enumerate(self.vnet.decoder.layers):
            x = layer(x, te, encm[:, l][None])
            if layer_hook is not None:
                x = layer_hook(l, x)
        v = self.vnet.eps_out(x, te, encm[:, -1][None])
        return (v, x) if return_hidden else v

    def sample_t(self, n, device):
        return self.policy._sample_fm_time(n).to(device)

    def interp(self, x0, x1, t):
        """(psi_t, target velocity) of the base's conditional path."""
        psi = self.policy._psi_t(x0, x1, t)
        return psi, x1 - (1 - self.sig_min) * x0

    def euler(self, field, encm, noise=None):
        """Integrate dz/dt = field(z, t) from Gaussian noise with the base's Euler schedule."""
        B = encm.shape[0]
        z = torch.randn(B, self.chunk, self.adim, device=encm.device) if noise is None else noise.clone()
        dt = 1.0 / self.n_steps
        t = torch.zeros(B, device=encm.device)
        for _ in range(self.n_steps):
            z = z + dt * field(z, t)
            t = t + dt
        return z

    def sample(self, encm, noise=None):
        return torch.clamp(self.euler(lambda z, t: self.decode(z, t, encm), encm, noise), -1, 1)

    def fm_loss(self, encm, actions, field=None):
        """Base FM loss (optionally with a replacement field) on normalized action chunks."""
        x1 = torch.clamp(actions, -1, 1)
        x0 = torch.randn_like(x1)
        t = self.sample_t(x1.shape[0], x1.device)
        psi, u = self.interp(x0, x1, t)
        v = self.decode(psi, t, encm) if field is None else field(psi, t)
        return ((v - u) ** 2).mean()
