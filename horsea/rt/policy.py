"""Flow-matching policy for RoboTwin 2.0 (aloha-agilex, 14-D joint-position actions), mirroring the LIBERO base.

Same velocity network as the LIBERO base (DiTNoiseNet: width 256, 4 encoder + 4 decoder blocks, 4 heads, MLP 512,
dropout 0.1), same FM path and time sampler (sig_min 0.001; t = (1 - sig_min)(1 - b), b ~ Beta(1.5, 1)), K = 10
Euler steps, chunk 16. Observation tokens (as in the LIBERO base: one token per camera + proprio + language):
    head / left-wrist / right-wrist RGB -> shared ImageNet ResNet-18, FiLM-conditioned on the instruction
        (language fusion as in the LIBERO base encoder), global-avg-pooled -> 256, + camera embedding
    joint state (14) -> MLP -> 256;  CLIP ViT-B/32 text embedding (512) -> 256
Actions: next 16 joint-position targets (vector[t+1 .. t+16]), min-max normalized to [-1, 1] per dimension.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from horsea.rt.dit_modules import DiTNoiseNet

CAMS = ("head_camera", "left_camera", "right_camera")
IMG_HW = (120, 160)          # 240x320 D435 frames at half resolution
ADIM, CHUNK = 14, 16
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FiLMResNet18(nn.Module):
    """ImageNet ResNet-18 trunk; after each residual stage x <- x * (1 + gamma(l)) + beta(l) from the language embedding
    (zero-initialised FiLM, so it starts as the pretrained trunk)."""

    def __init__(self, lang_dim=256, out_dim=256):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.stages = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.film = nn.ModuleList([nn.Linear(lang_dim, 2 * c) for c in (64, 128, 256, 512)])
        for f in self.film:
            nn.init.zeros_(f.weight)
            nn.init.zeros_(f.bias)
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x, lang):
        x = self.stem(x)
        for stage, film in zip(self.stages, self.film):
            x = stage(x)
            g, b = film(lang).chunk(2, -1)
            x = x * (1 + g[:, :, None, None]) + b[:, :, None, None]
        return self.proj(x.mean((2, 3)))


class RTFlowPolicy(nn.Module):
    def __init__(self, stats, dim=256, num_blocks=4, nhead=4, ff=512, dropout=0.1, K=10, sig_min=0.001, alpha=1.5):
        super().__init__()
        self.K, self.sig_min, self.alpha = K, sig_min, alpha
        for k in ("a_min", "a_max", "s_min", "s_max"):
            self.register_buffer(k, torch.as_tensor(stats[k], dtype=torch.float32))
        self.lang = nn.Sequential(nn.Linear(512, dim), nn.GELU(), nn.Linear(dim, dim))
        self.vision = FiLMResNet18(dim, dim)
        self.cam_emb = nn.Parameter(torch.zeros(len(CAMS), dim))
        self.state = nn.Sequential(nn.Linear(ADIM, dim), nn.GELU(), nn.Linear(dim, dim))
        self.velocity_net = DiTNoiseNet(ac_dim=ADIM, ac_chunk=CHUNK, time_dim=dim, hidden_dim=dim, num_blocks=num_blocks,
                                        dropout=dropout, dim_feedforward=ff, nhead=nhead)
        self.register_buffer("im_mean", IMNET_MEAN)
        self.register_buffer("im_std", IMNET_STD)

    # ---- normalization --------------------------------------------------------------------------------------
    def norm_a(self, a):
        return 2 * (a - self.a_min) / (self.a_max - self.a_min).clamp_min(1e-6) - 1

    def unnorm_a(self, a):
        return (a + 1) / 2 * (self.a_max - self.a_min).clamp_min(1e-6) + self.a_min

    def norm_s(self, s):
        return 2 * (s - self.s_min) / (self.s_max - self.s_min).clamp_min(1e-6) - 1

    # ---- observation tokens ---------------------------------------------------------------------------------
    def obs_tokens(self, imgs, state, lang_emb):
        """imgs (B, 3 cams, 3, H, W) float in [0, 1]; state (B, 14) raw; lang_emb (B, 512) CLIP text embedding."""
        B = imgs.shape[0]
        lang = self.lang(F.normalize(lang_emb.float(), dim=-1))
        x = (imgs.flatten(0, 1) - self.im_mean) / self.im_std
        v = self.vision(x, lang.repeat_interleave(len(CAMS), 0)).view(B, len(CAMS), -1) + self.cam_emb
        s = self.state(self.norm_s(state.float()).clamp(-1.5, 1.5))[:, None]
        return torch.cat([v, s, lang[:, None]], 1)                     # (B, 5, dim)

    # ---- flow matching --------------------------------------------------------------------------------------
    def sample_t(self, n, device, gen=None):
        u = torch.rand(n, device=device, generator=gen)
        return (1 - self.sig_min) * (1 - u ** (1.0 / self.alpha))       # Beta(alpha, 1) by inverse CDF

    def loss(self, imgs, state, lang_emb, actions, gen=None):
        cond = self.obs_tokens(imgs, state, lang_emb)
        x1 = self.norm_a(actions.float()).clamp(-1, 1)
        x0 = torch.randn(x1.shape, device=x1.device, generator=gen)
        t = self.sample_t(x1.shape[0], x1.device, gen)
        tt = t[:, None, None]
        psi = (1 - (1 - self.sig_min) * tt) * x0 + tt * x1
        _, v = self.velocity_net(psi, t, cond)
        return ((v - (x1 - (1 - self.sig_min) * x0)) ** 2).mean()

    @torch.no_grad()
    def sample(self, imgs, state, lang_emb, noise=None):
        cond = self.obs_tokens(imgs, state, lang_emb)
        enc = self.velocity_net.forward_enc(cond)
        B = cond.shape[0]
        z = torch.randn(B, CHUNK, ADIM, device=cond.device) if noise is None else noise
        t = torch.zeros(B, device=cond.device)
        for _ in range(self.K):
            z = z + (1.0 / self.K) * self.velocity_net.forward_dec(z, t, enc)
            t = t + 1.0 / self.K
        return self.unnorm_a(z.clamp(-1, 1))                            # (B, 16, 14) joint targets
