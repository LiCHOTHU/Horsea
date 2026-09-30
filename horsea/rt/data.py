"""RoboTwin 2.0 demonstrations (aloha-agilex, clean, 50 per task, all 50 tasks) for the FM policy.

Episode split per task: episodes 0-44 train, 45-49 held out (diagnostics / later studies).
Sample (task, episode, t): 3 camera JPEGs decoded exactly as RoboTwin's DP baseline does (cv2.imdecode, no colour
conversion -- the channel order the environment returns at evaluation) and resized to 120x160; state = joint
vector[t]; actions = vector[t+1 .. t+16] (padded with the last frame); instruction = a random 'seen' instruction of
that episode, as its cached CLIP ViT-B/32 text embedding.
"""
import glob
import json
import os

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from horsea.rt.policy import CAMS, CHUNK, IMG_HW

ROOT = os.path.expanduser("~/workspace/RoboTwin/data")
CFG = "aloha-agilex_clean_50"
CACHE = os.path.expanduser("~/workspace/Horsea/experiments/rt/cache")
TRAIN_EPS, HELD_EPS = list(range(45)), list(range(45, 50))


def tasks():
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{ROOT}/*/{CFG}"))


def ep_path(task, ep):
    return f"{ROOT}/{task}/{CFG}/data/episode{ep}.hdf5"


def instructions(task, ep, kind="seen"):
    return json.load(open(f"{ROOT}/{task}/{CFG}/instructions/episode{ep}.json"))[kind]


def clip_cache(texts, device="cuda:0"):
    """text -> 512-d CLIP ViT-B/32 embedding, cached on disk."""
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/clip_text.pt"
    cache = torch.load(path, weights_only=False) if os.path.exists(path) else {}
    todo = sorted(set(texts) - set(cache))
    if todo:
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        m = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        with torch.no_grad():
            for i in range(0, len(todo), 256):
                b = todo[i:i + 256]
                e = m(**tok(b, return_tensors="pt", padding=True, truncation=True).to(device)).text_embeds.cpu()
                cache.update(dict(zip(b, e)))
        torch.save(cache, path)
    return cache


def decode(buf):
    im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(im, (IMG_HW[1], IMG_HW[0]), interpolation=cv2.INTER_AREA)


def compute_stats(task_list, eps=TRAIN_EPS):
    path = f"{CACHE}/stats.pt"
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    V = np.concatenate([h5py.File(ep_path(t, e), "r")["joint_action/vector"][()] for t in task_list for e in eps])
    lo, hi = V.min(0), V.max(0)
    pad = 0.02 * (hi - lo) + 1e-4
    stats = {"a_min": lo - pad, "a_max": hi + pad, "s_min": lo - pad, "s_max": hi + pad, "tasks": task_list}
    os.makedirs(CACHE, exist_ok=True)
    torch.save(stats, path)
    return stats


class RTDataset(Dataset):
    def __init__(self, task_list, eps, clip):
        self.items, self.lengths, self.instr = [], {}, {}
        self.clip = clip
        for ti, t in enumerate(task_list):
            for e in eps:
                n = h5py.File(ep_path(t, e), "r")["joint_action/vector"].shape[0]
                self.lengths[(t, e)] = n
                self.instr[(t, e)] = instructions(t, e)
                self.items += [(t, e, i, ti) for i in range(n - 1)]    # every frame with at least one next action
        self._h5 = {}

    def __len__(self):
        return len(self.items)

    def _f(self, t, e):
        k = (t, e)
        if k not in self._h5:           # opened lazily per worker
            if len(self._h5) > 64:
                self._h5.pop(next(iter(self._h5))).close()
            self._h5[k] = h5py.File(ep_path(t, e), "r")
        return self._h5[k]

    @staticmethod
    def _chunk(vec, i, n):
        a = vec[i + 1:min(i + 1 + CHUNK, n)].astype(np.float32)          # contiguous read (h5py needs increasing idx)
        if len(a) < CHUNK:
            a = np.concatenate([a, np.repeat(a[-1:], CHUNK - len(a), 0)])  # pad with the last frame
        return a

    def __getitem__(self, idx):
        t, e, i, ti = self.items[idx]
        f = self._f(t, e)
        vec = f["joint_action/vector"]
        n = vec.shape[0]
        imgs = np.stack([decode(f[f"observation/{c}/rgb"][i]) for c in CAMS])          # (3, H, W, 3) uint8
        instr = self.instr[(t, e)][np.random.randint(len(self.instr[(t, e)]))]
        return {"imgs": torch.from_numpy(imgs), "state": torch.from_numpy(vec[i].astype(np.float32)),
                "actions": torch.from_numpy(self._chunk(vec, i, n)), "lang": self.clip[instr], "task": ti}
