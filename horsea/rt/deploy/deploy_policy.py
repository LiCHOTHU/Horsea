"""RoboTwin deploy wrapper for the Horsea FM policy (horsea/rt). Receding horizon: sample a 16-step chunk of joint
targets, execute the first `execute_steps` (default 8) with qpos control, re-plan (no temporal aggregation, no action
queue carried across calls). Images decoded/resized exactly as in training; CLIP ViT-B/32 embedding of the instruction.

Graph study (2026-10-01): `graph` (name in horsea.rt.graph.library('all'), default G0) and `K` (outer Euler
evaluations, default 10) come from the yml/overrides. The initial FM noise of every action generation comes from a
dedicated generator keyed by (eval seed, task, episode index, decision index, noise replicate) -- identical across
graphs and never advanced by anything else. Per-decision latency/block calls are appended to HORSEA_EPLOG if set.
"""
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/licho/workspace/Horsea")
from horsea.rt.graph import Tracer, install, library  # noqa: E402
from horsea.rt.policy import ADIM, CAMS, CHUNK, IMG_HW, RTFlowPolicy  # noqa: E402


# Record each episode's env seed on the task object (RoboTwin does not store it). Wrapping at import time leaves the
# RoboTwin sources untouched; task classes reach this through super()._init_task_env_(**kwags).
try:
    import envs._base_task as _bt
    if not getattr(_bt.Base_Task._init_task_env_, "_horsea_wrapped", False):
        _orig_init = _bt.Base_Task._init_task_env_

        def _init_task_env_(self, **kwags):
            self.horsea_seed = kwags.get("seed")
            return _orig_init(self, **kwags)

        _init_task_env_._horsea_wrapped = True
        _bt.Base_Task._init_task_env_ = _init_task_env_
except ImportError:  # imported outside RoboTwin (unit tests)
    pass


def _key(*xs):
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:15], 16)


class _Model:
    def __init__(self, ckpt, execute_steps=8, device="cuda:0", graph="G0", K=10, noise_rep=0, seed=0, task=""):
        torch.backends.cudnn.benchmark = False        # reproducibility across processes (spec check 9)
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        s = torch.load(ckpt, map_location=device, weights_only=False)
        self.policy = RTFlowPolicy(s["stats"]).to(device).eval()
        self.policy.load_state_dict(s["model"])
        self.policy.K = int(K)
        self.graph_name = graph
        self.tracer = Tracer()
        install(self.policy.velocity_net, library("all")[graph], self.tracer)
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        self.tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.dev, self.execute_steps, self._lang = device, execute_steps, {}
        self.noise_rep, self.seed, self.task = int(noise_rep), seed, task
        self.decision = 0
        self.log = os.environ.get("HORSEA_EPLOG")

    def lang(self, text):
        if text not in self._lang:
            with torch.no_grad():
                self._lang[text] = self.clip(**self.tok([text], return_tensors="pt", padding=True, truncation=True)
                                             .to(self.dev)).text_embeds
        return self._lang[text]

    def noise(self, env_seed):
        """Initial FM noise keyed by the INITIAL STATE (env seed), not the episode index: RoboTwin's expert check that
        admits seeds is not deterministic, so the same episode index can be a different scene in different arms."""
        g = torch.Generator(device=self.dev).manual_seed(_key(self.seed, self.task, env_seed, self.decision, self.noise_rep))
        return torch.randn(1, CHUNK, ADIM, device=self.dev, generator=g)


def encode_obs(observation):
    imgs = np.stack([cv2.resize(observation["observation"][c]["rgb"], (IMG_HW[1], IMG_HW[0]), interpolation=cv2.INTER_AREA)
                     for c in CAMS])
    return {"imgs": imgs, "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32)}


def get_model(usr_args):
    return _Model(usr_args["ckpt_path"], int(usr_args.get("execute_steps", 8)), usr_args.get("device", "cuda:0"),
                  usr_args.get("graph", "G0"), usr_args.get("K", 10), usr_args.get("noise_rep", 0),
                  usr_args.get("seed", 0), usr_args.get("task_name", ""))


def eval(TASK_ENV, model, observation):
    if not getattr(model, "_cache_freed", False):  # release allocator blocks cached by the planner warm-up (memory only)
        torch.cuda.empty_cache()
        model._cache_freed = True
    obs = encode_obs(observation)
    imgs = torch.from_numpy(obs["imgs"]).to(model.dev).permute(0, 3, 1, 2).float()[None] / 255.0
    state = torch.from_numpy(obs["state"]).to(model.dev)[None]
    episode = TASK_ENV.test_num
    env_seed = getattr(TASK_ENV, "horsea_seed", None)
    assert env_seed is not None, "env seed not captured (seed wrapper inactive)"
    noise = model.noise(env_seed)
    model.tracer.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    actions = model.policy.sample(imgs, state, model.lang(TASK_ENV.get_instruction()), noise=noise)[0].cpu().numpy()
    dt = time.perf_counter() - t0
    if model.log:
        with open(model.log, "a") as f:
            f.write(json.dumps({"graph": model.graph_name, "K": model.policy.K, "episode": episode, "env_seed": env_seed,
                                "decision": model.decision, "sec": round(dt, 5),
                                "block_calls": model.tracer.calls()}) + "\n")
    if os.environ.get("HORSEA_DUMP") and not getattr(model, "_dumped", False):
        np.savez(os.environ["HORSEA_DUMP"], imgs=obs["imgs"], state=obs["state"], actions=actions,
                 raw_head=observation["observation"]["head_camera"]["rgb"], instr=TASK_ENV.get_instruction())
        model._dumped = True
    if os.environ.get("HORSEA_RENDERCHECK") and model.decision == 0:   # diagnostic: render the same scene twice
        o2 = TASK_ENV.get_obs()
        d = {c: float(np.abs(observation["observation"][c]["rgb"].astype(np.int16) -
                             o2["observation"][c]["rgb"].astype(np.int16)).mean()) for c in CAMS}
        print("RENDERCHECK", json.dumps(d), flush=True)
    model.decision += 1
    for a in actions[:model.execute_steps]:
        TASK_ENV.take_action(a, action_type="qpos")
        if TASK_ENV.eval_success:
            break


def reset_model(model):
    model.decision = 0  # per-episode decision counter (noise keys); there is no action queue or aggregation buffer
    torch.cuda.empty_cache()  # memory only: drop allocator blocks cached by the per-episode expert-check planning
