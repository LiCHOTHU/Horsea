"""RoboTwin deploy wrapper for the Horsea FM policy (horsea/rt). Receding horizon: sample a 16-step chunk of joint
targets, execute the first `execute_steps` (default 8) with qpos control, re-plan. Images decoded/resized exactly as
in training (env RGB arrays == cv2-decoded training JPEGs), CLIP ViT-B/32 embedding of the evaluation instruction."""
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/licho/workspace/Horsea")
from horsea.rt.policy import CAMS, IMG_HW, RTFlowPolicy  # noqa: E402


class _Model:
    def __init__(self, ckpt, execute_steps=8, device="cuda:0"):
        s = torch.load(ckpt, map_location=device, weights_only=False)
        self.policy = RTFlowPolicy(s["stats"]).to(device).eval()
        self.policy.load_state_dict(s["model"])
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        self.tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.dev, self.execute_steps, self._lang = device, execute_steps, {}

    def lang(self, text):
        if text not in self._lang:
            with torch.no_grad():
                self._lang[text] = self.clip(**self.tok([text], return_tensors="pt", padding=True, truncation=True)
                                             .to(self.dev)).text_embeds
        return self._lang[text]


def encode_obs(observation):
    imgs = np.stack([cv2.resize(observation["observation"][c]["rgb"], (IMG_HW[1], IMG_HW[0]), interpolation=cv2.INTER_AREA)
                     for c in CAMS])
    return {"imgs": imgs, "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32)}


def get_model(usr_args):
    return _Model(usr_args["ckpt_path"], int(usr_args.get("execute_steps", 8)), usr_args.get("device", "cuda:0"))


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    imgs = torch.from_numpy(obs["imgs"]).to(model.dev).permute(0, 3, 1, 2).float()[None] / 255.0
    state = torch.from_numpy(obs["state"]).to(model.dev)[None]
    actions = model.policy.sample(imgs, state, model.lang(TASK_ENV.get_instruction()))[0].cpu().numpy()
    for a in actions[:model.execute_steps]:
        TASK_ENV.take_action(a, action_type="qpos")
        if TASK_ENV.eval_success:
            break


def reset_model(model):
    pass
