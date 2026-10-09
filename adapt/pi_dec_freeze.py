"""Freeze a source run's current actor for Policy Decorator: copy the actor weights and the critics' fixed input
transform out of train_ckpt.pt into a small, hashed file, so the base never changes under a running stream even
while source training continues.

  python adapt/pi_dec_freeze.py --src runs/bc02/source_s1 --out runs/bc02/pi_dec/pd_base_s1.pt
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapt import common  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

ck_path = os.path.join(args.src, "train_ckpt.pt")
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
state = json.load(open(os.path.join(args.src, "state.json")))
ag = ck["agent"]
critic_norm = {k[len("norm."):]: v for k, v in ag["qf1"].items() if k.startswith("norm.")}
last = state["eval_log"][-1] if state.get("eval_log") else {}
record = dict(source_dir=args.src, train_ckpt_hash=common.file_hash(ck_path), global_step=state["global_step"],
              phase=state["phase"], last_eval_step=last.get("step"), last_eval_success=last.get("success"),
              last_eval_per_object=last.get("per_object"), alpha=float(ag["log_alpha"].exp()))
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
common.atomic_torch_save({"actor": ag["actor"], "critic_norm": critic_norm, "record": record}, args.out)
print(json.dumps(record | {"out": args.out, "out_hash": common.file_hash(args.out)}, default=str))
