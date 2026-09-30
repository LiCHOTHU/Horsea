"""Demonstration splits for internal-loop continuation (spec 2026-09-30, sec. 6).

Same pipeline as the base checkpoint's training (imitation get_dataset -> SequenceDataset(seq 16, pad, low_dim
cache), SequenceVLDataset with the train-time CLIP task embeddings), over the base's 80-task LIBERO-90 split.
Episode-level split per task: demos 0-44 = continuation training, demos 45-49 = diagnostic (FM-loss screening).
All 50 demos of every task were in the base checkpoint's historical pretraining; the diagnostic episodes are held
out from CONTINUATION training only (not unseen by the base).
"""
import os

import torch
from torch.utils.data import ConcatDataset

import horsea  # noqa: F401
import horsea.benchmarks  # noqa: F401  (registers libero_90_train80)
from imitation.dataset.sequence_vl_dataset import SequenceVLDataset
import imitation.utils.obs_utils as ObsUtils
from imitation.envs.libero.utils import get_dataset, get_task_embs
from libero.libero.benchmark import get_benchmark

TRAIN_DEMOS = [f"demo_{i}" for i in range(45)]
DIAG_DEMOS = [f"demo_{i}" for i in range(45, 50)]


def build(cfg, demos, task_idx=None):
    """cfg: checkpoint config['task']['dataset']; demos: demo keys; task_idx: train80 indices (default all)."""
    bench = get_benchmark(cfg["benchmark_name"])()
    sm = cfg["shape_meta"]
    obs_modality = {"rgb": list(sm["observation"]["rgb"].keys()) if cfg.get("load_image", True) else [],
                    "depth": list(sm["observation"]["depth"].keys()) if cfg.get("load_depth", False) else [],
                    "low_dim": list(sm["observation"]["lowdim"].keys())}
    for k, v in (cfg.get("extra_obs_modality") or {}).items():
        obs_modality[k] = obs_modality[k] + list(v)
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})
    idx = list(range(bench.n_tasks)) if task_idx is None else list(task_idx)
    descriptions = [bench.get_task(i).language for i in range(bench.n_tasks)]
    embs = get_task_embs(cfg["task_embedding_format"], descriptions, train=True)
    out = []
    for i in idx:
        ds = get_dataset(dataset_path=os.path.join(cfg["data_prefix"], cfg["suite_name"], bench.get_task_demonstration(i)),
                         obs_modality=obs_modality, seq_len=cfg["seq_len"], obs_seq_len=cfg.get("obs_seq_len", 1),
                         frame_stack=cfg["frame_stack"], load_obs=cfg.get("load_obs", True), few_demos=list(demos),
                         n_demos=None, hdf5_cache_mode=cfg.get("hdf5_cache_mode", "low_dim"),
                         load_next_obs=False, dataset_keys=(), action_keys=cfg.get("action_keys", "actions"))
        out.append(SequenceVLDataset(ds, task_id=i, **embs[i]))
    return ConcatDataset(out)


def to_device(batch, dev):
    if torch.is_tensor(batch):
        return batch.to(dev, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, dev) for k, v in batch.items()}
    return batch
