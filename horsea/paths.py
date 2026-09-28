"""Paths and task splits shared by every stage."""
import os

HORSEA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMITATION = os.path.expanduser("~/workspace/imitation")
DATA_PREFIX = os.path.join(IMITATION, "data")
EXP = os.environ.get("HORSEA_EXP", os.path.join(HORSEA, "experiments"))
FEAT_DIR = os.path.join(EXP, "features")

# One held-out task per scene, from 10 scenes that keep >= 3 training tasks, so every novel
# task is a new instruction in a familiar scene. Task 51 (butter -> basket) is avoided: every
# policy scored 0/10 on it in July (suspected data defect).
HELDOUT_90 = [2, 9, 13, 25, 43, 48, 57, 66, 79, 84]
TRAIN_90 = [i for i in range(90) if i not in HELDOUT_90]
LIBERO_10 = list(range(10))  # far-novel set: long-horizon compositions, never trained on

# Old-task subset used to measure retention after consolidation (fixed, spread over scenes).
RETENTION_90 = TRAIN_90[::4]  # 20 tasks

BASE_CKPT = os.path.join(EXP, "libero", "libero_90_train80", "base80", "fm", "multitask_model.pth")
