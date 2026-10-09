"""Train the base fm_policy_S on the 80 training tasks of LIBERO-90.

Runs the imitation repo's own train.py unchanged (same recipe as the July 93% LIBERO-90 run:
50 epochs, fm_policy_S, rollouts off); only the train80 benchmark is registered first.
Resumable: train.py resumes from multitask_model_latest.pth in the experiment dir.
"""
import os
import runpy
import sys

HORSEA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Python puts THIS file's directory (scripts/) on sys.path, where scripts/queue.py shadows the
# stdlib `queue` that torch.fx imports ("from queue import Queue"). Drop it before importing torch.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if not p or os.path.abspath(p) != _HERE]
sys.path.insert(0, HORSEA)
import horsea  # noqa: E402,F401  (LIBERO_CONFIG_PATH, MUJOCO_GL)
import horsea.benchmarks  # noqa: E402,F401  (registers libero_90_train80)
from horsea.paths import DATA_PREFIX, EXP, IMITATION  # noqa: E402

overrides = [
    "task=libero",
    "task.benchmark_name=libero_90_train80",
    "algo=fm_policy_S",
    "training.n_epochs=50",
    "training.save_interval=10",
    "training.save_all_checkpoints=false",
    "training.use_tqdm=false",
    "rollout.enabled=false",
    "logging.mode=disabled",
    # The dataset uses hdf5_cache_mode=low_dim, so every sample reads its two camera images from a
    # compressed hdf5 on shared scratch: training is I/O bound and needs parallel readers to hide
    # the latency. num_workers=0 measured ~4 MiB/s with the GPU at 0%, which is far too slow.
    # Keep config/train.yaml's fork context (spawn is impossible: SequenceDataset holds a lambda and
    # cannot be pickled) and match the workers to the CPUs the job asks for.
    "train_dataloader.num_workers=8",
    "make_unique_experiment_dir=false",
    "exp_name=base80",
    "variant_name=fm",
    f"data_prefix={DATA_PREFIX}",
    f"output_prefix={EXP}",
    f"hydra.run.dir={EXP}/hydra/train_base",
] + sys.argv[1:]

sys.path.insert(0, IMITATION)
sys.argv = [os.path.join(IMITATION, "train.py"), "--config-name=train"] + overrides
runpy.run_path(os.path.join(IMITATION, "train.py"), run_name="__main__")
