"""LIBERO-90 index-subset benchmarks (training split for the base policy)."""
import os

import torch
from libero.libero import get_libero_path
from libero.libero.benchmark import Benchmark, register_benchmark
from libero.libero.benchmark import task_maps as libero_task_maps

import imitation.envs.libero  # noqa: F401  (registers the repo's custom benchmarks)
from horsea.paths import TRAIN_90


class _Libero90IndexSubset(Benchmark):
    """A non-contiguous subset of the stock libero_90 suite.

    Reuses the stock task definitions, so bddl files, init states and demonstration paths
    resolve exactly as for the full benchmark. Not decorated itself: register_benchmark
    returns None, so a decorated class cannot be subclassed.
    """

    task_ids = ()

    def _make_benchmark(self):
        stock = list(libero_task_maps["libero_90"].values())
        self.tasks = [stock[i] for i in self.task_ids]
        self.n_tasks = len(self.tasks)

    def get_task_init_states(self, i):
        path = os.path.join(
            get_libero_path("init_states"), self.tasks[i].problem_folder, self.tasks[i].init_states_file
        )
        return torch.load(path, weights_only=False)


@register_benchmark
class LIBERO_90_TRAIN80(_Libero90IndexSubset):
    """The 80 LIBERO-90 tasks the base policy and all memories are trained on."""

    task_ids = tuple(TRAIN_90)

    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_90_train80"
        self._make_benchmark()
