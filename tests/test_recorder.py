"""Regression test for selfplay.run_batch (audit P1): an episode's recorded transitions must not depend on
how long the OTHER episodes in its vectorized batch run.

Fake env: eef x-coordinate = number of env steps taken (all envs step together, as in LIBERO's
vector env). Env b reports success at a chosen step. The recorded proprio of an episode that ends at
step k must be [0, 1, ..., k] regardless of its batch partners.

    python -m pytest tests/test_recorder.py -q     (or: python tests/test_recorder.py)
"""
import types

import numpy as np
import torch

from horsea.selfplay import run_batch


class FakeVecEnv:
    def __init__(self, B, succ_at):
        self.B, self.succ_at, self.t = B, succ_at, 0

    def _obs(self):
        x = np.full((self.B, 1, 3), float(self.t))
        return {"robot0_eef_pos": x, "robot0_gripper_qpos": np.zeros((self.B, 1, 2))}

    def reset(self, init_states=None):
        self.t = 0
        return self._obs(), {}

    def step(self, act):
        self.t += 1
        info = [{"success": self.succ_at[b] is not None and self.t >= self.succ_at[b]} for b in range(self.B)]
        return self._obs(), 0.0, None, None, info

    def close(self):
        pass


def fake_runner():
    bench = types.SimpleNamespace(get_task_init_states=lambda t: np.zeros((50, 97)),
                                  get_task_emb=lambda t: {"task_emb": torch.zeros(1, 4)})
    return types.SimpleNamespace(benchmark=bench)


def fake_rec(B):
    pol = types.SimpleNamespace(reset=lambda: None, get_action=lambda obs, task, **k: np.zeros((B, 7)))
    return types.SimpleNamespace(policy=pol, calls=None, shifts=None, mode="novice", step=0)


def record(succ_at, horizon=10):
    B = len(succ_at)
    out = run_batch(fake_runner(), fake_rec(B), 0, list(range(B)), ["none"] * B, horizon=horizon,
                    env_ctor=lambda n: FakeVecEnv(n, succ_at))
    return [e["proprio"][:, 0].tolist() for e in out]


def test_terminal_observation_is_per_env():
    alone = record([2, 2])[0]                 # partner finishes at the same time
    early_vs_late = record([2, 6])            # partner runs 4 steps longer
    early_vs_never = record([2, None])        # partner never succeeds (runs to horizon)
    assert alone == [0.0, 1.0, 2.0], alone
    assert early_vs_late[0] == alone, early_vs_late[0]
    assert early_vs_never[0] == alone, early_vs_never[0]
    assert early_vs_late[1] == [float(i) for i in range(7)], early_vs_late[1]
    assert early_vs_never[1] == [float(i) for i in range(11)], early_vs_never[1]


if __name__ == "__main__":
    test_terminal_observation_is_per_env()
    print("ok")
