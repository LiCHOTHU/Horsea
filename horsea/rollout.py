"""LIBERO closed-loop evaluation / data collection for base, fine-tuned and memory-adapted policies.

Reuses imitation's LiberoRunner (same env wrapper, temporal aggregation, horizons and init
states as the July LIBERO-90 evaluation); adds an init-state offset (collection must not reuse
evaluation initial states), per-episode records, and a sampler hook that routes the policy's
action sampling through a memory.
"""
import functools
import time
import types

import numpy as np
import torch

import imitation.envs.libero.runner as _runner_mod
import imitation.envs.libero.wrappers as lw
from imitation.envs.libero.runner import LiberoRunner
from imitation.envs.libero.utils import get_benchmark_instance

HORIZON = {"libero_90": 300, "libero_10": 520}
_runner_mod.trange = lambda n, **kw: range(n)  # the runner draws a progress bar per env step


class OffsetRunner(LiberoRunner):
    def __init__(self, *args, init_offset=0, init_indices=None, init_states=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_offset = init_offset
        self.init_indices = None if init_indices is None else np.asarray(init_indices)  # explicit fold
        self.init_states = init_states  # custom (N, D) start states (e.g. generated test starts)

    def run_policy_in_env(self, env_name, policy, render=False, fault_tolerant=False):
        """Parent's loop with (i) an init-state offset or explicit fold and (ii) env shutdown at the end."""
        env_id = self.env_names.index(env_name)
        env_num = min(self.num_parallel_envs, self.rollouts_per_env)
        env_fn = lambda: lw.LiberoFrameStack(self.env_factory(task_id=env_id, benchmark=self.benchmark), self.frame_stack)
        env = lw.LiberoVectorWrapper(env_fn, env_num)
        inits = self.init_states if self.init_states is not None else self.benchmark.get_task_init_states(env_id)
        n_loops = (self.rollouts_per_env + env_num - 1) // env_num
        try:
            for c in range(n_loops):
                if self.init_indices is not None:
                    idx = self.init_indices[np.arange(c * env_num, (c + 1) * env_num) % len(self.init_indices)]
                else:
                    idx = (self.init_offset + np.arange(c * env_num, (c + 1) * env_num)) % inits.shape[0]
                success, total_reward, episode = self.run_episode(env, env_name, policy, inits[idx], env_num, render)
                for k in range(env_num):
                    if c * env_num + k >= self.rollouts_per_env:
                        break
                    yield bool(success[k]), int(len(episode["actions"])), int(idx[k])
        finally:
            try:
                env._env.close()
            except Exception:
                pass


def make_runner(shape_meta, suite, n_rollouts, n_par=5, init_offset=0, device="cuda:0", horizon=None,
                init_indices=None, init_states=None):
    env_factory = functools.partial(
        lw.LiberoWrapper, shape_meta=shape_meta, img_height=128, img_width=128, abs_action=False,
        robot="Panda", camera_pose_variations=False, device=device,
    )
    return OffsetRunner(
        env_factory=env_factory, benchmark=get_benchmark_instance(suite), mode="all",
        rollouts_per_env=n_rollouts, num_parallel_envs=n_par, max_episode_length=horizon or HORIZON[suite],
        frame_stack=1, fps=24, debug=False, task_embedding_format="clip", init_offset=init_offset,
        init_indices=init_indices, init_states=init_states,
    )


def install_sampler(policy, flow, memory=None, state=None, recorder=None, task_emb=None, recorder_real=None):
    """Route policy.sample_actions (called by ChunkPolicy.get_action) through flow [+ memory].
    task_emb: if given, replaces the env's instruction embedding (generic-instruction protocol)."""

    def sample_actions(self, data):
        with torch.no_grad():
            if recorder_real is not None:  # the same states under the env's real instruction
                import copy
                recorder_real.append(flow.encode(copy.deepcopy(data)).half().cpu())
            if task_emb is not None:
                B = data["task_emb"].shape[0]
                data["task_emb"] = task_emb.to(data["task_emb"].device).expand(B, -1)
            encm = flow.encode(data)
            if recorder is not None:
                recorder.append(encm.half().cpu())
            a = flow.sample(encm) if memory is None else memory.sample(flow, state, encm)
            return a.cpu().numpy()

    policy.sample_actions = types.MethodType(sample_actions, policy)
    policy.batch_size = None  # temporal-aggregation buffers are re-sized on the next call
    return policy


def run_task(runner, policy, task_id):
    """Roll out one task. Returns per-episode success, length and init-state index."""
    env_name = runner.env_names[task_id]
    t0 = time.time()
    eps = list(runner.run_policy_in_env(env_name, policy))
    return {
        "task": task_id,
        "env_name": env_name,
        "success": [e[0] for e in eps],
        "length": [e[1] for e in eps],
        "init": [e[2] for e in eps],
        "rate": float(np.mean([e[0] for e in eps])),
        "rollout_sec": round(time.time() - t0, 1),
    }
