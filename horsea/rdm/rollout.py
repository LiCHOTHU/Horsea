"""Five-attempt own-rollout metaepisodes for the RDM proof of concept (spec sec. 5 and 7).

One vector env = B independent sequences of the same task (one init state each). Per metaepisode:
  for attempt in 0..4: reset to the sequence's init state (H kept, B cleared, reset marker), run a FIXED horizon
  (304 env steps = 38 decisions x 8; running continues after success), each decision:
    c_n = frozen encoder summary of o_n;  eps ~ N(0, I);  mu = prefix_8(z_K);  u ~ N(mu, sigma^2);
    executed = clip(u) (then the base's own unnormalize/postprocess);  log N(u; mu, sigma) is logged;
    after the prefix has executed: append e_n = (c_n, p_n, clip(u), c_n+1, p_n+1, tags) to H.
  optional probe: one more attempt from a fresh held-out init state with a CLONE of H (never written back).
Online memory inputs never include reward/success. The sparse success reward is logged for training only.
"""
import copy
import time
import types

import numpy as np
import torch

import imitation.envs.libero.wrappers as lw
from horsea.rdm.model import EXEC, N_ATT, N_DEC, policy_logp, sample_action

HORIZON = N_DEC * EXEC  # 304


def prop_of(obs):
    return np.concatenate([obs["robot0_eef_pos"][:, -1], obs["robot0_gripper_qpos"][:, -1]], 1).astype(np.float32)


FIELDS = {"pre": ((4, 256), torch.float16), "post": ((4, 256), torch.float16), "act": ((EXEC, 7), torch.float32),
          "amask": ((EXEC,), torch.float32), "p_pre": ((5,), torch.float32), "p_post": ((5,), torch.float32),
          "attempt": ((), torch.long), "dstep": ((), torch.long), "reset": ((), torch.long)}


class Bank:
    """Causal experience bank of B synchronous sequences: raw events in preallocated tensors (tokens are
    recomputed under the current psi at training time; during a rollout psi is fixed, so they are cached)."""

    def __init__(self, B, dev, cap=(N_ATT + 1) * N_DEC):
        self.B, self.dev, self.cap, self.n = B, dev, cap, 0
        self.t = {k: torch.zeros((B, cap) + shp, dtype=dt, device=dev) for k, (shp, dt) in FIELDS.items()}

    def append_all(self, e):
        """e: dict of (B, ...) tensors -- one real event per sequence."""
        for k in FIELDS:
            self.t[k][:, self.n] = e[k].to(self.t[k].dtype)
        self.n += 1

    def n_valid(self):
        return torch.full((self.B,), self.n, dtype=torch.long, device=self.dev)

    def view(self, lo=0, hi=None):
        hi = self.n if hi is None else hi
        return {k: v[:, lo:hi] for k, v in self.t.items()}

    def cpu_events(self):
        return {k: v[:, :self.n].cpu() for k, v in self.t.items()}


class Controller:
    """Routes policy.sample_actions: RDM mean + Gaussian exploration; logs decisions; writes H after execution."""

    def __init__(self, model, flow, sigma, dev, deterministic=False):
        self.model, self.flow, self.sigma, self.dev = model, flow, sigma, dev
        self.deterministic = deterministic

    def start(self, B, bank=None):
        self.B = B
        self.bank = bank or Bank(B, self.dev)
        self.tok = None  # cached event tokens (B, cap, 9, d) under the fixed rollout psi
        self.ttt = getattr(self.model, "variant", None) == "ttt_info"
        self.state = self.model.init_state(B, requires_grad=True) if self.ttt else None
        self.records = [[] for _ in range(B)]       # policy records (training data)
        self.pending = None                          # decision awaiting its outcome
        self.prop = None
        self.attempt, self.first_of_attempt, self.dstep = 0, True, 0
        self.write = True

    def finalize(self, encm_next, prop_next, steps_done):
        """Close the pending decision: its prefix has executed; append the real event to H."""
        if self.pending is None:
            return
        p = self.pending
        n_exec = int(min(EXEC, steps_done))
        B, dev = self.B, self.dev
        amask = torch.zeros(B, EXEC, device=dev)
        amask[:, :n_exec] = 1
        full = lambda v: torch.full((B,), int(v), dtype=torch.long, device=dev)
        e = {"pre": p["encm"], "post": encm_next, "act": p["exec"], "amask": amask,
             "p_pre": torch.as_tensor(p["prop"], device=dev), "p_post": torch.as_tensor(prop_next, device=dev),
             "attempt": full(p["attempt"]), "dstep": full(p["dstep"]), "reset": full(p["first"])}
        if self.write:
            n = self.bank.n
            self.bank.append_all(e)
            if self.ttt:  # fast-weight write of the real event (one inner step)
                self.state = self.model.write(self.state, {k: self.bank.t[k][:, n] for k in e}, create_graph=False)
                self.pending = None
                return
            one = self.model.event_tokens({k: v[:, None] for k, v in e.items()})
            if one is not None:
                if self.tok is None:
                    self.tok = torch.zeros(B, self.bank.cap, *one.shape[2:], device=dev)
                self.tok[:, n] = one[:, 0]
        self.pending = None

    @torch.no_grad()
    def decide(self, encm):
        B = encm.shape[0]
        n_valid = self.bank.n_valid()
        tok = self.tok[:, :self.bank.n] if (self.tok is not None and self.bank.n > 0) else None
        eps = torch.randn(B, self.flow.chunk, self.flow.adim, device=self.dev)
        z = self.model.mean(encm, eps, self.state) if self.ttt else self.model.mean(encm, eps, tok, n_valid)
        mu = z[:, :EXEC]
        u = mu if self.deterministic else sample_action(mu, self.sigma)
        logp = policy_logp(u, mu, self.sigma) if not self.deterministic else torch.zeros(B, device=self.dev)
        ex = u.clamp(-1, 1)
        chunk = torch.cat([ex, z[:, EXEC:].clamp(-1, 1)], 1)       # suffix never executed
        for b in range(B):
            self.records[b].append({"encm": encm[b].half().cpu(), "eps": eps[b].cpu(), "u": u[b].cpu(), "mu": mu[b].cpu(),
                                    "logp": float(logp[b]), "n_hist": int(n_valid[b]), "attempt": self.attempt,
                                    "dstep": self.dstep, "reward": 0.0})
        self.pending = {"encm": encm, "exec": ex, "prop": self.prop, "attempt": self.attempt, "dstep": self.dstep,
                        "first": self.first_of_attempt}
        self.first_of_attempt = False
        self.dstep += 1
        return chunk


def install(policy, flow, ctrl):
    def sample_actions(self_, data):
        with torch.no_grad():
            encm = flow.encode(data).half().float()  # stored as fp16: use exactly the stored values online
            if ctrl.pending is not None:  # previous prefix finished executing within this attempt
                ctrl.finalize(encm, ctrl.prop, EXEC)
            return ctrl.decide(encm).cpu().numpy()

    policy.sample_actions = types.MethodType(sample_actions, policy)
    policy.temporal_agg, policy.action_horizon, policy.batch_size = False, EXEC, None


def encode_obs(policy, flow, obs, task, task_emb):
    batch = policy._make_batch(copy.deepcopy(obs), task, **task_emb)
    return flow.encode(batch).half().float()


def run_metaepisodes(runner, policy, flow, ctrl, task, init_ids, probe_ids=None, n_att=N_ATT, log=None):
    """Returns (records per sequence with rewards, per-attempt success (B, n_att), probe success or None)."""
    B = len(init_ids)
    env_fn = lambda: lw.LiberoFrameStack(runner.env_factory(task_id=task, benchmark=runner.benchmark), 1)
    env = lw.LiberoVectorWrapper(env_fn, B)
    all_inits = runner.benchmark.get_task_init_states(task)
    task_emb = {k: v.repeat(B, 1) for k, v in runner.benchmark.get_task_emb(task).items()}
    ctrl.start(B)
    succ = np.zeros((B, n_att), dtype=bool)
    probe = None
    t0 = time.time()
    try:
        plan = [(a, init_ids) for a in range(n_att)] + ([("probe", probe_ids)] if probe_ids is not None else [])
        for a, ids in plan:
            if a == "probe":  # fresh held-out state; H is read but never written (the probe cannot modify it)
                ctrl.write = False
                ctrl.attempt = N_ATT - 1  # attempt embedding saturates at the last position
            else:
                ctrl.attempt = a
            ctrl.first_of_attempt, ctrl.dstep = True, 0
            obs, _ = env.reset(init_states=all_inits[np.asarray(ids)])
            policy.reset()
            got = np.zeros(B, dtype=bool)
            for step in range(HORIZON):
                ctrl.prop = prop_of(obs)
                act = policy.get_action(obs, task, **task_emb)
                obs, _, _, _, info = env.step(act)
                for b in range(B):
                    if info[b]["success"] and not got[b]:
                        got[b] = True
                        if a != "probe":
                            ctrl.records[b][-1]["reward"] = 1.0   # the decision whose prefix reached success
            # attempt ends after a (possibly partial) prefix: close it with the real final observation
            steps_last = HORIZON - (ctrl.dstep - 1) * EXEC
            ctrl.finalize(encode_obs(policy, flow, obs, task, task_emb), prop_of(obs), steps_last)
            if a == "probe":
                probe = got.copy()
            else:
                succ[:, a] = got
            if log:
                log(f"task {task} attempt {a}: {got.mean():.2f} ({time.time() - t0:.0f}s)")
    finally:
        try:
            env._env.close()
        except Exception:  # noqa: BLE001
            pass
    recs = [r[: n_att * N_DEC] for r in ctrl.records]  # probe decisions are not training data
    return recs, ctrl.bank.cpu_events(), succ, probe
