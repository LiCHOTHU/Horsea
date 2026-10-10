"""Per-episode synchronized transition recorder (platform-independent).

Records, at the student's control rate, the pre-action observation o_t (student-native channels), the executed native
command a_t^exec, the controller source ('astra' | 'student'), the native step index and terminal status. o_{t+1} is the next
row's observation; the final post-episode observation is stored as `obs_final_*`. Rejected commands and unexecuted chunk tails
are never written (the adapters call `add` only after the command was dispatched to the simulator).
"""
import json
import time
from pathlib import Path

import numpy as np

SOURCES = {"astra": 0, "student": 1}


class EpisodeRecorder:
    def __init__(self, out_dir, meta):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta, started=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.obs, self.act, self.src, self.step, self.t = {}, [], [], [], []
        self.final_obs = None
        self.trace = []
        self.t0 = time.time()

    def add(self, obs, action, source, step):
        for k, v in obs.items():
            self.obs.setdefault(k, []).append(np.asarray(v))
        self.act.append(np.asarray(action, dtype=np.float32))
        self.src.append(SOURCES[source])
        self.step.append(int(step))
        self.t.append(time.time() - self.t0)

    def note(self, objective, controller, action, result, next_decision):
        """Concise trace line: objective -> controller -> executed action -> observed result -> next decision."""
        self.trace.append({"objective": objective, "controller": controller, "action": action, "result": result, "next": next_decision})

    def close(self, final_obs, success, terminal_status, **extra):
        self.final_obs = final_obs
        arrays = {f"obs_{k}": np.stack(v) for k, v in self.obs.items()} if self.act else {}
        if final_obs is not None:
            arrays.update({f"obs_final_{k}": np.asarray(v) for k, v in final_obs.items()})
        arrays.update(action=np.stack(self.act) if self.act else np.zeros((0, 0), np.float32), source=np.asarray(self.src, np.int8),
                      step=np.asarray(self.step, np.int32), wall=np.asarray(self.t, np.float32))
        np.savez_compressed(self.dir / "transitions.npz", **arrays)
        self.meta.update(extra, success=bool(success), terminal_status=terminal_status, native_steps=len(self.act),
                         astra_steps=int(sum(1 for s in self.src if s == 0)), student_steps=int(sum(1 for s in self.src if s == 1)),
                         wall_s=round(time.time() - self.t0, 1), trace=self.trace, ended=time.strftime("%Y-%m-%d %H:%M:%S"))
        (self.dir / "episode.json").write_text(json.dumps(self.meta, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        return self.meta


def load_episode(path):
    path = Path(path)
    meta = json.loads((path / "episode.json").read_text())
    data = dict(np.load(path / "transitions.npz"))
    return meta, data


def chunk_anchors(source, horizon, teacher_only=True):
    """Valid chunk anchors: every step t whose next `horizon` executed commands are contiguous within the episode and (for
    teacher data) all teacher-controlled. Returns anchor indices and a per-anchor valid mask of length horizon (the tail past
    the episode end or past a controller switch is masked, never filled with the other controller's actions)."""
    src = np.asarray(source)
    T = len(src)
    anchors, masks = [], []
    for t in range(T):
        if teacher_only and src[t] != 0:
            continue
        m = np.zeros(horizon, bool)
        for k in range(horizon):
            if t + k < T and (not teacher_only or src[t + k] == 0):
                m[k] = True
            else:
                break
        anchors.append(t)
        masks.append(m)
    return np.asarray(anchors, np.int64), np.asarray(masks, bool)
