import hashlib
import json
import os
import platform
import subprocess
import time

import numpy as np
import torch


def git_rev():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def cfg_hash(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def file_hash(path, n=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:12]


def versions():
    out = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "node": platform.node()}
    try:
        import mani_skill, sapien, gymnasium
        out.update(mani_skill=mani_skill.__version__, sapien=sapien.__version__,
                   gymnasium=gymnasium.__version__)
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


def atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.cpu().tolist()
    return str(o)


def rng_state():
    return {"torch": torch.get_rng_state(), "np": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["np"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def wall(self):
        return time.time() - self.t0


def gpu_guard():
    """Fail fast on allocation: nodes with ECC faults report is_available()=True but cannot allocate."""
    x = torch.zeros(1024, 1024, device="cuda")
    y = (x + 1).sum().item()
    assert y == 1024 * 1024, y
    torch.cuda.synchronize()
