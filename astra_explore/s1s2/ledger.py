"""Global Astra-call ledger and wall-clock window for the overnight study.

Append-only JSONL so concurrent readers and crashes never lose counts. Every Astra decision call (including retries and
failed calls) is charged here BEFORE the call is made; the per-platform reservation and the global cap are enforced at the
same point. Existing records from earlier runs are never charged; their acquisition cost is reported separately.
"""
import json
import os
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


class Ledger:
    def __init__(self, root, calls_global=600, calls_per_platform=150, window_hours=8.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "astra_ledger.jsonl"
        self.window_path = self.root / "window.json"
        self.calls_global, self.calls_per_platform = calls_global, calls_per_platform
        if not self.window_path.exists():
            self.window_path.write_text(json.dumps({"start": time.time(), "hours": window_hours,
                                                    "start_text": time.strftime("%Y-%m-%d %H:%M:%S")}))
        self.window = json.loads(self.window_path.read_text())

    # ------------------------------------------------------------------ wall clock
    def deadline(self):
        return self.window["start"] + 3600.0 * self.window["hours"]

    def seconds_left(self):
        return self.deadline() - time.time()

    # ------------------------------------------------------------------ calls
    def _events(self):
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def used(self, platform=None):
        ev = self._events()
        return sum(1 for e in ev if e.get("kind") == "call" and (platform is None or e.get("platform") == platform))

    def tokens(self, platform=None):
        tot = {}
        for e in self._events():
            if e.get("kind") == "usage" and (platform is None or e.get("platform") == platform):
                for k, v in (e.get("usage") or {}).items():
                    if isinstance(v, (int, float)):
                        tot[k] = tot.get(k, 0) + v
        return tot

    def can_call(self, platform, n=1):
        return self.used() + n <= self.calls_global and self.used(platform) + n <= self.calls_per_platform

    def charge(self, platform, cell, episode, decision):
        """Reserve one call; returns False (and charges nothing) if a cap would be exceeded."""
        with open(self.path, "a") as f:
            if fcntl:
                fcntl.flock(f, fcntl.LOCK_EX)
            if not self.can_call(platform):
                return False
            f.write(json.dumps({"kind": "call", "t": time.time(), "platform": platform, "cell": cell, "episode": episode,
                                "decision": decision}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return True

    def record_usage(self, platform, cell, episode, usage, latency_s, outcome):
        with open(self.path, "a") as f:
            f.write(json.dumps({"kind": "usage", "t": time.time(), "platform": platform, "cell": cell, "episode": episode,
                                "usage": usage or {}, "latency_s": latency_s, "outcome": outcome}) + "\n")

    def summary(self):
        ev = self._events()
        by = {}
        for e in ev:
            if e.get("kind") == "call":
                key = (e["platform"], e.get("cell"))
                by[key] = by.get(key, 0) + 1
        return {"global_used": self.used(), "global_cap": self.calls_global,
                "per_platform": {p: self.used(p) for p in {e.get("platform") for e in ev if e.get("platform")}},
                "per_platform_cap": self.calls_per_platform,
                "per_cell": {f"{p}/{c}": n for (p, c), n in sorted(by.items())},
                "tokens": self.tokens(), "seconds_left": round(self.seconds_left()),
                "window_start": self.window["start_text"], "window_hours": self.window["hours"]}
