"""Astra (System 2) as used by the transfer study: the existing CodexAstra integration, unchanged, behind the global
call ledger. Every call is charged before it is made; a refused charge ends the episode as budget-limited, never as an
infrastructure failure and never as a task failure.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from codex_astra import CodexAstra  # noqa: E402
from run import Budget  # noqa: E402


class LedgeredAstra:
    """Same interface as CodexAstra.act(decision_dir, images, prompt); charges the ledger first."""

    def __init__(self, ledger, platform, cell, schema_fn=None, model="gpt-6-astra", reasoning="medium", timeout=180.0):
        self.inner = CodexAstra(model, reasoning, timeout, schema_fn=schema_fn) if schema_fn else CodexAstra(model, reasoning, timeout)
        self.ledger, self.platform, self.cell = ledger, platform, cell
        self.episode, self.decision = None, 0
        self.binary = self.inner.binary

    def act(self, decision_dir, images, prompt):
        if not self.ledger.charge(self.platform, self.cell, self.episode, self.decision):
            return None, {"outcome": "budget_refused", "requested_model": self.inner.model, "latency_s": 0.0}
        raw, meta = self.inner.act(decision_dir, images, prompt)
        self.ledger.record_usage(self.platform, self.cell, self.episode, meta.get("usage"), meta.get("latency_s"), meta.get("outcome"))
        self.decision += 1
        return raw, meta


class LedgerBudget(Budget):
    """Per-episode call ceiling (the pilots' 40) that also respects the global / per-platform ledger."""

    def __init__(self, ledger, platform, calls_per_episode=40):
        super().__init__(1, calls_per_episode, 10 ** 9)
        self.ledger, self.platform = ledger, platform

    def can_call(self):
        return super().can_call() and self.ledger.can_call(self.platform)

    def view(self, attempt):
        v = super().view(attempt)
        v["model_calls_remaining_total"] = min(v["model_calls_remaining_this_attempt"],
                                               self.ledger.calls_per_platform - self.ledger.used(self.platform))
        return v


# ---------------------------------------------------------------------- teacher streams (Astra's own memory across episodes)
import copy  # noqa: E402
import json  # noqa: E402

from memory import History  # noqa: E402


def load_stream(path):
    """Astra's explicit history for one teacher stream (source collection, adaptation demos, E01 reference), persisted between
    worker activities so that the stream behaves like one continuous run of the existing harness."""
    h = History()
    p = Path(path)
    if p.exists():
        d = json.loads(p.read_text())
        h.interactions = d["interactions"]
        h.attempt_summaries = d["attempt_summaries"]
        h.running = d["running"]
        h.next_id = d["next_id"]
    return h


def save_stream(path, h):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"interactions": h.interactions, "attempt_summaries": h.attempt_summaries,
                               "running": h.running, "next_id": h.next_id}))
    tmp.replace(p)


def fresh_history():
    """Hybrid takeovers (E08 / E09 / corrections) start from an explicitly empty history: no hidden student or teacher context."""
    return History()


def student_history_package(records, k=3):
    """E09: the last k student interaction records before handover, as plain data (observations summarized as robot state,
    executed actions, subsequent observations); images of these records are attached separately by the platform adapter."""
    return copy.deepcopy(records[-k:])
