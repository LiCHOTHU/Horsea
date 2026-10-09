"""Export the four inspectable mechanism traces required before robot studies."""
import argparse
from pathlib import Path

from .events import Event, Option, Prefix, Tail
from .memory import StructuredMemory
from .protocol import atomic_json
from .selectors import select, voi


def event_trace(event):
    memory = StructuredMemory()
    before = memory.snapshot()
    memory.update(event)
    return {"source": "categorical unit fixture, not a robot outcome", "event": event.record(),
            "before": before, "after": memory.snapshot(),
            "scores_after": memory.scores().tolist(), "next_option_index": select(memory, "greedy").index}


def export(out):
    out = Path(out)
    for name, event in (
        ("1_prefix_failure", Event(Option(0, 0), Prefix.NOT_READY)),
        ("2_ready_unhelpful_tail", Event(Option(0, 0), Prefix.READY, Tail.AWAY)),
        ("3_unknown", Event(Option(0, 0), Prefix.UNKNOWN)),
    ):
        atomic_json(out / (name + ".json"), event_trace(event))
    m = StructuredMemory(2, 1, alpha=[[.01, 100, .01, .01], [.01, 1, .01, .01]],
                         gamma=[[[20, 20, 20, 20, 100, 20]], [[.05, .05, .05, .05, .15, .05]]])
    atomic_json(out / "4_informative_probe.json", {
        "source": "categorical unit fixture, not a robot outcome", "memory": m.snapshot(),
        "immediate_scores": m.scores().tolist(), "voi": voi(m).tolist(),
        "greedy_option_index": select(m, "greedy").index,
        "probe_option_index": select(m, "voi").index,
        "single_final_option_voi": voi(StructuredMemory(1, 1)).tolist()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    export(p.parse_args().out)
